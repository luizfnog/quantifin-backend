from fastapi import FastAPI, Depends, HTTPException, Query, Body
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
from datetime import date, datetime
from supabase import create_client
import os
import json
from services.supabase_client import get_user_transactions, get_user_current_balance
from services.forecast_engine import detectar_despesas_recorrentes, gerar_projecao_financeira_v2, detectar_receitas_recorrentes
from services.persistencia import upsert_fixed_expense_forecasts
from services.currency import get_rates_from_db_or_default, convert_amount
from services.classifier import (
    train_or_update,
    train_subcategory_model,
    predict,
    predict_subcategory,
    load_model,
)
from services.supabase_client import _get_client as get_client
from fastapi.middleware.cors import CORSMiddleware

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

app = FastAPI(
    title="Quantifin-AI Predictive Engine",
    description="API de previsão de despesas e receitas fixas (integrada ao Supabase)",
    version="1.4.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://finpilot-nu.vercel.app",   # domínio do frontend (produção)
        "http://localhost:5173",            # desenvolvimento React/Vite::
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "https://sessional-supercapably-preston.ngrok-free.dev",
    ],
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

class ForecastItem(BaseModel):
    descricao_base: str
    fingerprint: str
    valor_previsto: float
    data_prevista: date
    status: str
    tipo_despesa: str
    currency: Optional[str] = None
    bank_id: Optional[str] = None
    bank_name: Optional[str] = None
    
class BankForecast(BaseModel):
    bank_id: str
    bank_name: str
    currency: str
    saldo_atual: float
    total_despesas_previstas: float
    total_receitas_previstas: float
    saldo_liquido_projetado: float
    despesas_fixas: List[Dict[str, Any]]
    receitas_fixas: List[Dict[str, Any]]

class ForecastResponse(BaseModel):
    user_id: str
    banks: List[BankForecast]

@app.post("/api/fixed-expense-forecast", response_model=ForecastResponse)
@app.post("/fixed-expense-forecast", response_model=ForecastResponse)  # compatibilidade
async def fixed_expense_forecast(
    user_id: str = Query(..., description="UUID do usuário autenticado"),
):
    print(f"\n🚀 Iniciando análise preditiva para o usuário: {user_id}")

    # 1) carrega transações (para o motor de recorrência)
    df = get_user_transactions(user_id)
    if df.empty:
        print("⚠️ Nenhuma transação encontrada.")
        return ForecastResponse(user_id=user_id, banks=[])

    # garantias mínimas
    if "bank_id" not in df.columns:
        df["bank_id"] = "DEFAULT"
    if "currency" not in df.columns:
        df["currency"] = "EUR"

    # 2) pega saldo real por banco via RPC
    print("🔄 Chamando RPC get_accumulated_balance_multi_currency...")
    try:
        rpc_res = supabase.rpc(
            "get_accumulated_balance_multi_currency",
            {
                "p_user_id": user_id,
                # se sua função exige, troque aqui a moeda principal
                "p_home_currency": "BRL",
            },
        ).execute()
        rpc_data = rpc_res.data or []
        print(f"✅ RPC retornou {len(rpc_data)} linhas")
        print(f"📝 RPC bruto: {rpc_data}")
    except Exception as e:
        print(f"❌ Erro na RPC: {e}")
        rpc_data = []

    # dicionário por bank_id (uuid) -> saldo convertido (home currency)
    saldos_por_bank_id = {}

    for item in rpc_data:
        if item.get("breakdown"):
            for sub in item["breakdown"]:
                bid = str(sub.get("bank_id"))
                bal = float(sub.get("raw_balance", 0.0))
                saldos_por_bank_id[bid] = bal
        elif item.get("bank_id") is not None:
            # fallback caso retorne várias linhas diretas
            saldos_por_bank_id[str(item["bank_id"])] = float(item.get("raw_balance", 0.0))

    print("📊 Saldos recebidos do Supabase (por bank_id):")
    for bid, saldo in saldos_por_bank_id.items():
        print(f"   - {bid}: {saldo:.2f}")

    banks_payload: List[BankForecast] = []

    # 3) processa banco a banco
    for bank_group, df_bank in df.groupby("bank_id"):
        print(df_bank.head(10))
        if "bank_id" in df_bank.columns:
            bank_uuid = str(df_bank["bank_id"].iloc[0])
            bank_name = df_bank["bank_name"].iloc[0]
        else:
            import re
            if re.match(r"^[0-9a-fA-F-]{36}$", str(bank_group)):
                bank_uuid = str(bank_group)
                bank_name = (
                    df_bank["bank_name"].iloc[0]
                    if "bank_name" in df_bank.columns and df_bank["bank_name"].notna().any()
                    else bank_uuid
                )
            else:
                bank_uuid = None
                bank_name = str(bank_group)
        print(f"🔎 Identificação banco: uuid={bank_uuid}, name={bank_name}")
        # ✅ Define moeda do banco
        bank_currency = df_bank["currency"].iloc[0] if "currency" in df_bank.columns else "N/A"
        # Saldo atual
        saldo_atual_bank = 0.0
        if bank_uuid and bank_uuid in saldos_por_bank_id:
            saldo_atual_bank = float(saldos_por_bank_id[bank_uuid])
        elif bank_name in saldos_por_bank_id:
            saldo_atual_bank = float(saldos_por_bank_id[bank_name])

        print(f"\n🏦 Banco: {bank_name} ({bank_uuid}) [{bank_currency}]")
        print(f"💰 Saldo atual (RPC): {saldo_atual_bank:.2f}")

        # Analisa recorrências
        df_desp, _ = detectar_despesas_recorrentes(df_bank)
        df_rec, _ = detectar_receitas_recorrentes(df_bank)

        def convert_dataframe_dates_to_iso(df: pd.DataFrame) -> pd.DataFrame:
            """Converte a coluna 'data_prevista' para string ISO 8601."""
            if "data_prevista" in df.columns:
                df["data_prevista"] = df["data_prevista"].apply(
                    lambda d: d.strftime("%Y-%m-%d") if not pd.isna(d) else None
                )
            return df
        
        if not df_desp.empty:
            df_desp = convert_dataframe_dates_to_iso(df_desp) # <--- CONVERSÃO APLICADA
            df_desp["bank_id"] = bank_uuid
            df_desp["bank_name"] = bank_name
            df_desp["currency"] = bank_currency

        if not df_rec.empty:
            df_rec = convert_dataframe_dates_to_iso(df_rec) # <--- CONVERSÃO APLICADA
            df_rec["bank_id"] = bank_uuid
            df_rec["bank_name"] = bank_name
            df_rec["currency"] = bank_currency

        total_desp = float(df_desp["valor_previsto"].sum()) if not df_desp.empty else 0.0
        total_rec = float(df_rec["valor_previsto"].sum()) if not df_rec.empty else 0.0

        print(f"📉 Despesas previstas: {total_desp:.2f}")
        print(f"📈 Receitas previstas: {total_rec:.2f}")

        saldo_liquido_projetado = round(saldo_atual_bank - total_desp + total_rec, 2)
        print(f"🧮 Saldo líquido projetado: {saldo_liquido_projetado:.2f}")

        # 5) persiste do jeito que já estava
        if not df_desp.empty:
            upsert_fixed_expense_forecasts(user_id, df_desp)
        if not df_rec.empty:
            upsert_fixed_expense_forecasts(user_id, df_rec)

        # 6) monta o payload para o frontend

        banks_payload.append(
            BankForecast(
                bank_id=bank_uuid,
                bank_name=bank_name,
                currency=bank_currency,
                saldo_atual=round(saldo_atual_bank, 2),
                total_despesas_previstas=round(total_desp, 2),
                total_receitas_previstas=round(total_rec, 2),
                saldo_liquido_projetado=saldo_liquido_projetado,
                despesas_fixas=df_desp.to_dict(orient="records") if not df_desp.empty else [],
                receitas_fixas=df_rec.to_dict(orient="records") if not df_rec.empty else [],
            )
        )

    print("\n🏁 Debug final — Resumo de todos os bancos:")
    for b in banks_payload:
        print(
            f" {b.bank_name} | {b.currency} | Saldo: {b.saldo_atual:.2f} | "
            f"Despesas: {b.total_despesas_previstas:.2f} | Receitas: {b.total_receitas_previstas:.2f} | "
            f"Saldo projetado: {b.saldo_liquido_projetado:.2f}"
        )
       
    return ForecastResponse(user_id=user_id, banks=banks_payload)


@app.get("/api/all-transactions/{user_id}")
@app.get("/all-transactions/{user_id}")  # mantém compatibilidade
def get_all_transactions(user_id: str):
    print(f"🔍 Recebido user_id: {user_id}")
    try:
        response = (
            supabase.table("transactions")
            .select("*")
            .eq("user_id", user_id)
            .order("date", desc=False)
            .limit("5000")
            .execute()
        )

        # ✅ resposta moderna da SDK: só usa .data
        data = getattr(response, "data", None) or []
        print(f"📊 Supabase retornou {len(data)} registros")

        return {"transactions": data, "count": len(data)}

    except Exception as e:
        print("💥 Erro no endpoint /api/all-transactions:", e)
        return {"error": str(e)}

@app.post("/api/process-transactions")
def process_transactions(
    user_id: str = Query(...),
    payload: Dict[str, Any] = Body(...),
):
    try:
        # TODO: todo o conteúdo atual do endpoint permanece aqui
         pass   # <- evita erro, placeholder válido

    except Exception as e:
        print("\n🔥 ERRO NO process_transactions >>>")
        print(str(e))
        import traceback
        traceback.print_exc()

        # Retorna erro direto pro frontend (mensagem visível)
        raise HTTPException(status_code=500, detail=str(e))

    client = get_client()

    # 1) garantir que EXISTE modelo de CATEGORIA treinado
    hist_cat = (
        client.table("transactions")
        .select("id, description, amount, date, category_id")
        .eq("user_id", user_id)
        .not_.is_("category_id", "null")
        .execute()
        .data
        or []
    )
    df_hist_cat = pd.DataFrame(hist_cat)
    model = load_model(user_id)
    if model is None:
        pipe, status = train_or_update(user_id, df_hist_cat)
        if pipe is None:
            # ainda não tem dado suficiente rotulado
            return {"updated": 0, "reason": "insufficient_training_data"}
        model = pipe  # só pra documentar

    # 2) garantir que EXISTE modelo de SUBCATEGORIA treinado (se tiver dados)
    hist_sub = (
        client.table("transactions")
        .select("id, description, amount, date, category_id, subcategory_id")
        .eq("user_id", user_id)
        .not_.is_("subcategory_id", "null")
        .execute()
        .data
        or []
    )
    df_hist_sub = pd.DataFrame(hist_sub)
    if not df_hist_sub.empty and df_hist_sub["subcategory_id"].nunique() >= 2:
        # treinar/atualizar modelo de subcategoria
        train_subcategory_model(user_id, df_hist_sub)
    # se não tiver dados suficientes, o predict_subcategory vai retornar arrays vazios

    # 3) buscar as transações a classificar
    data_new = (
        client.table("transactions")
        .select("id, description, amount, date, bank_id, category_id, subcategory_id")
        .in_("id", trx_ids)
        .eq("user_id", user_id)
        .execute()
        .data
        or []
    )
    df_new = pd.DataFrame(data_new)
    if df_new.empty:
        return {"updated": 0, "reason": "no_transactions_found"}

    # 4) predição de CATEGORIA
    y_cat, conf_cat = predict(user_id, df_new)
    if len(y_cat) != len(df_new):
        return {"updated": 0, "reason": "prediction_failed_category"}

    # 5) predição de SUBCATEGORIA (podem vir arrays vazios se não houver modelo)
    y_sub, conf_sub = predict_subcategory(user_id, df_new)

    # 6) carregar árvore de categorias do usuário (pai -> filhos)
    cats = (
        client.table("categories")
        .select("id, parent_id")
        .eq("user_id", user_id)
        .execute()
        .data
        or []
    )
    children_by_parent: Dict[str, list[str]] = {}
    for c in cats:
        cid = str(c["id"])
        pid = c["parent_id"]
        if pid is not None:
            pid = str(pid)
            children_by_parent.setdefault(pid, []).append(cid)

    # 7) montar updates garantindo:
    #    - subcategoria NUNCA sai do pai previsto
    #    - se pai não tiver filhos -> subcategory_id = None
    updates: list[Dict[str, Any]] = []

    for idx, row in df_new.reset_index(drop=True).iterrows():
        trx_id = row["id"]
        cat_id = str(y_cat[idx])
        cat_conf = float(conf_cat[idx]) if len(conf_cat) == len(df_new) else 0.0

        # candidatos de subcategoria para ESTE pai
        allowed_children = children_by_parent.get(cat_id, [])

        sub_id_final: Optional[str] = None

        if len(y_sub) == len(df_new):
            raw_sub = str(y_sub[idx])
            # só aceita se estiver na lista de filhos do pai previsto
            if raw_sub in allowed_children:
                sub_id_final = raw_sub
            else:
                # ❗ NUNCA sugerir subcategoria que não pertença ao pai
                sub_id_final = None
        else:
            # sem modelo de subcategoria ou erro: deixa como None
            sub_id_final = None

        status = "approved" if cat_conf >= 85.0 else "pending"

        updates.append(
            {
                "id": trx_id,
                "user_id": user_id,
                "category_id": cat_id,
                "subcategory_id": sub_id_final,
                "ai_confidence": int(round(cat_conf)),  # inteiro, conforme schema
                "ai_status": status,
                "is_reviewed": True if status == "approved" else False,
            }
        )

    # 8) aplicar updates um-a-um (garantindo user_id no filtro)
    for u in updates:
        client.table("transactions").update(
            {
                "category_id": u["category_id"],
                "subcategory_id": u["subcategory_id"],
                "ai_confidence": u["ai_confidence"],
                "ai_status": u["ai_status"],
                "is_reviewed": u["is_reviewed"],
            }
        ).eq("id", u["id"]).eq("user_id", user_id).execute()

    print(f"✅ Atualizadas {len(updates)} transações com IA (categoria + subcategoria)")

    return {
        "updated": len(updates),
        "items": updates,  # cada item traz category_id + subcategory_id previstos
    }
    
@app.post("/api/ai-feedback")
def ai_feedback(
    payload: Dict[str, Any] = Body(...),
    user_id: str = Query(...),
):
    """
    Registra feedback do usuário (approve/correct) e opcionalmente retreina.
    payload: {
      transaction_id, action: 'approve'|'correct',
      final_category_id, final_subcategory_id, suggested_category_id, suggested_subcategory_id,
      confidence
    }
    """
    client = get_client()
    trx_id = payload.get("transaction_id")
    action = payload.get("action")
    if not trx_id or action not in ("approve", "correct"):
        raise HTTPException(400, "payload inválido")

    # 1) atualiza transação conforme ação
    update_row = {
        "id": trx_id,
        "user_id": user_id,  # ← OBRIGATÓRIO
        "is_reviewed": True,
        "ai_status": "approved",
        "ai_confidence": float(payload.get("confidence") or 100.0),
    }
    if payload.get("final_category_id"):
        update_row["category_id"] = payload["final_category_id"]
    if payload.get("final_subcategory_id"):
        update_row["subcategory_id"] = payload["final_subcategory_id"]

    client.table("transactions").update(update_row).eq("id", trx_id).execute()

    # 2) grava evento de feedback supervisionado
    fb = {
        "user_id": user_id,
        "transaction_id": trx_id,
        "suggested_category_id": payload.get("suggested_category_id"),
        "suggested_subcategory_id": payload.get("suggested_subcategory_id"),
        "final_category_id": payload.get("final_category_id"),
        "final_subcategory_id": payload.get("final_subcategory_id"),
        "model_name": "tfidf+logreg",
        "model_version": "v1",
        "confidence": float(payload.get("confidence") or 100.0),
        "action": action
    }
    client.table("transaction_ai_feedback").insert(fb).execute()

    # 3) (opcional) retreinar incrementalmente — aqui faremos simples: re-treina do zero usando histórico do usuário
    hist = client.table("transactions").select(
        "id, description, amount, date, category_id"
    ).eq("user_id", user_id).not_.is_("category_id", "null").execute().data or []
    df_hist = pd.DataFrame(hist)
    train_or_update(user_id, df_hist)

    return {"ok": True}