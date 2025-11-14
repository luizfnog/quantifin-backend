# backend/services/persistencia.py
import pandas as pd
from datetime import datetime
from .supabase_client import _get_client
from pandas.api.types import is_datetime64_any_dtype

def upsert_fixed_expense_forecasts(user_id: str, df_fixas: pd.DataFrame):

    if "data_prevista" in df_fixas.columns and is_datetime64_any_dtype(df_fixas["data_prevista"]):
        df_fixas["data_prevista"] = df_fixas["data_prevista"].dt.strftime("%Y-%m-%d")
        
    # if "data_prevista" in df_fixas.columns:
    #     df_fixas["data_prevista"] = df_fixas["data_prevista"].apply(
    #         lambda d: d.strftime("%Y-%m-%d") if not pd.isna(d) else None
    #     )    
     
    supabase = _get_client()

    rows = []
    total = 0.0
    for _, row in df_fixas.iterrows():
        total += float(row["valor_previsto"])

        # 🧠 conversão segura de data para string ISO
        data_prevista = row["data_prevista"]
        if isinstance(data_prevista, pd.Timestamp):
            data_prevista = data_prevista.strftime("%Y-%m-%d")
        elif isinstance(data_prevista, datetime):
            data_prevista = data_prevista.date().isoformat()

        rows.append({
            "user_id": user_id,
            "fingerprint": row["fingerprint"],
            "descricao_base": row["descricao_base"],
            "valor_previsto": float(row["valor_previsto"]),
            "data_prevista": data_prevista,
            "status": row.get("status", "Ativa"),  # ✅ evita KeyError
            "tipo_despesa": row.get("tipo_despesa", "Despesa Fixa"),
            "bank_name": row.get("bank_name") or row.get("bank_id") or "DEFAULT",
            "currency": row.get("currency") or "EUR",
        })

    if rows:
        supabase.table("financial_forecasts").upsert(
            rows, on_conflict="user_id,fingerprint"
        ).execute()