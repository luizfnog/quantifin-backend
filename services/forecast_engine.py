# backend/services/forecast_engine.py

import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# =============================
# PARÂMETROS DE CONTROLE
# =============================
MIN_DIAS = 26
MAX_DIAS = 35
VALOR_TOL_BASE = 0.25
MESES_MAX_HISTORICO = 12
PERCENTUAL_MIN_PRESENCA = 0.60
MESES_ATRASO_MAX = 2

# =============================
# 1. Normalização e fingerprint
# =============================
def normalizar_texto(texto: str) -> str:
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    texto = re.sub(r"[^\w\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def fingerprint_descricao(desc: str) -> str:
    """
    Cria fingerprints mais semânticos:
    - diferencia 'transferencia DE' (entrada)
    - de 'transferencia A FAVOR DE' (saída)
    - e identifica padrões de aluguel, recibo, etc.
    """
    desc = normalizar_texto(desc)

    # remove números longos e termos genéricos
    desc = re.sub(r"\d{4,}", " ", desc)
    desc = re.sub(r"\b(ref|mandato|codigo|num|tarjeta|recibo)\b", " ", desc)

    # --- transferências ---
    if "transferencia" in desc:
        # saída (ex: aluguel, pagamentos)
        if "a favor de" in desc:
            match = re.search(r"transferencia a favor de ([a-záéíóúñüç\s]{3,40})", desc)
            if match:
                nome = match.group(1).strip()
                nome = " ".join(nome.split()[:4])
                if "alquiler" in desc:
                    return f"transferencia a favor de {nome} alquiler"
                return f"transferencia a favor de {nome}"
        # entrada (ex: salário)
        elif "de" in desc:
            match = re.search(r"transferencia de ([a-záéíóúñüç\s]{3,40})", desc)
            if match:
                nome = match.group(1).strip()
                nome = " ".join(nome.split()[:4])
                return f"transferencia de {nome}"

    # --- recibos e débitos automáticos ---
    if "recibo" in desc:
        m = re.search(r"recibo ([a-z\s]+)", desc)
        if m:
            nome = " ".join(m.group(1).split()[:3])
            return f"recibo {nome}"

    # fallback
    palavras = desc.split()
    return " ".join(palavras[:5]).strip()

# =============================
# 2. Limite de histórico
# =============================
def limitar_historico(df: pd.DataFrame, meses: int = MESES_MAX_HISTORICO) -> pd.DataFrame:
    if df.empty:
        return df
    max_data = df["data"].max()
    limite = max_data - relativedelta(months=meses)
    return df[df["data"] >= limite].copy()

# =============================
# 3. Função de recorrência base
# =============================
def eh_recorrente(grupo: pd.DataFrame) -> dict | None:
    desc = grupo["descricao"].iloc[0]
    # print(f"\n🔎 Analisando grupo: {desc[:60]} ({len(grupo)} lançamentos)")

    grupo = grupo.sort_values("data")
    tipo_transacao = grupo["tipo"].iloc[0].lower().strip() if "tipo" in grupo.columns else "expense"

    # exige pelo menos 3 ocorrências
    if len(grupo) < 3:
        return None

    diffs = grupo["data"].diff().dt.days.dropna()
    if diffs.empty:
        return None

    mediana_intervalo = diffs.median()
    if not (20 <= mediana_intervalo <= 45):
        return None

    valores = np.abs(grupo["valor"].values)
    media_valor = np.mean(valores)
    variacao = np.std(valores) / media_valor
    if variacao > 0.25:
        return None

    ultima_data = grupo["data"].max()
    if (datetime.today() - ultima_data.to_pydatetime()).days > 90:
        return None

    data_prevista = ultima_data + timedelta(days=int(mediana_intervalo))

    tipo_label = "Receita Fixa" if tipo_transacao == "income" else "Despesa Fixa"

    # 🔹 Ajuste para transferências de aluguel
    descricao_teste = " ".join(grupo["descricao"].head(1).astype(str).tolist()).lower()
    if "transferencia a favor de" in descricao_teste and "alquiler" in descricao_teste:
        tipo_label = "Despesa Fixa (Aluguel)"

    return {
        "descricao_base": grupo["descricao"].iloc[-1][:80],
        "fingerprint": grupo["fingerprint"].iloc[0],
        "qtd_ocorrencias": len(grupo),
        "ultima_data": ultima_data,
        "data_prevista": data_prevista,
        "valor_previsto": round(media_valor, 2),
        "intervalo_tipico_dias": int(mediana_intervalo),
        "tipo_despesa": tipo_label,
    }

# =============================
# 4. Classificação temporal
# =============================
def classificar_despesas_fixas(df_rec: pd.DataFrame, df_original: pd.DataFrame) -> pd.DataFrame:
    if df_rec.empty:
        return df_rec

    data_min = df_original["data"].min().date()
    data_max = df_original["data"].max().date()
    n_meses_total = (data_max.year - data_min.year) * 12 + (data_max.month - data_min.month) + 1

    df_original["ano_mes"] = df_original["data"].dt.to_period("M")
    meses_por_fp = df_original.groupby("fingerprint")["ano_mes"].nunique().to_dict()

    df_rec["meses_ativos"] = df_rec["fingerprint"].map(meses_por_fp).fillna(0).astype(int)
    df_rec["total_meses_extrato"] = n_meses_total
    df_rec["percentual_presenca"] = (df_rec["meses_ativos"] / n_meses_total).round(2)

    hoje = pd.Timestamp(datetime.now().date())

    def definir_status(row):
        if row["data_prevista"] < hoje - relativedelta(months=MESES_ATRASO_MAX):
            return "Inativa"
        elif row["data_prevista"] < hoje:
            return "Em atraso"
        elif (row["data_prevista"] - hoje).days <= 7:
            return "Próximo"
        return "Ativa"

    df_rec["status"] = df_rec.apply(definir_status, axis=1)
    df_rec = df_rec[df_rec["status"] != "Inativa"].copy()
    return df_rec

# =============================
# 5. Detecção principal de despesas
# =============================
def detectar_despesas_recorrentes(df_transacoes: pd.DataFrame):
    required = {"data", "descricao", "valor"}
    if not required.issubset(df_transacoes.columns):
        raise ValueError(f"DataFrame não tem as colunas necessárias: {required}")

    df = df_transacoes.copy()
    if "tipo" in df.columns:
        df = df[df["tipo"].str.lower().isin(["expense", "gasto", "saida"])]
    else:
        print("⚠️ Coluna 'tipo' não encontrada — analisando todas as transações (modo fallback).")

    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df = df.dropna(subset=["data"])
    df = limitar_historico(df, MESES_MAX_HISTORICO)
    df["fingerprint"] = df["descricao"].apply(fingerprint_descricao)

    recorrencias = []
    for fp, grp in df.groupby("fingerprint"):
        rec = eh_recorrente(grp)
        if rec:
            recorrencias.append(rec)

    if not recorrencias:
        colunas = [
            "descricao_base", "fingerprint", "qtd_ocorrencias", "ultima_data",
            "data_prevista", "valor_previsto", "intervalo_tipico_dias",
            "tipo_despesa", "status"
        ]
        return pd.DataFrame(columns=colunas), df

    df_rec = pd.DataFrame(recorrencias)
    df_rec = classificar_despesas_fixas(df_rec, df)

    if "tipo_despesa" not in df_rec.columns:
        df_rec["tipo_despesa"] = "Indefinido"
    if "status" not in df_rec.columns:
        df_rec["status"] = "Desconhecido"

    return df_rec, df

# =============================
# 6. Receitas recorrentes
# =============================
def detectar_receitas_recorrentes(df: pd.DataFrame):
    print("\n🧩 Iniciando análise de receitas recorrentes...")
    if df.empty:
        print("⚠️ Nenhuma transação encontrada.")
        return pd.DataFrame(), df

    print(f"🔹 Total de transações recebidas: {len(df)}")
    print(f"🔹 Colunas disponíveis: {list(df.columns)}")
    
    
    # apenas receitas reais
    df_income = df[df["tipo"].str.lower().isin(["income", "entrada", "receita"])].copy()
    df_income = df_income[~df_income["descricao"].str.lower().str.contains("a favor de")]
    if df_income.empty:
        print("⚠️ Nenhuma linha de receita após filtro de tipo/descrição.")
        return pd.DataFrame(), df_income

    df_income["data"] = pd.to_datetime(df_income["data"], errors="coerce")
    df_income = df_income.dropna(subset=["data", "valor", "descricao"])
    print(f"📆 Após limpeza (data/valor/descricao): {len(df_income)} registros")
     
    df_income["fingerprint"] = df_income["descricao"].apply(fingerprint_descricao)


    print(f"📊 Total de receitas válidas: {len(df_income)}")
    print(df_income["fingerprint"].value_counts().head(10))

    grupos = []
    hoje = pd.Timestamp(datetime.now().date())
    total_meses = (df_income["data"].max().year - df_income["data"].min().year) * 12 + (
        df_income["data"].max().month - df_income["data"].min().month + 1
    )

    for fp, grupo in df_income.groupby("fingerprint"):
        #print(f"\n🧠 Analisando grupo: '{fp}' com {len(grupo)} lançamentos")

        if len(grupo) < 3:
            print("  ⏩ Ignorado (menos de 3 lançamentos)")
            continue

        grupo = grupo.sort_values("data")

        # 🔹 PASSO 1 — Identificar e remover valores anômalos (reembolsos)
        mediana_valor = grupo["valor"].median()
        tolerancia = 0.35  # ±35% em torno da mediana (para permitir pequenas variações salariais)
        lim_inf = mediana_valor * (1 - tolerancia)
        lim_sup = mediana_valor * (1 + tolerancia)

        grupo_filtrado = grupo[(grupo["valor"] >= lim_inf) & (grupo["valor"] <= lim_sup)].copy()
        if len(grupo_filtrado) < 3:
            print("  ⚠️ Removidos reembolsos ou valores fora da faixa, menos de 3 válidos")
            continue

        # 🔹 PASSO 2 — Calcular intervalo entre os lançamentos válidos
        grupo_filtrado["diff_dias"] = grupo_filtrado["data"].diff().dt.days
        intervalo_medio = grupo_filtrado["diff_dias"].median()

        if not (20 <= intervalo_medio <= 45):
            print(f"  ❌ Intervalo médio {intervalo_medio:.1f} dias — fora da faixa mensal (20–45)")
            continue

        # 🔹 PASSO 3 — Calcular estatísticas da receita
        valor_med = grupo_filtrado["valor"].median()
        ultima_data = grupo_filtrado["data"].max()
        proxima_data = ultima_data + relativedelta(months=1)

        # Status
        status = "Ativa"
        if proxima_data < pd.Timestamp(datetime.now().date()):
            status = "Em atraso"
        elif (proxima_data - pd.Timestamp(datetime.now().date())).days <= 7:
            status = "Próximo"

        grupos.append({
            "descricao_base": grupo_filtrado["descricao"].iloc[-1][:100],
            "fingerprint": fp,
            "qtd_ocorrencias": len(grupo_filtrado),
            "ultima_data": ultima_data,
            "data_prevista": proxima_data,
            "valor_previsto": round(valor_med, 2),
            "intervalo_tipico_dias": int(intervalo_medio),
            "tipo_despesa": "Receita Fixa",
            "status": status
        })
        print(f"  ✅ Receita detectada: média={valor_med:.2f}, intervalo={intervalo_medio:.1f} dias, status={status}")
        
    df_rec = pd.DataFrame(grupos)
    if df_rec.empty:
        print("ℹ️ Nenhuma receita fixa detectada.")
    else:
        print(f"💰 Receitas fixas detectadas: {len(df_rec)}")
        print(df_rec[["fingerprint", "valor_previsto", "status"]])

    return df_rec, df_income

# ============================================================
# 🔹 7. Geração da projeção financeira mensal
# ============================================================

def gerar_projecao_financeira_v2(df_rec: pd.DataFrame, saldo_atual: float):
    """
    Gera a projeção financeira mensal com base nas despesas fixas detectadas.
    Retorna saldo projetado = saldo_atual - soma das despesas previstas do mês.
    """
    if df_rec.empty:
        return {
            "saldo_atual": saldo_atual,
            "total_despesas_previstas": 0.0,
            "saldo_liquido_projetado": saldo_atual
        }

    total_previsto = df_rec["valor_previsto"].sum()
    saldo_liquido = saldo_atual - total_previsto

    return {
        "saldo_atual": round(saldo_atual, 2),
        "total_despesas_previstas": round(total_previsto, 2),
        "saldo_liquido_projetado": round(saldo_liquido, 2)
    }


# ============================================================
# 🔹 8. Execução manual (modo teste)
# ============================================================

if __name__ == "__main__":
    # Simulação local com CSV (modo debug)
    caminho_csv = "meu_extrato.csv"
    df_extrato = pd.read_csv(caminho_csv)
    df_extrato["data"] = pd.to_datetime(df_extrato["data"])

    df_rec, _ = detectar_despesas_recorrentes(df_extrato)
    print("Despesas fixas detectadas:")
    print(df_rec[["descricao_base", "valor_previsto", "proxima_data_prevista", "status"]])

    saldo_atual = 2500.00
    projecao = gerar_projecao_financeira_v2(df_rec, saldo_atual)
    print("\nProjeção Financeira:")
    print(projecao)
