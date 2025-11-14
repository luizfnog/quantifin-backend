import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime

def gerar_projecao_financeira(df_rec, meses_previstos=6, plotar=True):
    """
    Gera projeção financeira mensal com base no histórico de despesas fixas.
    Usa Prophet para prever a tendência dos próximos meses.
    """

    # Garantir apenas despesas fixas
    df_fixas = df_rec[df_rec["tipo_despesa"] == "Despesa Fixa"].copy()
    if df_fixas.empty:
        print("⚠️ Nenhuma despesa fixa detectada.")
        return None

    # Construir série mensal agregada
    df_fixas["ano_mes"] = pd.to_datetime(df_fixas["ultima_data"]).dt.to_period("M").dt.to_timestamp()
    serie = df_fixas.groupby("ano_mes")["valor_previsto"].sum().reset_index()
    serie = serie.rename(columns={"ano_mes": "ds", "valor_previsto": "y"})

    # Normalizar série para frequência mensal e interpolar valores ausentes
    serie = serie.set_index("ds").asfreq("M").interpolate().reset_index()

    if len(serie) < 3:
        print("⚠️ Histórico insuficiente para projeção (mínimo 3 meses).")
        return None

    # Criar e ajustar modelo Prophet
    modelo = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=1.0
    )
    modelo.fit(serie)

    # Criar datas futuras
    futuro = modelo.make_future_dataframe(periods=meses_previstos, freq="M")
    previsao = modelo.predict(futuro)

    # Merge histórico + previsão simplificada
    df_prev = previsao[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    df_prev = df_prev.merge(serie, on="ds", how="left")

    # Plotar
    if plotar:
        plt.figure(figsize=(10, 5))
        plt.plot(df_prev["ds"], df_prev["yhat"], "--", color="blue", label="Projeção")
        plt.plot(df_prev["ds"], df_prev["y"], color="black", label="Histórico real")
        plt.fill_between(df_prev["ds"], df_prev["yhat_lower"], df_prev["yhat_upper"],
                         color="skyblue", alpha=0.2, label="Intervalo confiança")
        plt.title("📈 Projeção Mensal de Despesas Fixas")
        plt.xlabel("Mês")
        plt.ylabel("Total (€)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Exibir resumo numérico
    resumo = df_prev.tail(meses_previstos)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    resumo["yhat"] = resumo["yhat"].round(2)
    resumo["yhat_lower"] = resumo["yhat_lower"].round(2)
    resumo["yhat_upper"] = resumo["yhat_upper"].round(2)
    print("\n📊 Previsão de Despesas Fixas (Próximos Meses):\n")
    print(resumo.to_string(index=False))

    return resumo
