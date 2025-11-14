# backend/services/supabase_client.py
import os
import pandas as pd
from supabase import create_client, Client
import requests
import logging


SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

def _get_client() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def get_user_transactions(user_id: str) -> pd.DataFrame:
    try:
        url = f"{SUPABASE_URL}/rest/v1/transactions"
        headers = {
            "apikey": SUPABASE_SERVICE_ROLE_KEY,
            "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        }
        params = {
            "user_id": f"eq.{user_id}",
            "select": "*",
            "order": "date.desc",
            "limit": 5000,
        }

        resp = requests.get(url, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()
        print(f"🔹 Supabase retornou {len(data)} transações para o user_id={user_id}")

        if not data:
            return pd.DataFrame(columns=["data", "descricao", "valor", "tipo", "currency", "bank_id", "bank_name"])
        
        df = pd.DataFrame(data)
        df.rename(columns={"date": "data", "description": "descricao", "amount": "valor"}, inplace=True)
        df["data"] = pd.to_datetime(df["data"], errors="coerce")

        try:
            url_banks = f"{SUPABASE_URL}/rest/v1/banks"
            banks_resp = requests.get(url_banks, headers=headers, params={"select": "id,name"}).json()
            df_banks = pd.DataFrame(banks_resp)
            if not df_banks.empty:
                df_banks.rename(columns={"id": "bank_id", "name": "bank_name"}, inplace=True)
                df = df.merge(df_banks, on="bank_id", how="left")
                # ADICIONE ISTO:
                print("🔎 DEBUG 1: Colunas após o merge:")
                print(df.columns.tolist())
                print("🔎 DEBUG 2: Amostra de bank_id e bank_name após o merge:")
                print(df[["bank_id", "bank_name"]].head(10).to_string())
                # FIM DO DEBUG
                print(f"🏦 Enriquecido com nomes de {len(df_banks)} bancos.")
            else:
                print("⚠️ Nenhum banco retornado da tabela 'banks'.")
        except Exception as e:
            print(f"⚠️ Falha ao enriquecer nomes de bancos: {e}")


        if "tipo" not in df.columns and "type" in df.columns:
            df["tipo"] = df["type"]

        df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
        df.loc[df["tipo"] == "expense", "valor"] = -df.loc[df["tipo"] == "expense", "valor"]
        df.loc[df["tipo"] == "income", "valor"] = df.loc[df["tipo"] == "income", "valor"].abs()

      
        # Log de controle
        print(f"✅ {len(df)} transações normalizadas para processamento.")
        print("📋 Colunas finais:", list(df.columns))
        print(df[["descricao", "bank_id", "bank_name", "currency"]].head(5))

        return df

    except Exception as e:
        print(f"❌ Erro ao obter transações do Supabase: {e}")
        return pd.DataFrame(columns=["data", "descricao", "valor", "tipo", "currency", "bank_id", "bank_name"])


def get_user_current_balance(user_id: str) -> float:
    supabase = _get_client()
    resp = supabase.rpc("get_accumulated_balance_by_user", {"p_user_id": user_id}).execute()
    print("🔹 RPC get_accumulated_balance_by_user:", resp.data)
    try:
        return float(resp.data or 0.0)
    except Exception:
        return 0.0