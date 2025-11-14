import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================================================
# 🧭 Caminho universal para salvar modelos
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)


# =========================================================
# 🔹 Funções auxiliares
# =========================================================

def _prep_text_series(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str).str.lower()
    s = s.str.replace(r"[^\w\s]", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    return s


def _num_features(df: pd.DataFrame) -> np.ndarray:
    amt = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    d  = pd.to_datetime(df["date"], errors="coerce")
    dow = d.dt.weekday.fillna(0).astype(int)
    mth = d.dt.month.fillna(1).astype(int)
    desc = _prep_text_series(df["description"])
    freq = desc.map(desc.value_counts()).fillna(1).astype(int)
    X = np.vstack([amt.values, dow.values, mth.values, freq.values]).T
    return X


def _select_description(df: pd.DataFrame) -> pd.Series:
    """Seleciona e normaliza a coluna de descrição para o pipeline."""
    if isinstance(df, pd.DataFrame) and "description" in df.columns:
        return _prep_text_series(df["description"])
    elif isinstance(df, pd.Series):
        return _prep_text_series(df)
    else:
        return pd.Series([""], dtype=str)


# =========================================================
# 🤖 Pipeline de Classificação IA
# =========================================================

def _make_pipeline() -> Pipeline:
    text_pipe = Pipeline([
        ("sel", FunctionTransformer(_select_description, validate=False)),
        ("tfidf", TfidfVectorizer(max_features=8000, ngram_range=(1,2))),
    ])
    num_pipe = Pipeline([
        ("sel", FunctionTransformer(_num_features, validate=False)),
        ("scaler", StandardScaler(with_mean=False)),  # compatível com sparse
    ])
    union = FeatureUnion([
        ("txt", text_pipe),
        ("num", num_pipe),
    ])
    clf = LogisticRegression(max_iter=200, n_jobs=None)
    return Pipeline([("feat", union), ("clf", clf)])


# =========================================================
# 💾 Treino / Carga / Inferência
# =========================================================

def _model_path(user_id: str, name="tfidf_logreg_v1") -> str:
    return os.path.join(MODEL_DIR, f"{user_id}_{name}.joblib")

def _model_path_sub(user_id: str, name="subcat_v1"):
    return os.path.join(MODEL_DIR, f"{user_id}_{name}.joblib")
    
def train_or_update(user_id: str, df_hist: pd.DataFrame) -> Tuple[Optional[Pipeline], str]:
    """
    Treina ou atualiza o modelo e sempre exibe métricas quando possível.
    Mesmo com poucos dados, imprime o que conseguir.
    """

    df = df_hist.dropna(subset=["description", "amount", "date", "category_id"]).copy()
    if df.empty or df["category_id"].nunique() < 2:
        print("⚠️ Dados insuficientes para treinar modelo IA.")
        return None, "insufficient_data"

    y = df["category_id"].astype(str)
    pipe = _make_pipeline()

    ### 🔍 CASO 1 — Dados suficientes para treino/teste estratificado
    if len(df) >= 20 and all(y.value_counts() >= 2):

        print("📌 Modo treino COMPLETO (com métricas).")

        df_train, df_test, y_train, y_test = train_test_split(
            df, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        pipe.fit(df_train, y_train)

        y_pred = pipe.predict(df_test)

        # métricas
        acc = accuracy_score(y_test, y_pred)
        cls_report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        print("\n============================")
        print("📊  MÉTRICAS DO MODELO IA")
        print("============================")
        print(f"🔥 Accuracy: {acc * 100:.2f}%\n")
        print("📌 Classification Report:")
        print(cls_report)
        print("🧩 Matriz de Confusão:")
        print(cm)
        print("============================\n")

    ### 🔍 CASO 2 — Poucos dados → Treinar tudo sem split
    else:
        print("📌 Poucos dados — Treinando sem split (sem métricas completas).")
        print(f"📌 Total de itens rotulados: {len(df)}")
        print(df['category_id'].value_counts())

        pipe.fit(df, y)

        print("⚠️ Nenhuma métrica disponível (dados insuficientes).")

    # salvar modelo
    model_path = _model_path(user_id)
    joblib.dump(pipe, model_path)
    print(f"✅ Modelo IA salvo em: {model_path}")

    return pipe, "ok"


def train_subcategory_model(user_id: str, df_hist: pd.DataFrame):
    df = df_hist.dropna(subset=["description", "amount", "date", "subcategory_id"]).copy()
    if df.empty or df["subcategory_id"].nunique() < 2:
        return None, "insufficient_data"

    y = df["subcategory_id"].astype(str)
    pipe = _make_pipeline()
    pipe.fit(df, y)
    joblib.dump(pipe, _model_path_sub(user_id))
    return pipe, "ok"
    

def load_model(user_id: str) -> Optional[Pipeline]:
    path = _model_path(user_id)
    if os.path.exists(path) and os.path.getsize(path) > 100:
        print(f"📦 Carregando modelo IA: {path}")
        return joblib.load(path)
    print(f"⚠️ Nenhum modelo válido encontrado para {user_id}.")
    return None


def predict(user_id: str, df_new: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    pipe = load_model(user_id)
    if pipe is None:
        return np.array([]), np.array([])

    try:
        proba = pipe.predict_proba(df_new)
        y_pred = pipe.classes_[proba.argmax(axis=1)]
        conf   = proba.max(axis=1) * 100.0
        return y_pred, conf
    except Exception as e:
        print(f"⚠️ Falha ao calcular probas: {e}")
        y_pred = pipe.predict(df_new)
        conf   = np.full(shape=(len(y_pred),), fill_value=70.0)
        return y_pred, conf

def predict_subcategory(user_id: str, df_new: pd.DataFrame):
    path = _model_path_sub(user_id)
    if not os.path.exists(path):
        return np.array([]), np.array([])

    pipe = joblib.load(path)
    try:
        proba = pipe.predict_proba(df_new)
        y_pred = pipe.classes_[proba.argmax(axis=1)]
        conf = proba.max(axis=1) * 100.0
        return y_pred, conf
    except:
        y_pred = pipe.predict(df_new)
        conf = np.full(len(y_pred), 60.0)
        return y_pred, conf