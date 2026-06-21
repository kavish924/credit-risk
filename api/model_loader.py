import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MLFLOW_URI, MODEL_PATH  # noqa: E402

logger = logging.getLogger(__name__)

MODELS_DIR = PROJECT_ROOT / (MODEL_PATH or "models")
LOCAL_MODEL_PATH = MODELS_DIR / "best_model.pkl"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PREPROCESSOR_PATH = PROCESSED_DIR / "preprocessor.pkl"

REGISTERED_MODEL_NAME = "credit-risk-model"
MODEL_STAGE = "Production"


class ModelWrapper:
    def __init__(self, model, version: str, name: str, preprocessor=None):
        self.model = model
        self.version = version
        self.name = name
        self.preprocessor = preprocessor

        self._num_cols = []
        self._cat_cols = []
        if preprocessor is not None:
            for name_t, _, cols in preprocessor.transformers:
                if name_t == "num":
                    self._num_cols = list(cols)
                elif name_t == "cat":
                    self._cat_cols = list(cols)
            logger.info(
                f"Preprocessor expects {len(self._num_cols)} numeric + " f"{len(self._cat_cols)} categorical columns"
            )

    def _align_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        all_expected = self._num_cols + self._cat_cols
        for col in all_expected:
            if col not in df.columns:
                df[col] = np.nan
        return df

    def predict_proba(self, X):
        if self.preprocessor is not None:
            if isinstance(X, pd.DataFrame):
                X = self._align_columns(X)
            X = self.preprocessor.transform(X)
        return self.model.predict_proba(X)[:, 1]


def _load_preprocessor():
    """Load the saved sklearn preprocessor pipeline."""
    if PREPROCESSOR_PATH.exists():
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        logger.info(f"✅ Loaded preprocessor from {PREPROCESSOR_PATH}")
        return preprocessor
    else:
        logger.warning(f"⚠️ Preprocessor not found at {PREPROCESSOR_PATH}")
        return None


def load_from_mlflow() -> ModelWrapper:
    import mlflow

    tracking_uri = MLFLOW_URI or "http://127.0.0.1:5000"
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    champion = next(
        (v for v in sorted(versions, key=lambda v: int(v.version), reverse=True) if v.tags.get("champion") == "true"),
        None,
    )

    if champion is None and versions:
        champion = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]

    if champion is None:
        raise RuntimeError("No model versions found in MLflow registry.")

    model_uri = f"models:/{REGISTERED_MODEL_NAME}/{champion.version}"
    logger.info(f"Loading model from MLflow: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)
    model_name = type(model).__name__

    preprocessor = _load_preprocessor()

    logger.info(f"✅ Loaded {model_name} v{champion.version} from MLflow")
    return ModelWrapper(model=model, version=str(champion.version), name=model_name, preprocessor=preprocessor)


def load_from_local() -> ModelWrapper:
    if not LOCAL_MODEL_PATH.exists():
        raise FileNotFoundError(f"Local model not found at {LOCAL_MODEL_PATH}. " "Run: python -m src.ml.train")
    raw = joblib.load(LOCAL_MODEL_PATH)

    if isinstance(raw, dict) and "model" in raw:
        model = raw["model"]
        threshold = raw.get("threshold", 0.5)
        logger.info(f"Loaded threshold={threshold:.4f} from saved model dict")
    else:
        model = raw

    preprocessor = _load_preprocessor()

    model_name = type(model).__name__
    logger.info(f"✅ Loaded {model_name} from local: {LOCAL_MODEL_PATH}")
    return ModelWrapper(model=model, version="local", name=model_name, preprocessor=preprocessor)


def load_model() -> ModelWrapper:
    try:
        return load_from_mlflow()
    except Exception as e:
        logger.warning(f"MLflow load failed ({e}). Falling back to local model.")
        return load_from_local()
