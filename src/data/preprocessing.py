

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_PATH  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

PROCESSED_DIR = PROJECT_ROOT / (DATA_PATH or "data") / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

DROP_COLS = ["SK_ID_CURR"]
TARGET_COL = "TARGET"
MISSING_THRESHOLD = 0.50  # drop columns with more than 50% missing values

MAX_OHE_CARDINALITY = 10



def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived financial features.
    All operations are safe — fills 0 on division errors.
    """
    df = df.copy()

    # --- 1. Handle the anomalous DAYS_EMPLOYED (positive = data error) -------
    if "DAYS_EMPLOYED" in df.columns:
        # Create a flag for the anomaly (positive values)
        df["DAYS_EMPLOYED_ANOMALY"] = (df["DAYS_EMPLOYED"] > 0).astype(int)
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace({365243: np.nan})

    # --- 2. Debt-to-income ratio ---------------------------------------------
    if "AMT_CREDIT" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
        df["DEBT_TO_INCOME"] = (
            df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
        ).fillna(0)

    # --- 3. Annuity-to-income ratio ------------------------------------------
    if "AMT_ANNUITY" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
        df["ANNUITY_TO_INCOME"] = (
            df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
        ).fillna(0)

    # --- 4. Credit term (months) ---------------------------------------------
    if "AMT_CREDIT" in df.columns and "AMT_ANNUITY" in df.columns:
        df["CREDIT_TERM"] = (
            df["AMT_CREDIT"] / df["AMT_ANNUITY"].replace(0, np.nan)
        ).fillna(0)

    # --- 5. Age in years (DAYS_BIRTH is negative) ----------------------------
    if "DAYS_BIRTH" in df.columns:
        df["AGE_YEARS"] = (-df["DAYS_BIRTH"] / 365).round(1)

    return df


def build_preprocessor(X: pd.DataFrame):
    """
    Build and return an un-fitted sklearn ColumnTransformer for the given DataFrame.
    """
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = [
        c for c in X.select_dtypes(include=["object"]).columns
        if X[c].nunique() < MAX_OHE_CARDINALITY
    ]

    logger.info(f"  Numeric features  : {len(numeric_cols)}")
    logger.info(f"  Categorical (OHE) : {len(categorical_cols)} — {categorical_cols}")

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor, numeric_cols, categorical_cols


def run_preprocessing(
    df: pd.DataFrame,
    fit: bool = True,
    preprocessor=None,
) -> tuple:

    # 1. Drop high-missing columns
    missing_pct = df.isnull().mean()
    cols_to_drop = missing_pct[missing_pct > MISSING_THRESHOLD].index.tolist()
    if cols_to_drop:
        logger.info(f"Dropping {len(cols_to_drop)} columns with >{MISSING_THRESHOLD*100:.0f}% missing")
    df = df.drop(columns=cols_to_drop, errors="ignore")

    # 2. Drop ID columns
    df = df.drop(columns=DROP_COLS, errors="ignore")

    # 3. Separate target
    y = None
    if TARGET_COL in df.columns:
        y = df[TARGET_COL].reset_index(drop=True)   # ✅ FIX — align with numpy array
        df = df.drop(columns=[TARGET_COL])

    # 4. Feature engineering
    logger.info("Running feature engineering...")
    df = engineer_features(df)

    # 5. Build/use preprocessor
    if fit:
        logger.info("Fitting preprocessor...")
        preprocessor, numeric_cols, categorical_cols = build_preprocessor(df)
        X_processed = preprocessor.fit_transform(df)
        feature_names = preprocessor.get_feature_names_out().tolist()
    else:
        if preprocessor is None:
            raise ValueError("preprocessor must be provided when fit=False")
        logger.info("Transforming with existing preprocessor...")
        X_processed = preprocessor.transform(df)
        feature_names = preprocessor.get_feature_names_out().tolist()

    logger.info(f"Processed shape: {X_processed.shape}")
    return X_processed, y, preprocessor, feature_names


def main():
    from sklearn.model_selection import train_test_split
    from src.data.load_data import load_train

    logger.info("=" * 60)
    logger.info("Starting Preprocessing Pipeline")
    logger.info("=" * 60)

    # Load raw data
    df = load_train()

    # Train / val split BEFORE fitting preprocessor (no leakage)
    df_train, df_val = train_test_split(df, test_size=0.2, stratify=df[TARGET_COL], random_state=42)
    logger.info(f"Train split: {df_train.shape} | Val split: {df_val.shape}")

    # Fit on train
    X_train, y_train, preprocessor, feature_names = run_preprocessing(df_train, fit=True)

    # Transform val
    X_val, y_val, _, _ = run_preprocessing(df_val, fit=False, preprocessor=preprocessor)

    # Save artifacts
    joblib.dump(preprocessor, PROCESSED_DIR / "preprocessor.pkl")
    joblib.dump((X_train, y_train), PROCESSED_DIR / "train_processed.pkl")
    joblib.dump((X_val, y_val), PROCESSED_DIR / "val_processed.pkl")
    joblib.dump(feature_names, PROCESSED_DIR / "feature_names.pkl")

    logger.info(f"✅ Saved preprocessor → {PROCESSED_DIR / 'preprocessor.pkl'}")
    logger.info(f"✅ Saved train data  → {PROCESSED_DIR / 'train_processed.pkl'}")
    logger.info(f"✅ Saved val data    → {PROCESSED_DIR / 'val_processed.pkl'}")
    logger.info(f"✅ Feature count     : {len(feature_names)}")

    return preprocessor, feature_names


if __name__ == "__main__":
    main()
