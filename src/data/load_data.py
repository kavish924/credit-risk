import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_PATH  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


RAW_DIR = PROJECT_ROOT / (DATA_PATH or "data") / "raw"
TRAIN_FILE = RAW_DIR / "application_train.csv"
TEST_FILE = RAW_DIR / "application_test.csv"

EXPECTED_COLUMNS = ["SK_ID_CURR", "TARGET", "AMT_CREDIT", "AMT_INCOME_TOTAL", "DAYS_EMPLOYED"]
MIN_ROWS = 100_000  # sanity guard — real dataset has ~307k rows


def load_raw(path: Path, label: str = "dataset") -> pd.DataFrame:
    """Load a CSV file and return a DataFrame with basic validation."""
    if not path.exists():
        raise FileNotFoundError(f"❌ {label} not found at: {path}")

    logger.info(f"Loading {label} from {path} ...")
    df = pd.read_csv(path)
    logger.info(f"✅ Loaded {label}: shape={df.shape}")

    # --- Basic schema checks -------------------------------------------------
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        # TARGET is absent in test set — only warn, don't crash
        train_only = {"TARGET"}
        fatal = [c for c in missing_cols if c not in train_only]
        if fatal:
            raise ValueError(f"Missing critical columns in {label}: {fatal}")
        logger.warning(f"Columns absent (expected for test split): {missing_cols}")

    if len(df) < MIN_ROWS:
        logger.warning(f"⚠️  {label} has only {len(df)} rows — expected >= {MIN_ROWS}.")

    target_col = "TARGET"
    if target_col in df.columns:
        dist = df[target_col].value_counts(normalize=True).to_dict()
        logger.info(f"Target distribution: {dist}")

    return df


def load_train() -> pd.DataFrame:
    return load_raw(TRAIN_FILE, label="train")


def load_test() -> pd.DataFrame:
    return load_raw(TEST_FILE, label="test")


if __name__ == "__main__":
    train_df = load_train()
    test_df = load_test()
    print(f"\nTrain shape : {train_df.shape}")
    print(f"Test  shape : {test_df.shape}")
    print(f"Train columns (first 5): {list(train_df.columns[:5])}")
