"""
validate.py — Data Validation Layer
Uses Great Expectations to validate schema, distributions, and quality of raw data.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Pure-pandas validation (no GE dependency) — always runs
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_COLUMNS = [
    "SK_ID_CURR",
    "TARGET",
    "AMT_CREDIT",
    "AMT_INCOME_TOTAL",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
]

VALIDATION_RULES = {
    "min_rows": 100_000,
    "target_values": {0, 1},
    "credit_min": 0,
    "income_min": 0,
}


def validate_basic(df: pd.DataFrame, label: str = "dataset") -> dict:
    """
    Run rule-based validation checks on the DataFrame.
    Returns a dict of { check_name: passed (bool) }.
    """
    results = {}

    # 1. Minimum rows
    results["min_rows"] = len(df) >= VALIDATION_RULES["min_rows"]

    # 2. Required columns present
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    results["required_columns"] = len(missing) == 0
    if missing:
        logger.warning(f"Missing columns: {missing}")

    # 3. TARGET is binary
    if "TARGET" in df.columns:
        actual_vals = set(df["TARGET"].dropna().unique())
        results["target_binary"] = actual_vals.issubset(VALIDATION_RULES["target_values"])

    # 4. No duplicates on SK_ID_CURR
    if "SK_ID_CURR" in df.columns:
        results["no_duplicate_ids"] = df["SK_ID_CURR"].nunique() == len(df)

    # 5. AMT_CREDIT positive
    if "AMT_CREDIT" in df.columns:
        results["credit_positive"] = (df["AMT_CREDIT"] >= 0).all()

    # 6. AMT_INCOME_TOTAL positive
    if "AMT_INCOME_TOTAL" in df.columns:
        results["income_positive"] = (df["AMT_INCOME_TOTAL"] >= 0).all()

    # 7. Missing-value summary
    high_missing = (df.isnull().mean() > 0.8).sum()
    results["no_catastrophic_missing"] = high_missing == 0

    # ── Print summary ──────────────────────────────────────────────────────
    passed = sum(results.values())
    total = len(results)
    status = "✅ PASSED" if passed == total else f"⚠️  {total - passed}/{total} checks FAILED"
    logger.info(f"\n{'='*50}")
    logger.info(f"Validation report for '{label}': {status}")
    for check, ok in results.items():
        icon = "✅" if ok else "❌"
        logger.info(f"  {icon}  {check}")
    logger.info(f"{'='*50}\n")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Great Expectations validation (optional — degrades gracefully)
# ─────────────────────────────────────────────────────────────────────────────


def validate_with_ge(df: pd.DataFrame) -> bool:
    """
    Run Great Expectations suite on the DataFrame.
    Returns True if all expectations pass, False otherwise.
    Falls back gracefully if GE is not installed.
    """
    try:
        import great_expectations as ge  # type: ignore

        ge_df = ge.from_pandas(df)

        expectations = [
            ge_df.expect_column_to_exist("TARGET"),
            ge_df.expect_column_values_to_be_in_set("TARGET", [0, 1]),
            ge_df.expect_column_to_exist("AMT_CREDIT"),
            ge_df.expect_column_values_to_be_between("AMT_CREDIT", min_value=0),
            ge_df.expect_column_to_exist("AMT_INCOME_TOTAL"),
            ge_df.expect_column_values_to_be_between("AMT_INCOME_TOTAL", min_value=0),
            ge_df.expect_column_to_exist("SK_ID_CURR"),
            ge_df.expect_column_values_to_be_unique("SK_ID_CURR"),
            ge_df.expect_table_row_count_to_be_between(min_value=100_000, max_value=400_000),
        ]

        passed = all(e["success"] for e in expectations)
        logger.info(f"Great Expectations: {'✅ All checks passed' if passed else '❌ Some checks failed'}")
        return passed

    except ImportError:
        logger.warning("great-expectations not installed — skipping GE validation.")
        return True  # non-fatal
    except Exception as exc:
        logger.error(f"GE validation error: {exc}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main():
    from src.data.load_data import load_train

    df = load_train()
    basic_results = validate_basic(df, label="application_train")
    ge_ok = validate_with_ge(df)

    all_pass = all(basic_results.values()) and ge_ok
    if not all_pass:
        logger.error("❌ Data validation FAILED — aborting pipeline.")
        sys.exit(1)
    logger.info("✅ All validation checks passed.")


if __name__ == "__main__":
    main()
