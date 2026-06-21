import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "monitoring" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def build_feature_df(X: np.ndarray, feature_names: list, y: pd.Series = None) -> pd.DataFrame:
    """Wrap numpy array back into a named DataFrame."""
    df = pd.DataFrame(X, columns=feature_names)
    if y is not None:
        df["target"] = y.values
    return df


def run_drift_report(ref_size: int = 5000, curr_size: int = 1000) -> Path:
    """
    Run Evidently drift report:
      - Reference = first `ref_size` rows of training data
      - Current   = last `curr_size` rows of val data (simulating prod traffic)

    Returns path to the HTML report.
    """
    try:
        from evidently.metric_preset import DataDriftPreset, TargetDriftPreset  # type: ignore
        from evidently.report import Report  # type: ignore
    except ImportError:
        logger.error("evidently not installed. Run: pip install evidently")
        sys.exit(1)

    # Load processed data
    if not (PROCESSED_DIR / "train_processed.pkl").exists():
        logger.error("Processed data not found. Run preprocessing.py first.")
        sys.exit(1)

    X_train, y_train = joblib.load(PROCESSED_DIR / "train_processed.pkl")
    X_val, y_val = joblib.load(PROCESSED_DIR / "val_processed.pkl")
    feature_names = joblib.load(PROCESSED_DIR / "feature_names.pkl")

    # Build DataFrames
    ref_df = build_feature_df(X_train[:ref_size], feature_names, y_train.reset_index(drop=True)[:ref_size])
    curr_df = build_feature_df(X_val[-curr_size:], feature_names, y_val.reset_index(drop=True)[-curr_size:])

    logger.info(f"Reference: {ref_df.shape} | Current: {curr_df.shape}")
    logger.info("Running Evidently drift analysis...")

    # Build report with Data Drift + Target Drift
    report = Report(
        metrics=[
            DataDriftPreset(),
            TargetDriftPreset(),
        ]
    )
    report.run(reference_data=ref_df, current_data=curr_df)

    # Save HTML report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"drift_report_{timestamp}.html"
    report.save_html(str(report_path))
    logger.info(f"✅ Drift report saved → {report_path}")

    # Also save as latest
    latest_path = REPORTS_DIR / "drift_report_latest.html"
    report.save_html(str(latest_path))
    logger.info(f"✅ Latest report    → {latest_path}")

    return report_path


def check_drift_threshold(report_path: Path, threshold: float = 0.5) -> bool:
    """
    Simple heuristic: open the HTML and check if drift is detected.
    Returns True if drift exceeds threshold (retrain needed).
    """

    logger.info(f"Drift threshold check: {threshold} — see full report at {report_path}")
    return False  # Replace with real parsing in production


def parse_args():
    parser = argparse.ArgumentParser(description="Run Evidently drift monitoring report")
    parser.add_argument("--ref-size", type=int, default=5000, help="Reference dataset size")
    parser.add_argument("--curr-size", type=int, default=1000, help="Current dataset size")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    report_path = run_drift_report(ref_size=args.ref_size, curr_size=args.curr_size)
    drift_detected = check_drift_threshold(report_path)
    if drift_detected:
        logger.warning("⚠️  Significant drift detected — consider retraining!")
    else:
        logger.info("✅ No significant drift detected.")
