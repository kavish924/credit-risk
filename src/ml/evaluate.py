
import logging
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "monitoring" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate(model_path: Path = MODELS_DIR / "best_model.pkl"):
    """Full evaluation: metrics + ROC curve + confusion matrix + SHAP."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run train.py first.")

    model = joblib.load(model_path)
    X_val, y_val = joblib.load(PROCESSED_DIR / "val_processed.pkl")
    feature_names = joblib.load(PROCESSED_DIR / "feature_names.pkl")

    y_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_val, y_proba)
    ap = average_precision_score(y_val, y_proba)

    print("\n" + "=" * 60)
    print("MODEL EVALUATION REPORT")
    print("=" * 60)
    print(f"  ROC-AUC            : {auc:.4f}")
    print(f"  Avg Precision      : {ap:.4f}")
    print(f"\n{classification_report(y_val, y_pred, zero_division=0)}")

  
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    fpr, tpr, _ = roc_curve(y_val, y_proba)
    axes[0].plot(fpr, tpr, color="#6366f1", lw=2, label=f"ROC AUC = {auc:.4f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve — Credit Risk Model")
    axes[0].legend()

    cm = confusion_matrix(y_val, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["No Default", "Default"]).plot(ax=axes[1])
    axes[1].set_title("Confusion Matrix")

    plt.tight_layout()
    roc_path = REPORTS_DIR / "roc_confusion.png"
    plt.savefig(roc_path, dpi=150)
    logger.info(f"Saved ROC + Confusion Matrix → {roc_path}")
    plt.close()

    try:
        import shap  # type: ignore

        logger.info("Computing SHAP values (this may take a few minutes)...")
        sample_size = min(500, len(X_val))
        X_sample = X_val[:sample_size]

        explainer = shap.Explainer(model, X_sample, feature_names=feature_names)
        shap_values = explainer(X_sample)

        # Summary plot
        plt.figure(figsize=(12, 7))
        shap.summary_plot(
            shap_values, X_sample, feature_names=feature_names,
            show=False, max_display=20,
        )
        shap_path = REPORTS_DIR / "shap_summary.png"
        plt.savefig(shap_path, bbox_inches="tight", dpi=150)
        logger.info(f"Saved SHAP summary → {shap_path}")
        plt.close()

    except ImportError:
        logger.warning("shap not installed — skipping SHAP plots.")
    except Exception as e:
        logger.warning(f"SHAP failed: {e}")

    return {"roc_auc": auc, "avg_precision": ap}


if __name__ == "__main__":
    evaluate()
