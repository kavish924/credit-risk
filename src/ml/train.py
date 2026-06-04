

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MLFLOW_URI, MODEL_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

logger = logging.getLogger(__name__)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / (MODEL_PATH or "models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENT_NAME = "credit-risk-classification"
REGISTERED_MODEL_NAME = "credit-risk-model"

SAMPLING_STRATEGY = "smote"   # options: "smote" | "undersample" | "combined" | "none"

def get_models(args, y_train):
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos

    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            C=0.1,
            random_state=42,
            class_weight="balanced",
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=2,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=5,
            gamma=0.2,
            reg_alpha=0.5,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            random_state=42,
            n_jobs=-1,
        ),
    }


def get_resampler(strategy: str):
    """
    Returns a fitted resampler based on strategy.

    Options:
        "smote"       — Oversample minority class using SMOTE
        "undersample" — Undersample majority class randomly
        "combined"    — SMOTE + undersampling (best of both)
        "none"        — No resampling (rely on class_weight only)

    Data profile: ~1:11 imbalance (8% minority)
    Target: maximise recall on class 1 (credit default detection)
    """
    if strategy == "smote":
        return SMOTE(
            sampling_strategy=0.5,   # minority → 50% of majority (1:2 ratio)
            k_neighbors=5,
            random_state=42,
        )
    elif strategy == "undersample":
        return RandomUnderSampler(
            sampling_strategy=0.7,   # minority → 70% of majority (1:1.4 ratio)
            random_state=42,
        )
    elif strategy == "combined":

        return ImbPipeline([
            ("smote", SMOTE(
                sampling_strategy=0.3,
                k_neighbors=5,
                random_state=42,
            )),
            ("under", RandomUnderSampler(
                sampling_strategy=0.7,
                random_state=42,
            )),
        ])
    elif strategy == "none":
        return None
    else:
        raise ValueError(
            f"Unknown sampling strategy: {strategy}. "
            "Choose from: smote, undersample, combined, none"
        )


def apply_resampling(X, y, strategy: str):
    """Apply resampling and log class distribution before/after."""
    logger.info(
        f"Before resampling → "
        f"class 0: {(y==0).sum()} | class 1: {(y==1).sum()}"
    )

    resampler = get_resampler(strategy)

    if resampler is None:
        logger.info("Resampling skipped (strategy=none)")
        return X, y

    X_res, y_res = resampler.fit_resample(X, y)

    logger.info(
        f"After resampling  → "
        f"class 0: {(y_res==0).sum()} | class 1: {(y_res==1).sum()}"
    )

    return X_res, y_res



def find_best_threshold(model, X_val, y_val):
    """
    Find the classification threshold that maximises F1 for class 1
    on the validation set using the Precision-Recall curve.

    Returns the best threshold and its F1 score.
    """
    y_proba = model.predict_proba(X_val)[:, 1]

    precisions, recalls, thresholds = precision_recall_curve(
        y_val, y_proba
    )


    f1_scores = np.where(
        (precisions + recalls) > 0,
        2 * (precisions * recalls) / (precisions + recalls),
        0,
    )

    best_idx = np.argmax(f1_scores[:-1])
    best_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])

    logger.info(
        f"Best threshold = {best_threshold:.4f} "
        f"(F1 class=1: {best_f1:.4f})"
    )

    return best_threshold, best_f1


def evaluate_model(model, X_val, y_val, threshold=None):
    """
    Evaluate model on validation set.
    If threshold is None, uses the optimal threshold found via PR curve.
    """
    y_proba = model.predict_proba(X_val)[:, 1]

    if threshold is None:
        threshold, _ = find_best_threshold(model, X_val, y_val)

    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_val, y_proba),
        "avg_precision": average_precision_score(y_val, y_proba),
        "f1_score": f1_score(y_val, y_pred, zero_division=0),
        "f1_weighted": f1_score(
            y_val, y_pred, average="weighted", zero_division=0
        ),
        "best_threshold": threshold,
    }

    report = classification_report(
        y_val, y_pred, output_dict=True, zero_division=0
    )

    metrics["precision_class1"] = report["1"]["precision"]
    metrics["recall_class1"] = report["1"]["recall"]

    
    logger.info("\n" + classification_report(
        y_val, y_pred, zero_division=0
    ))

    return metrics, threshold


def log_model_to_mlflow(model, model_name, X_sample, use_registry):
    """
    Log model to MLflow using the correct logger per model type.
    XGBoost uses mlflow.xgboost, others use mlflow.sklearn.
    """
    try:
        signature = mlflow.models.infer_signature(
            X_sample,
            model.predict_proba(X_sample),
        )

        kwargs = dict(
            artifact_path="model",
            signature=signature,
        )

        if use_registry:
            kwargs["registered_model_name"] = REGISTERED_MODEL_NAME

        if model_name == "XGBoost":
            mlflow.xgboost.log_model(
                xgb_model=model, **kwargs
            )
        else:
            mlflow.sklearn.log_model(
                sk_model=model, **kwargs
            )

    except Exception as e:
        logger.warning(f"MLflow model logging failed: {e}")



def train_all_models(args):
    tracking_uri = MLFLOW_URI or "http://127.0.0.1:5000"

    os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "1"
    os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "5"

    mlruns_dir = PROJECT_ROOT / "mlruns"
    mlruns_dir.mkdir(exist_ok=True)
    local_file_uri = mlruns_dir.as_uri()

    use_registry = False

    try:
        import urllib.request
        urllib.request.urlopen(tracking_uri, timeout=3)
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(EXPERIMENT_NAME)
        logger.info(f"MLflow tracking URI: {tracking_uri}")
        use_registry = True
    except Exception as e:
        logger.warning(f"MLflow unavailable ({e}). Using local store.")
        mlflow.set_tracking_uri(local_file_uri)
        mlflow.set_experiment(EXPERIMENT_NAME)

    train_pkl = PROCESSED_DIR / "train_processed.pkl"
    val_pkl   = PROCESSED_DIR / "val_processed.pkl"

    if not train_pkl.exists():
        logger.error("Processed files missing. Run preprocessing first.")
        sys.exit(1)

    X_train, y_train = joblib.load(train_pkl)
    X_val,   y_val   = joblib.load(val_pkl)

    logger.info(f"Train: {X_train.shape} | Val: {X_val.shape}")
    logger.info(
        f"Class balance → "
        f"0={(y_train==0).sum()} "
        f"1={(y_train==1).sum()}"
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = get_models(args, y_train)

    best_avg_precision = 0   # ← use Avg Precision as champion metric
    best_model_name = None
    best_run_id = None

    for model_name, model in models.items():
        logger.info("=" * 60)
        logger.info(f"Training {model_name} | sampling={SAMPLING_STRATEGY}")

        with mlflow.start_run(run_name=model_name) as run:

            # Log hyperparams + sampling strategy
            params = {
                "model_type":       model_name,
                "sampling_strategy": SAMPLING_STRATEGY,
                "n_estimators":     getattr(model, "n_estimators", None),
                "max_depth":        getattr(model, "max_depth", None),
                "learning_rate":    getattr(model, "learning_rate", None),
            }
            mlflow.log_params(
                {k: v for k, v in params.items() if v is not None}
            )

           
            fold_aucs = []
            fold_aps  = []   

            for fold, (train_idx, cv_idx) in enumerate(
                cv.split(X_train, y_train), start=1
            ):
                X_tr = X_train[train_idx]
                y_tr = y_train[train_idx]
                X_cv = X_train[cv_idx]
                y_cv = y_train[cv_idx]

                
                X_tr_res, y_tr_res = apply_resampling(
                    X_tr, y_tr, SAMPLING_STRATEGY
                )

                model.fit(X_tr_res, y_tr_res)

                
                cv_metrics, _ = evaluate_model(model, X_cv, y_cv)

                fold_aucs.append(cv_metrics["roc_auc"])
                fold_aps.append(cv_metrics["avg_precision"])

                logger.info(
                    f"Fold {fold} | "
                    f"ROC-AUC={cv_metrics['roc_auc']:.4f} | "
                    f"Avg Precision={cv_metrics['avg_precision']:.4f}"
                )

            avg_cv_auc = float(np.mean(fold_aucs))
            avg_cv_ap  = float(np.mean(fold_aps))

            mlflow.log_metric("cv_mean_roc_auc",       avg_cv_auc)
            mlflow.log_metric("cv_mean_avg_precision", avg_cv_ap)

            logger.info(
                f"CV Mean ROC-AUC={avg_cv_auc:.4f} | "
                f"CV Mean Avg Precision={avg_cv_ap:.4f}"
            )

          
            X_train_res, y_train_res = apply_resampling(
                X_train, y_train, SAMPLING_STRATEGY
            )
            model.fit(X_train_res, y_train_res)

            metrics, best_threshold = evaluate_model(
                model, X_val, y_val, threshold=None
            )

            mlflow.log_metrics(metrics)

            logger.info(f"ROC-AUC        : {metrics['roc_auc']:.4f}")
            logger.info(f"Avg Precision  : {metrics['avg_precision']:.4f}")
            logger.info(f"F1 class=1     : {metrics['f1_score']:.4f}")
            logger.info(f"Precision cl=1 : {metrics['precision_class1']:.4f}")
            logger.info(f"Recall cl=1    : {metrics['recall_class1']:.4f}")
            logger.info(f"Best threshold : {best_threshold:.4f}")

            local_path = MODELS_DIR / f"{model_name.lower()}.pkl"
            joblib.dump(
                {"model": model, "threshold": best_threshold},
                local_path,
            )


            log_model_to_mlflow(
                model, model_name, X_train[:5], use_registry
            )

            try:
                mlflow.log_artifact(str(local_path))
            except Exception as e:
                logger.warning(f"Artifact upload failed: {e}")

            if metrics["avg_precision"] > best_avg_precision:
                best_avg_precision = metrics["avg_precision"]
                best_model_name    = model_name
                best_run_id        = run.info.run_id


    logger.info("=" * 60)
    logger.info(
        f"Best model: {best_model_name} "
        f"Avg Precision={best_avg_precision:.4f}"
    )

    best_local = MODELS_DIR / f"{best_model_name.lower()}.pkl"

    if best_local.exists():
        shutil.copy(best_local, MODELS_DIR / "best_model.pkl")
        logger.info("Saved best_model.pkl")

    return best_avg_precision, best_model_name


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--n-estimators",  type=int,   default=500)
    parser.add_argument("--max-depth",     type=int,   default=4)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument(
        "--sampling",
        type=str,
        default=SAMPLING_STRATEGY,
        choices=["smote", "undersample", "combined", "none"],
        help="Resampling strategy for class imbalance",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()


    globals()["SAMPLING_STRATEGY"] = args.sampling

    logger.info("=" * 60)
    logger.info("Credit Risk Training")
    logger.info(f"Sampling strategy : {SAMPLING_STRATEGY}")
    logger.info("=" * 60)

    best_ap, best_name = train_all_models(args)

    logger.info(
        f"Training complete → "
        f"{best_name} Avg Precision={best_ap:.4f}"
    )