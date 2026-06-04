

import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from airflow import DAG  # type: ignore
from airflow.operators.python import PythonOperator, ShortCircuitOperator  # type: ignore



default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}




def task_ingest(**context):
    """Load raw data and basic validation."""
    from src.data.load_data import load_train, load_test
    df_train = load_train()
    df_test = load_test()
    context["ti"].xcom_push(key="train_rows", value=len(df_train))
    context["ti"].xcom_push(key="test_rows", value=len(df_test))
    print(f"✅ Ingested: train={df_train.shape}, test={df_test.shape}")


def task_validate(**context):
    """Run data quality validation. Short-circuits DAG if validation fails."""
    from src.data.validate import validate_basic, validate_with_ge
    from src.data.load_data import load_train
    df = load_train()
    results = validate_basic(df, label="application_train")
    ge_ok = validate_with_ge(df)
    all_pass = all(results.values()) and ge_ok
    if not all_pass:
        raise ValueError(f"❌ Data validation failed: {results}")
    print("✅ All validation checks passed.")
    return True


def task_preprocess(**context):
    """Run feature engineering and preprocessing pipeline."""
    from src.data.preprocessing import main as run_preprocessing
    run_preprocessing()
    print("✅ Preprocessing complete.")


def task_train(**context):
    """Train XGBoost + baselines with MLflow tracking."""
    import argparse
    from src.ml.train import train_all_models

    class Args:
        n_estimators = 300
        max_depth = 6
        learning_rate = 0.05

    best_auc, best_name = train_all_models(Args())
    context["ti"].xcom_push(key="best_auc", value=best_auc)
    context["ti"].xcom_push(key="best_model", value=best_name)
    print(f"✅ Training complete. Best: {best_name} @ ROC-AUC={best_auc:.4f}")


def task_evaluate(**context):
    """Evaluate best model and generate reports."""
    from src.ml.evaluate import evaluate
    metrics = evaluate()
    context["ti"].xcom_push(key="eval_auc", value=metrics["roc_auc"])

    # Gate: if AUC < 0.70, raise alert (don't deploy)
    if metrics["roc_auc"] < 0.70:
        raise ValueError(
            f"❌ Model AUC {metrics['roc_auc']:.4f} below threshold 0.70 — skipping deployment."
        )
    print(f"✅ Evaluation passed. ROC-AUC: {metrics['roc_auc']:.4f}")


def task_monitor(**context):
    """Run Evidently drift report on latest data."""
    from monitoring.drift import run_drift_report, check_drift_threshold
    report_path = run_drift_report(ref_size=5000, curr_size=1000)
    drift_detected = check_drift_threshold(report_path)
    context["ti"].xcom_push(key="drift_detected", value=drift_detected)
    if drift_detected:
        print("⚠️  Data drift detected — next run will retrain.")
    else:
        print("✅ No significant drift detected.")


def task_notify(**context):
    """Send a summary notification (Slack/email stub)."""
    ti = context["ti"]
    best_auc = ti.xcom_pull(task_ids="train", key="best_auc") or "N/A"
    eval_auc = ti.xcom_pull(task_ids="evaluate", key="eval_auc") or "N/A"
    drift = ti.xcom_pull(task_ids="monitor", key="drift_detected") or False

    message = (
        f"📊 Credit Risk MLOps Pipeline Complete\n"
        f"  Best model AUC : {best_auc}\n"
        f"  Eval AUC       : {eval_auc}\n"
        f"  Drift detected : {drift}\n"
        f"  Run at         : {datetime.now().isoformat()}"
    )
    print(message)
    # TODO: integrate with Slack webhook or SMTP




with DAG(
    dag_id="credit_risk_weekly_retrain",
    default_args=default_args,
    description="Weekly credit risk model retraining pipeline",
    schedule_interval="0 0 * * 0",  # Every Sunday midnight
    catchup=False,
    tags=["mlops", "credit-risk", "xgboost"],
) as dag:

    ingest = PythonOperator(
        task_id="ingest",
        python_callable=task_ingest,
    )

    validate = PythonOperator(
        task_id="validate",
        python_callable=task_validate,
    )

    preprocess = PythonOperator(
        task_id="preprocess",
        python_callable=task_preprocess,
    )

    train = PythonOperator(
        task_id="train",
        python_callable=task_train,
        execution_timeout=timedelta(hours=2),
    )

    evaluate = PythonOperator(
        task_id="evaluate",
        python_callable=task_evaluate,
    )

    monitor = PythonOperator(
        task_id="monitor",
        python_callable=task_monitor,
    )

    notify = PythonOperator(
        task_id="notify",
        python_callable=task_notify,
        trigger_rule="all_done",  # Always run notify, even if upstream fails
    )

    
    ingest >> validate >> preprocess >> train >> evaluate >> monitor >> notify
