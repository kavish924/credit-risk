import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import engineer_features, run_preprocessing


@pytest.fixture
def minimal_df():
    """Minimal DataFrame mimicking the Home Credit dataset structure."""
    return pd.DataFrame(
        {
            "SK_ID_CURR": [100001, 100002, 100003],
            "TARGET": [0, 1, 0],
            "AMT_CREDIT": [500_000, 250_000, 750_000],
            "AMT_INCOME_TOTAL": [100_000, 50_000, 200_000],
            "AMT_ANNUITY": [25_000, 12_500, 37_500],
            "DAYS_BIRTH": [-12_000, -15_000, -10_000],
            "DAYS_EMPLOYED": [-3_000, -1_000, -5_000],
            "NAME_CONTRACT_TYPE": ["Cash loans", "Revolving loans", "Cash loans"],
            "CODE_GENDER": ["M", "F", "M"],
            "FLAG_OWN_CAR": ["Y", "N", "Y"],
            "FLAG_OWN_REALTY": ["Y", "Y", "N"],
        }
    )


@pytest.fixture
def high_missing_df(minimal_df):
    """DataFrame with a column that has >50% missing values."""
    df = minimal_df.copy()
    df["HIGH_MISSING_COL"] = [np.nan, np.nan, np.nan]  # 100% missing
    return df


class TestFeatureEngineering:
    def test_debt_to_income_created(self, minimal_df):
        result = engineer_features(minimal_df)
        assert "DEBT_TO_INCOME" in result.columns

    def test_debt_to_income_correct_value(self, minimal_df):
        result = engineer_features(minimal_df)
        expected = 500_000 / 100_000
        assert abs(result["DEBT_TO_INCOME"].iloc[0] - expected) < 1e-6

    def test_annuity_to_income_created(self, minimal_df):
        result = engineer_features(minimal_df)
        assert "ANNUITY_TO_INCOME" in result.columns

    def test_credit_term_created(self, minimal_df):
        result = engineer_features(minimal_df)
        assert "CREDIT_TERM" in result.columns

    def test_age_years_created(self, minimal_df):
        result = engineer_features(minimal_df)
        assert "AGE_YEARS" in result.columns
        assert result["AGE_YEARS"].iloc[0] > 0  # Should be positive

    def test_days_employed_anomaly_flag(self, minimal_df):
        df = minimal_df.copy()
        df.loc[0, "DAYS_EMPLOYED"] = 365243  # anomalous positive value
        result = engineer_features(df)
        assert "DAYS_EMPLOYED_ANOMALY" in result.columns

        assert result["DAYS_EMPLOYED_ANOMALY"].iloc[0] == 1

    def test_no_inf_values(self, minimal_df):
        result = engineer_features(minimal_df)
        numeric_cols = result.select_dtypes(include="number").columns
        assert not np.isinf(result[numeric_cols].values).any()


class TestPreprocessingPipeline:
    def test_output_shape_is_2d(self, minimal_df):
        X, y, _, _ = run_preprocessing(minimal_df, fit=True)
        assert X.ndim == 2

    def test_target_extracted(self, minimal_df):
        X, y, _, _ = run_preprocessing(minimal_df, fit=True)
        assert y is not None
        assert len(y) == 3

    def test_feature_names_returned(self, minimal_df):
        _, _, _, feature_names = run_preprocessing(minimal_df, fit=True)
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0

    def test_high_missing_columns_dropped(self, high_missing_df):
        X, y, preprocessor, feature_names = run_preprocessing(high_missing_df, fit=True)
        # HIGH_MISSING_COL should have been dropped before fitting
        assert "HIGH_MISSING_COL" not in feature_names

    def test_transform_only_mode(self, minimal_df):
        X_train, y_train, preprocessor, feature_names = run_preprocessing(minimal_df, fit=True)

        # Create a new test sample (same columns)
        test_df = minimal_df.drop(columns=["TARGET"])
        X_test, y_test, _, _ = run_preprocessing(test_df, fit=False, preprocessor=preprocessor)

        assert X_test.shape[1] == X_train.shape[1]

    def test_no_nans_in_output(self, minimal_df):
        X, y, _, _ = run_preprocessing(minimal_df, fit=True)
        assert not np.isnan(X).any()

    def test_raises_without_preprocessor_in_transform_mode(self, minimal_df):
        with pytest.raises(ValueError, match="preprocessor must be provided"):
            run_preprocessing(minimal_df, fit=False, preprocessor=None)
