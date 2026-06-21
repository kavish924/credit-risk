from typing import Literal

from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    # Loan info
    AMT_CREDIT: float = Field(..., gt=0, description="Loan credit amount (>0)")
    AMT_INCOME_TOTAL: float = Field(..., gt=0, description="Applicant total annual income (>0)")
    AMT_ANNUITY: float | None = Field(None, description="Loan annuity amount")

    # Applicant demographics
    DAYS_BIRTH: int = Field(..., lt=0, description="Days relative to application date (negative = past)")
    DAYS_EMPLOYED: int | None = Field(None, description="Days of employment (negative = past)")

    # Loan characteristics
    NAME_CONTRACT_TYPE: Literal["Cash loans", "Revolving loans"] | None = None
    CODE_GENDER: Literal["M", "F", "XNA"] | None = None
    FLAG_OWN_CAR: Literal["Y", "N"] | None = None
    FLAG_OWN_REALTY: Literal["Y", "N"] | None = None

    # Credit bureau data
    AMT_GOODS_PRICE: float | None = Field(None, ge=0)
    REGION_POPULATION_RELATIVE: float | None = Field(None, ge=0, le=1)
    DAYS_REGISTRATION: float | None = None
    DAYS_ID_PUBLISH: int | None = None

    # Social info
    CNT_CHILDREN: int | None = Field(None, ge=0)
    CNT_FAM_MEMBERS: float | None = Field(None, ge=0)
    LIVE_CITY_NOT_WORK_CITY: int | None = Field(None, ge=0, le=1)

    model_config = {
        "json_schema_extra": {
            "example": {
                "AMT_CREDIT": 540000.0,
                "AMT_INCOME_TOTAL": 135000.0,
                "AMT_ANNUITY": 26000.0,
                "DAYS_BIRTH": -12000,
                "DAYS_EMPLOYED": -3000,
                "NAME_CONTRACT_TYPE": "Cash loans",
                "CODE_GENDER": "M",
                "FLAG_OWN_CAR": "N",
                "FLAG_OWN_REALTY": "Y",
            }
        }
    }

    @field_validator("DAYS_BIRTH")
    @classmethod
    def birth_must_be_negative(cls, v: int) -> int:
        if v >= 0:
            raise ValueError("DAYS_BIRTH must be negative (days in the past).")
        return v


class PredictionResponse(BaseModel):
    """Single prediction result."""

    default_probability: float = Field(..., ge=0.0, le=1.0, description="Probability of loan default (0–1)")
    risk_label: Literal["LOW", "MEDIUM", "HIGH"] = Field(
        ..., description="Risk tier: LOW (<20%), MEDIUM (20–50%), HIGH (>50%)"
    )
    risk_score: int = Field(..., ge=0, le=1000, description="Credit risk score (1000 = safest)")
    model_version: str = Field(..., description="Model version used for prediction")
    model_name: str = Field(..., description="Model algorithm name")


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    model_loaded: bool
    model_version: str
    uptime_seconds: float


class BatchPredictionRequest(BaseModel):
    """Batch of up to 100 loan applications."""

    applications: list[PredictionRequest] = Field(..., min_length=1, max_length=100)


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int
    avg_default_probability: float
