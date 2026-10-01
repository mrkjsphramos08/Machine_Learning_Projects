"""Pydantic schemas for request validation and response formatting."""

from pydantic import BaseModel, Field


class ModelInput(BaseModel):
    features: list[float] = Field(
        ...,
        min_length=10,
        max_length=10,
        description="List of 10 scaled clinical features: [age, sex, bmi, bp, s1, s2, s3, s4, s5, s6]",
        examples=[
            [0.038, 0.050, 0.061, 0.021, -0.044, -0.034, -0.043, -0.002, 0.019, -0.017]
        ],
    )


class PredictionResponse(BaseModel):
    prediction: float = Field(
        ..., description="Predicted disease progression score after 1 year"
    )
    model_uri: str = Field(
        ..., description="MLflow registry URI used for this prediction"
    )
