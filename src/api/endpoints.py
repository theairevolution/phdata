import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()
logger = logging.getLogger(__name__)

IMPUTER_FEATURES = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot",
    "floors", "sqft_above", "sqft_basement",
]

# Paths resolved relative to this file so they work regardless of working
# directory (local dev, Docker production, Docker test with symlinks).
_SRC = Path(__file__).parent.parent
_MODEL_DIR = _SRC / "model"
_DATA_DIR = _SRC / "data"

with open(_MODEL_DIR / "model.pkl", "rb") as _f:
    _MODEL = pickle.load(_f)
logger.info("Loaded model from %s", _MODEL_DIR / "model.pkl")

with open(_MODEL_DIR / "imputer.pkl", "rb") as _f:
    _IMPUTER = pickle.load(_f)
logger.info("Loaded imputer from %s", _MODEL_DIR / "imputer.pkl")

with open(_MODEL_DIR / "model_features.json") as _f:
    _MODEL_FEATURES = json.load(_f)

_DEMOGRAPHICS = (
    pd.read_csv(_DATA_DIR / "zipcode_demographics.csv", dtype={"zipcode": str})
    .set_index("zipcode")
    .to_dict(orient="index")
)
logger.info("Loaded demographics for %d zipcodes", len(_DEMOGRAPHICS))


class HomeFeatures(BaseModel):
    bedrooms: Optional[int] = Field(default=None, ge=0)
    bathrooms: Optional[float] = Field(default=None, ge=0)
    sqft_living: Optional[float] = Field(default=None, ge=0)
    sqft_lot: Optional[float] = Field(default=None, ge=0)
    floors: Optional[float] = Field(default=None, ge=0)
    sqft_above: Optional[float] = Field(default=None, ge=0)
    sqft_basement: Optional[float] = Field(default=None, ge=0)
    zipcode: str


@router.get("/health")
async def health_check():
    """
    Health check endpoint for container orchestration.
    Returns 200 if API is ready to accept requests.
    """
    return {"status": "healthy"}


@router.post("/predict")
async def predict(home_features: HomeFeatures) -> dict:
    if home_features.zipcode not in _DEMOGRAPHICS:
        logger.warning("Unknown zipcode requested: %s", home_features.zipcode)
        raise HTTPException(
            status_code=422,
            detail=f"Unknown zipcode: {home_features.zipcode}",
        )

    try:
        input_data = pd.DataFrame([home_features.model_dump() if hasattr(home_features, "model_dump") else home_features.dict()])

        input_data[IMPUTER_FEATURES] = _IMPUTER.transform(input_data[IMPUTER_FEATURES])

        demographic_info = pd.DataFrame([_DEMOGRAPHICS[home_features.zipcode]])
        input_data = pd.concat([input_data, demographic_info], axis=1)
        input_data = input_data[_MODEL_FEATURES]

        prediction = _MODEL.predict(input_data)
        logger.debug("predict zipcode=%s price=%.2f", home_features.zipcode, prediction[0])
        return {"predicted_price": float(prediction[0])}
    except Exception as exc:
        logger.error("Prediction failed for zipcode %s: %s", home_features.zipcode, exc)
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
