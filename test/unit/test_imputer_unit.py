import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.api.endpoints import IMPUTER_FEATURES

IMPUTER_PATH = Path("src/model/imputer.pkl")


@pytest.fixture(scope="module")
def imputer():
    with open(IMPUTER_PATH, "rb") as f:
        return pickle.load(f)


@pytest.fixture
def complete_row():
    return pd.DataFrame([{
        "bedrooms": 3,
        "bathrooms": 2.0,
        "sqft_living": 1500.0,
        "sqft_lot": 5000.0,
        "floors": 1.0,
        "sqft_above": 1200.0,
        "sqft_basement": 300.0,
    }])


def test_imputer_binary_loads():
    with open(IMPUTER_PATH, "rb") as f:
        imp = pickle.load(f)
    assert imp is not None


def test_imputer_no_change_on_complete_row(imputer, complete_row):
    result = imputer.transform(complete_row)
    assert result[0, IMPUTER_FEATURES.index("bedrooms")] == 3
    assert result[0, IMPUTER_FEATURES.index("bathrooms")] == 2.0
    assert result[0, IMPUTER_FEATURES.index("sqft_living")] == 1500.0
    assert result[0, IMPUTER_FEATURES.index("sqft_lot")] == 5000.0
    assert result[0, IMPUTER_FEATURES.index("floors")] == 1.0
    assert result[0, IMPUTER_FEATURES.index("sqft_above")] == 1200.0
    assert result[0, IMPUTER_FEATURES.index("sqft_basement")] == 300.0


def test_imputer_fills_single_missing(imputer, complete_row):
    complete_row.loc[0, "bedrooms"] = np.nan
    result = imputer.transform(complete_row)
    assert not np.isnan(result).any()


def test_imputer_fills_multiple_missing(imputer):
    row = pd.DataFrame([{
        "bedrooms": np.nan,
        "bathrooms": np.nan,
        "sqft_living": 1500.0,
        "sqft_lot": np.nan,
        "floors": 1.0,
        "sqft_above": np.nan,
        "sqft_basement": 300.0,
    }])
    result = imputer.transform(row)
    assert not np.isnan(result).any()


def test_imputed_bedrooms_plausible(imputer):
    row = pd.DataFrame([{
        "bedrooms": np.nan,
        "bathrooms": 2.0,
        "sqft_living": 1500.0,
        "sqft_lot": 5000.0,
        "floors": 1.0,
        "sqft_above": 1200.0,
        "sqft_basement": 300.0,
    }])
    result = imputer.transform(row)
    imputed_bedrooms = result[0, IMPUTER_FEATURES.index("bedrooms")]
    assert 0 <= imputed_bedrooms <= 33  # range from training data


def test_imputed_sqft_living_plausible(imputer):
    row = pd.DataFrame([{
        "bedrooms": 3,
        "bathrooms": 2.0,
        "sqft_living": np.nan,
        "sqft_lot": 5000.0,
        "floors": 1.0,
        "sqft_above": 1200.0,
        "sqft_basement": 300.0,
    }])
    result = imputer.transform(row)
    imputed_sqft = result[0, IMPUTER_FEATURES.index("sqft_living")]
    assert 290 <= imputed_sqft <= 13540  # min/max from training data


def test_predict_with_missing_fields(test_client):
    response = test_client.post("/predict", json={
        "bedrooms": None,
        "bathrooms": None,
        "sqft_living": 1500.0,
        "sqft_lot": 5000.0,
        "floors": 1.0,
        "sqft_above": 1200.0,
        "sqft_basement": 300.0,
        "zipcode": "98042",
    })
    assert response.status_code == 200
    data = response.json()
    assert "predicted_price" in data
    assert isinstance(data["predicted_price"], float)
    assert data["predicted_price"] > 0


def test_predict_all_home_features_missing(test_client):
    response = test_client.post("/predict", json={"zipcode": "98042"})
    assert response.status_code == 200
    data = response.json()
    assert "predicted_price" in data
    assert isinstance(data["predicted_price"], float)
    assert data["predicted_price"] > 0


def test_predict_imputed_vs_complete_price_similar(test_client, sample_home_features):
    complete_response = test_client.post("/predict", json=sample_home_features)
    assert complete_response.status_code == 200
    complete_price = complete_response.json()["predicted_price"]

    partial = {k: None for k in sample_home_features if k != "zipcode"}
    partial["sqft_living"] = sample_home_features["sqft_living"]
    partial["sqft_lot"] = sample_home_features["sqft_lot"]
    partial["zipcode"] = sample_home_features["zipcode"]

    partial_response = test_client.post("/predict", json=partial)
    assert partial_response.status_code == 200
    partial_price = partial_response.json()["predicted_price"]

    # Imputed prediction should be in the same ballpark (within 50%)
    assert abs(partial_price - complete_price) / complete_price < 0.50
