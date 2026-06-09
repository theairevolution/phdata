import pytest


def test_health_endpoint(test_client):
    """Test the /health endpoint returns correct status."""
    response = test_client.get("/health")
    assert response.status_code == 200
    response_data = response.json()
    assert "status" in response_data
    assert response_data["status"] == "healthy"


def test_predict_endpoint_valid_input(test_client, sample_home_features):
    """Test the /predict endpoint with valid input."""
    response = test_client.post("/predict", json=sample_home_features)
    assert response.status_code == 200
    response_data = response.json()
    assert "predicted_price" in response_data
    assert isinstance(response_data["predicted_price"], float)


def test_predict_endpoint_price_positive(test_client, sample_home_features):
    """Predicted price must always be a positive number."""
    response = test_client.post("/predict", json=sample_home_features)
    assert response.status_code == 200
    assert response.json()["predicted_price"] > 0


def test_predict_endpoint_invalid_zipcode(test_client, sample_home_features):
    """Unknown zipcodes should return 422 with an error detail."""
    payload = {**sample_home_features, "zipcode": "00000"}
    response = test_client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "zipcode" in response.json()["detail"].lower()


def test_predict_endpoint_negative_bedrooms_rejected(test_client, sample_home_features):
    """Negative bedroom count must fail Pydantic field validation."""
    payload = {**sample_home_features, "bedrooms": -1}
    response = test_client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_endpoint_negative_sqft_rejected(test_client, sample_home_features):
    """Negative square footage must fail Pydantic field validation."""
    payload = {**sample_home_features, "sqft_living": -500.0}
    response = test_client.post("/predict", json=payload)
    assert response.status_code == 422
