"""Tests for FastAPI endpoints."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["driver_model_loaded"] is True

    def test_health_shows_available_years(self, client):
        response = client.get("/health")
        data = response.json()
        assert 2014 in data["available_years"]
        assert 2026 in data["available_years"]


class TestDriverPrediction:
    def test_predict_driver_default(self, client):
        response = client.get("/predict/driver")
        assert response.status_code == 200
        data = response.json()
        assert data["championship_type"] == "driver"
        assert data["year"] == 2026
        assert len(data["predictions"]) > 0

    def test_predict_driver_specific_year_round(self, client):
        response = client.get("/predict/driver?year=2024&round=10")
        assert response.status_code == 200
        data = response.json()
        assert data["year"] == 2024
        assert data["round"] == 10

    def test_predict_driver_invalid_year(self, client):
        response = client.get("/predict/driver?year=2010")
        assert response.status_code == 404

    def test_predict_driver_invalid_round(self, client):
        response = client.get("/predict/driver?year=2024&round=99")
        assert response.status_code == 404

    def test_predict_driver_probabilities_sum_to_one(self, client):
        response = client.get("/predict/driver?year=2024&round=10")
        data = response.json()
        total = sum(p["normalized_probability"] for p in data["predictions"])
        assert abs(total - 1.0) < 0.01

    def test_predict_driver_top_n(self, client):
        response = client.get("/predict/driver?year=2024&round=10&top_n=5")
        data = response.json()
        assert len(data["predictions"]) <= 5

    def test_predict_single_driver(self, client):
        response = client.get("/predict/driver/VER?year=2024&round=10")
        assert response.status_code == 200
        data = response.json()
        assert data["driver"] == "VER"
        assert 0 <= data["championship_probability"] <= 1

    def test_predict_single_driver_not_found(self, client):
        response = client.get("/predict/driver/XYZ?year=2024&round=10")
        assert response.status_code == 404


class TestConstructorPrediction:
    def test_predict_constructor_returns_response(self, client):
        response = client.get("/predict/constructor?year=2024&round=10")
        if response.status_code == 200:
            data = response.json()
            assert data["championship_type"] == "constructor"
            assert len(data["predictions"]) > 0
        else:
            assert response.status_code == 503


class TestSeasonsEndpoint:
    def test_list_seasons(self, client):
        response = client.get("/seasons")
        assert response.status_code == 200
        data = response.json()
        assert len(data["seasons"]) >= 12

    def test_season_standings(self, client):
        response = client.get("/seasons/2024/standings?round=10")
        assert response.status_code == 200
        data = response.json()
        assert data["year"] == 2024
        assert data["round"] == 10
        assert len(data["standings"]) > 0
        assert data["standings"][0]["position"] == 1

    def test_season_standings_invalid_year(self, client):
        response = client.get("/seasons/2010/standings")
        assert response.status_code == 404
