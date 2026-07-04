"""
FastAPI service for F1 Championship Prediction.
Serves predictions via REST API for both Driver and Constructor championships.
"""

import torch
import pandas as pd
import numpy as np
import os
import sys
from functools import lru_cache
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from model import F1ChampionshipLSTM
from feature_engineering import load_standings, extract_features, create_prediction_input
from constructor_feature_engineering import (
    build_constructor_standings, extract_constructor_features, create_constructor_prediction_input
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')

_models = {}
_data = {}


def load_driver_model():
    checkpoint = torch.load(
        os.path.join(MODELS_DIR, 'best_model.pth'),
        weights_only=False, map_location='cpu'
    )
    model = F1ChampionshipLSTM(input_size=12, hidden_size=64, num_layers=2, dropout=0.3)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def load_constructor_model():
    model_path = os.path.join(MODELS_DIR, 'best_constructor_model.pth')
    if not os.path.exists(model_path):
        return None
    checkpoint = torch.load(model_path, weights_only=False, map_location='cpu')
    model = F1ChampionshipLSTM(input_size=12, hidden_size=64, num_layers=2, dropout=0.3)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    _models['driver'] = load_driver_model()
    _models['constructor'] = load_constructor_model()
    _data['standings'] = load_standings()
    _data['constructor_standings'] = build_constructor_standings(_data['standings'])
    yield
    _models.clear()
    _data.clear()


app = FastAPI(
    title="F1 Championship Prediction API",
    description="LSTM-based predictions for F1 Driver and Constructor World Championships",
    version="1.0.0",
    lifespan=lifespan,
)


class DriverPrediction(BaseModel):
    driver: str
    team: str
    points: float
    probability: float
    normalized_probability: float


class ConstructorPrediction(BaseModel):
    team: str
    points: float
    probability: float
    normalized_probability: float


class PredictionResponse(BaseModel):
    year: int
    round: int
    championship_type: str
    predictions: list
    model_info: dict


class HealthResponse(BaseModel):
    status: str
    driver_model_loaded: bool
    constructor_model_loaded: bool
    data_latest_round: int
    available_years: list[int]


@app.get("/health", response_model=HealthResponse)
async def health_check():
    standings = _data['standings']
    latest_2026 = standings[standings['Year'] == 2026]['Round'].max() if 2026 in standings['Year'].values else 0
    return HealthResponse(
        status="healthy",
        driver_model_loaded='driver' in _models and _models['driver'] is not None,
        constructor_model_loaded='constructor' in _models and _models['constructor'] is not None,
        data_latest_round=int(latest_2026) if pd.notna(latest_2026) else 0,
        available_years=sorted(standings['Year'].unique().tolist()),
    )


@app.get("/predict/driver", response_model=PredictionResponse)
async def predict_driver_championship(
    year: int = Query(2026, description="Season year"),
    round: int = Query(None, description="Race round number (defaults to latest available)"),
    top_n: int = Query(10, description="Number of top drivers to predict", ge=1, le=20),
):
    standings = _data['standings']
    model = _models['driver']

    year_data = standings[standings['Year'] == year]
    if year_data.empty:
        raise HTTPException(status_code=404, detail=f"No data for year {year}")

    available_rounds = sorted(year_data['Round'].unique())
    if round is None:
        round = int(available_rounds[-1])
    elif round not in available_rounds:
        raise HTTPException(
            status_code=404,
            detail=f"Round {round} not available for {year}. Available: {available_rounds}"
        )

    pred_inputs = create_prediction_input(standings, year, round, top_n=top_n)
    if not pred_inputs:
        raise HTTPException(status_code=404, detail="No prediction data available")

    results = []
    for p in pred_inputs:
        features = torch.FloatTensor(p['features']).unsqueeze(0)
        seq_len = torch.LongTensor([p['features'].shape[0]])
        with torch.no_grad():
            prob = torch.sigmoid(model(features, seq_len)).item()
        results.append({
            'driver': p['driver'],
            'team': p['team'],
            'points': float(p['current_points']),
            'probability': prob,
        })

    results.sort(key=lambda x: x['probability'], reverse=True)
    total_prob = sum(r['probability'] for r in results)

    predictions = [
        DriverPrediction(
            driver=r['driver'],
            team=r['team'],
            points=r['points'],
            probability=r['probability'],
            normalized_probability=r['probability'] / total_prob if total_prob > 0 else 0,
        )
        for r in results
    ]

    return PredictionResponse(
        year=year,
        round=round,
        championship_type="driver",
        predictions=[p.model_dump() for p in predictions],
        model_info={
            "architecture": "LSTM (2 layers, 64 hidden)",
            "parameters": 53313,
            "trained_on": "2014-2024",
            "validated_on": "2025 (80% accuracy)",
        },
    )


@app.get("/predict/constructor", response_model=PredictionResponse)
async def predict_constructor_championship(
    year: int = Query(2026, description="Season year"),
    round: int = Query(None, description="Race round number (defaults to latest available)"),
    top_n: int = Query(10, description="Number of top teams to predict", ge=1, le=10),
):
    if _models.get('constructor') is None:
        raise HTTPException(
            status_code=503,
            detail="Constructor model not trained yet. Run: python src/train_constructor.py"
        )

    constructor_standings = _data['constructor_standings']
    model = _models['constructor']

    year_data = constructor_standings[constructor_standings['Year'] == year]
    if year_data.empty:
        raise HTTPException(status_code=404, detail=f"No constructor data for year {year}")

    available_rounds = sorted(year_data['Round'].unique())
    if round is None:
        round = int(available_rounds[-1])
    elif round not in available_rounds:
        raise HTTPException(
            status_code=404,
            detail=f"Round {round} not available for {year}. Available: {available_rounds}"
        )

    pred_inputs = create_constructor_prediction_input(constructor_standings, year, round, top_n=top_n)
    if not pred_inputs:
        raise HTTPException(status_code=404, detail="No constructor prediction data available")

    results = []
    for p in pred_inputs:
        features = torch.FloatTensor(p['features']).unsqueeze(0)
        seq_len = torch.LongTensor([p['features'].shape[0]])
        with torch.no_grad():
            prob = torch.sigmoid(model(features, seq_len)).item()
        results.append({
            'team': p['team'],
            'points': float(p['current_points']),
            'probability': prob,
        })

    results.sort(key=lambda x: x['probability'], reverse=True)
    total_prob = sum(r['probability'] for r in results)

    predictions = [
        ConstructorPrediction(
            team=r['team'],
            points=r['points'],
            probability=r['probability'],
            normalized_probability=r['probability'] / total_prob if total_prob > 0 else 0,
        )
        for r in results
    ]

    return PredictionResponse(
        year=year,
        round=round,
        championship_type="constructor",
        predictions=[p.model_dump() for p in predictions],
        model_info={
            "architecture": "LSTM (2 layers, 64 hidden)",
            "parameters": 53313,
            "trained_on": "2014-2024",
            "validated_on": "2025",
        },
    )


@app.get("/predict/driver/{driver_code}")
async def predict_single_driver(
    driver_code: str,
    year: int = Query(2026, description="Season year"),
    round: int = Query(None, description="Race round number"),
):
    """Get championship probability for a specific driver."""
    standings = _data['standings']
    model = _models['driver']

    driver_code = driver_code.upper()
    year_data = standings[standings['Year'] == year]
    if year_data.empty:
        raise HTTPException(status_code=404, detail=f"No data for year {year}")

    available_rounds = sorted(year_data['Round'].unique())
    if round is None:
        round = int(available_rounds[-1])

    driver_data = year_data[
        (year_data['Driver'] == driver_code) & (year_data['Round'] <= round)
    ].sort_values('Round')

    if driver_data.empty:
        raise HTTPException(status_code=404, detail=f"Driver {driver_code} not found in {year} season")

    total_rounds_estimate = 22
    features = extract_features(driver_data, total_rounds_estimate)
    x = torch.FloatTensor(features).unsqueeze(0)
    sl = torch.LongTensor([features.shape[0]])

    with torch.no_grad():
        prob = torch.sigmoid(model(x, sl)).item()

    last_row = driver_data.iloc[-1]
    return {
        "driver": driver_code,
        "team": last_row['Team'],
        "year": year,
        "round": round,
        "points": float(last_row['CumulativePoints']),
        "championship_position": int(last_row['ChampionshipPosition']),
        "championship_probability": prob,
        "wins": int(last_row['Wins']),
        "podiums": int(last_row['Podiums']),
    }


@app.get("/seasons")
async def list_seasons():
    """List all available seasons and their details."""
    standings = _data['standings']
    seasons = []
    for year in sorted(standings['Year'].unique()):
        year_data = standings[standings['Year'] == year]
        max_round = int(year_data['Round'].max())
        final = year_data[year_data['Round'] == max_round]
        leader = final.sort_values('CumulativePoints', ascending=False).iloc[0]
        seasons.append({
            "year": int(year),
            "rounds_completed": max_round,
            "leader": leader['Driver'],
            "leader_team": leader['Team'],
            "leader_points": float(leader['CumulativePoints']),
        })
    return {"seasons": seasons}


@app.get("/seasons/{year}/standings")
async def get_season_standings(
    year: int,
    round: int = Query(None, description="Round number (defaults to latest)"),
):
    """Get championship standings for a specific season and round."""
    standings = _data['standings']
    year_data = standings[standings['Year'] == year]
    if year_data.empty:
        raise HTTPException(status_code=404, detail=f"No data for year {year}")

    available_rounds = sorted(year_data['Round'].unique())
    if round is None:
        round = int(available_rounds[-1])
    elif round not in available_rounds:
        raise HTTPException(status_code=404, detail=f"Round {round} not available")

    round_data = year_data[year_data['Round'] == round].sort_values('ChampionshipPosition')

    return {
        "year": year,
        "round": round,
        "standings": [
            {
                "position": int(row['ChampionshipPosition']),
                "driver": row['Driver'],
                "team": row['Team'],
                "points": float(row['CumulativePoints']),
                "wins": int(row['Wins']),
                "podiums": int(row['Podiums']),
                "points_per_race": float(row['PointsPerRace']),
            }
            for _, row in round_data.iterrows()
        ],
    }
