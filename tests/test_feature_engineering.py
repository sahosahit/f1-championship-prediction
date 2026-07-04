"""Tests for feature engineering pipeline."""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from feature_engineering import extract_features, create_sequences, create_prediction_input, load_standings
from constructor_feature_engineering import (
    build_constructor_standings, extract_constructor_features,
    create_constructor_sequences, create_constructor_prediction_input
)


@pytest.fixture
def standings():
    return load_standings()


@pytest.fixture
def constructor_standings(standings):
    return build_constructor_standings(standings)


class TestExtractFeatures:
    def test_output_shape(self, standings):
        driver_history = standings[
            (standings['Year'] == 2024) &
            (standings['Driver'] == 'VER') &
            (standings['Round'] <= 10)
        ].sort_values('Round')
        features = extract_features(driver_history, 24)
        assert features.shape[0] == len(driver_history)
        assert features.shape[1] == 12

    def test_features_in_valid_range(self, standings):
        driver_history = standings[
            (standings['Year'] == 2023) &
            (standings['Driver'] == 'VER') &
            (standings['Round'] <= 5)
        ].sort_values('Round')
        features = extract_features(driver_history, 22)
        assert features.dtype == np.float32
        assert not np.isnan(features).any()
        assert not np.isinf(features).any()

    def test_season_progress_increases(self, standings):
        driver_history = standings[
            (standings['Year'] == 2022) &
            (standings['Driver'] == 'VER')
        ].sort_values('Round')
        features = extract_features(driver_history, 22)
        season_progress = features[:, 6]
        assert all(season_progress[i] <= season_progress[i+1] for i in range(len(season_progress)-1))


class TestCreateSequences:
    def test_sequences_created(self, standings):
        sequences = create_sequences(standings, min_races=4, top_n_drivers=5)
        assert len(sequences) > 0

    def test_sequence_structure(self, standings):
        sequences = create_sequences(standings, min_races=4, top_n_drivers=5)
        seq = sequences[0]
        assert 'features' in seq
        assert 'target' in seq
        assert 'year' in seq
        assert 'driver' in seq
        assert seq['target'] in [0, 1]
        assert seq['features'].shape[1] == 12

    def test_class_balance(self, standings):
        sequences = create_sequences(standings, min_races=4, top_n_drivers=10)
        positives = sum(s['target'] for s in sequences)
        negatives = len(sequences) - positives
        assert positives > 0
        assert negatives > positives  # class imbalance expected


class TestCreatePredictionInput:
    def test_prediction_input_structure(self, standings):
        inputs = create_prediction_input(standings, 2024, 10, top_n=5)
        assert len(inputs) > 0
        for p in inputs:
            assert 'driver' in p
            assert 'team' in p
            assert 'features' in p
            assert 'current_points' in p
            assert p['features'].shape[1] == 12


class TestConstructorFeatureEngineering:
    def test_build_constructor_standings(self, constructor_standings):
        assert not constructor_standings.empty
        assert 'ConstructorPosition' in constructor_standings.columns
        assert 'PointsGapToLeader' in constructor_standings.columns

    def test_constructor_features_shape(self, constructor_standings):
        team_history = constructor_standings[
            (constructor_standings['Year'] == 2024) &
            (constructor_standings['Team'] == 'McLaren') &
            (constructor_standings['Round'] <= 10)
        ].sort_values('Round')
        if not team_history.empty:
            features = extract_constructor_features(team_history, 24)
            assert features.shape[1] == 12
            assert not np.isnan(features).any()

    def test_constructor_sequences(self, constructor_standings):
        sequences = create_constructor_sequences(constructor_standings, min_races=4, top_n_teams=5)
        assert len(sequences) > 0
        seq = sequences[0]
        assert 'team' in seq
        assert seq['target'] in [0, 1]
        assert seq['features'].shape[1] == 12

    def test_constructor_prediction_input(self, constructor_standings):
        inputs = create_constructor_prediction_input(constructor_standings, 2024, 10, top_n=5)
        assert len(inputs) > 0
        for p in inputs:
            assert 'team' in p
            assert 'features' in p
            assert 'current_points' in p
