"""Tests for the prediction file and the README blocks generated from it."""

import json
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.report import (
    BACKTEST_PATH, PREDICTION_PATH, README_PATH, build_prediction, read_block,
    render_backtest_block, render_prediction_block,
)


def _results(rows):
    return pd.DataFrame([
        {"Year": 2030, "Round": rnd, "EventName": f"Race {rnd}", "Driver": driver, "Team": team,
         "Position": position, "Classified": True, "RacePoints": points, "SprintPoints": 0.0,
         "Status": "Finished"}
        for rnd, driver, team, position, points in rows
    ])


def _calendar(n_rounds):
    return pd.DataFrame([
        {"Year": 2030, "Round": r, "EventName": f"Race {r}", "Date": "2030-01-01",
         "IsSprint": False} for r in range(1, n_rounds + 1)
    ])


ROWS = [
    (1, "AAA", "Red", 1, 25), (1, "BBB", "Blue", 2, 18),
    (2, "AAA", "Red", 1, 25), (2, "BBB", "Blue", 2, 18),
    (3, "AAA", "Red", 1, 25), (3, "BBB", "Blue", 2, 18),
]


def test_title_is_clinched_when_the_lead_exceeds_the_points_left():
    prediction = build_prediction(_results(ROWS), _calendar(3), 2030, n_sims=500)
    assert prediction["round"] == 3 and prediction["rounds_remaining"] == 0
    assert prediction["drivers_title_clinched"] is True
    assert prediction["drivers"][0] == {
        "driver": "AAA", "team": "Red", "points": 75.0, "probability": 1.0}


def test_title_is_open_while_the_chaser_can_still_catch_up():
    prediction = build_prediction(_results(ROWS), _calendar(5), 2030, n_sims=500)
    assert prediction["points_still_available"] == 50
    assert prediction["drivers_title_clinched"] is False
    assert prediction["drivers"][0]["driver"] == "AAA"
    assert sum(d["probability"] for d in prediction["drivers"]) == pytest.approx(1.0)
    assert [t["team"] for t in prediction["teams"]] == ["Red", "Blue"]


def test_prediction_block_says_when_the_title_is_decided():
    prediction = build_prediction(_results(ROWS), _calendar(3), 2030, n_sims=500)
    block = render_prediction_block(prediction)
    assert "AAA has won the 2030 drivers' championship" in block
    assert "| AAA | Red | 75 | 100.0% |" in block


def test_readme_blocks_match_the_committed_results_files():
    with open(README_PATH) as handle:
        readme = handle.read()
    with open(PREDICTION_PATH) as handle:
        assert read_block(readme, "prediction") == render_prediction_block(json.load(handle))
    with open(BACKTEST_PATH) as handle:
        assert read_block(readme, "backtest") == render_backtest_block(json.load(handle))


def test_too_early_in_the_season_no_probabilities_are_published():
    rows = [(1, "AAA", "Red", 1, 25), (1, "BBB", "Blue", 2, 18)]
    prediction = build_prediction(_results(rows), _calendar(20), 2030, n_sims=200)
    assert prediction["too_early"] is True
    block = render_prediction_block(prediction)
    assert "too early" in block.lower()
    assert "%" not in block.split("|", 1)[-1]


def test_one_round_left_reads_naturally():
    prediction = build_prediction(_results(ROWS), _calendar(4), 2030, n_sims=200)
    assert "1 round left" in render_prediction_block(prediction)
