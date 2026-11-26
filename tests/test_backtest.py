"""Tests for the backtest scoring."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtest import brier_score, leader_probabilities, score, stage_of, summarise


def test_brier_score_is_zero_for_a_certain_correct_call():
    assert brier_score({"A": 1.0, "B": 0.0}, "A") == 0.0


def test_brier_score_is_two_for_a_certain_wrong_call():
    assert brier_score({"A": 1.0, "B": 0.0}, "B") == 2.0


def test_brier_score_rewards_hedging_when_wrong():
    assert brier_score({"A": 0.6, "B": 0.4}, "B") == pytest.approx(0.72)


def test_leader_baseline_puts_everything_on_the_points_leader():
    assert leader_probabilities({"A": 50.0, "B": 75.0, "C": 10.0}) == {"A": 0.0, "B": 1.0, "C": 0.0}


def test_score_records_pick_and_probability_on_the_champion():
    row = score({"A": 0.7, "B": 0.3}, "B")
    assert row == {"correct": False, "p_champion": 0.3, "brier": pytest.approx(0.98)}


@pytest.mark.parametrize("rnd,total,stage", [
    (4, 22, "First third"), (8, 22, "Middle third"), (15, 22, "Final third"), (21, 22, "Final third"),
])
def test_stage_splits_the_season_into_thirds(rnd, total, stage):
    assert stage_of(rnd, total) == stage


def test_summarise_averages_each_method():
    rows = [
        {"method": "sim", "correct": True, "p_champion": 0.8, "brier": 0.08},
        {"method": "sim", "correct": False, "p_champion": 0.2, "brier": 1.28},
        {"method": "leader", "correct": True, "p_champion": 1.0, "brier": 0.0},
    ]
    summary = summarise(rows)
    assert summary["sim"] == {"n": 2, "accuracy": 0.5, "p_champion": 0.5,
                              "brier": pytest.approx(0.68)}
    assert summary["leader"]["accuracy"] == 1.0
