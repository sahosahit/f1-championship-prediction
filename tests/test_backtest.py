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


from src.backtest import calibration_table, hedged_leader_probabilities
from src.points import max_points_remaining


def test_hedged_leader_splits_between_first_and_second():
    probabilities = hedged_leader_probabilities({"A": 50.0, "B": 75.0, "C": 10.0}, 0.8)
    assert probabilities == {"A": pytest.approx(0.2), "B": 0.8, "C": 0.0}


def test_calibration_table_compares_confidence_with_how_often_it_was_right():
    rows = [{"method": "simulation", "p_pick": 0.95, "correct": True},
            {"method": "simulation", "p_pick": 0.92, "correct": False},
            {"method": "simulation", "p_pick": 0.55, "correct": True},
            {"method": "leader", "p_pick": 1.0, "correct": True}]
    table = calibration_table(rows, "simulation")
    top = next(band for band in table if band["band"] == "90–99%")
    assert top == {"band": "90–99%", "n": 2, "mean_confidence": pytest.approx(0.935),
                   "share_correct": 0.5}


@pytest.mark.parametrize("year,rounds,sprints,expected", [
    (2026, [22, 23], [False, False], 50),          # 25 per race
    (2023, [21, 22], [True, False], 26 + 8 + 26),  # fastest-lap point era, one sprint
    (2021, [22], [True], 26 + 3),                  # 2021 sprints paid 3 to the winner
    (2014, [18, 19], [False, False], 25 + 50),     # double points at the 2014 finale
    (2026, [], [], 0),
])
def test_max_points_remaining_follows_each_seasons_rules(year, rounds, sprints, expected):
    assert max_points_remaining(year, rounds, sprints, final_round=rounds[-1] if rounds else 23) \
        == expected
