"""Tests for the rest-of-season simulation."""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.season_simulation import simulate_title


def _points(rows):
    frame = pd.DataFrame(rows).T
    frame.index.name = "Round"
    return frame.fillna(0.0)


def _run(points, remaining, teams=None, past_sprint=None, **kwargs):
    teams = teams if teams is not None else points.copy()
    past_sprint = past_sprint or {rnd: False for rnd in points.index}
    return simulate_title(points, teams, past_sprint, remaining, n_sims=4000, seed=1, **kwargs)


def test_no_races_left_means_the_leader_has_won():
    points = _points({1: {"A": 25, "B": 18}, 2: {"A": 18, "B": 15}})
    result = _run(points, remaining=[])
    assert result["drivers"] == {"A": 1.0, "B": 0.0}


def test_driver_who_cannot_catch_up_gets_zero():
    # C is 68 behind with two rounds left and nobody has scored more than 25 in a round.
    points = _points({1: {"A": 25, "B": 18, "C": 0}, 2: {"A": 18, "B": 25, "C": 0},
                      3: {"A": 25, "B": 18, "C": 0}})
    result = _run(points, remaining=[False, False])
    assert result["drivers"]["C"] == 0.0
    assert 0.0 < result["drivers"]["B"] < result["drivers"]["A"]


def test_probabilities_sum_to_one():
    points = _points({1: {"A": 25, "B": 18, "C": 15}, 2: {"A": 15, "B": 25, "C": 18}})
    result = _run(points, remaining=[False] * 5)
    assert sum(result["drivers"].values()) == pytest.approx(1.0)
    assert sum(result["teams"].values()) == pytest.approx(1.0)


def test_same_seed_gives_the_same_answer():
    points = _points({1: {"A": 25, "B": 18}, 2: {"A": 18, "B": 25}})
    assert _run(points, remaining=[False] * 3) == _run(points, remaining=[False] * 3)


def test_level_on_points_with_no_races_left_is_split_evenly():
    points = _points({1: {"A": 25, "B": 18}, 2: {"A": 18, "B": 25}})
    result = _run(points, remaining=[])
    assert result["drivers"]["A"] == pytest.approx(0.5, abs=0.03)


def test_sprint_weekend_is_drawn_from_past_sprint_weekends():
    # A wins every normal round; B wins the only sprint weekend by a lot.
    points = _points({1: {"A": 25, "B": 18}, 2: {"A": 0, "B": 33}, 3: {"A": 25, "B": 18}})
    past_sprint = {1: False, 2: True, 3: False}
    sprint_left = _run(points, remaining=[True], past_sprint=past_sprint)
    normal_left = _run(points, remaining=[False], past_sprint=past_sprint)
    assert sprint_left["drivers"]["B"] == 1.0      # 69 + 33 beats 50 + 0
    assert normal_left["drivers"]["B"] == 1.0      # 69 + 18 beats 50 + 25
    points.loc[2, "B"] = 20                        # now B leads 56 to 50
    assert _run(points, remaining=[False], past_sprint=past_sprint)["drivers"]["A"] == 1.0
    assert _run(points, remaining=[True], past_sprint=past_sprint)["drivers"]["A"] == 0.0


def test_team_title_uses_team_points_not_driver_points():
    drivers = _points({1: {"A": 25, "B": 18, "C": 15}})
    teams = _points({1: {"Red": 25, "Blue": 33}})
    result = _run(drivers, remaining=[], teams=teams)
    assert result["teams"] == {"Blue": 1.0, "Red": 0.0}
