"""Tests for building championship standings from race results."""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.results_data import load_results
from src.standings import STANDINGS_COLUMNS, build_standings, round_points, team_round_points


def _row(rnd, driver, team, position, points, sprint=0.0, classified=True):
    return {"Year": 2030, "Round": rnd, "EventName": f"R{rnd}", "Driver": driver, "Team": team,
            "Position": position, "Classified": classified, "RacePoints": points,
            "SprintPoints": sprint, "Status": "Finished" if classified else "Retired"}


@pytest.fixture
def results():
    return pd.DataFrame([
        _row(1, "AAA", "Red", 1, 25), _row(1, "BBB", "Blue", 2, 18), _row(1, "CCC", "Red", 3, 15),
        _row(2, "BBB", "Blue", 1, 25, sprint=8), _row(2, "CCC", "Red", 2, 18),      # AAA absent
        _row(3, "AAA", "Blue", 1, 25), _row(3, "BBB", "Blue", 20, 0, classified=False),
        _row(3, "CCC", "Red", 2, 18),
    ])


def _at(standings, rnd, driver):
    rows = standings[(standings["Round"] == rnd) & (standings["Driver"] == driver)]
    assert len(rows) == 1
    return rows.iloc[0]


def test_columns_match_the_existing_dataset(results):
    assert list(build_standings(results).columns) == STANDINGS_COLUMNS


def test_driver_who_misses_a_round_keeps_their_points(results):
    standings = build_standings(results)
    assert _at(standings, 2, "AAA")["CumulativePoints"] == 25
    assert _at(standings, 3, "AAA")["CumulativePoints"] == 50
    assert _at(standings, 3, "AAA")["RacesCompleted"] == 2


def test_sprint_points_count(results):
    assert _at(build_standings(results), 2, "BBB")["CumulativePoints"] == 18 + 25 + 8


def test_wins_podiums_and_retirements(results):
    last = _at(build_standings(results), 3, "BBB")
    assert (last["Wins"], last["Podiums"], last["DNFs"]) == (1, 2, 1)


def test_position_and_gap_to_leader(results):
    standings = build_standings(results)
    assert _at(standings, 3, "BBB")["ChampionshipPosition"] == 1
    assert _at(standings, 3, "AAA")["PointsGapToLeader"] == 1
    assert _at(standings, 3, "CCC")["ChampionshipPosition"] == 2


def test_round_points_has_zero_for_absent_drivers(results):
    points = round_points(results, 2030)
    assert points.loc[2, "AAA"] == 0
    assert points.loc[2, "BBB"] == 33
    assert list(points.index) == [1, 2, 3]


def test_team_points_follow_the_team_the_driver_was_with(results):
    teams = team_round_points(results, 2030)
    assert teams.loc[1, "Red"] == 40      # AAA and CCC
    assert teams.loc[3, "Blue"] == 25     # AAA has moved to Blue


def test_official_totals_for_known_champions():
    standings = build_standings(load_results())
    final_2025 = standings[(standings["Year"] == 2025) & (standings["Round"] == 24)]
    top = final_2025.sort_values("ChampionshipPosition").head(3)
    assert list(zip(top["Driver"], top["CumulativePoints"])) == [
        ("NOR", 423.0), ("VER", 421.0), ("PIA", 410.0)]
