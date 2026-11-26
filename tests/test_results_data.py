"""Tests for parsing official results (Jolpica / Ergast format) into flat tables."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.results_data import parse_calendar, parse_race_results, parse_sprint_results


def _result(code, team, position, text, points, status="Finished"):
    return {
        "position": str(position), "positionText": text, "points": str(points),
        "status": status, "Driver": {"code": code, "driverId": code.lower()},
        "Constructor": {"name": team},
    }


RACES = [{
    "season": "2026", "round": "2", "raceName": "Chinese Grand Prix",
    "Results": [
        _result("ANT", "Mercedes", 1, "1", 25),
        _result("VER", "Red Bull", 2, "2", 18),
        _result("ALO", "Aston Martin", 20, "R", 0, status="Retired"),
    ],
}]

SPRINTS = [{
    "season": "2026", "round": "2", "raceName": "Chinese Grand Prix",
    "SprintResults": [_result("VER", "Red Bull", 1, "1", 8), _result("ANT", "Mercedes", 2, "2", 7)],
}]


def test_race_rows_carry_points_position_and_team():
    rows = parse_race_results(RACES)
    winner = rows[0]
    assert winner == {
        "Year": 2026, "Round": 2, "EventName": "Chinese Grand Prix", "Driver": "ANT",
        "Team": "Mercedes", "Position": 1, "Classified": True, "RacePoints": 25.0,
        "Status": "Finished",
    }


def test_retirement_is_not_classified():
    retired = parse_race_results(RACES)[2]
    assert retired["Classified"] is False
    assert retired["RacePoints"] == 0.0


def test_driver_without_a_code_falls_back_to_their_id():
    races = [{"season": "2014", "round": "1", "raceName": "X", "Results": [
        {"position": "1", "positionText": "1", "points": "25", "status": "Finished",
         "Driver": {"driverId": "kobayashi"}, "Constructor": {"name": "Caterham"}}]}]
    assert parse_race_results(races)[0]["Driver"] == "KOB"


def test_sprint_rows_carry_only_sprint_points():
    assert parse_sprint_results(SPRINTS) == [
        {"Year": 2026, "Round": 2, "Driver": "VER", "SprintPoints": 8.0},
        {"Year": 2026, "Round": 2, "Driver": "ANT", "SprintPoints": 7.0},
    ]


def test_calendar_marks_sprint_weekends():
    races = [
        {"season": "2026", "round": "1", "raceName": "A", "date": "2026-03-08"},
        {"season": "2026", "round": "2", "raceName": "B", "date": "2026-03-15",
         "Sprint": {"date": "2026-03-14"}},
    ]
    assert parse_calendar(races) == [
        {"Year": 2026, "Round": 1, "EventName": "A", "Date": "2026-03-08", "IsSprint": False},
        {"Year": 2026, "Round": 2, "EventName": "B", "Date": "2026-03-15", "IsSprint": True},
    ]


import pandas as pd
import pytest

from src import results_data


def _season(rounds):
    return pd.DataFrame([{"Year": 2030, "Round": r, "EventName": f"R{r}", "Driver": "AAA",
                          "Team": "Red", "Position": 1, "Classified": True, "RacePoints": 25.0,
                          "Status": "Finished", "SprintPoints": 0.0} for r in rounds])


def _calendar_frame(n):
    return pd.DataFrame([{"Year": 2030, "Round": r, "EventName": f"R{r}", "Date": "2030-01-01",
                          "IsSprint": False} for r in range(1, n + 1)])


@pytest.fixture
def stored(tmp_path, monkeypatch):
    monkeypatch.setattr(results_data, "RESULTS_PATH", str(tmp_path / "results.csv"))
    monkeypatch.setattr(results_data, "CALENDAR_PATH", str(tmp_path / "calendar.csv"))
    monkeypatch.setattr(results_data, "fetch_calendar", lambda year: _calendar_frame(10))
    monkeypatch.setattr(results_data, "fetch_season", lambda year: _season([1, 2, 3]))
    results_data.refresh_season(2030)
    return monkeypatch


def test_refresh_replaces_the_season(stored):
    stored.setattr(results_data, "fetch_season", lambda year: _season([1, 2, 3, 4]))
    results_data.refresh_season(2030)
    assert sorted(results_data.load_results()["Round"]) == [1, 2, 3, 4]


def test_refresh_refuses_a_response_with_fewer_rounds_than_already_stored(stored):
    stored.setattr(results_data, "fetch_season", lambda year: _season([1, 2]))
    with pytest.raises(RuntimeError, match="fewer rounds"):
        results_data.refresh_season(2030)
    assert sorted(results_data.load_results()["Round"]) == [1, 2, 3]


def test_refresh_refuses_an_empty_response(stored):
    stored.setattr(results_data, "fetch_season", lambda year: _season([]))
    with pytest.raises(RuntimeError):
        results_data.refresh_season(2030)
