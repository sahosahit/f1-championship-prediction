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
