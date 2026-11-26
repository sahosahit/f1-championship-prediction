"""
Official race, sprint and calendar data from the Jolpica API (the successor
to Ergast): https://api.jolpi.ca

Everything the project knows about a season is rebuilt from these results,
so the standings can never drift away from the official ones.
"""

import os
import time

import pandas as pd
import requests

API_URL = "https://api.jolpi.ca/ergast/f1"
PAGE_SIZE = 100
REQUEST_INTERVAL_SEC = 0.6   # the API allows a few requests per second
RETRY_WAITS_SEC = (5, 30)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_PATH = os.path.join(DATA_DIR, "season_results.csv")
CALENDAR_PATH = os.path.join(DATA_DIR, "calendar.csv")


def _driver_code(driver):
    return driver.get("code") or driver["driverId"][:3].upper()


def parse_race_results(races):
    """One row per driver per race."""
    rows = []
    for race in races:
        for result in race.get("Results", []):
            rows.append({
                "Year": int(race["season"]),
                "Round": int(race["round"]),
                "EventName": race["raceName"],
                "Driver": _driver_code(result["Driver"]),
                "Team": result["Constructor"]["name"],
                "Position": int(result["position"]),
                "Classified": result["positionText"].isdigit(),
                "RacePoints": float(result["points"]),
                "Status": result["status"],
            })
    return rows


def parse_sprint_results(races):
    """Sprint points per driver per round."""
    return [
        {
            "Year": int(race["season"]),
            "Round": int(race["round"]),
            "Driver": _driver_code(result["Driver"]),
            "SprintPoints": float(result["points"]),
        }
        for race in races
        for result in race.get("SprintResults", [])
    ]


def parse_calendar(races):
    """One row per scheduled round, with whether it is a sprint weekend."""
    return [
        {
            "Year": int(race["season"]),
            "Round": int(race["round"]),
            "EventName": race["raceName"],
            "Date": race["date"],
            "IsSprint": "Sprint" in race,
        }
        for race in races
    ]


def _get_all(path):
    """Fetch every page of an endpoint and return the combined list of races."""
    races, offset = [], 0
    while True:
        for wait in RETRY_WAITS_SEC + (None,):
            time.sleep(REQUEST_INTERVAL_SEC)
            response = requests.get(
                f"{API_URL}/{path}.json",
                params={"limit": PAGE_SIZE, "offset": offset},
                timeout=60,
            )
            busy = response.status_code == 429 or response.status_code >= 500
            if busy and wait is not None:
                time.sleep(wait)
                continue
            response.raise_for_status()
            break
        data = response.json()["MRData"]
        races.extend(data["RaceTable"]["Races"])
        offset += PAGE_SIZE
        if offset >= int(data["total"]):
            return races


def fetch_season(year):
    """Race and sprint results for one season, sprint points merged in."""
    results = pd.DataFrame(parse_race_results(_get_all(f"{year}/results")))
    results["SprintPoints"] = 0.0
    sprints = pd.DataFrame(parse_sprint_results(_get_all(f"{year}/sprint")))
    if not sprints.empty:
        results = results.drop(columns="SprintPoints").merge(
            sprints, on=["Year", "Round", "Driver"], how="left"
        )
        results["SprintPoints"] = results["SprintPoints"].fillna(0.0)
    return results.sort_values(["Round", "Position"]).reset_index(drop=True)


def fetch_calendar(year):
    return pd.DataFrame(parse_calendar(_get_all(str(year))))


def load_results():
    return pd.read_csv(RESULTS_PATH)


def load_calendar():
    return pd.read_csv(CALENDAR_PATH)


def refresh_season(year):
    """Replace one season in the stored results and calendar with fresh data."""
    for path, fresh in ((RESULTS_PATH, fetch_season(year)), (CALENDAR_PATH, fetch_calendar(year))):
        if os.path.exists(path):
            stored = pd.read_csv(path)
            fresh = pd.concat([stored[stored["Year"] != year], fresh], ignore_index=True)
        fresh.sort_values(["Year", "Round"], kind="stable").to_csv(path, index=False)
