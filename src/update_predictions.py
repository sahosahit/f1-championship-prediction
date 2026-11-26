"""
Automated prediction update, run by GitHub Actions every Monday.

    python src/update_predictions.py

Steps:
1. Re-download the current season's results and calendar from the official
   results API. The whole season is replaced each time, so late corrections
   (penalties, disqualifications) are picked up and nothing depends on the
   previous run.
2. Rebuild data/championship_standings.csv from the results.
3. Run the rest-of-season simulation and write results/prediction.json.
4. Re-render the generated blocks in README.md and the prediction chart.

It is safe to run when there is no new race: the outputs simply do not change.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.report import (
    PREDICTION_PATH, RESULTS_DIR, build_prediction, refresh_readme, save_json,
)
from src.results_data import DATA_DIR, load_calendar, load_results, refresh_season
from src.standings import build_standings

SEASON = 2026
STANDINGS_PATH = os.path.join(DATA_DIR, "championship_standings.csv")
CHART_PATH = os.path.join(RESULTS_DIR, f"{SEASON}_prediction.png")


def latest_round(results, year):
    season = results[results["Year"] == year]
    return 0 if season.empty else int(season["Round"].max())


def update_prediction_chart(prediction, path=CHART_PATH):
    """Bar chart of title probability for the leading drivers."""
    drivers = prediction["drivers"][:8][::-1]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.barh([d["driver"] for d in drivers], [d["probability"] * 100 for d in drivers],
              color="#1f77b4")
    for index, d in enumerate(drivers):
        axis.text(d["probability"] * 100 + 1, index, f"{d['probability']:.1%}", va="center")
    axis.set_xlim(0, 112)
    axis.set_xlabel("Title probability (%)")
    axis.set_title(
        f"{prediction['season']} drivers' title after round {prediction['round']} "
        f"of {prediction['rounds_total']}"
    )
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main():
    before = latest_round(load_results(), SEASON)
    print(f"Current latest round in data: {before}")
    print(f"Checking for Round {before + 1}...")

    refresh_season(SEASON)
    results, calendar = load_results(), load_calendar()
    after = latest_round(results, SEASON)
    if after > before:
        event = results[(results["Year"] == SEASON) & (results["Round"] == after)]
        print(f"Round {after} ({event['EventName'].iloc[0]}): Found!")
    else:
        print(f"Round {before + 1}: No data available yet.")

    build_standings(results).to_csv(STANDINGS_PATH, index=False)

    prediction = build_prediction(results, calendar, SEASON)
    save_json(prediction, PREDICTION_PATH)
    refresh_readme()
    update_prediction_chart(prediction)

    print(f"\n{SEASON} title probabilities after round {prediction['round']}:")
    for d in prediction["drivers"][:5]:
        print(f"  {d['driver']:4s} {d['team']:20s} {d['points']:4.0f} pts  {d['probability']:.1%}")


if __name__ == "__main__":
    main()
