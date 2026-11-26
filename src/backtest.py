"""
How good are the title predictions? Replay past seasons.

After every round from round 4 onward of each past season, each method
says who it thinks will be champion and how sure it is, using only what
was known at that point. Three methods are compared:

- simulation: the rest-of-season simulation in src.season_simulation;
- leader:     "whoever leads the standings wins", with full confidence;
- lstm:       the LSTM classifier, outputs rescaled to sum to 1.

The simulation has nothing to train, so every season is new to it. The
LSTM was trained on 2014-2024 and 2025 was used to choose which training
run to keep, so it has no season that is completely new to it.

Scores:
- accuracy:   how often the top pick was the eventual champion;
- p_champion: the probability given to the eventual champion;
- brier:      squared error of the whole probability table (0 is perfect,
              2 is a certain wrong call). It punishes being confidently
              wrong, which accuracy does not.
"""

import json
import os
from collections import defaultdict

import numpy as np

from .results_data import load_calendar, load_results
from .season_simulation import simulate_title
from .standings import build_standings, round_points, team_round_points

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
BACKTEST_PATH = os.path.join(RESULTS_DIR, "backtest.json")
FIRST_ROUND = 4
BACKTEST_SIMS = 5000
LSTM_TRAINED_THROUGH = 2024


def brier_score(probabilities, champion):
    return float(sum(
        (p - (1.0 if name == champion else 0.0)) ** 2 for name, p in probabilities.items()
    ))


def leader_probabilities(points):
    leader = max(points, key=points.get)
    return {name: (1.0 if name == leader else 0.0) for name in points}


def score(probabilities, champion):
    pick = max(probabilities, key=probabilities.get)
    return {
        "correct": pick == champion,
        "p_champion": float(probabilities.get(champion, 0.0)),
        "brier": brier_score(probabilities, champion),
    }


def stage_of(rnd, total_rounds):
    share = rnd / total_rounds
    if share <= 1 / 3:
        return "First third"
    return "Middle third" if share <= 2 / 3 else "Final third"


def summarise(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    return {
        method: {
            "n": len(items),
            "accuracy": float(np.mean([r["correct"] for r in items])),
            "p_champion": float(np.mean([r["p_champion"] for r in items])),
            "brier": float(np.mean([r["brier"] for r in items])),
        }
        for method, items in grouped.items()
    }


def simulation_probabilities(results, calendar, year, rnd, n_sims=BACKTEST_SIMS):
    """Title probabilities after `rnd` rounds of `year`, from the simulation."""
    sprint = calendar[calendar["Year"] == year].set_index("Round")["IsSprint"]
    return simulate_title(
        round_points(results, year, through_round=rnd),
        team_round_points(results, year, through_round=rnd),
        {r: bool(sprint[r]) for r in sprint.index if r <= rnd},
        [bool(sprint[r]) for r in sprint.index if r > rnd],
        n_sims=n_sims,
    )


def lstm_probabilities(model, standings, year, rnd, total_rounds):
    """The LSTM's outputs for the top ten drivers, rescaled to sum to 1."""
    import torch

    from .feature_engineering import extract_features

    season = standings[(standings["Year"] == year) & (standings["Round"] <= rnd)]
    current = season[season["Round"] == rnd].sort_values("ChampionshipPosition")
    raw = {}
    for driver in current.head(10)["Driver"]:
        history = season[season["Driver"] == driver].sort_values("Round")
        features = extract_features(history, total_rounds)
        with torch.no_grad():
            logit = model(torch.FloatTensor(features).unsqueeze(0),
                          torch.LongTensor([features.shape[0]]))
        raw[driver] = torch.sigmoid(logit).item()
    total = sum(raw.values())
    return {driver: value / total for driver, value in raw.items()}


def load_lstm():
    """The trained LSTM, or None if torch or the checkpoint is unavailable."""
    try:
        import torch

        from .model import F1ChampionshipLSTM
        checkpoint = torch.load(os.path.join(BASE_DIR, "models", "best_model.pth"),
                                weights_only=False, map_location="cpu")
    except (ImportError, FileNotFoundError):
        return None
    model = F1ChampionshipLSTM(input_size=12, hidden_size=64, num_layers=2, dropout=0.3)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def run_backtest(years, results=None, calendar=None):
    results = load_results() if results is None else results
    calendar = load_calendar() if calendar is None else calendar
    standings = build_standings(results)
    model = load_lstm()

    rows = []
    for year in years:
        season = standings[standings["Year"] == year]
        total_rounds = int(season["Round"].max())
        final = season[season["Round"] == total_rounds]
        champion = final.sort_values("ChampionshipPosition").iloc[0]["Driver"]
        for rnd in range(FIRST_ROUND, total_rounds):
            now = season[season["Round"] == rnd]
            points = dict(zip(now["Driver"], now["CumulativePoints"]))
            methods = {
                "simulation": simulation_probabilities(results, calendar, year, rnd)["drivers"],
                "leader": leader_probabilities(points),
            }
            if model is not None:
                methods["lstm"] = lstm_probabilities(model, standings, year, rnd, total_rounds)
            for method, probabilities in methods.items():
                rows.append({"year": int(year), "round": rnd, "method": method,
                             "stage": stage_of(rnd, total_rounds), "champion": champion,
                             **score(probabilities, champion)})

    def subset(**conditions):
        return [r for r in rows if all(r[k] == v for k, v in conditions.items())]

    held_out = [r for r in rows if r["year"] > LSTM_TRAINED_THROUGH]
    return {
        "years": [int(y) for y in years],
        "first_round": FIRST_ROUND,
        "lstm_trained_through": LSTM_TRAINED_THROUGH,
        "overall": summarise(rows),
        "lstm_validation_season": summarise(held_out),
        "by_stage": {stage: summarise(subset(stage=stage))
                     for stage in ("First third", "Middle third", "Final third")},
        "by_season": {str(year): {"champion": subset(year=year)[0]["champion"],
                                  **summarise(subset(year=year))} for year in years},
    }


def main():
    backtest = run_backtest(range(2014, 2026))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(BACKTEST_PATH, "w") as handle:
        json.dump(backtest, handle, indent=2)
        handle.write("\n")
    for method, s in backtest["overall"].items():
        print(f"{method:11} n={s['n']} accuracy={s['accuracy']:.3f} "
              f"p_champion={s['p_champion']:.3f} brier={s['brier']:.3f}")


if __name__ == "__main__":
    main()
