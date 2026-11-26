"""
How good are the title predictions? Replay past seasons.

After every round from round 4 onward of each past season, each method
says who it thinks will be champion and how sure it is, using only what
was known at that point. Three methods are compared:

- simulation:    the rest-of-season simulation in src.season_simulation;
- leader:        "whoever leads the standings wins", with full confidence;
- leader_hedged: the same pick, but only as confident as the leader rule
                 has been right in the other seasons, with the rest of the
                 probability on second place. This is the fair comparison
                 for a method that outputs probabilities;
- lstm:          the LSTM classifier, outputs rescaled to sum to 1.

Rounds where the title was already decided by arithmetic are left out:
every method gets those right and they would flatter all of them.

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

from .points import max_points_remaining
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


def hedged_leader_probabilities(points, confidence):
    """Leader at `confidence`, the rest on whoever is second, everyone else at zero."""
    ranked = sorted(points, key=points.get, reverse=True)
    probabilities = {name: 0.0 for name in points}
    probabilities[ranked[0]] = confidence
    if len(ranked) > 1:
        probabilities[ranked[1]] = 1.0 - confidence
    return probabilities


CONFIDENCE_BANDS = (("under 50%", 0.0, 0.5), ("50–70%", 0.5, 0.7), ("70–90%", 0.7, 0.9),
                    ("90–99%", 0.9, 0.99), ("99–100%", 0.99, 1.0000001))


def calibration_table(rows, method):
    """
    For one method: when it was this confident in its top pick, how often
    was the pick right? A well-calibrated method is right about as often as
    its confidence says.
    """
    table = []
    for band, low, high in CONFIDENCE_BANDS:
        inside = [r for r in rows if r["method"] == method and low <= r["p_pick"] < high]
        if inside:
            table.append({
                "band": band, "n": len(inside),
                "mean_confidence": float(np.mean([r["p_pick"] for r in inside])),
                "share_correct": float(np.mean([r["correct"] for r in inside])),
            })
    return table


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

    rows, snapshots, decided = [], [], 0
    for year in years:
        season = standings[standings["Year"] == year]
        sprint = calendar[calendar["Year"] == year].set_index("Round")["IsSprint"]
        total_rounds = int(season["Round"].max())
        final = season[season["Round"] == total_rounds]
        champion = final.sort_values("ChampionshipPosition").iloc[0]["Driver"]
        for rnd in range(FIRST_ROUND, total_rounds):
            now = season[season["Round"] == rnd]
            points = dict(zip(now["Driver"], now["CumulativePoints"]))
            ranked = sorted(points.values(), reverse=True)
            left = [r for r in sprint.index if r > rnd]
            available = max_points_remaining(
                year, left, [bool(sprint[r]) for r in left], total_rounds)
            if ranked[0] - ranked[1] > available:
                decided += 1
                continue
            snapshot = {"year": int(year), "round": rnd, "champion": champion,
                        "stage": stage_of(rnd, total_rounds), "points": points}
            snapshots.append(snapshot)
            methods = {
                "simulation": simulation_probabilities(results, calendar, year, rnd)["drivers"],
                "leader": leader_probabilities(points),
            }
            if model is not None:
                methods["lstm"] = lstm_probabilities(model, standings, year, rnd, total_rounds)
            for method, probabilities in methods.items():
                rows.append(_row(snapshot, method, probabilities))

    # Hedged leader: confidence learned only from the other seasons.
    leader_rows = [r for r in rows if r["method"] == "leader"]
    for snapshot in snapshots:
        others = [r["correct"] for r in leader_rows if r["year"] != snapshot["year"]]
        confidence = float(np.mean(others))
        rows.append(_row(snapshot, "leader_hedged",
                         hedged_leader_probabilities(snapshot["points"], confidence)))

    def subset(**conditions):
        return [r for r in rows if all(r[k] == v for k, v in conditions.items())]

    return {
        "years": [int(y) for y in years],
        "first_round": FIRST_ROUND,
        "n_sims": BACKTEST_SIMS,
        "n_predictions": len(snapshots),
        "n_decided_excluded": decided,
        "lstm_trained_through": LSTM_TRAINED_THROUGH,
        "overall": summarise(rows),
        "calibration": calibration_table(rows, "simulation"),
        "by_stage": {stage: summarise(subset(stage=stage))
                     for stage in ("First third", "Middle third", "Final third")},
        "by_season": {str(year): {"champion": subset(year=year)[0]["champion"],
                                  **summarise(subset(year=year))}
                      for year in years if subset(year=year)},
    }


def _row(snapshot, method, probabilities):
    return {"year": snapshot["year"], "round": snapshot["round"], "method": method,
            "stage": snapshot["stage"], "champion": snapshot["champion"],
            "p_pick": float(max(probabilities.values())),
            **score(probabilities, snapshot["champion"])}


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
