"""
The current prediction and the README sections generated from results files.

README.md has two generated blocks, between <!-- prediction:start/end -->
and <!-- backtest:start/end -->. They are always rendered from
results/prediction.json and results/backtest.json, so the README cannot
disagree with the numbers of record.
"""

import json
import os

from .points import max_points_remaining
from .season_simulation import DEFAULT_SIMS, simulate_title
from .standings import build_standings, round_points, team_round_points

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PREDICTION_PATH = os.path.join(RESULTS_DIR, "prediction.json")
BACKTEST_PATH = os.path.join(RESULTS_DIR, "backtest.json")
README_PATH = os.path.join(BASE_DIR, "README.md")

MIN_ROUND_TO_PUBLISH = 6
METHOD_NAMES = {"simulation": "Season simulation", "leader_hedged": "Points leader, hedged",
                "leader": "Points leader, certain", "lstm": "LSTM classifier"}
METHOD_ORDER = ("simulation", "leader_hedged", "leader", "lstm")


def build_prediction(results, calendar, year, n_sims=DEFAULT_SIMS):
    """Title probabilities for `year` after the latest round in `results`."""
    season = results[results["Year"] == year]
    rnd = int(season["Round"].max())
    sprint = calendar[calendar["Year"] == year].set_index("Round")["IsSprint"]
    rounds_left = [r for r in sprint.index if r > rnd]
    remaining = [bool(sprint[r]) for r in rounds_left]
    simulated = simulate_title(
        round_points(results, year), team_round_points(results, year),
        {r: bool(sprint[r]) for r in sprint.index if r <= rnd}, remaining, n_sims=n_sims,
    )

    standings = build_standings(season)
    now = standings[standings["Round"] == rnd].sort_values("ChampionshipPosition")
    drivers = [
        {"driver": row.Driver, "team": row.Team, "points": float(row.CumulativePoints),
         "probability": simulated["drivers"].get(row.Driver, 0.0)}
        for row in now.itertuples()
    ]
    drivers.sort(key=lambda d: (-d["probability"], -d["points"]))
    team_points = team_round_points(results, year).sum()
    teams = sorted(
        ({"team": team, "points": float(points),
          "probability": simulated["teams"].get(team, 0.0)}
         for team, points in team_points.items()),
        key=lambda t: (-t["probability"], -t["points"]),
    )

    # Arithmetic check that does not depend on the simulation.
    per_driver = max_points_remaining(year, rounds_left, remaining, int(sprint.index.max()))
    by_points = sorted((d["points"] for d in drivers), reverse=True)
    lead = by_points[0] - by_points[1] if len(by_points) > 1 else float("inf")
    return {
        "season": int(year),
        "round": rnd,
        "event": str(season[season["Round"] == rnd]["EventName"].iloc[0]),
        "rounds_total": int(sprint.index.max()),
        "rounds_remaining": len(remaining),
        "sprints_remaining": int(sum(remaining)),
        "points_still_available": int(per_driver),
        "drivers_title_clinched": bool(lead > per_driver),
        "too_early": bool(rnd < MIN_ROUND_TO_PUBLISH and lead <= per_driver),
        "n_sims": n_sims,
        "drivers": drivers,
        "teams": teams,
    }


def _plural(count, word):
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def render_prediction_block(prediction):
    season, leader = prediction["season"], prediction["drivers"][0]
    left = _plural(prediction["rounds_remaining"], "round") + " left"
    if prediction["sprints_remaining"]:
        left += f" ({_plural(prediction['sprints_remaining'], 'sprint weekend')} among them)"
    lines = [
        f"**{season} season after round {prediction['round']} of "
        f"{prediction['rounds_total']} ({prediction['event']}).** {left}; a driver can still "
        f"score at most {prediction['points_still_available']} points.",
        "",
    ]
    if prediction["too_early"]:
        by_points = sorted(prediction["drivers"], key=lambda d: -d["points"])
        lines += [
            f"**Too early to call.** Probabilities are published from round "
            f"{MIN_ROUND_TO_PUBLISH}. Before that the simulation has too few rounds to draw "
            "on and, replayed on past seasons, it was confidently wrong this early.",
            "",
            "| Driver | Team | Points |", "|---|---|---|",
        ]
        lines += [f"| {d['driver']} | {d['team']} | {d['points']:.0f} |" for d in by_points[:6]]
        return "\n".join(lines)

    if prediction["drivers_title_clinched"]:
        lines.append(
            f"**{leader['driver']} has won the {season} drivers' championship**: the lead is "
            "larger than the points still available."
        )
    else:
        lines.append(
            f"The simulation makes {leader['driver']} the favourite at "
            f"{leader['probability']:.1%}. Read that with the calibration table below in "
            "mind: the simulation tends to be more confident than it should be."
        )
    lines += ["", "| Driver | Team | Points | Title probability |", "|---|---|---|---|"]
    for d in prediction["drivers"][:6]:
        lines.append(
            f"| {d['driver']} | {d['team']} | {d['points']:.0f} | {d['probability']:.1%} |"
        )
    lines += ["", "| Team | Points | Constructors' title probability |", "|---|---|---|"]
    for t in prediction["teams"][:4]:
        lines.append(f"| {t['team']} | {t['points']:.0f} | {t['probability']:.1%} |")
    lines += ["", f"From {prediction['n_sims']:,} simulated seasons. Only the top six drivers "
                  "and top four teams are shown."]
    return "\n".join(lines)


def _method_table(summary, header):
    lines = [f"| {header} | Top pick was the champion | Probability given to the champion "
             "| Brier score (lower is better) |", "|---|---|---|---|"]
    for method in METHOD_ORDER:
        if method in summary:
            s = summary[method]
            lines.append(f"| {METHOD_NAMES[method]} | {s['accuracy']:.1%} "
                         f"| {s['p_champion']:.1%} | {s['brier']:.3f} |")
    return lines


def _cells(summary, key, fmt):
    return " | ".join(
        format(summary[m][key], fmt) if m in summary else "—" for m in METHOD_ORDER
    )


def render_backtest_block(backtest):
    years = backtest["years"]
    header = " | ".join(METHOD_NAMES[m] for m in METHOD_ORDER)
    rule = "|".join(["---"] * (len(METHOD_ORDER) + 2))
    lines = [
        f"Seasons {years[0]}–{years[-1]}, one prediction after every round from round "
        f"{backtest['first_round']} to the second-last: {backtest['n_predictions']} "
        f"predictions per method. A further {backtest['n_decided_excluded']} rounds where "
        "the title was already decided by arithmetic are left out, since every method gets "
        f"those right. The simulation uses {backtest['n_sims']:,} runs per prediction here.",
        "",
        *_method_table(backtest["overall"], "All seasons"),
        "",
        "**Is the simulation as sure as it should be?** When its top pick had this much "
        "probability, this is how often the pick went on to win:",
        "",
        "| Simulation's confidence in its pick | Predictions | Average confidence "
        "| Pick was right |",
        "|---|---|---|---|",
    ]
    for band in backtest["calibration"]:
        lines.append(f"| {band['band']} | {band['n']} | {band['mean_confidence']:.0%} "
                     f"| {band['share_correct']:.0%} |")
    lines += ["", "By stage of the season (Brier score, lower is better):", "",
              f"| Stage | Predictions | {header} |", f"|{rule}|"]
    for stage, summary in backtest["by_stage"].items():
        lines.append(f"| {stage} | {summary['simulation']['n']} "
                     f"| {_cells(summary, 'brier', '.3f')} |")
    lines += ["", "By season (how often the top pick was the eventual champion):", "",
              f"| Season | Champion | {header} |", f"|{rule}|"]
    for year, summary in backtest["by_season"].items():
        lines.append(f"| {year} | {summary['champion']} "
                     f"| {_cells(summary, 'accuracy', '.0%')} |")
    return "\n".join(lines)


def read_block(readme_text, name):
    start_marker, end_marker = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
    start = readme_text.index(start_marker) + len(start_marker)
    return readme_text[start:readme_text.index(end_marker)].strip("\n")


def write_block(name, block, path=README_PATH):
    start_marker, end_marker = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
    with open(path) as handle:
        text = handle.read()
    start = text.index(start_marker) + len(start_marker)
    end = text.index(end_marker)
    with open(path, "w") as handle:
        handle.write(f"{text[:start]}\n{block}\n{text[end:]}")


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def refresh_readme():
    """Re-render both README blocks from the results files."""
    with open(PREDICTION_PATH) as handle:
        write_block("prediction", render_prediction_block(json.load(handle)))
    with open(BACKTEST_PATH) as handle:
        write_block("backtest", render_backtest_block(json.load(handle)))
