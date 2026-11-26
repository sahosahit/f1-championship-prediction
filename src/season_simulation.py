"""
Who wins the championship? Simulate the rest of the season.

Each remaining round is filled in with a round that has already happened
this season, drawn at random: every driver and team gets the points they
actually scored in the drawn round. Repeating this thousands of times gives
the share of simulated seasons each driver (and team) finishes on top.

Why this design:
- The probabilities add up to 100%, because each simulated season has one
  champion.
- A driver who cannot catch up gets 0%, because nobody can score more in a
  simulated round than someone really scored in a round.
- Points within a round stay consistent (two drivers cannot both win),
  because whole rounds are drawn, not individual results.
- Sprint weekends are drawn from past sprint weekends when there are any,
  since more points are on offer.

Limit: it assumes the rest of the season looks like the season so far. It
knows nothing about car upgrades, driver changes or which circuits remain,
and early in a season it has very few rounds to draw from.
"""

import numpy as np

DEFAULT_SIMS = 20000


def _draw_rounds(past_sprint, remaining, n_sims, rng):
    """For each remaining round, the index of the past round drawn in each simulation."""
    flags = np.array(list(past_sprint), dtype=bool)
    every_round = np.arange(len(flags))
    draws = []
    for is_sprint in remaining:
        pool = every_round[flags == is_sprint]
        if len(pool) == 0:          # no past weekend of this kind yet
            pool = every_round
        draws.append(pool[rng.integers(len(pool), size=n_sims)])
    return draws


def _title_shares(points, draws, n_sims, rng):
    """Share of simulations each column (driver or team) finishes with the most points."""
    values = points.to_numpy(dtype=float)
    totals = np.tile(values.sum(axis=0), (n_sims, 1))
    for drawn in draws:
        totals += values[drawn]
    # Level on points: split at random. (Real ties go to countback on wins.)
    totals += rng.uniform(0.0, 1e-6, size=totals.shape)
    winners = np.bincount(totals.argmax(axis=1), minlength=values.shape[1])
    return {name: float(count / n_sims) for name, count in zip(points.columns, winners)}


def simulate_title(driver_points, team_points, past_sprint, remaining,
                   n_sims=DEFAULT_SIMS, seed=42):
    """
    Args:
        driver_points: rounds x drivers table of points scored so far this season
        team_points: rounds x teams table for the same rounds
        past_sprint: {round: whether it was a sprint weekend} for those rounds
        remaining: one bool per round still to run (True for a sprint weekend)

    Returns:
        {"drivers": {driver: probability}, "teams": {team: probability},
         "n_sims": ..., "rounds_remaining": ...}
    """
    rng = np.random.default_rng(seed)
    flags = [bool(past_sprint[rnd]) for rnd in driver_points.index]
    draws = _draw_rounds(flags, [bool(r) for r in remaining], n_sims, rng)
    return {
        "drivers": _title_shares(driver_points, draws, n_sims, rng),
        "teams": _title_shares(team_points, draws, n_sims, rng),
        "n_sims": n_sims,
        "rounds_remaining": len(remaining),
    }
