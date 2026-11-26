"""How many points a driver can still score, under each season's rules."""


def race_win_points(year):
    """25 for a win, plus the fastest-lap point that existed from 2019 to 2024."""
    return 26 if 2019 <= year <= 2024 else 25


def sprint_win_points(year):
    """Sprint winners scored 3 points in 2021 and 8 from 2022."""
    return 3 if year == 2021 else 8


def max_points_remaining(year, rounds, sprints, final_round):
    """
    The most one driver can score in the given remaining rounds.

    Args:
        rounds: round numbers still to run
        sprints: for each of those rounds, whether it is a sprint weekend
        final_round: the last round of the season (2014's finale paid double)
    """
    total = 0
    for rnd, is_sprint in zip(rounds, sprints):
        race = 50 if (year == 2014 and rnd == final_round) else race_win_points(year)
        total += race + (sprint_win_points(year) if is_sprint else 0)
    return total
