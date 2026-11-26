"""
Championship standings built from race results.

Standings are always rebuilt from the full list of results, never updated
one round at a time, so a driver who misses a round keeps their points and
a corrected result upstream corrects everything downstream.
"""

import pandas as pd

STANDINGS_COLUMNS = [
    "Year", "Round", "Driver", "Team", "CumulativePoints", "Wins", "Podiums", "DNFs",
    "RacesCompleted", "ChampionshipPosition", "PointsGapToLeader", "WinRate", "PodiumRate",
    "PointsPerRace",
]


def round_points(results, year, through_round=None):
    """Points each driver scored in each round (race plus sprint); 0 if they did not race."""
    season = results[results["Year"] == year]
    if through_round is not None:
        season = season[season["Round"] <= through_round]
    points = season.assign(Points=season["RacePoints"] + season["SprintPoints"])
    return points.pivot_table(
        index="Round", columns="Driver", values="Points", aggfunc="sum", fill_value=0.0
    ).sort_index()


def team_round_points(results, year, through_round=None):
    """Points each team scored in each round, credited to the team the driver was with."""
    season = results[results["Year"] == year]
    if through_round is not None:
        season = season[season["Round"] <= through_round]
    points = season.assign(Points=season["RacePoints"] + season["SprintPoints"])
    return points.pivot_table(
        index="Round", columns="Team", values="Points", aggfunc="sum", fill_value=0.0
    ).sort_index()


def _season_standings(season, year):
    entered = season.pivot_table(index="Round", columns="Driver", values="Position",
                                 aggfunc="count", fill_value=0).sort_index()
    rounds, drivers = entered.index, entered.columns

    def per_round(values):
        return (season.assign(Value=values)
                .pivot_table(index="Round", columns="Driver", values="Value",
                             aggfunc="sum", fill_value=0)
                .reindex(index=rounds, columns=drivers, fill_value=0))

    classified = season["Classified"].astype(bool)
    totals = {
        "CumulativePoints": per_round(season["RacePoints"] + season["SprintPoints"]).cumsum(),
        "Wins": per_round((season["Position"] == 1).astype(int)).cumsum(),
        "Podiums": per_round(((season["Position"] <= 3) & classified).astype(int)).cumsum(),
        "DNFs": per_round((~classified).astype(int)).cumsum(),
        "RacesCompleted": entered.cumsum(),
    }
    team = (season.pivot_table(index="Round", columns="Driver", values="Team", aggfunc="last")
            .reindex(index=rounds, columns=drivers).ffill())

    tables = []
    for rnd in rounds:
        table = pd.DataFrame({name: frame.loc[rnd] for name, frame in totals.items()})
        table = table[table["RacesCompleted"] > 0].copy()   # only drivers who have raced so far
        table["Team"] = team.loc[rnd]
        table = table.sort_values(["CumulativePoints", "Wins"], ascending=False)
        table["ChampionshipPosition"] = range(1, len(table) + 1)
        table["PointsGapToLeader"] = table["CumulativePoints"].max() - table["CumulativePoints"]
        table["WinRate"] = table["Wins"] / table["RacesCompleted"]
        table["PodiumRate"] = table["Podiums"] / table["RacesCompleted"]
        table["PointsPerRace"] = table["CumulativePoints"] / table["RacesCompleted"]
        table["Year"], table["Round"] = year, rnd
        tables.append(table.rename_axis("Driver").reset_index())
    return pd.concat(tables, ignore_index=True)


def build_standings(results):
    """Standings after every round of every season in `results`."""
    seasons = [
        _season_standings(season, year) for year, season in results.groupby("Year", sort=True)
    ]
    return pd.concat(seasons, ignore_index=True)[STANDINGS_COLUMNS]
