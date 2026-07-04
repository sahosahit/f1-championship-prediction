"""
Feature Engineering for F1 Constructor Championship Prediction.
Aggregates team-level performance into sequential features for LSTM.
"""

import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')


def load_standings():
    return pd.read_csv(os.path.join(DATA_DIR, 'championship_standings.csv'))


def build_constructor_standings(standings):
    """
    Aggregate driver standings into constructor (team) standings.
    Sums points, wins, podiums across both drivers per team per round.
    """
    constructor_data = []

    for (year, round_num), group in standings.groupby(['Year', 'Round']):
        team_stats = group.groupby('Team').agg(
            CumulativePoints=('CumulativePoints', 'sum'),
            Wins=('Wins', 'sum'),
            Podiums=('Podiums', 'sum'),
            DNFs=('DNFs', 'sum'),
            RacesCompleted=('RacesCompleted', 'sum'),
            NumDrivers=('Driver', 'count'),
            BestDriverPosition=('ChampionshipPosition', 'min'),
        ).reset_index()

        team_stats = team_stats.sort_values('CumulativePoints', ascending=False).reset_index(drop=True)
        team_stats['ConstructorPosition'] = team_stats.index + 1
        max_pts = team_stats['CumulativePoints'].max()
        team_stats['PointsGapToLeader'] = max_pts - team_stats['CumulativePoints']
        team_stats['WinRate'] = team_stats['Wins'] / team_stats['RacesCompleted'].clip(lower=1)
        team_stats['PodiumRate'] = team_stats['Podiums'] / team_stats['RacesCompleted'].clip(lower=1)
        team_stats['PointsPerRace'] = team_stats['CumulativePoints'] / (team_stats['RacesCompleted'] / team_stats['NumDrivers']).clip(lower=1)
        team_stats['Year'] = year
        team_stats['Round'] = round_num

        constructor_data.append(team_stats)

    return pd.concat(constructor_data, ignore_index=True)


def get_constructor_champion(constructor_standings, year):
    year_data = constructor_standings[constructor_standings['Year'] == year]
    final_round = year_data['Round'].max()
    final = year_data[year_data['Round'] == final_round]
    return final.sort_values('CumulativePoints', ascending=False).iloc[0]['Team']


def extract_constructor_features(team_history, total_rounds):
    """
    Extract normalized features from a team's race-by-race history.

    Features per time step (12 features):
    1. points_normalized: CumulativePoints / max_possible
    2. position_normalized: 1 - (position-1) / 9 (10 teams)
    3. gap_normalized: PointsGapToLeader / max_possible_gap
    4. win_rate: Team win rate
    5. podium_rate: Team podium rate
    6. points_per_race_norm: PPR / (26*2) (max for 2 drivers)
    7. season_progress: Round / total_rounds
    8. consistency: 1 - DNF rate
    9. momentum: Change in PPR vs previous step
    10. is_leader: 1 if P1 in constructors, 0 otherwise
    11. points_share: Team points / total available
    12. driver_spread: How evenly points are distributed (entropy proxy)
    """
    features = []
    max_points_per_race = 52  # 25+1(FL) + 18+1(FL) theoretical max for 2 drivers

    prev_ppr = None
    prev_points = 0

    for _, row in team_history.iterrows():
        round_num = row['Round']
        races_per_driver = row['RacesCompleted'] / max(row['NumDrivers'], 1)
        max_possible = races_per_driver * max_points_per_race

        current_ppr = row['PointsPerRace']
        if prev_ppr is not None:
            momentum = (current_ppr - prev_ppr) / max_points_per_race
        else:
            momentum = 0.0
        prev_ppr = current_ppr

        points_this_round = row['CumulativePoints'] - prev_points
        recent_form = points_this_round / max_points_per_race
        prev_points = row['CumulativePoints']

        step_features = [
            row['CumulativePoints'] / max_possible if max_possible > 0 else 0,
            1 - (row['ConstructorPosition'] - 1) / 9.0,
            min(row['PointsGapToLeader'] / (max_possible * 0.5), 1.0) if max_possible > 0 else 0,
            row['WinRate'],
            row['PodiumRate'],
            row['PointsPerRace'] / max_points_per_race,
            round_num / total_rounds,
            1 - (row['DNFs'] / row['RacesCompleted'] if row['RacesCompleted'] > 0 else 0),
            np.clip(momentum, -1, 1),
            1.0 if row['ConstructorPosition'] == 1 else 0.0,
            row['CumulativePoints'] / (max_possible * 0.5) if max_possible > 0 else 0,
            recent_form,
        ]

        features.append(step_features)

    return np.array(features, dtype=np.float32)


def create_constructor_sequences(constructor_standings, min_races=4, top_n_teams=10):
    """
    Create sequential training data for Constructor Championship LSTM.
    """
    sequences = []
    complete_seasons = sorted(constructor_standings['Year'].unique())

    champions = {}
    for year in complete_seasons:
        try:
            champions[year] = get_constructor_champion(constructor_standings, year)
        except Exception:
            pass

    for year in complete_seasons:
        if year not in champions:
            continue

        year_data = constructor_standings[constructor_standings['Year'] == year]
        all_rounds = sorted(year_data['Round'].unique())
        total_rounds = len(all_rounds)
        champion = champions[year]

        for pred_idx in range(min_races, total_rounds):
            current_round = all_rounds[pred_idx - 1]
            current_standings = year_data[year_data['Round'] == current_round]
            current_standings = current_standings.sort_values('ConstructorPosition')
            top_teams = current_standings.head(top_n_teams)['Team'].tolist()

            for team in top_teams:
                team_history = year_data[
                    (year_data['Team'] == team) &
                    (year_data['Round'] <= current_round)
                ].sort_values('Round')

                if len(team_history) < min_races:
                    continue

                features = extract_constructor_features(team_history, total_rounds)
                target = 1 if team == champion else 0

                sequences.append({
                    'features': features,
                    'target': target,
                    'year': year,
                    'team': team,
                    'prediction_round': current_round,
                    'total_rounds': total_rounds,
                    'season_progress': current_round / total_rounds,
                })

    return sequences


def create_constructor_prediction_input(constructor_standings, year, round_num, top_n=10):
    """Create input features for predicting ongoing constructor championship."""
    year_data = constructor_standings[constructor_standings['Year'] == year]
    total_rounds_estimate = 22

    current = year_data[year_data['Round'] == round_num]
    current = current.sort_values('ConstructorPosition')
    top_teams = current.head(top_n)['Team'].tolist()

    prediction_inputs = []

    for team in top_teams:
        team_history = year_data[
            (year_data['Team'] == team) &
            (year_data['Round'] <= round_num)
        ].sort_values('Round')

        if team_history.empty:
            continue

        features = extract_constructor_features(team_history, total_rounds_estimate)
        prediction_inputs.append({
            'team': team,
            'features': features,
            'current_points': team_history.iloc[-1]['CumulativePoints'],
            'current_position': team_history.iloc[-1]['ConstructorPosition'],
        })

    return prediction_inputs


if __name__ == "__main__":
    standings = load_standings()
    constructor_standings = build_constructor_standings(standings)

    print(f"Constructor standings shape: {constructor_standings.shape}")
    print(f"\nSeasons: {sorted(constructor_standings['Year'].unique())}")

    sequences = create_constructor_sequences(constructor_standings, min_races=4, top_n_teams=10)
    print(f"\nTotal sequences: {len(sequences)}")
    print(f"Positive (champion): {sum(s['target'] for s in sequences)}")
    print(f"Negative: {sum(1 - s['target'] for s in sequences)}")
    print(f"Feature shape: {sequences[0]['features'].shape}")
