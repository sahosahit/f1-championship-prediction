"""
Feature Engineering for F1 Championship Prediction
Creates sequential features for LSTM model training.
"""

import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')


def load_standings():
    """Load championship standings data."""
    return pd.read_csv(os.path.join(DATA_DIR, 'championship_standings.csv'))


def get_champion(standings, year):
    """Get the championship winner for a given year."""
    year_data = standings[standings['Year'] == year]
    final_round = year_data['Round'].max()
    final_standings = year_data[year_data['Round'] == final_round]
    winner = final_standings.sort_values('CumulativePoints', ascending=False).iloc[0]
    return winner['Driver']


def create_sequences(standings, min_races=4, top_n_drivers=10):
    """
    Create sequential training data for LSTM.

    For each season, at each prediction point (after Race K):
    - Input: Driver's feature sequence from Race 1 to Race K
    - Target: Did this driver win the championship? (1/0)

    Args:
        standings: DataFrame with championship standings
        min_races: Minimum races before making predictions (default: 4)
        top_n_drivers: Only consider top N drivers at each prediction point

    Returns:
        sequences: List of dicts with 'features', 'target', 'metadata'
    """
    sequences = []

    # Get complete seasons (exclude 2026 which is ongoing)
    complete_seasons = sorted(standings['Year'].unique())

    # Get champion for each complete season
    champions = {}
    for year in complete_seasons:
        try:
            champions[year] = get_champion(standings, year)
        except:
            pass  # 2026 is ongoing, skip

    print(f"Champions: {champions}")
    print(f"Creating sequences (min_races={min_races}, top_n={top_n_drivers})...")

    for year in complete_seasons:
        if year not in champions:
            continue  # Skip incomplete seasons for training

        year_data = standings[standings['Year'] == year]
        all_rounds = sorted(year_data['Round'].unique())
        total_rounds = len(all_rounds)
        champion = champions[year]

        # For each prediction point (after min_races to second-to-last race)
        for pred_idx in range(min_races, total_rounds):
            current_round = all_rounds[pred_idx - 1]  # 0-indexed

            # Get standings up to current round
            current_standings = year_data[year_data['Round'] == current_round]
            current_standings = current_standings.sort_values('ChampionshipPosition')

            # Only consider top N drivers (reduces noise from backmarkers)
            top_drivers = current_standings.head(top_n_drivers)['Driver'].tolist()

            for driver in top_drivers:
                # Get this driver's sequence from Race 1 to current_round
                driver_history = year_data[
                    (year_data['Driver'] == driver) &
                    (year_data['Round'] <= current_round)
                ].sort_values('Round')

                if len(driver_history) < min_races:
                    continue

                # Extract features for each time step
                features = extract_features(driver_history, total_rounds)

                # Target: 1 if this driver won championship, 0 otherwise
                target = 1 if driver == champion else 0

                sequences.append({
                    'features': features,
                    'target': target,
                    'year': year,
                    'driver': driver,
                    'prediction_round': current_round,
                    'total_rounds': total_rounds,
                    'season_progress': current_round / total_rounds,
                })

    print(f"Total sequences created: {len(sequences)}")
    print(f"Positive (champion): {sum(s['target'] for s in sequences)}")
    print(f"Negative (non-champion): {sum(1 - s['target'] for s in sequences)}")

    return sequences


def extract_features(driver_history, total_rounds):
    """
    Extract normalized features from a driver's race-by-race history.

    Features per time step (12 features):
    1. points_normalized: CumulativePoints / max_possible_points_so_far
    2. position_normalized: 1 - (ChampionshipPosition - 1) / 19
    3. gap_normalized: PointsGapToLeader / max_possible_gap
    4. win_rate: WinRate (0-1)
    5. podium_rate: PodiumRate (0-1)
    6. points_per_race_norm: PointsPerRace / 26
    7. season_progress: Round / total_rounds
    8. consistency: 1 - DNF rate
    9. momentum: Change in PPR vs previous step (positive = improving)
    10. is_leader: 1 if P1, 0 otherwise
    11. points_share: Driver's points as fraction of total grid points
    12. recent_form: Points scored in last 3 races / max possible in 3 races
    """
    features = []
    max_points_per_race = 26

    prev_ppr = None
    prev_points = 0

    for idx, (_, row) in enumerate(driver_history.iterrows()):
        round_num = row['Round']
        races_completed = row['RacesCompleted']
        max_possible = races_completed * max_points_per_race

        # Momentum: change in points per race vs previous time step
        current_ppr = row['PointsPerRace']
        if prev_ppr is not None:
            momentum = (current_ppr - prev_ppr) / max_points_per_race
        else:
            momentum = 0.0
        prev_ppr = current_ppr

        # Recent form: points scored this race / max per race
        points_this_race = row['CumulativePoints'] - prev_points
        recent_form = points_this_race / max_points_per_race
        prev_points = row['CumulativePoints']

        step_features = [
            # 1. Points normalized
            row['CumulativePoints'] / max_possible if max_possible > 0 else 0,
            # 2. Position normalized (higher = better)
            1 - (row['ChampionshipPosition'] - 1) / 19.0,
            # 3. Gap to leader normalized (0 = leader)
            min(row['PointsGapToLeader'] / (max_possible * 0.5), 1.0) if max_possible > 0 else 0,
            # 4. Win rate
            row['WinRate'],
            # 5. Podium rate
            row['PodiumRate'],
            # 6. Points per race normalized
            row['PointsPerRace'] / max_points_per_race,
            # 7. Season progress
            round_num / total_rounds,
            # 8. Consistency
            1 - (row['DNFs'] / races_completed if races_completed > 0 else 0),
            # 9. Momentum (positive = improving form)
            np.clip(momentum, -1, 1),
            # 10. Is championship leader
            1.0 if row['ChampionshipPosition'] == 1 else 0.0,
            # 11. Points share (how much of available points this driver has)
            row['CumulativePoints'] / (max_possible * 0.5) if max_possible > 0 else 0,
            # 12. Recent form (this race's points / max)
            recent_form,
        ]

        features.append(step_features)

    return np.array(features, dtype=np.float32)


def create_prediction_input(standings, year, round_num, top_n=10):
    """
    Create input features for making predictions on an ongoing season.
    Used for 2026 prediction.

    Returns list of (driver, features) tuples.
    """
    year_data = standings[standings['Year'] == year]
    all_rounds = sorted(year_data['Round'].unique())
    total_rounds_estimate = 22  # Estimate for ongoing season

    # Get current standings
    current = year_data[year_data['Round'] == round_num]
    current = current.sort_values('ChampionshipPosition')
    top_drivers = current.head(top_n)['Driver'].tolist()

    prediction_inputs = []

    for driver in top_drivers:
        driver_history = year_data[
            (year_data['Driver'] == driver) &
            (year_data['Round'] <= round_num)
        ].sort_values('Round')

        if driver_history.empty:
            continue

        features = extract_features(driver_history, total_rounds_estimate)
        prediction_inputs.append({
            'driver': driver,
            'team': driver_history.iloc[-1]['Team'],
            'features': features,
            'current_points': driver_history.iloc[-1]['CumulativePoints'],
            'current_position': driver_history.iloc[-1]['ChampionshipPosition'],
        })

    return prediction_inputs


if __name__ == "__main__":
    standings = load_standings()

    # Create training sequences
    sequences = create_sequences(standings, min_races=4, top_n_drivers=10)

    # Print summary
    print(f"\n{'='*50}")
    print("SEQUENCE SUMMARY:")
    print(f"{'='*50}")

    # By year
    print("\nSequences per year:")
    year_counts = {}
    for s in sequences:
        year_counts[s['year']] = year_counts.get(s['year'], 0) + 1
    for year in sorted(year_counts.keys()):
        champ_count = sum(1 for s in sequences if s['year'] == year and s['target'] == 1)
        print(f"  {year}: {year_counts[year]} sequences ({champ_count} positive)")

    # Feature shape
    print(f"\nFeature shape example: {sequences[0]['features'].shape}")
    print(f"Sequence lengths: {min(s['features'].shape[0] for s in sequences)} - {max(s['features'].shape[0] for s in sequences)}")

    # Create 2026 prediction input
    print(f"\n{'='*50}")
    print("2026 PREDICTION INPUT (After Race 6):")
    print(f"{'='*50}")
    pred_inputs = create_prediction_input(standings, 2026, 6, top_n=10)
    for p in pred_inputs:
        print(f"  {p['driver']:4s} ({p['team']:20s}) | Pts: {p['current_points']:5.0f} | Pos: P{int(p['current_position'])} | Seq len: {p['features'].shape[0]}")
