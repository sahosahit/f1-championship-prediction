"""
Automated Prediction Update Script
Triggered by GitHub Actions after each race weekend.

Steps:
1. Check if new 2026 race data is available via FastF1
2. Update championship_standings.csv with new race
3. Run model prediction with updated data
4. Update results/evaluation_results.json
5. Update README.md with latest predictions
"""

import fastf1
import pandas as pd
import numpy as np
import torch
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import F1ChampionshipLSTM
from feature_engineering import extract_features

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
CACHE_DIR = os.path.join(BASE_DIR, 'cache')

os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)


def get_latest_round_in_data():
    """Get the latest 2026 round in our dataset."""
    standings = pd.read_csv(os.path.join(DATA_DIR, 'championship_standings.csv'))
    data_2026 = standings[standings['Year'] == 2026]
    if data_2026.empty:
        return 0
    return int(data_2026['Round'].max())


def check_for_new_race():
    """Check if there's a new 2026 race available that we don't have."""
    current_latest = get_latest_round_in_data()
    next_round = current_latest + 1

    print(f"Current latest round in data: {current_latest}")
    print(f"Checking for Round {next_round}...")

    try:
        session = fastf1.get_session(2026, next_round, 'R')
        session.load(telemetry=False, weather=False, messages=False, laps=False)
        results = session.results

        if results is None or results.empty:
            print(f"Round {next_round}: No data available yet.")
            return None, None, None

        race_data = results[['DriverNumber', 'Abbreviation', 'TeamName',
                            'Position', 'Points', 'Status', 'GridPosition']].copy()
        race_data['Year'] = 2026
        race_data['Round'] = next_round
        race_data['EventName'] = session.event['EventName']

        print(f"Round {next_round} ({session.event['EventName']}): Found! {len(race_data)} drivers")

        # Check for sprint race
        sprint_data = None
        try:
            sprint = fastf1.get_session(2026, next_round, 'S')
            sprint.load(telemetry=False, weather=False, messages=False, laps=False)
            sprint_results = sprint.results
            if sprint_results is not None and not sprint_results.empty:
                sprint_data = sprint_results[['Abbreviation', 'Points']].copy()
                print(f"  Sprint race found! {len(sprint_data)} drivers")
        except Exception:
            print(f"  No sprint race for this round")

        return next_round, race_data, sprint_data

    except Exception as e:
        print(f"Round {next_round}: Not available ({e})")
        return None, None, None


def update_standings(new_race_data, new_round, sprint_data=None):
    """Add new race + sprint results to championship standings."""
    standings = pd.read_csv(os.path.join(DATA_DIR, 'championship_standings.csv'))

    # Get previous 2026 standings to calculate cumulative stats
    data_2026 = standings[standings['Year'] == 2026]
    prev_round = data_2026['Round'].max()
    prev_standings = data_2026[data_2026['Round'] == prev_round]

    # Collect sprint points per driver
    sprint_points = {}
    if sprint_data is not None:
        for _, row in sprint_data.iterrows():
            pts = float(row['Points']) if pd.notna(row['Points']) else 0.0
            sprint_points[row['Abbreviation']] = pts

    # Build cumulative stats for the new round
    new_rows = []
    for _, row in new_race_data.iterrows():
        driver = row['Abbreviation']
        team = row['TeamName']
        race_points = float(row['Points']) if pd.notna(row['Points']) else 0.0
        total_points = race_points + sprint_points.get(driver, 0.0)
        position = int(row['Position']) if pd.notna(row['Position']) and row['Position'] > 0 else 99

        # Get previous stats for this driver
        prev = prev_standings[prev_standings['Driver'] == driver]
        if not prev.empty:
            prev = prev.iloc[0]
            cum_points = prev['CumulativePoints'] + total_points
            wins = int(prev['Wins']) + (1 if position == 1 else 0)
            podiums = int(prev['Podiums']) + (1 if position <= 3 else 0)
            dnfs = int(prev['DNFs']) + (1 if position == 99 else 0)
            races = int(prev['RacesCompleted']) + 1
        else:
            cum_points = total_points
            wins = 1 if position == 1 else 0
            podiums = 1 if position <= 3 else 0
            dnfs = 1 if position == 99 else 0
            races = 1

        new_rows.append({
            'Year': 2026, 'Round': new_round, 'Driver': driver, 'Team': team,
            'CumulativePoints': cum_points, 'Wins': wins, 'Podiums': podiums,
            'DNFs': dnfs, 'RacesCompleted': races,
        })

    new_df = pd.DataFrame(new_rows)
    new_df = new_df.sort_values('CumulativePoints', ascending=False).reset_index(drop=True)
    new_df['ChampionshipPosition'] = new_df.index + 1
    max_pts = new_df['CumulativePoints'].max()
    new_df['PointsGapToLeader'] = max_pts - new_df['CumulativePoints']
    new_df['WinRate'] = new_df['Wins'] / new_df['RacesCompleted']
    new_df['PodiumRate'] = new_df['Podiums'] / new_df['RacesCompleted']
    new_df['PointsPerRace'] = new_df['CumulativePoints'] / new_df['RacesCompleted']

    # Append to standings
    standings = pd.concat([standings, new_df], ignore_index=True)
    standings.to_csv(os.path.join(DATA_DIR, 'championship_standings.csv'), index=False)

    print(f"Updated standings with Round {new_round} ({len(new_rows)} drivers)")
    return standings


def run_prediction(standings, round_num):
    """Run model prediction on latest 2026 standings."""
    # Load model
    checkpoint = torch.load(os.path.join(MODELS_DIR, 'best_model.pth'),
                           weights_only=False, map_location='cpu')
    model = F1ChampionshipLSTM(input_size=12, hidden_size=64, num_layers=2, dropout=0.3)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Get prediction inputs
    year_data = standings[standings['Year'] == 2026]
    current = year_data[year_data['Round'] == round_num]
    current = current.sort_values('ChampionshipPosition')
    top_drivers = current.head(10)['Driver'].tolist()

    total_rounds_est = 22
    results = []

    for driver in top_drivers:
        driver_history = year_data[
            (year_data['Driver'] == driver) & (year_data['Round'] <= round_num)
        ].sort_values('Round')

        if driver_history.empty:
            continue

        features = extract_features(driver_history, total_rounds_est)
        x = torch.FloatTensor(features).unsqueeze(0)
        sl = torch.LongTensor([features.shape[0]])

        with torch.no_grad():
            prob = torch.sigmoid(model(x, sl)).item()

        results.append({
            'driver': driver,
            'team': driver_history.iloc[-1]['Team'],
            'points': float(driver_history.iloc[-1]['CumulativePoints']),
            'probability': prob
        })

    results.sort(key=lambda x: x['probability'], reverse=True)
    return results


def update_results_json(predictions, round_num):
    """Update evaluation_results.json with new predictions."""
    results_path = os.path.join(RESULTS_DIR, 'evaluation_results.json')

    if os.path.exists(results_path):
        with open(results_path) as f:
            eval_results = json.load(f)
    else:
        eval_results = {}

    eval_results['prediction_2026'] = predictions
    eval_results['last_updated'] = datetime.now().isoformat()
    eval_results['prediction_round'] = round_num

    with open(results_path, 'w') as f:
        json.dump(eval_results, f, indent=2)

    print(f"Updated evaluation_results.json (Round {round_num})")


def update_readme(predictions, round_num):
    """Update the prediction table in README.md."""
    readme_path = os.path.join(BASE_DIR, 'README.md')

    with open(readme_path, 'r') as f:
        content = f.read()

    # Build new prediction table
    total_prob = sum(p['probability'] for p in predictions)
    table_lines = [
        f"\n**2026 Prediction (After {round_num} Races):**\n",
        "| Driver | Team | Points | Championship Probability |",
        "|--------|------|--------|--------------------------|",
    ]
    for p in predictions[:5]:
        norm = p['probability'] / total_prob * 100
        table_lines.append(f"| {p['driver']} | {p['team']} | {p['points']:.0f} | {norm:.1f}% |")

    top_team = predictions[0]['team']
    team_prob = sum(p['probability']/total_prob*100 for p in predictions if p['team'] == top_team)
    table_lines.append(f"\n**Model sees a {team_prob:.0f}% chance a {top_team} driver wins 2026.**")

    new_table = "\n".join(table_lines)

    # Replace existing prediction table
    start_marker = "\n**2026 Prediction (After"
    end_marker = "**Model Performance:**"

    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker)

    if start_idx != -1 and end_idx != -1:
        content = content[:start_idx] + new_table + "\n\n" + content[end_idx:]

    with open(readme_path, 'w') as f:
        f.write(content)

    print(f"Updated README.md with Round {round_num} predictions")


def update_prediction_chart(predictions, round_num):
    """Regenerate the 2026 prediction bar chart."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")

    team_colors = {
        'Mercedes': '#00D2BE', 'Ferrari': '#DC0000', 'McLaren': '#FF8700',
        'Red Bull Racing': '#0600EF', 'Haas F1 Team': '#B6BABD',
        'Alpine': '#0090FF', 'Racing Bulls': '#2B4562', 'Williams': '#005AFF',
        'Audi': '#006F62', 'Aston Martin': '#006F62', 'Cadillac': '#C0C0C0',
    }

    total_prob = sum(p['probability'] for p in predictions)
    drivers = [p['driver'] for p in predictions]
    norm_probs = [p['probability'] / total_prob * 100 for p in predictions]
    teams = [p['team'] for p in predictions]
    points = [p['points'] for p in predictions]
    colors = [team_colors.get(t, '#888888') for t in teams]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    bars = ax.barh(range(len(drivers)), norm_probs, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(drivers)))
    ax.set_yticklabels([f"{d} ({t})" for d, t in zip(drivers, teams)])
    ax.set_xlabel('Championship Probability (%)')
    ax.set_title(f'2026 F1 World Championship Prediction\n(After {round_num} Races - PyTorch LSTM Model)')
    ax.invert_yaxis()

    for bar, prob, pts in zip(bars, norm_probs, points):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{prob:.1f}% ({int(pts)} pts)', va='center', fontsize=9)

    ax.set_xlim(0, max(norm_probs) * 1.3)
    ax.text(0.98, 0.02, f'Model: 2-layer LSTM (64 hidden)\nTrained: 2014-2024\nValidated: 2025 (80% accuracy)',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    chart_path = os.path.join(RESULTS_DIR, '2026_prediction.png')
    plt.savefig(chart_path, bbox_inches='tight')
    plt.close()
    print(f"Updated 2026_prediction.png")


def main():
    print("=" * 60)
    print("F1 CHAMPIONSHIP PREDICTION - AUTOMATED UPDATE")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # Step 1: Check for new race
    new_round, race_data, sprint_data = check_for_new_race()

    if new_round is None:
        print("\nNo new race data available. Nothing to update.")
        return

    # Step 2: Update standings (race + sprint)
    standings = update_standings(race_data, new_round, sprint_data)

    # Step 3: Run predictions
    print(f"\nRunning model prediction (Round {new_round})...")
    predictions = run_prediction(standings, new_round)

    total_prob = sum(p['probability'] for p in predictions)
    print(f"\n2026 Championship Probabilities (After Round {new_round}):")
    for i, p in enumerate(predictions[:5], 1):
        norm = p['probability'] / total_prob * 100
        print(f"  {i}. {p['driver']:4s} ({p['team']:20s}) | {p['points']:3.0f} pts | {norm:.1f}%")

    # Step 4: Update results JSON
    update_results_json(predictions, new_round)

    # Step 5: Update README
    update_readme(predictions, new_round)

    # Step 6: Update prediction chart
    update_prediction_chart(predictions, new_round)

    print(f"\n{'='*60}")
    print("UPDATE COMPLETE!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
