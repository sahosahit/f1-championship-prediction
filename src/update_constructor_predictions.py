"""
Automated Constructor Championship Prediction Update.
Triggered alongside driver predictions after each race weekend.
"""

import torch
import pandas as pd
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import F1ChampionshipLSTM
from constructor_feature_engineering import (
    build_constructor_standings, extract_constructor_features,
    create_constructor_prediction_input
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')


def run_constructor_prediction(round_num):
    """Run constructor championship prediction for current 2026 standings."""
    model_path = os.path.join(MODELS_DIR, 'best_constructor_model.pth')
    if not os.path.exists(model_path):
        print("Constructor model not found. Skipping constructor predictions.")
        print("Train it first: python src/train_constructor.py")
        return None

    checkpoint = torch.load(model_path, weights_only=False, map_location='cpu')
    model = F1ChampionshipLSTM(input_size=12, hidden_size=64, num_layers=2, dropout=0.3)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    standings = pd.read_csv(os.path.join(DATA_DIR, 'championship_standings.csv'))
    constructor_standings = build_constructor_standings(standings)

    pred_inputs = create_constructor_prediction_input(constructor_standings, 2026, round_num, top_n=10)
    if not pred_inputs:
        print("No constructor prediction data available.")
        return None

    results = []
    for p in pred_inputs:
        features = torch.FloatTensor(p['features']).unsqueeze(0)
        seq_len = torch.LongTensor([p['features'].shape[0]])
        with torch.no_grad():
            prob = torch.sigmoid(model(features, seq_len)).item()
        results.append({
            'team': p['team'],
            'points': float(p['current_points']),
            'probability': prob,
        })

    results.sort(key=lambda x: x['probability'], reverse=True)
    return results


def update_constructor_results(predictions, round_num):
    """Save constructor predictions to results JSON."""
    results_path = os.path.join(RESULTS_DIR, 'evaluation_results.json')

    if os.path.exists(results_path):
        with open(results_path) as f:
            eval_results = json.load(f)
    else:
        eval_results = {}

    eval_results['constructor_prediction_2026'] = predictions
    eval_results['constructor_prediction_round'] = round_num
    eval_results['constructor_last_updated'] = datetime.now().isoformat()

    with open(results_path, 'w') as f:
        json.dump(eval_results, f, indent=2)

    print(f"Updated constructor predictions in evaluation_results.json")


def update_constructor_chart(predictions, round_num):
    """Generate constructor championship prediction chart."""
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
    teams = [p['team'] for p in predictions]
    norm_probs = [p['probability'] / total_prob * 100 for p in predictions]
    points = [p['points'] for p in predictions]
    colors = [team_colors.get(t, '#888888') for t in teams]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    bars = ax.barh(range(len(teams)), norm_probs, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(teams)))
    ax.set_yticklabels(teams)
    ax.set_xlabel('Constructor Championship Probability (%)')
    ax.set_title(f'2026 F1 Constructor Championship Prediction\n(After {round_num} Races - PyTorch LSTM Model)')
    ax.invert_yaxis()

    for bar, prob, pts in zip(bars, norm_probs, points):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{prob:.1f}% ({int(pts)} pts)', va='center', fontsize=9)

    ax.set_xlim(0, max(norm_probs) * 1.3)
    plt.tight_layout()
    chart_path = os.path.join(RESULTS_DIR, '2026_constructor_prediction.png')
    plt.savefig(chart_path, bbox_inches='tight')
    plt.close()
    print(f"Updated 2026_constructor_prediction.png")


def main():
    print("=" * 60)
    print("CONSTRUCTOR CHAMPIONSHIP PREDICTION UPDATE")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    standings = pd.read_csv(os.path.join(DATA_DIR, 'championship_standings.csv'))
    data_2026 = standings[standings['Year'] == 2026]
    if data_2026.empty:
        print("No 2026 data available.")
        return

    latest_round = int(data_2026['Round'].max())
    print(f"Latest round in data: {latest_round}")

    predictions = run_constructor_prediction(latest_round)
    if predictions is None:
        return

    total_prob = sum(p['probability'] for p in predictions)
    print(f"\n2026 Constructor Championship Probabilities (After Round {latest_round}):")
    for i, p in enumerate(predictions[:5], 1):
        norm = p['probability'] / total_prob * 100
        print(f"  {i}. {p['team']:<22s} | {p['points']:5.0f} pts | {norm:.1f}%")

    update_constructor_results(predictions, latest_round)
    update_constructor_chart(predictions, latest_round)

    print(f"\n{'='*60}")
    print("CONSTRUCTOR UPDATE COMPLETE!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
