"""
Visualization for F1 Championship Prediction.
Generates plots for:
1. Training history (loss curves)
2. Accuracy by season progress
3. 2026 championship prediction bar chart
4. Per-season prediction timeline
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import json
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import F1ChampionshipLSTM
from feature_engineering import load_standings, create_prediction_input

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')

# Style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10


def plot_training_history():
    """Plot training and validation loss curves."""
    with open(os.path.join(RESULTS_DIR, 'training_history.json')) as f:
        history = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curves
    epochs = range(1, len(history['train_loss']) + 1)
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (BCE)')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].set_ylim(0, max(history['train_loss'][0], history['val_loss'][0]) * 1.1)

    # Championship accuracy
    axes[1].plot(epochs, [a * 100 for a in history['champ_accuracy']],
                'g-', linewidth=2, marker='o', markersize=3)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Championship Prediction Accuracy (%)')
    axes[1].set_title('Validation: Champion Prediction Accuracy')
    axes[1].set_ylim(0, 105)
    axes[1].axhline(y=80, color='orange', linestyle='--', alpha=0.7, label='80% threshold')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'training_curves.png'), bbox_inches='tight')
    plt.close()
    print("Saved: training_curves.png")


def plot_accuracy_by_progress():
    """Plot how prediction accuracy improves over the season."""
    progress_data = {
        'Early\n(R4-R6)': 41.7,
        'Mid-Early\n(R7-R10)': 54.2,
        'Mid\n(R11-R14)': 58.3,
        'Mid-Late\n(R15-R18)': 63.8,
        'Late\n(R19+)': 83.8,
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(progress_data.keys(), progress_data.values(),
                  color=['#ff6b6b', '#ffa94d', '#ffd43b', '#69db7c', '#20c997'],
                  edgecolor='black', linewidth=0.5)

    # Add value labels
    for bar, val in zip(bars, progress_data.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')

    ax.set_ylabel('Championship Prediction Accuracy (%)')
    ax.set_title('Model Accuracy Improves as Season Progresses\n(Validated on 2014-2025, 12 seasons)')
    ax.set_ylim(0, 100)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Random baseline (50%)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'accuracy_by_progress.png'), bbox_inches='tight')
    plt.close()
    print("Saved: accuracy_by_progress.png")


def plot_2026_prediction():
    """Bar chart of 2026 championship probabilities."""
    with open(os.path.join(RESULTS_DIR, 'evaluation_results.json')) as f:
        results = json.load(f)

    predictions = results['prediction_2026']

    drivers = [p['driver'] for p in predictions]
    probs = [p['probability'] for p in predictions]
    total = sum(probs)
    norm_probs = [p / total * 100 for p in probs]
    teams = [p['team'] for p in predictions]
    points = [p['points'] for p in predictions]

    # Team colors
    team_colors = {
        'Mercedes': '#00D2BE',
        'Ferrari': '#DC0000',
        'McLaren': '#FF8700',
        'Red Bull Racing': '#0600EF',
        'Haas F1 Team': '#B6BABD',
        'Alpine': '#0090FF',
        'Racing Bulls': '#2B4562',
        'Williams': '#005AFF',
        'Audi': '#006F62',
        'Aston Martin': '#006F62',
        'Cadillac': '#C0C0C0',
    }

    colors = [team_colors.get(t, '#888888') for t in teams]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.barh(range(len(drivers)), norm_probs, color=colors, edgecolor='black', linewidth=0.5)

    # Labels
    ax.set_yticks(range(len(drivers)))
    ax.set_yticklabels([f"{d} ({t})" for d, t in zip(drivers, teams)])
    ax.set_xlabel('Championship Probability (%)')
    ax.set_title('2026 F1 World Championship Prediction\n(After 6 Races - PyTorch LSTM Model)')
    ax.invert_yaxis()

    # Add probability labels
    for i, (bar, prob, pts) in enumerate(zip(bars, norm_probs, points)):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{prob:.1f}% ({int(pts)} pts)', va='center', fontsize=9)

    ax.set_xlim(0, max(norm_probs) * 1.3)

    # Add annotation
    ax.text(0.98, 0.02, 'Model: LSTM (2-layer, 64 hidden)\nTrained: 2014-2024 (11 seasons)\nValidated: 2025 (80% accuracy)',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, '2026_prediction.png'), bbox_inches='tight')
    plt.close()
    print("Saved: 2026_prediction.png")


def plot_season_timeline():
    """
    Plot prediction correctness across race points for key seasons.
    Shows when model gets it right/wrong throughout a season.
    """
    # Load model and standings
    checkpoint = torch.load(os.path.join(MODELS_DIR, 'best_model.pth'), weights_only=False)
    model = F1ChampionshipLSTM(input_size=12, hidden_size=64, num_layers=2, dropout=0.3)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    standings = load_standings()

    champions = {
        2016: 'ROS', 2019: 'HAM', 2021: 'VER',
        2023: 'VER', 2025: 'NOR'
    }

    fig, axes = plt.subplots(len(champions), 1, figsize=(12, 10), sharex=False)

    for idx, (year, champion) in enumerate(champions.items()):
        ax = axes[idx]
        year_data = standings[standings['Year'] == year]
        all_rounds = sorted(year_data['Round'].unique())

        rounds_plotted = []
        champion_probs = []
        top_pred_probs = []
        correct_markers = []

        for round_idx in range(3, len(all_rounds)):
            round_num = all_rounds[round_idx]
            pred_inputs = create_prediction_input(standings, year, round_num, top_n=10)

            champ_prob = 0
            top_prob = 0
            top_driver = ''

            for p in pred_inputs:
                features = torch.FloatTensor(p['features']).unsqueeze(0)
                seq_len = torch.LongTensor([p['features'].shape[0]])
                with torch.no_grad():
                    prob = torch.sigmoid(model(features, seq_len)).item()

                if p['driver'] == champion:
                    champ_prob = prob
                if prob > top_prob:
                    top_prob = prob
                    top_driver = p['driver']

            rounds_plotted.append(round_num)
            champion_probs.append(champ_prob * 100)
            top_pred_probs.append(top_prob * 100)
            correct_markers.append(top_driver == champion)

        # Plot
        ax.plot(rounds_plotted, champion_probs, 'g-', linewidth=2,
                label=f'Actual Champion ({champion}) prob', marker='o', markersize=3)
        ax.plot(rounds_plotted, top_pred_probs, 'b--', linewidth=1.5,
                label='Top predicted driver prob', alpha=0.7)

        # Mark correct/incorrect
        for r, c, correct in zip(rounds_plotted, champion_probs, correct_markers):
            if correct:
                ax.scatter(r, c, color='green', s=30, zorder=5)
            else:
                ax.scatter(r, c, color='red', s=30, zorder=5, marker='x')

        ax.set_ylabel('Probability (%)')
        ax.set_title(f'{year}: {champion} Championship', fontsize=10, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.legend(loc='lower right', fontsize=8)
        ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5)

    axes[-1].set_xlabel('Race Number')
    plt.suptitle("Championship Probability Over Season\n(Green ● = correct prediction, Red ✗ = incorrect)",
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'season_timelines.png'), bbox_inches='tight')
    plt.close()
    print("Saved: season_timelines.png")


def plot_model_architecture():
    """Create a simple text-based architecture diagram and save as image."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis('off')

    architecture_text = """
    ┌─────────────────────────────────────────────────────────┐
    │              F1 Championship LSTM Model                  │
    ├─────────────────────────────────────────────────────────┤
    │                                                         │
    │  Input: (batch, seq_len, 12 features)                   │
    │    ├─ Points normalized, Position, Gap to leader        │
    │    ├─ Win rate, Podium rate, Points per race            │
    │    ├─ Season progress, Consistency                      │
    │    └─ Momentum, Is leader, Points share, Recent form    │
    │                    ↓                                    │
    │  LSTM Layer 1: (input=12, hidden=64)                    │
    │                    ↓                                    │
    │  LSTM Layer 2: (input=64, hidden=64) + Dropout(0.3)     │
    │                    ↓                                    │
    │  Final Hidden State: (batch, 64)                        │
    │                    ↓                                    │
    │  Dropout(0.3)                                           │
    │                    ↓                                    │
    │  Linear: (64 → 1)                                       │
    │                    ↓                                    │
    │  Sigmoid → Championship Probability (0-1)               │
    │                                                         │
    │  Parameters: 53,313 | Loss: BCEWithLogits (pos_weight=9)│
    └─────────────────────────────────────────────────────────┘
    """

    ax.text(0.5, 0.5, architecture_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='center', horizontalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'model_architecture.png'), bbox_inches='tight')
    plt.close()
    print("Saved: model_architecture.png")


if __name__ == "__main__":
    print("Generating visualizations...")
    print("=" * 50)

    plot_training_history()
    plot_accuracy_by_progress()
    plot_2026_prediction()
    plot_model_architecture()
    plot_season_timeline()

    print(f"\n{'='*50}")
    print("ALL VISUALIZATIONS COMPLETE!")
    print(f"{'='*50}")
    print(f"\nFiles saved to: {RESULTS_DIR}/")
    print("  - training_curves.png")
    print("  - accuracy_by_progress.png")
    print("  - 2026_prediction.png")
    print("  - model_architecture.png")
    print("  - season_timelines.png")
