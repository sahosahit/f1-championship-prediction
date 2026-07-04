"""
Training script for F1 Constructor Championship Prediction LSTM.
Reuses the same LSTM architecture but trains on team-aggregated data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import F1ChampionshipLSTM
from dataset import F1ChampionshipDataset, collate_fn, get_data_loaders, create_data_splits
from constructor_feature_engineering import (
    load_standings, build_constructor_standings, create_constructor_sequences
)
from train import train_one_epoch, validate

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def train_constructor(config=None):
    if config is None:
        config = {
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.3,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'patience': 15,
            'pos_weight': 9.0,
            'min_races': 4,
            'top_n_teams': 10,
        }

    print("=" * 60)
    print("F1 CONSTRUCTOR CHAMPIONSHIP PREDICTION - TRAINING")
    print("=" * 60)
    print(f"\nConfig: {json.dumps(config, indent=2)}")

    device = torch.device('cuda' if torch.cuda.is_available() else
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    print("\nLoading data...")
    standings = load_standings()
    constructor_standings = build_constructor_standings(standings)
    sequences = create_constructor_sequences(
        constructor_standings,
        min_races=config['min_races'],
        top_n_teams=config['top_n_teams']
    )

    train_years = list(range(2014, 2025))
    val_years = [2025]

    train_dataset, val_dataset, _ = create_data_splits(sequences, train_years, val_years)

    print(f"Train: {len(train_dataset)} samples ({train_years[0]}-{train_years[-1]})")
    print(f"Val:   {len(val_dataset)} samples ({val_years})")

    train_loader, val_loader, _ = get_data_loaders(
        train_dataset, val_dataset, batch_size=config['batch_size']
    )

    input_size = sequences[0]['features'].shape[1]
    print(f"Input features: {input_size}")

    model = F1ChampionshipLSTM(
        input_size=input_size,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")

    pos_weight = torch.tensor([config['pos_weight']]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    print(f"\n{'='*60}")
    print("TRAINING STARTED")
    print(f"{'='*60}")
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10} | {'Val Acc':>7} | {'Champ Acc':>9} | {'LR':>8}")
    print("-" * 60)

    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'champ_accuracy': []}
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, config['epochs'] + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, champ_acc = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        history['champ_accuracy'].append(champ_acc)

        print(f"{epoch:5d} | {train_loss:10.4f} | {val_loss:10.4f} | {val_acc:6.1%} | {champ_acc:8.1%} | {current_lr:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'champ_accuracy': champ_acc,
                'config': config,
            }, os.path.join(MODELS_DIR, 'best_constructor_model.pth'))
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    print(f"\n{'='*60}")
    print("CONSTRUCTOR MODEL TRAINING COMPLETE")
    print(f"{'='*60}")

    checkpoint = torch.load(os.path.join(MODELS_DIR, 'best_constructor_model.pth'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Best model from epoch {checkpoint['epoch']} (val_loss={checkpoint['val_loss']:.4f})")

    history_path = os.path.join(RESULTS_DIR, 'constructor_training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    config_path = os.path.join(RESULTS_DIR, 'constructor_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    return model, history


if __name__ == "__main__":
    model, history = train_constructor()
