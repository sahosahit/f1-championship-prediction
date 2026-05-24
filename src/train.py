"""
Training script for F1 Championship Prediction LSTM.

Handles:
- Training loop with BCE loss (weighted for class imbalance)
- Validation with accuracy and AUC metrics
- Early stopping
- Model checkpointing
- Training history logging
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

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import F1ChampionshipLSTM
from dataset import create_data_splits, get_data_loaders
from feature_engineering import load_standings, create_sequences

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for features, targets, seq_lens in train_loader:
        features = features.to(device)
        targets = targets.to(device)
        seq_lens = seq_lens.to(device)

        # Forward pass
        optimizer.zero_grad()
        logits = model(features, seq_lens)
        loss = criterion(logits, targets)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Track metrics
        total_loss += loss.item() * features.size(0)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    all_probs = []
    all_targets = []
    correct = 0
    total = 0

    with torch.no_grad():
        for features, targets, seq_lens in val_loader:
            features = features.to(device)
            targets = targets.to(device)
            seq_lens = seq_lens.to(device)

            logits = model(features, seq_lens)
            loss = criterion(logits, targets)

            total_loss += loss.item() * features.size(0)

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            correct += (preds == targets).sum().item()
            total += targets.size(0)

            all_probs.extend(probs.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())

    avg_loss = total_loss / total
    accuracy = correct / total

    # Calculate championship prediction accuracy
    # (Did the highest-probability driver actually win?)
    champ_accuracy = calculate_championship_accuracy(all_probs, all_targets, val_loader)

    return avg_loss, accuracy, champ_accuracy


def calculate_championship_accuracy(all_probs, all_targets, data_loader):
    """
    Check if the driver with highest probability in each season/round
    actually won the championship.
    """
    # Group predictions by (year, round) from the dataset metadata
    dataset = data_loader.dataset
    predictions_by_group = {}

    idx = 0
    for seq in dataset.sequences:
        year = seq['year']
        round_num = seq['prediction_round']
        key = (year, round_num)

        if key not in predictions_by_group:
            predictions_by_group[key] = []

        if idx < len(all_probs):
            predictions_by_group[key].append({
                'prob': all_probs[idx],
                'target': all_targets[idx],
                'driver': seq['driver']
            })
        idx += 1

    # For each group, check if highest prob driver is the actual champion
    correct = 0
    total = 0
    for key, preds in predictions_by_group.items():
        if not preds:
            continue
        top_pred = max(preds, key=lambda x: x['prob'])
        if top_pred['target'] == 1:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def train(config=None):
    """
    Main training function.

    Args:
        config: dict with hyperparameters (optional, uses defaults)
    """
    # Default configuration
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
            'top_n_drivers': 10,
        }

    print("=" * 60)
    print("F1 CHAMPIONSHIP PREDICTION - TRAINING")
    print("=" * 60)
    print(f"\nConfig: {json.dumps(config, indent=2)}")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    print("\nLoading data...")
    standings = load_standings()
    sequences = create_sequences(standings, min_races=config['min_races'],
                                top_n_drivers=config['top_n_drivers'])

    # Data splits
    train_years = list(range(2014, 2025))  # 2014-2024
    val_years = [2025]
    test_years = None  # 2026 used for live prediction only

    train_dataset, val_dataset, test_dataset = create_data_splits(
        sequences, train_years, val_years, test_years
    )

    print(f"Train: {len(train_dataset)} samples ({train_years[0]}-{train_years[-1]})")
    print(f"Val:   {len(val_dataset)} samples ({val_years})")
    if test_dataset:
        print(f"Test:  {len(test_dataset)} samples ({test_years})")

    # DataLoaders
    train_loader, val_loader, test_loader = get_data_loaders(
        train_dataset, val_dataset, test_dataset, batch_size=config['batch_size']
    )

    # Determine input size from data
    input_size = sequences[0]['features'].shape[1]
    print(f"Input features: {input_size}")

    # Model
    model = F1ChampionshipLSTM(
        input_size=input_size,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")

    # Loss function (weighted BCE for class imbalance)
    pos_weight = torch.tensor([config['pos_weight']]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Training loop
    print(f"\n{'='*60}")
    print("TRAINING STARTED")
    print(f"{'='*60}")
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10} | {'Val Acc':>7} | {'Champ Acc':>9} | {'LR':>8}")
    print("-" * 60)

    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'champ_accuracy': []}
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, config['epochs'] + 1):
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_acc, champ_acc = validate(model, val_loader, criterion, device)

        # Scheduler step
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Log
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        history['champ_accuracy'].append(champ_acc)

        print(f"{epoch:5d} | {train_loss:10.4f} | {val_loss:10.4f} | {val_acc:6.1%} | {champ_acc:8.1%} | {current_lr:.6f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'champ_accuracy': champ_acc,
                'config': config,
            }, os.path.join(MODELS_DIR, 'best_model.pth'))
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {config['patience']} epochs)")
                break

    # Load best model for evaluation
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")

    checkpoint = torch.load(os.path.join(MODELS_DIR, 'best_model.pth'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Best model from epoch {checkpoint['epoch']} (val_loss={checkpoint['val_loss']:.4f})")

    # Validation evaluation (2025 season - includes the comeback pattern)
    print(f"\n{'='*60}")
    print("VALIDATION EVALUATION (2025 Season)")
    print(f"{'='*60}")

    val_loss_final, val_acc_final, val_champ_acc_final = validate(model, val_loader, criterion, device)
    print(f"  Val Loss:     {val_loss_final:.4f}")
    print(f"  Val Accuracy: {val_acc_final:.1%}")
    print(f"  Champion Prediction Accuracy: {val_champ_acc_final:.1%}")

    # Detailed validation predictions
    print(f"\n  2025 Championship Predictions (at each race point):")
    evaluate_season(model, val_dataset, device, year=2025)

    # Test set if available
    if test_loader:
        test_loss, test_acc, test_champ_acc = validate(model, test_loader, criterion, device)
        print(f"\n  Test Loss:     {test_loss:.4f}")
        print(f"  Test Accuracy: {test_acc:.1%}")
        print(f"  Champion Prediction Accuracy: {test_champ_acc:.1%}")

    # Save training history
    history_path = os.path.join(RESULTS_DIR, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to: {history_path}")

    # Save config
    config_path = os.path.join(RESULTS_DIR, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    return model, history


def evaluate_season(model, dataset, device, year):
    """
    Evaluate model predictions for a specific season.
    Shows top predicted driver at each race point.
    """
    model.eval()

    # Group sequences by prediction round
    rounds_data = {}
    for seq in dataset.sequences:
        if seq['year'] != year:
            continue
        round_num = seq['prediction_round']
        if round_num not in rounds_data:
            rounds_data[round_num] = []
        rounds_data[round_num].append(seq)

    for round_num in sorted(rounds_data.keys()):
        drivers = rounds_data[round_num]
        predictions = []

        for d in drivers:
            features = torch.FloatTensor(d['features']).unsqueeze(0).to(device)
            seq_len = torch.LongTensor([d['features'].shape[0]]).to(device)
            prob = model.predict_proba(features, seq_len).item()
            predictions.append({
                'driver': d['driver'],
                'prob': prob,
                'is_champion': d['target'] == 1
            })

        predictions.sort(key=lambda x: x['prob'], reverse=True)
        top = predictions[0]
        marker = "✓" if top['is_champion'] else "✗"
        print(f"    After R{round_num:02d}: {top['driver']} ({top['prob']:.1%}) {marker}")


if __name__ == "__main__":
    model, history = train()
