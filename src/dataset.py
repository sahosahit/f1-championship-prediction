"""
PyTorch Dataset and DataLoader for F1 Championship Prediction.
Handles variable-length sequences with padding for batch processing.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from feature_engineering import load_standings, create_sequences, create_prediction_input


class F1ChampionshipDataset(Dataset):
    """
    PyTorch Dataset for F1 Championship prediction.

    Each sample is:
    - features: (seq_len, num_features) - driver's stats progression over races
    - target: 0 or 1 (did this driver win the championship?)
    - seq_len: actual sequence length (before padding)
    """

    def __init__(self, sequences):
        """
        Args:
            sequences: List of dicts from create_sequences()
                       Each has 'features' (np.array) and 'target' (int)
        """
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        features = torch.FloatTensor(seq['features'])  # (seq_len, num_features)
        target = torch.FloatTensor([seq['target']])     # (1,)
        seq_len = features.shape[0]

        return features, target, seq_len


def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences.
    Pads sequences to max length in batch.

    Returns:
        features: (batch_size, max_seq_len, num_features) - padded
        targets: (batch_size, 1)
        seq_lens: (batch_size,) - actual lengths for masking
    """
    features_list, targets_list, seq_lens_list = zip(*batch)

    # Pad sequences to max length in this batch
    features_padded = pad_sequence(features_list, batch_first=True, padding_value=0.0)

    # Stack targets and sequence lengths
    targets = torch.stack(targets_list)
    seq_lens = torch.LongTensor(seq_lens_list)

    return features_padded, targets, seq_lens


def create_data_splits(sequences, train_years, val_years, test_years=None):
    """
    Split sequences by year into train/val/test sets.

    Args:
        sequences: All sequences from create_sequences()
        train_years: List of years for training (e.g., [2014, ..., 2023])
        val_years: List of years for validation (e.g., [2024])
        test_years: List of years for testing (e.g., [2025])

    Returns:
        train_dataset, val_dataset, test_dataset (or None)
    """
    train_seqs = [s for s in sequences if s['year'] in train_years]
    val_seqs = [s for s in sequences if s['year'] in val_years]
    test_seqs = [s for s in sequences if s['year'] in test_years] if test_years else []

    train_dataset = F1ChampionshipDataset(train_seqs)
    val_dataset = F1ChampionshipDataset(val_seqs)
    test_dataset = F1ChampionshipDataset(test_seqs) if test_seqs else None

    return train_dataset, val_dataset, test_dataset


def get_data_loaders(train_dataset, val_dataset, test_dataset=None, batch_size=32):
    """Create DataLoaders with custom collate function."""

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False
    )

    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=False
        )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Load data and create sequences
    standings = load_standings()
    sequences = create_sequences(standings, min_races=4, top_n_drivers=10)

    # Define splits
    train_years = list(range(2014, 2024))  # 2014-2023 (10 seasons)
    val_years = [2024]                      # 2024 (validation)
    test_years = [2025]                     # 2025 (test)

    print(f"\nData Split:")
    print(f"  Train: {train_years} ({sum(1 for s in sequences if s['year'] in train_years)} sequences)")
    print(f"  Val:   {val_years} ({sum(1 for s in sequences if s['year'] in val_years)} sequences)")
    print(f"  Test:  {test_years} ({sum(1 for s in sequences if s['year'] in test_years)} sequences)")

    # Create datasets
    train_dataset, val_dataset, test_dataset = create_data_splits(
        sequences, train_years, val_years, test_years
    )

    print(f"\n  Train dataset size: {len(train_dataset)}")
    print(f"  Val dataset size:   {len(val_dataset)}")
    print(f"  Test dataset size:  {len(test_dataset)}")

    # Create DataLoaders
    train_loader, val_loader, test_loader = get_data_loaders(
        train_dataset, val_dataset, test_dataset, batch_size=32
    )

    # Test a batch
    print(f"\n{'='*50}")
    print("BATCH TEST:")
    print(f"{'='*50}")

    for features, targets, seq_lens in train_loader:
        print(f"  Batch features shape: {features.shape}")  # (batch_size, max_seq_len, 8)
        print(f"  Batch targets shape:  {targets.shape}")   # (batch_size, 1)
        print(f"  Sequence lengths:     {seq_lens[:5]}...")  # Actual lengths
        print(f"  Target distribution:  {targets.sum().item():.0f} positive / {len(targets)} total")
        print(f"  Features (first sample, first step): {features[0, 0, :].tolist()}")
        break

    # Class imbalance check
    print(f"\n{'='*50}")
    print("CLASS BALANCE:")
    print(f"{'='*50}")
    train_positive = sum(1 for s in sequences if s['year'] in train_years and s['target'] == 1)
    train_negative = sum(1 for s in sequences if s['year'] in train_years and s['target'] == 0)
    print(f"  Train - Positive: {train_positive} ({train_positive/(train_positive+train_negative)*100:.1f}%)")
    print(f"  Train - Negative: {train_negative} ({train_negative/(train_positive+train_negative)*100:.1f}%)")
    print(f"  Imbalance ratio: 1:{train_negative//train_positive}")
    print(f"  -> Will use pos_weight={train_negative/train_positive:.1f} in BCEWithLogitsLoss")
