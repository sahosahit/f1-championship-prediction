"""Tests for PyTorch Dataset and DataLoader."""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from dataset import F1ChampionshipDataset, collate_fn, create_data_splits, get_data_loaders
from feature_engineering import load_standings, create_sequences
import numpy as np


@pytest.fixture
def sequences():
    standings = load_standings()
    return create_sequences(standings, min_races=4, top_n_drivers=5)


class TestF1ChampionshipDataset:
    def test_dataset_length(self, sequences):
        dataset = F1ChampionshipDataset(sequences)
        assert len(dataset) == len(sequences)

    def test_dataset_getitem(self, sequences):
        dataset = F1ChampionshipDataset(sequences)
        features, target, seq_len = dataset[0]
        assert isinstance(features, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        assert features.shape[1] == 12
        assert target.shape == (1,)
        assert seq_len == features.shape[0]


class TestCollateFn:
    def test_collate_pads_sequences(self, sequences):
        dataset = F1ChampionshipDataset(sequences)
        batch = [dataset[i] for i in range(4)]
        features, targets, seq_lens = collate_fn(batch)
        assert features.shape[0] == 4
        assert features.shape[2] == 12
        assert targets.shape == (4, 1)
        assert seq_lens.shape == (4,)
        assert features.shape[1] == max(seq_lens).item()


class TestDataSplits:
    def test_split_by_year(self, sequences):
        train_years = list(range(2014, 2023))
        val_years = [2023, 2024]
        train_ds, val_ds, _ = create_data_splits(sequences, train_years, val_years)
        assert len(train_ds) > 0
        assert len(val_ds) > 0
        for seq in train_ds.sequences:
            assert seq['year'] in train_years
        for seq in val_ds.sequences:
            assert seq['year'] in val_years

    def test_data_loaders(self, sequences):
        train_years = list(range(2014, 2024))
        val_years = [2024]
        train_ds, val_ds, _ = create_data_splits(sequences, train_years, val_years)
        train_loader, val_loader, _ = get_data_loaders(train_ds, val_ds, batch_size=16)
        batch = next(iter(train_loader))
        features, targets, seq_lens = batch
        assert features.dim() == 3
        assert targets.dim() == 2
        assert seq_lens.dim() == 1
