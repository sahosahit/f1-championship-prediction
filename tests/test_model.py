"""Tests for LSTM model architecture and forward pass."""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from model import F1ChampionshipLSTM


class TestF1ChampionshipLSTM:
    def test_model_creation(self):
        model = F1ChampionshipLSTM(input_size=12, hidden_size=64, num_layers=2, dropout=0.3)
        assert model is not None
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params == 53313

    def test_forward_pass_with_seq_lens(self):
        model = F1ChampionshipLSTM(input_size=12, hidden_size=64, num_layers=2, dropout=0.3)
        x = torch.randn(4, 10, 12)
        seq_lens = torch.LongTensor([10, 8, 6, 4])
        logits = model(x, seq_lens)
        assert logits.shape == (4, 1)

    def test_forward_pass_without_seq_lens(self):
        model = F1ChampionshipLSTM(input_size=12, hidden_size=64, num_layers=2, dropout=0.3)
        x = torch.randn(4, 10, 12)
        logits = model(x)
        assert logits.shape == (4, 1)

    def test_predict_proba_returns_0_to_1(self):
        model = F1ChampionshipLSTM(input_size=12, hidden_size=64, num_layers=2, dropout=0.3)
        x = torch.randn(4, 10, 12)
        seq_lens = torch.LongTensor([10, 8, 6, 4])
        probs = model.predict_proba(x, seq_lens)
        assert probs.shape == (4, 1)
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_variable_sequence_lengths(self):
        model = F1ChampionshipLSTM(input_size=12, hidden_size=64, num_layers=2, dropout=0.3)
        for seq_len in [4, 8, 12, 20]:
            x = torch.randn(1, seq_len, 12)
            lens = torch.LongTensor([seq_len])
            logits = model(x, lens)
            assert logits.shape == (1, 1)

    def test_single_sample_batch(self):
        model = F1ChampionshipLSTM(input_size=12, hidden_size=64, num_layers=2, dropout=0.3)
        x = torch.randn(1, 5, 12)
        seq_lens = torch.LongTensor([5])
        logits = model(x, seq_lens)
        assert logits.shape == (1, 1)

    def test_different_input_sizes(self):
        for input_size in [8, 12, 16]:
            model = F1ChampionshipLSTM(input_size=input_size, hidden_size=64, num_layers=2)
            x = torch.randn(2, 6, input_size)
            seq_lens = torch.LongTensor([6, 4])
            logits = model(x, seq_lens)
            assert logits.shape == (2, 1)

    def test_model_deterministic_in_eval(self):
        model = F1ChampionshipLSTM(input_size=12, hidden_size=64, num_layers=2, dropout=0.3)
        model.eval()
        x = torch.randn(2, 6, 12)
        seq_lens = torch.LongTensor([6, 4])
        with torch.no_grad():
            out1 = model(x, seq_lens)
            out2 = model(x, seq_lens)
        assert torch.allclose(out1, out2)
