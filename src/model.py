"""
PyTorch LSTM Model for F1 Championship Prediction.

Architecture:
    Input (seq_len, 8 features) → LSTM → Dropout → Linear → Sigmoid

Predicts probability that a driver wins the championship
based on their standings progression through the season.
"""

import torch
import torch.nn as nn


class F1ChampionshipLSTM(nn.Module):
    """
    LSTM model for predicting F1 championship winner.

    Takes a sequence of driver standings features and outputs
    the probability of winning the championship.
    """

    def __init__(self, input_size=8, hidden_size=64, num_layers=2, dropout=0.3):
        """
        Args:
            input_size: Number of features per time step (default: 8)
            hidden_size: LSTM hidden state dimension (default: 64)
            num_layers: Number of stacked LSTM layers (default: 2)
            dropout: Dropout rate between LSTM layers (default: 0.3)
        """
        super(F1ChampionshipLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, seq_lens=None):
        """
        Forward pass.

        Args:
            x: (batch_size, max_seq_len, input_size) - padded sequences
            seq_lens: (batch_size,) - actual sequence lengths (for masking)

        Returns:
            logits: (batch_size, 1) - raw logits (apply sigmoid for probability)
        """
        # Pack padded sequences for efficient LSTM processing
        if seq_lens is not None:
            # Sort by sequence length (required for pack_padded_sequence)
            sorted_lens, sort_idx = seq_lens.sort(descending=True)
            x_sorted = x[sort_idx]

            packed = nn.utils.rnn.pack_padded_sequence(
                x_sorted, sorted_lens.cpu(), batch_first=True, enforce_sorted=True
            )
            lstm_out, (hidden, _) = self.lstm(packed)

            # Unsort to restore original order
            _, unsort_idx = sort_idx.sort()
            hidden = hidden[:, unsort_idx, :]
        else:
            _, (hidden, _) = self.lstm(x)

        # Use final hidden state from last LSTM layer
        final_hidden = hidden[-1]  # (batch_size, hidden_size)

        # Output layer
        out = self.dropout(final_hidden)
        logits = self.fc(out)  # (batch_size, 1)

        return logits

    def predict_proba(self, x, seq_lens=None):
        """Get championship probability (0-1)."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, seq_lens)
            probs = torch.sigmoid(logits)
        return probs


if __name__ == "__main__":
    # Test the model
    print("=" * 50)
    print("MODEL ARCHITECTURE TEST")
    print("=" * 50)

    model = F1ChampionshipLSTM(input_size=8, hidden_size=64, num_layers=2, dropout=0.3)
    print(f"\n{model}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass
    batch_size = 4
    seq_len = 10
    input_size = 8

    x = torch.randn(batch_size, seq_len, input_size)
    seq_lens = torch.LongTensor([10, 8, 6, 4])

    logits = model(x, seq_lens)
    probs = torch.sigmoid(logits)

    print(f"\nInput shape:  {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Probabilities: {probs.squeeze().tolist()}")

    # Test with different sequence lengths
    print(f"\n{'='*50}")
    print("VARIABLE LENGTH TEST")
    print(f"{'='*50}")
    for sl in [4, 8, 12, 20]:
        x_test = torch.randn(1, sl, input_size)
        lens_test = torch.LongTensor([sl])
        prob = model.predict_proba(x_test, lens_test)
        print(f"  Seq length {sl:2d}: prob = {prob.item():.4f}")
