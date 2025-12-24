import math
from typing import Optional

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


def _generate_causal_mask(sz: int, device: torch.device) -> torch.Tensor:
    mask = torch.triu(torch.ones(sz, sz, device=device) * float("-inf"), diagonal=1)
    return mask


class CausalTransformerAutoencoder(nn.Module):
    """
    Causal transformer that reconstructs the input sequence from past context only.
    Suitable for online anomaly scoring: at time t, predicts/reconstructs x_t using x_{<=t}.
    """

    def __init__(
        self,
        input_dim: int = 4,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_len: int = 1000,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        self.in_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, input_dim)
        Returns reconstruction y: (B, T, input_dim)
        """
        b, t, _ = x.shape
        h = self.in_proj(x)
        h = self.pos_encoding(h)
        mask = _generate_causal_mask(t, device=x.device)
        z = self.encoder(h, mask=mask)
        y = self.out_proj(z)
        return y

    @torch.no_grad()
    def reconstruct_last(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, input_dim)
        Returns reconstruction for the last time step only: (B, input_dim)
        """
        y = self.forward(x)
        return y[:, -1, :]

