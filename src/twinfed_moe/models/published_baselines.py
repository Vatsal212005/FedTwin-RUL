import math
import torch
from torch import nn


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()

        position = torch.arange(
            max_len,
            dtype=torch.float32,
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2,
                dtype=torch.float32,
            )
            * (
                -math.log(10000.0)
                / d_model
            )
        )

        pe = torch.zeros(
            max_len,
            d_model,
            dtype=torch.float32,
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        self.register_buffer(
            "pe",
            pe.unsqueeze(0),
            persistent=False,
        )

    def forward(self, x):
        return x + self.pe[
            :,
            :x.size(1),
        ].to(dtype=x.dtype)


class TransLighterReproduction(nn.Module):
    """
    Protocol-aligned reproduction of the architecture family described in:
    Nguyen et al., Computers in Industry 148 (2023), 103888.

    Publicly documented core:
      lightweight Transformer encoder + linear RUL prediction layer
      inside a federated-learning architecture.

    This is NOT claimed to reproduce the paper's Bayesian-optimized
    hyperparameters or original 4-device topology. It uses the exact same
    FD004 preprocessing, 10-client partition, heterogeneity levels, seeds,
    and training budget as FedTwin-RUL for a controlled comparison.
    """

    def __init__(
        self,
        num_sensors: int,
        seq_len: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()

        self.input_projection = nn.Linear(
            num_sensors,
            d_model,
        )

        self.position = SinusoidalPositionEncoding(
            d_model=d_model,
            max_len=seq_len,
        )

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(
            d_model
        )

        self.regressor = nn.Sequential(
            nn.Linear(
                d_model,
                1,
            ),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x,
        twin_aux=None,
    ):
        h = self.input_projection(
            x
        )

        h = self.position(
            h
        )

        h = self.encoder(
            h
        )

        h = self.norm(
            h.mean(
                dim=1
            )
        )

        return (
            self.regressor(
                h
            ).squeeze(-1)
        )


class StandardTransformerRUL(nn.Module):
    """
    Plain Transformer backbone used for the SCAFFOLD baseline.
    No Digital-Twin inputs are used.
    """

    def __init__(
        self,
        num_sensors: int,
        seq_len: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()

        self.input_projection = nn.Linear(
            num_sensors,
            d_model,
        )

        self.position = SinusoidalPositionEncoding(
            d_model=d_model,
            max_len=seq_len,
        )

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(
            d_model
        )

        self.regressor = nn.Sequential(
            nn.Linear(
                d_model,
                1,
            ),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x,
        twin_aux=None,
    ):
        h = self.input_projection(
            x
        )

        h = self.position(
            h
        )

        h = self.encoder(
            h
        )

        h = self.norm(
            h.mean(
                dim=1
            )
        )

        return (
            self.regressor(
                h
            ).squeeze(-1)
        )
