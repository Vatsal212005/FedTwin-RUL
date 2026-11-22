import torch
from torch import nn

class TemporalTransformer(nn.Module):
    def __init__(
        self,
        num_sensors: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.input_proj = nn.Linear(num_sensors, d_model)
        self.positional = nn.Parameter(torch.zeros(1, max_seq_len, d_model))

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = h + self.positional[:, : h.size(1)]
        h = self.encoder(h)
        h = self.norm(h)
        return h.mean(dim=1)
