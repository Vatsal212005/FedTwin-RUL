import torch
from torch import nn

class Expert(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ExpertBank(nn.Module):
    def __init__(self, num_experts: int, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.experts = nn.ModuleList(
            [Expert(dim, hidden_dim, dropout) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, M, D]
        return torch.stack([expert(x) for expert in self.experts], dim=1)
