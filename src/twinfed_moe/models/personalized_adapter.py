from torch import nn

class PersonalizedAdapter(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return x + self.adapter(x)
