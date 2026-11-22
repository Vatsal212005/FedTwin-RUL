import torch
from torch import nn

class TwinAwareRouter(nn.Module):
    def __init__(self, feature_dim: int, twin_dim: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(feature_dim + twin_dim, num_experts)

    def forward(self, temporal_features: torch.Tensor, twin_state: torch.Tensor):
        logits = self.gate(torch.cat([temporal_features, twin_state], dim=-1))
        probs = torch.softmax(logits, dim=-1)

        top_values, top_indices = torch.topk(probs, k=self.top_k, dim=-1)
        sparse = torch.zeros_like(probs).scatter(1, top_indices, top_values)
        sparse = sparse / sparse.sum(dim=1, keepdim=True).clamp_min(1e-8)

        # Mean client-level utilization statistic used by federated expert aggregation.
        utilization = sparse.detach().mean(dim=0)

        return sparse, utilization, probs
