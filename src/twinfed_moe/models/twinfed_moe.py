import torch
from torch import nn

from .digital_twin import DigitalTwinEncoder
from .transformer_encoder import TemporalTransformer
from .router import TwinAwareRouter
from .experts import ExpertBank
from .personalized_adapter import PersonalizedAdapter


class TwinFedMoE(nn.Module):
    def __init__(
        self,
        num_sensors: int,
        twin_aux_dim: int,
        seq_len: int,
        d_model: int = 128,
        nhead: int = 4,
        transformer_layers: int = 2,
        dim_feedforward: int = 256,
        num_experts: int = 4,
        top_k: int = 2,
        twin_latent_dim: int = 24,
        expert_hidden_dim: int = 256,
        adapter_hidden_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.digital_twin = DigitalTwinEncoder(
            num_sensors=num_sensors,
            aux_dim=twin_aux_dim,
            latent_dim=twin_latent_dim,
        )

        self.temporal_encoder = TemporalTransformer(
            num_sensors=num_sensors,
            d_model=d_model,
            nhead=nhead,
            num_layers=transformer_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_len=seq_len,
        )

        self.router = TwinAwareRouter(
            feature_dim=d_model,
            twin_dim=twin_latent_dim,
            num_experts=num_experts,
            top_k=top_k,
        )

        self.experts = ExpertBank(
            num_experts=num_experts,
            dim=d_model,
            hidden_dim=expert_hidden_dim,
            dropout=dropout,
        )

        # This module is intentionally local-only under federation.
        self.personal_adapter = PersonalizedAdapter(
            dim=d_model,
            hidden_dim=adapter_hidden_dim,
        )

        # Network predicts normalized RUL in [0,1].
        self.regressor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x,
        twin_aux,
        use_personal_adapter: bool = True,
    ):
        twin_state, reconstruction, sensor_summary = self.digital_twin(
            x,
            twin_aux,
        )

        temporal = self.temporal_encoder(x)

        sparse_gate, utilization, dense_gate = self.router(
            temporal,
            twin_state,
        )

        expert_outputs = self.experts(temporal)

        mixed = torch.sum(
            expert_outputs * sparse_gate.unsqueeze(-1),
            dim=1,
        )

        if use_personal_adapter:
            representation = self.personal_adapter(mixed)
        else:
            # Unseen assets have no learned client-specific adapter.
            representation = mixed

        prediction = self.regressor(
            representation
        ).squeeze(-1)

        return {
            "prediction": prediction,
            "twin_state": twin_state,
            "reconstruction": reconstruction,
            "sensor_summary": sensor_summary,
            "dense_gate": dense_gate,
            "sparse_gate": sparse_gate,
            "expert_utilization": utilization,
            "mixed_representation": mixed,
            "final_representation": representation,
        }
