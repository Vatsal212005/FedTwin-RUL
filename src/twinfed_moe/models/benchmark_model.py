import torch
from torch import nn

from .digital_twin import DigitalTwinEncoder
from .transformer_encoder import TemporalTransformer
from .router import TwinAwareRouter
from .experts import ExpertBank
from .personalized_adapter import PersonalizedAdapter


class TemporalOnlyRouter(nn.Module):
    """Sparse Top-k router that does not consume Digital Twin state."""

    def __init__(
        self,
        feature_dim: int,
        num_experts: int,
        top_k: int,
    ):
        super().__init__()

        self.num_experts = int(num_experts)
        self.top_k = int(
            min(top_k, num_experts)
        )

        self.gate = nn.Linear(
            feature_dim,
            num_experts,
        )

    def forward(
        self,
        feature,
    ):
        logits = self.gate(
            feature
        )

        dense_gate = torch.softmax(
            logits,
            dim=-1,
        )

        values, indices = torch.topk(
            dense_gate,
            k=self.top_k,
            dim=-1,
        )

        sparse_gate = torch.zeros_like(
            dense_gate
        )

        sparse_gate.scatter_(
            1,
            indices,
            values,
        )

        sparse_gate = (
            sparse_gate
            / sparse_gate.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-12)
        )

        utilization = (
            sparse_gate.mean(
                dim=0
            )
        )

        return (
            sparse_gate,
            utilization,
            dense_gate,
        )


class BenchmarkRULModel(nn.Module):
    """
    Unified model used only by the benchmark suite.

    Variants are created by changing:
      use_digital_twin
      use_moe
      use_personalization
      router_mode: "twin" | "temporal"

    The full configuration uses the same architecture family as TwinFed-MoE.
    """

    def __init__(
        self,
        num_sensors: int,
        twin_aux_dim: int,
        seq_len: int,
        d_model: int,
        nhead: int,
        transformer_layers: int,
        dim_feedforward: int,
        num_experts: int,
        top_k: int,
        twin_latent_dim: int,
        expert_hidden_dim: int,
        adapter_hidden_dim: int,
        dropout: float,
        use_digital_twin: bool,
        use_moe: bool,
        use_personalization: bool,
        router_mode: str,
    ):
        super().__init__()

        self.use_digital_twin = bool(
            use_digital_twin
        )

        self.use_moe = bool(
            use_moe
        )

        self.use_personalization = bool(
            use_personalization
        )

        self.router_mode = str(
            router_mode
        )

        self.num_experts = (
            int(num_experts)
            if self.use_moe
            else 1
        )

        self.twin_latent_dim = int(
            twin_latent_dim
        )

        if self.use_digital_twin:
            self.digital_twin = (
                DigitalTwinEncoder(
                    num_sensors=num_sensors,
                    aux_dim=twin_aux_dim,
                    latent_dim=twin_latent_dim,
                )
            )
        else:
            self.digital_twin = None

        self.temporal_encoder = (
            TemporalTransformer(
                num_sensors=num_sensors,
                d_model=d_model,
                nhead=nhead,
                num_layers=transformer_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                max_seq_len=seq_len,
            )
        )

        if self.use_moe:
            self.experts = ExpertBank(
                num_experts=num_experts,
                dim=d_model,
                hidden_dim=expert_hidden_dim,
                dropout=dropout,
            )

            if (
                self.router_mode == "twin"
                and self.use_digital_twin
            ):
                self.router = (
                    TwinAwareRouter(
                        feature_dim=d_model,
                        twin_dim=twin_latent_dim,
                        num_experts=num_experts,
                        top_k=top_k,
                    )
                )
            else:
                self.router = (
                    TemporalOnlyRouter(
                        feature_dim=d_model,
                        num_experts=num_experts,
                        top_k=top_k,
                    )
                )
        else:
            self.experts = None
            self.router = None

        if self.use_personalization:
            self.personal_adapter = (
                PersonalizedAdapter(
                    dim=d_model,
                    hidden_dim=adapter_hidden_dim,
                )
            )
        else:
            self.personal_adapter = (
                nn.Identity()
            )

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
        batch_size = x.size(0)

        if self.use_digital_twin:
            (
                twin_state,
                reconstruction,
                sensor_summary,
            ) = self.digital_twin(
                x,
                twin_aux,
            )
        else:
            sensor_summary = x.mean(
                dim=1
            )

            reconstruction = (
                sensor_summary
            )

            twin_state = x.new_zeros(
                (
                    batch_size,
                    self.twin_latent_dim,
                )
            )

        temporal = (
            self.temporal_encoder(
                x
            )
        )

        if self.use_moe:
            if (
                self.router_mode == "twin"
                and self.use_digital_twin
            ):
                (
                    sparse_gate,
                    utilization,
                    dense_gate,
                ) = self.router(
                    temporal,
                    twin_state,
                )
            else:
                (
                    sparse_gate,
                    utilization,
                    dense_gate,
                ) = self.router(
                    temporal
                )

            expert_outputs = (
                self.experts(
                    temporal
                )
            )

            mixed = torch.sum(
                expert_outputs
                * sparse_gate.unsqueeze(-1),
                dim=1,
            )
        else:
            mixed = temporal

            dense_gate = x.new_ones(
                (batch_size, 1)
            )

            sparse_gate = x.new_ones(
                (batch_size, 1)
            )

            utilization = x.new_ones(
                (1,)
            )

        if (
            self.use_personalization
            and use_personal_adapter
        ):
            representation = (
                self.personal_adapter(
                    mixed
                )
            )
        else:
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
