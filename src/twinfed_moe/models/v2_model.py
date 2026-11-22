import torch
from torch import nn

from .digital_twin import DigitalTwinEncoder
from .transformer_encoder import TemporalTransformer
from .router import TwinAwareRouter
from .experts import ExpertBank
from .personalized_adapter import PersonalizedAdapter


class TemporalOnlyRouter(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_experts: int,
        top_k: int,
    ):
        super().__init__()

        self.num_experts = int(num_experts)
        self.top_k = min(
            int(top_k),
            int(num_experts),
        )

        self.gate = nn.Linear(
            feature_dim,
            num_experts,
        )

    def forward(self, feature):
        dense_gate = torch.softmax(
            self.gate(feature),
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

        utilization = sparse_gate.mean(
            dim=0
        )

        return (
            sparse_gate,
            utilization,
            dense_gate,
        )


class TwinFedMoEV2(nn.Module):
    """
    TwinFed-MoE v2.

    Shared prediction:
        y_g = sigmoid(f_g(h))

    Optional client-local residual:
        delta_k = s * tanh(f_k(A_k(h)))
        y_k = clip(y_g + delta_k, 0, 1)

    The global/shared head is therefore always a valid directly supervised
    predictor for a previously unseen engine.
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
        personal_residual_scale: float = 0.20,
        use_digital_twin: bool = True,
        use_moe: bool = True,
        use_personalization: bool = True,
        router_mode: str = "twin",
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

        self.personal_residual_scale = float(
            personal_residual_scale
        )

        if self.use_digital_twin:
            self.digital_twin = DigitalTwinEncoder(
                num_sensors=num_sensors,
                aux_dim=twin_aux_dim,
                latent_dim=twin_latent_dim,
            )
        else:
            self.digital_twin = None

        self.temporal_encoder = TemporalTransformer(
            num_sensors=num_sensors,
            d_model=d_model,
            nhead=nhead,
            num_layers=transformer_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_len=seq_len,
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
                self.router = TwinAwareRouter(
                    feature_dim=d_model,
                    twin_dim=twin_latent_dim,
                    num_experts=num_experts,
                    top_k=top_k,
                )
            else:
                self.router = TemporalOnlyRouter(
                    feature_dim=d_model,
                    num_experts=num_experts,
                    top_k=top_k,
                )
        else:
            self.experts = None
            self.router = None

        # Always-valid global RUL head.
        self.shared_regressor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

        if self.use_personalization:
            # Entire correction pathway is client-local.
            self.personal_adapter = PersonalizedAdapter(
                dim=d_model,
                hidden_dim=adapter_hidden_dim,
            )

            self.personal_delta_head = nn.Linear(
                d_model,
                1,
            )

            # Start personalization as exact zero correction.
            nn.init.zeros_(
                self.personal_delta_head.weight
            )
            nn.init.zeros_(
                self.personal_delta_head.bias
            )
        else:
            self.personal_adapter = None
            self.personal_delta_head = None

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
            reconstruction = sensor_summary
            twin_state = x.new_zeros(
                (
                    batch_size,
                    self.twin_latent_dim,
                )
            )

        temporal = self.temporal_encoder(
            x
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

            expert_outputs = self.experts(
                temporal
            )

            shared_representation = torch.sum(
                expert_outputs
                * sparse_gate.unsqueeze(-1),
                dim=1,
            )
        else:
            shared_representation = temporal
            dense_gate = x.new_ones(
                (batch_size, 1)
            )
            sparse_gate = x.new_ones(
                (batch_size, 1)
            )
            utilization = x.new_ones(
                (1,)
            )

        shared_prediction = (
            self.shared_regressor(
                shared_representation
            ).squeeze(-1)
        )

        if (
            self.use_personalization
            and use_personal_adapter
        ):
            local_representation = (
                self.personal_adapter(
                    shared_representation
                )
            )

            residual_delta = (
                self.personal_residual_scale
                * torch.tanh(
                    self.personal_delta_head(
                        local_representation
                    ).squeeze(-1)
                )
            )

            personalized_prediction = torch.clamp(
                shared_prediction
                + residual_delta,
                min=0.0,
                max=1.0,
            )
        else:
            local_representation = (
                shared_representation
            )
            residual_delta = torch.zeros_like(
                shared_prediction
            )
            personalized_prediction = (
                shared_prediction
            )

        return {
            "prediction": personalized_prediction,
            "shared_prediction": shared_prediction,
            "personalized_prediction": personalized_prediction,
            "residual_delta": residual_delta,
            "twin_state": twin_state,
            "reconstruction": reconstruction,
            "sensor_summary": sensor_summary,
            "dense_gate": dense_gate,
            "sparse_gate": sparse_gate,
            "expert_utilization": utilization,
            "shared_representation": shared_representation,
            "local_representation": local_representation,
        }
