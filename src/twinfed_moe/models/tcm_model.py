import torch
from torch import nn

from .digital_twin import DigitalTwinEncoder
from .transformer_encoder import TemporalTransformer


class TwinConditionedAffine(nn.Module):
    """
    Feature-wise modulation:
        h_tcm = gamma(z) * LN(h) + beta(z)

    gamma is initialized around 1 and beta around 0 so the model starts
    near the plain Transformer representation rather than from a destructive
    random modulation.
    """

    def __init__(
        self,
        feature_dim: int,
        twin_dim: int,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(
            feature_dim
        )

        self.conditioner = nn.Sequential(
            nn.Linear(
                twin_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                hidden_dim,
                2 * feature_dim,
            ),
        )

        # Identity-like initialization.
        final = self.conditioner[-1]
        nn.init.zeros_(
            final.weight
        )
        nn.init.zeros_(
            final.bias
        )

    def forward(
        self,
        feature,
        twin_state,
    ):
        params = self.conditioner(
            twin_state
        )

        gamma_delta, beta = (
            params.chunk(
                2,
                dim=-1,
            )
        )

        gamma = (
            1.0
            + 0.25
            * torch.tanh(
                gamma_delta
            )
        )

        beta = (
            0.25
            * torch.tanh(
                beta
            )
        )

        normalized = self.norm(
            feature
        )

        modulated = (
            gamma * normalized
            + beta
        )

        return (
            modulated,
            gamma,
            beta,
        )


class TwinFedTCM(nn.Module):
    """
    Unified model for the TCM comparison suite.

    fusion_mode:
      "concat"   -> concatenate h and z, then fuse through an MLP
      "tcm"      -> twin-conditioned affine modulation
      "identity" -> plain normalized Transformer representation

    The Digital Twin branch remains lightweight and data-driven.
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
        twin_latent_dim: int,
        fusion_hidden_dim: int,
        dropout: float,
        fusion_mode: str = "tcm",
        use_dt_reconstruction: bool = True,
    ):
        super().__init__()

        self.fusion_mode = str(
            fusion_mode
        )

        self.use_dt_reconstruction = bool(
            use_dt_reconstruction
        )

        self.digital_twin = (
            DigitalTwinEncoder(
                num_sensors=num_sensors,
                aux_dim=twin_aux_dim,
                latent_dim=twin_latent_dim,
            )
        )

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

        if self.fusion_mode == "tcm":
            self.tcm = (
                TwinConditionedAffine(
                    feature_dim=d_model,
                    twin_dim=twin_latent_dim,
                    hidden_dim=fusion_hidden_dim,
                    dropout=dropout,
                )
            )

            self.concat_fusion = None
            self.identity_norm = None

        elif self.fusion_mode == "concat":
            self.tcm = None

            self.concat_fusion = nn.Sequential(
                nn.Linear(
                    d_model
                    + twin_latent_dim,
                    fusion_hidden_dim,
                ),
                nn.GELU(),
                nn.Dropout(
                    dropout
                ),
                nn.Linear(
                    fusion_hidden_dim,
                    d_model,
                ),
                nn.LayerNorm(
                    d_model
                ),
            )

            self.identity_norm = None

        elif self.fusion_mode == "identity":
            self.tcm = None
            self.concat_fusion = None
            self.identity_norm = nn.LayerNorm(
                d_model
            )

        else:
            raise ValueError(
                f"Unknown fusion_mode: "
                f"{self.fusion_mode}"
            )

        self.regressor = nn.Sequential(
            nn.LayerNorm(
                d_model
            ),
            nn.Linear(
                d_model,
                1,
            ),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x,
        twin_aux,
    ):
        (
            twin_state,
            reconstruction,
            sensor_summary,
        ) = self.digital_twin(
            x,
            twin_aux,
        )

        temporal = self.temporal_encoder(
            x
        )

        if self.fusion_mode == "tcm":
            (
                fused,
                gamma,
                beta,
            ) = self.tcm(
                temporal,
                twin_state,
            )

        elif self.fusion_mode == "concat":
            fused = self.concat_fusion(
                torch.cat(
                    [
                        temporal,
                        twin_state,
                    ],
                    dim=-1,
                )
            )

            gamma = torch.ones_like(
                temporal
            )
            beta = torch.zeros_like(
                temporal
            )

        else:
            fused = self.identity_norm(
                temporal
            )

            gamma = torch.ones_like(
                temporal
            )
            beta = torch.zeros_like(
                temporal
            )

        prediction = self.regressor(
            fused
        ).squeeze(-1)

        return {
            "prediction": prediction,
            "twin_state": twin_state,
            "reconstruction": reconstruction,
            "sensor_summary": sensor_summary,
            "temporal_feature": temporal,
            "fused_feature": fused,
            "gamma": gamma,
            "beta": beta,
        }
