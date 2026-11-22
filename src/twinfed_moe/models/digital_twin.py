import torch
from torch import nn

class DigitalTwinEncoder(nn.Module):
    """
    Lightweight data-driven digital twin.

    Input:
        raw/current sensor window summary + auxiliary operating-state features
    Output:
        twin_state vector z
        reconstruction for DT consistency/reconstruction loss
    """
    def __init__(self, num_sensors: int, aux_dim: int, latent_dim: int):
        super().__init__()
        self.sensor_encoder = nn.Sequential(
            nn.Linear(num_sensors, 64),
            nn.GELU(),
            nn.Linear(64, latent_dim),
        )
        self.aux_encoder = nn.Sequential(
            nn.Linear(aux_dim, latent_dim),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_sensors),
        )

    def forward(self, x_window: torch.Tensor, twin_aux: torch.Tensor):
        # summarize the current machine window
        sensor_summary = x_window.mean(dim=1)
        sensor_latent = self.sensor_encoder(sensor_summary)
        aux_latent = self.aux_encoder(twin_aux)

        twin_state = 0.5 * (sensor_latent + aux_latent)
        reconstruction = self.decoder(sensor_latent)

        return twin_state, reconstruction, sensor_summary
