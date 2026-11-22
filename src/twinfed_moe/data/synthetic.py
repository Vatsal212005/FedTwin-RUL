import numpy as np
import torch
from torch.utils.data import TensorDataset

def make_synthetic_rul_dataset(
    num_samples: int = 1600,
    seq_len: int = 30,
    num_sensors: int = 14,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)

    # latent degradation level in [0, 1]
    degradation = rng.uniform(0.0, 1.0, size=(num_samples, 1, 1)).astype(np.float32)
    operating = rng.normal(0.0, 0.7, size=(num_samples, 1, 3)).astype(np.float32)

    time = np.linspace(0, 1, seq_len, dtype=np.float32)[None, :, None]
    base = rng.normal(0, 0.25, size=(num_samples, seq_len, num_sensors)).astype(np.float32)

    sensor_trend = np.linspace(0.3, 1.2, num_sensors, dtype=np.float32)[None, None, :]
    x = base + degradation * time * sensor_trend

    # inject operating-regime effect into first 3 channels
    x[:, :, :3] += operating

    # higher degradation => lower RUL
    y = (125.0 * (1.0 - degradation[:, 0, 0]) + rng.normal(0, 3.0, num_samples)).astype(np.float32)

    # twin auxiliary features: mean operating channels + health proxy + drift proxy
    op = x[:, :, :3].mean(axis=1)
    health = (1.0 - degradation[:, 0, 0])[:, None]
    drift = x.std(axis=1).mean(axis=1, keepdims=True)
    twin_aux = np.concatenate([op, health, drift], axis=1).astype(np.float32)

    return TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(twin_aux, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
