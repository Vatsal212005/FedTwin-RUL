import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from twinfed_moe.models.twinfed_moe import TwinFedMoE

def test_forward():
    model = TwinFedMoE(
        num_sensors=14,
        twin_aux_dim=5,
        seq_len=30,
        d_model=64,
        nhead=4,
        transformer_layers=2,
        dim_feedforward=128,
        num_experts=4,
        top_k=2,
        twin_latent_dim=16,
        expert_hidden_dim=128,
        adapter_hidden_dim=32,
    )

    x = torch.randn(8, 30, 14)
    twin_aux = torch.randn(8, 5)
    out = model(x, twin_aux)

    assert out["prediction"].shape == (8,)
    assert out["sparse_gate"].shape == (8, 4)
    assert torch.allclose(
        out["sparse_gate"].sum(dim=1),
        torch.ones(8),
        atol=1e-5,
    )
