import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from twinfed_moe.models.twinfed_moe import TwinFedMoE
from twinfed_moe.federated.aggregation import aggregate_twinfed

def make_model():
    return TwinFedMoE(
        num_sensors=14,
        twin_aux_dim=5,
        seq_len=30,
        d_model=32,
        nhead=4,
        transformer_layers=1,
        dim_feedforward=64,
        num_experts=2,
        top_k=1,
        twin_latent_dim=8,
        expert_hidden_dim=64,
        adapter_hidden_dim=16,
    )

def test_aggregation_runs():
    a = make_model().state_dict()
    b = make_model().state_dict()
    util_a = torch.tensor([0.9, 0.1])
    util_b = torch.tensor([0.1, 0.9])

    out = aggregate_twinfed(
        [a, b],
        [100, 100],
        [util_a, util_b],
    )
    assert set(out.keys()) == set(a.keys())
