import torch
import torch.nn.functional as F

def rul_mse_loss(pred, target):
    return F.mse_loss(pred, target)

def digital_twin_reconstruction_loss(reconstruction, sensor_summary):
    return F.mse_loss(reconstruction, sensor_summary)

def load_balance_loss(dense_gate):
    mean_prob = dense_gate.mean(dim=0)
    target = torch.full_like(mean_prob, 1.0 / mean_prob.numel())
    return F.mse_loss(mean_prob, target)

def twin_guided_consistency_loss(
    twin_state, dense_gate, sigma: float = 1.0, max_pairs: int = 256
):
    """Monte-Carlo estimate of pairwise TGC loss; O(P), not O(B^2)."""
    b = twin_state.size(0)
    if b < 2:
        return twin_state.new_tensor(0.0)

    p = min(max_pairs, b * (b - 1))
    i = torch.randint(0, b, (p,), device=twin_state.device)
    j = torch.randint(0, b - 1, (p,), device=twin_state.device)
    j = j + (j >= i).long()

    dz2 = torch.sum((twin_state[i] - twin_state[j]) ** 2, dim=-1)
    similarity = torch.exp(-dz2 / (2.0 * sigma ** 2))
    dg2 = torch.sum((dense_gate[i] - dense_gate[j]) ** 2, dim=-1)
    return (similarity * dg2).mean()

def prepare_fedprox_reference(global_state, device):
    if global_state is None:
        return None
    return {
        name: tensor.to(device=device, non_blocking=True)
        for name, tensor in global_state.items()
        if not name.startswith("personal_adapter.")
    }

def fedprox_loss(model, prox_reference, mu: float):
    if mu <= 0 or prox_reference is None:
        return next(model.parameters()).new_tensor(0.0)

    penalty = next(model.parameters()).new_tensor(0.0)
    for name, param in model.named_parameters():
        if name in prox_reference and not name.startswith("personal_adapter."):
            penalty = penalty + torch.sum((param - prox_reference[name]) ** 2)
    return 0.5 * mu * penalty
