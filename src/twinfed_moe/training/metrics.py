import math
import torch


def rmse(pred, target):
    return torch.sqrt(
        torch.mean((pred - target) ** 2)
    ).item()


def mae(pred, target):
    return torch.mean(
        torch.abs(pred - target)
    ).item()


def r2_score_torch(pred, target):
    ss_res = torch.sum(
        (target - pred) ** 2
    )

    ss_tot = torch.sum(
        (target - torch.mean(target)) ** 2
    )

    if float(ss_tot) <= 1e-12:
        return float("nan")

    return float(
        1.0 - ss_res / ss_tot
    )


def nasa_score(pred, target):
    """
    NASA C-MAPSS asymmetric score.

    d = predicted RUL - true RUL

    d < 0  -> exp(-d/13)-1
    d >= 0 -> exp(d/10)-1

    Lower is better.
    """
    d = pred - target

    score = torch.where(
        d < 0,
        torch.exp(-d / 13.0) - 1.0,
        torch.exp(d / 10.0) - 1.0,
    )

    return float(
        score.sum()
    )


def routing_entropy(probabilities):
    """
    Mean per-sample categorical entropy of expert routing.
    """
    p = probabilities.clamp_min(1e-12)

    entropy = -torch.sum(
        p * torch.log(p),
        dim=-1,
    )

    return float(
        entropy.mean()
    )


def metric_dict(pred, target):
    return {
        "rmse": rmse(pred, target),
        "mae": mae(pred, target),
        "r2": r2_score_torch(pred, target),
        "nasa_score": nasa_score(pred, target),
    }
