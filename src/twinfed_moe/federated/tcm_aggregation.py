import torch


def _weighted_average(
    tensors,
    weights,
):
    first = tensors[
        0
    ]

    if not (
        first.is_floating_point()
        or first.is_complex()
    ):
        return first.clone()

    total = float(
        sum(weights)
    )

    if total <= 0:
        weights = [
            1.0
            for _ in tensors
        ]
        total = float(
            len(tensors)
        )

    result = torch.zeros_like(
        first
    )

    for tensor, weight in zip(
        tensors,
        weights,
    ):
        result.add_(
            tensor,
            alpha=(
                float(weight)
                / total
            ),
        )

    return result


def fedavg_state_dicts(
    states,
    sample_counts,
):
    aggregated = {}

    for name in (
        states[
            0
        ].keys()
    ):
        aggregated[
            name
        ] = _weighted_average(
            [
                state[
                    name
                ]
                for state
                in states
            ],
            sample_counts,
        )

    return aggregated
