import re

import torch

from twinfed_moe.training.v2_trainer import (
    is_local_parameter,
)


_EXPERT_PATTERNS = [
    re.compile(
        r"(?:^|\.)experts\.experts\.(\d+)\."
    ),
    re.compile(
        r"(?:^|\.)experts\.(\d+)\."
    ),
]


def _expert_index(
    name,
):
    for pattern in (
        _EXPERT_PATTERNS
    ):
        match = pattern.search(
            name
        )

        if match:
            return int(
                match.group(1)
            )

    return None


def _weighted_average(
    tensors,
    weights,
):
    first = tensors[0]

    if not (
        first.is_floating_point()
        or first.is_complex()
    ):
        return first.clone()

    total_weight = float(
        sum(weights)
    )

    if total_weight <= 0:
        weights = [
            1.0
            for _ in tensors
        ]
        total_weight = float(
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
                / total_weight
            ),
        )

    return result


def aggregate_v2(
    client_states,
    sample_counts,
    expert_utilizations,
    mode,
):
    result = {}

    for name in (
        client_states[
            0
        ].keys()
    ):
        if is_local_parameter(
            name
        ):
            # Server copy is only a placeholder.
            result[name] = (
                client_states[
                    0
                ][name].clone()
            )
            continue

        expert_idx = (
            _expert_index(
                name
            )
        )

        if (
            mode
            == "expertwise"
            and expert_idx
            is not None
        ):
            weights = []

            for n_k, util in zip(
                sample_counts,
                expert_utilizations,
            ):
                if (
                    expert_idx
                    < util.numel()
                ):
                    u = float(
                        util[
                            expert_idx
                        ]
                    )
                else:
                    u = 0.0

                weights.append(
                    float(n_k)
                    * max(
                        u,
                        1e-8,
                    )
                )
        else:
            weights = [
                float(x)
                for x in sample_counts
            ]

        result[name] = (
            _weighted_average(
                [
                    state[name]
                    for state
                    in client_states
                ],
                weights,
            )
        )

    return result
