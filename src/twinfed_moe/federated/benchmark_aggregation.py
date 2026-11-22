import re

import torch


_EXPERT_PATTERNS = [
    re.compile(
        r"(?:^|\.)experts\.experts\.(\d+)\."
    ),
    re.compile(
        r"(?:^|\.)experts\.(\d+)\."
    ),
]


def _expert_index(
    parameter_name,
):
    for pattern in (
        _EXPERT_PATTERNS
    ):
        match = pattern.search(
            parameter_name
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


def aggregate_fedavg(
    client_states,
    sample_counts,
):
    """
    Sample-count weighted FedAvg.

    personalized adapter tensors are placeholders in the server/global copy;
    actual client-local adapters are preserved separately.
    """
    result = {}

    keys = list(
        client_states[0].keys()
    )

    for name in keys:
        if name.startswith(
            "personal_adapter."
        ):
            result[name] = (
                client_states[0][
                    name
                ].clone()
            )

            continue

        result[name] = (
            _weighted_average(
                [
                    state[name]
                    for state
                    in client_states
                ],
                sample_counts,
            )
        )

    return result


def aggregate_expertwise(
    client_states,
    sample_counts,
    expert_utilizations,
):
    """
    TwinFed-MoE aggregation:
    - ordinary shared tensors: sample-weighted FedAvg
    - expert m tensors: weighted by N_k * u_{k,m}
    - local adapters: never meaningfully aggregated
    """
    result = {}

    keys = list(
        client_states[0].keys()
    )

    for name in keys:
        if name.startswith(
            "personal_adapter."
        ):
            result[name] = (
                client_states[0][
                    name
                ].clone()
            )

            continue

        expert_idx = (
            _expert_index(
                name
            )
        )

        if expert_idx is None:
            weights = (
                sample_counts
            )
        else:
            weights = []

            for n_k, util in zip(
                sample_counts,
                expert_utilizations,
            ):
                if (
                    expert_idx
                    < int(
                        util.numel()
                    )
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
