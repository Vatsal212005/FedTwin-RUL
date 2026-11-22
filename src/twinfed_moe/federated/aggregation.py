from collections import OrderedDict
import torch

LOCAL_ONLY_PREFIXES = ("personal_adapter.",)

def is_local_only(name: str) -> bool:
    return name.startswith(LOCAL_ONLY_PREFIXES)

def is_expert_param(name: str) -> bool:
    return name.startswith("experts.experts.")

def get_expert_id(name: str):
    # experts.experts.<id>.net....
    if not is_expert_param(name):
        return None
    parts = name.split(".")
    return int(parts[2])

def aggregate_twinfed(client_states, client_sizes, utilizations):
    """
    Shared layers:
        sample-count weighted FedAvg

    Expert m:
        weight client k by n_k * utilization[k,m]

    Local adapter:
        excluded from server aggregation
    """
    if not client_states:
        raise ValueError("No client states supplied.")

    result = OrderedDict()
    names = client_states[0].keys()

    total_n = float(sum(client_sizes))

    for name in names:
        if is_local_only(name):
            # Keep server copy unchanged using first state as placeholder.
            result[name] = client_states[0][name].clone()
            continue

        expert_id = get_expert_id(name)

        if expert_id is None:
            weights = [n / total_n for n in client_sizes]
        else:
            raw = [
                float(n) * float(util[expert_id])
                for n, util in zip(client_sizes, utilizations)
            ]
            denom = sum(raw)
            if denom <= 1e-12:
                weights = [n / total_n for n in client_sizes]
            else:
                weights = [w / denom for w in raw]

        acc = None
        for state, w in zip(client_states, weights):
            tensor = state[name].float() * w
            acc = tensor if acc is None else acc + tensor

        # restore original dtype when needed
        result[name] = acc.to(client_states[0][name].dtype)

    return result
