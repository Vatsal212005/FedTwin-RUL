import numpy as np
from torch.utils.data import Subset

def dirichlet_partition(dataset, num_clients: int, alpha: float, seed: int = 42):
    """Synthetic-data helper."""
    rng = np.random.default_rng(seed)
    ys = np.array([float(dataset[i][2]) for i in range(len(dataset))])
    bins = np.quantile(ys, [0.2, 0.4, 0.6, 0.8])
    labels = np.digitize(ys, bins)

    client_indices = [[] for _ in range(num_clients)]
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        proportions = rng.dirichlet(np.full(num_clients, alpha))
        cuts = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        splits = np.split(idx, cuts)
        for k, split in enumerate(splits):
            client_indices[k].extend(split.tolist())

    if any(len(v) == 0 for v in client_indices):
        all_idx = np.arange(len(dataset))
        rng.shuffle(all_idx)
        client_indices = [x.tolist() for x in np.array_split(all_idx, num_clients)]

    return [Subset(dataset, sorted(x)) for x in client_indices]

def _dominant_regime_by_unit(unit_ids, regime_ids):
    result = {}
    for unit in np.unique(unit_ids):
        r = regime_ids[unit_ids == unit]
        values, counts = np.unique(r, return_counts=True)
        result[int(unit)] = int(values[np.argmax(counts)])
    return result

def balanced_regime_partition(
    dataset,
    num_clients: int,
    alpha: float,
    seed: int = 42,
    affinity_weight: float = 0.20,
):
    """
    Whole-engine, regime-aware, load-balanced partition.

    Engines remain intact. Dirichlet preferences preserve non-IID regime
    specialization, while a greedy load term plus a rebalancing pass prevents
    the extreme client-size skew seen in the earlier prototype.
    """
    rng = np.random.default_rng(seed)
    unit_ids = np.asarray(dataset.unit_ids)
    regime_ids = np.asarray(dataset.regime_ids)
    units = [int(u) for u in np.unique(unit_ids)]

    if len(units) < num_clients:
        raise ValueError("Not enough engines for requested clients.")

    dominant = _dominant_regime_by_unit(unit_ids, regime_ids)
    regimes = sorted(int(r) for r in np.unique(regime_ids))
    regime_to_col = {r: i for i, r in enumerate(regimes)}

    preferences = rng.dirichlet(
        np.full(len(regimes), alpha),
        size=num_clients,
    )

    unit_sizes = {
        u: int(np.sum(unit_ids == u))
        for u in units
    }

    ordered = sorted(units, key=lambda u: unit_sizes[u], reverse=True)
    target = len(dataset) / float(num_clients)

    client_units = [[] for _ in range(num_clients)]
    loads = np.zeros(num_clients, dtype=np.float64)

    # Guarantee each client gets an engine.
    for cid, unit in enumerate(ordered[:num_clients]):
        client_units[cid].append(unit)
        loads[cid] += unit_sizes[unit]

    for unit in ordered[num_clients:]:
        reg_col = regime_to_col[dominant[unit]]
        pref_cost = -np.log(
            np.clip(preferences[:, reg_col], 1e-8, 1.0)
        )
        projected = (loads + unit_sizes[unit]) / max(target, 1.0)
        score = projected + affinity_weight * pref_cost

        # Avoid assigning to a client already far above target when possible.
        viable = projected <= 1.30
        if np.any(viable):
            score = np.where(viable, score, np.inf)

        cid = int(np.argmin(score))
        client_units[cid].append(unit)
        loads[cid] += unit_sizes[unit]

    # Small balancing pass. Move engines only when it improves max/min balance.
    for _ in range(500):
        hi = int(np.argmax(loads))
        lo = int(np.argmin(loads))
        if loads[hi] <= 1.20 * target and loads[lo] >= 0.80 * target:
            break
        if len(client_units[hi]) <= 1:
            break

        old_gap = loads[hi] - loads[lo]
        best = None
        best_gap = old_gap

        for unit in client_units[hi]:
            size = unit_sizes[unit]
            new_hi = loads[hi] - size
            new_lo = loads[lo] + size
            new_gap = abs(new_hi - new_lo)
            if new_gap < best_gap:
                best_gap = new_gap
                best = unit

        if best is None:
            break

        client_units[hi].remove(best)
        client_units[lo].append(best)
        loads[hi] -= unit_sizes[best]
        loads[lo] += unit_sizes[best]

    subsets = []
    for units_for_client in client_units:
        idx = np.where(np.isin(unit_ids, units_for_client))[0].tolist()
        subsets.append(Subset(dataset, idx))
    return subsets

def unitwise_regime_partition(dataset, num_clients: int, alpha: float, seed: int = 42):
    return balanced_regime_partition(
        dataset=dataset,
        num_clients=num_clients,
        alpha=alpha,
        seed=seed,
    )

def split_subset_by_unit(subset, val_fraction: float = 0.2, seed: int = 42):
    base = subset.dataset
    subset_idx = np.asarray(subset.indices, dtype=np.int64)
    units = np.asarray(base.unit_ids)[subset_idx]
    unique_units = np.unique(units)

    if len(unique_units) < 2:
        return subset, subset

    rng = np.random.default_rng(seed)
    rng.shuffle(unique_units)

    n_val = max(1, int(round(len(unique_units) * val_fraction)))
    n_val = min(n_val, len(unique_units) - 1)
    val_units = set(unique_units[:n_val].tolist())

    train_idx, val_idx = [], []
    for idx, unit in zip(subset_idx.tolist(), units.tolist()):
        (val_idx if unit in val_units else train_idx).append(idx)

    return Subset(base, train_idx), Subset(base, val_idx)
