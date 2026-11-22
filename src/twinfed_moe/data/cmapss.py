from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class CMapssDataset(Dataset):
    def __init__(self, X, twin_aux, y, unit_ids, regime_ids):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.twin_aux = torch.as_tensor(twin_aux, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.unit_ids = np.asarray(unit_ids, dtype=np.int32)
        self.regime_ids = np.asarray(regime_ids, dtype=np.int16)

        n = len(self.X)
        if not (
            len(self.twin_aux) == n
            and len(self.y) == n
            and len(self.unit_ids) == n
            and len(self.regime_ids) == n
        ):
            raise ValueError("C-MAPSS arrays have inconsistent lengths.")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.twin_aux[idx], self.y[idx]


def load_processed_cmapss(path: str) -> CMapssDataset:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Run scripts/preprocess_cmapss.py first."
        )

    data = np.load(p)
    required = {"X", "twin_aux", "y", "unit_ids", "regime_ids"}
    missing = required.difference(data.files)

    if missing:
        raise ValueError(
            f"{p} is from the old preprocessing pass and is missing "
            f"{sorted(missing)}. Re-run preprocess_cmapss.py."
        )

    return CMapssDataset(
        X=data["X"],
        twin_aux=data["twin_aux"],
        y=data["y"],
        unit_ids=data["unit_ids"],
        regime_ids=data["regime_ids"],
    )
