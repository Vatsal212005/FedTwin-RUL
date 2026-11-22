import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from twinfed_moe.training.trainer import (
    train_local,
    evaluate,
    predict,
)


class FederatedClient:
    def __init__(
        self,
        client_id,
        train_dataset,
        val_dataset,
        model_factory,
        cfg,
        device,
    ):
        self.client_id = client_id
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.model = model_factory().to(device)
        self.cfg = cfg
        self.device = device

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=cfg["data"]["batch_size"],
            shuffle=True,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=cfg["data"]["batch_size"],
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

    def train_from_global(
        self,
        global_state,
    ):
        current = self.model.state_dict()

        merged = {
            name: (
                current[name]
                if name.startswith(
                    "personal_adapter."
                )
                else tensor
            )
            for name, tensor
            in global_state.items()
        }

        self.model.load_state_dict(
            merged,
            strict=True,
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg["training"]["lr"],
            weight_decay=self.cfg["training"]["weight_decay"],
        )

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        result = train_local(
            self.model,
            self.train_loader,
            optimizer,
            self.device,
            self.cfg,
            global_state=global_state,
        )

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = (
            time.perf_counter()
            - start
        )

        return {
            "client_id": self.client_id,
            "num_samples": len(
                self.train_dataset
            ),
            "state": {
                k: v.detach().cpu().clone()
                for k, v
                in self.model.state_dict().items()
            },
            "expert_utilization": result[
                "expert_utilization"
            ],
            "loss": result["loss"],
            "train_seconds": elapsed,
        }

    def evaluate(
        self,
        use_personal_adapter: bool = True,
    ):
        return evaluate(
            self.model,
            self.val_loader,
            self.device,
            cfg=self.cfg,
            use_personal_adapter=use_personal_adapter,
        )

    def predict_validation(
        self,
        use_personal_adapter: bool = True,
        collect_routing: bool = False,
    ):
        return predict(
            self.model,
            self.val_loader,
            self.device,
            cfg=self.cfg,
            use_personal_adapter=use_personal_adapter,
            collect_routing=collect_routing,
        )

    def adapter_state_dict(self):
        return {
            name: tensor.detach().cpu().clone()
            for name, tensor
            in self.model.state_dict().items()
            if name.startswith(
                "personal_adapter."
            )
        }

    def split_metadata(self):
        """
        Persist engine identities used in this client's train/validation split.
        This makes the reported validation protocol reproducible.
        """
        base = self.train_dataset.dataset

        train_indices = np.asarray(
            self.train_dataset.indices,
            dtype=np.int64,
        )

        val_indices = np.asarray(
            self.val_dataset.indices,
            dtype=np.int64,
        )

        train_units = sorted(
            set(
                np.asarray(base.unit_ids)[
                    train_indices
                ].tolist()
            )
        )

        val_units = sorted(
            set(
                np.asarray(base.unit_ids)[
                    val_indices
                ].tolist()
            )
        )

        return {
            "client_id": int(
                self.client_id
            ),
            "train_units": [
                int(x)
                for x in train_units
            ],
            "val_units": [
                int(x)
                for x in val_units
            ],
            "train_windows": int(
                len(self.train_dataset)
            ),
            "val_windows": int(
                len(self.val_dataset)
            ),
        }
