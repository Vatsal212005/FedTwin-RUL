import math
import random
import time

import torch

from .aggregation import aggregate_twinfed
from twinfed_moe.training.metrics import metric_dict


class FederatedServer:
    def __init__(
        self,
        global_model,
        clients,
        cfg,
    ):
        self.global_model = global_model
        self.clients = clients
        self.cfg = cfg

    def _broadcast_shared(
        self,
        aggregated,
    ):
        """
        Broadcast global/shared parameters to every client while preserving
        each client's local personalized adapter.
        """
        for client in self.clients:
            local = client.model.state_dict()

            merged = {
                name: (
                    local[name]
                    if name.startswith(
                        "personal_adapter."
                    )
                    else tensor
                )
                for name, tensor
                in aggregated.items()
            }

            client.model.load_state_dict(
                merged,
                strict=True,
            )

    def _validation_metrics(self):
        """
        Evaluate the SAME validation population every communication round.

        Returns:
        - pooled personalized metrics across all validation windows
        - mean-client personalized metrics
        - worst-client personalized RMSE
        - pooled shared-only metrics with adapters bypassed
        """
        personalized_outputs = []
        shared_outputs = []
        client_metrics = []

        for client in self.clients:
            personal = client.predict_validation(
                use_personal_adapter=True,
                collect_routing=False,
            )

            shared = client.predict_validation(
                use_personal_adapter=False,
                collect_routing=False,
            )

            personalized_outputs.append(
                personal
            )

            shared_outputs.append(
                shared
            )

            client_metrics.append(
                metric_dict(
                    personal["pred"],
                    personal["target"],
                )
            )

        pooled_pred = torch.cat([
            x["pred"]
            for x in personalized_outputs
        ])

        pooled_target = torch.cat([
            x["target"]
            for x in personalized_outputs
        ])

        shared_pred = torch.cat([
            x["pred"]
            for x in shared_outputs
        ])

        shared_target = torch.cat([
            x["target"]
            for x in shared_outputs
        ])

        pooled = metric_dict(
            pooled_pred,
            pooled_target,
        )

        shared_pooled = metric_dict(
            shared_pred,
            shared_target,
        )

        return {
            "pooled_rmse": pooled["rmse"],
            "pooled_mae": pooled["mae"],
            "pooled_r2": pooled["r2"],
            "pooled_nasa_score": pooled[
                "nasa_score"
            ],

            "mean_client_rmse": (
                sum(
                    x["rmse"]
                    for x in client_metrics
                )
                / len(client_metrics)
            ),

            "mean_client_mae": (
                sum(
                    x["mae"]
                    for x in client_metrics
                )
                / len(client_metrics)
            ),

            "mean_client_r2": (
                sum(
                    x["r2"]
                    for x in client_metrics
                )
                / len(client_metrics)
            ),

            "worst_client_rmse": max(
                x["rmse"]
                for x in client_metrics
            ),

            "shared_pooled_rmse": (
                shared_pooled["rmse"]
            ),

            "shared_pooled_mae": (
                shared_pooled["mae"]
            ),

            "shared_pooled_r2": (
                shared_pooled["r2"]
            ),

            "shared_pooled_nasa_score": (
                shared_pooled[
                    "nasa_score"
                ]
            ),

            # Positive value means personalization improves RMSE.
            "personalization_gain_rmse": (
                shared_pooled["rmse"]
                - pooled["rmse"]
            ),
        }

    def run_round(
        self,
        round_idx: int,
    ):
        start = time.perf_counter()

        rate = self.cfg[
            "federated"
        ]["participation_rate"]

        m = max(
            1,
            int(round(
                len(self.clients) * rate
            )),
        )

        selected = random.sample(
            self.clients,
            m,
        )

        print(
            f"Round {round_idx:03d}: "
            f"selected clients "
            f"{[c.client_id for c in selected]}",
            flush=True,
        )

        global_state = {
            k: v.detach().cpu().clone()
            for k, v
            in self.global_model.state_dict().items()
        }

        results = []

        for pos, client in enumerate(
            selected,
            start=1,
        ):
            print(
                f"  Client {client.client_id} "
                f"({pos}/{m}) | "
                f"{len(client.train_dataset):,} "
                f"train windows...",
                flush=True,
            )

            result = client.train_from_global(
                global_state
            )

            results.append(
                result
            )

            print(
                f"    done in "
                f"{result['train_seconds']:.1f}s | "
                f"loss={result['loss']:.5f}",
                flush=True,
            )

        aggregated = aggregate_twinfed(
            [r["state"] for r in results],
            [r["num_samples"] for r in results],
            [r["expert_utilization"] for r in results],
        )

        self.global_model.load_state_dict(
            aggregated,
            strict=True,
        )

        self._broadcast_shared(
            aggregated
        )

        validation = (
            self._validation_metrics()
        )

        # Weighted training-time expert utilization.
        total_samples = float(
            sum(
                r["num_samples"]
                for r in results
            )
        )

        weighted_util = None

        for result in results:
            weight = (
                result["num_samples"]
                / total_samples
            )

            util = (
                result["expert_utilization"]
                .float()
            )

            weighted_util = (
                util * weight
                if weighted_util is None
                else weighted_util
                + util * weight
            )

        util_prob = (
            weighted_util
            / weighted_util.sum().clamp_min(
                1e-12
            )
        )

        expert_entropy = float(
            -torch.sum(
                util_prob
                * torch.log(
                    util_prob.clamp_min(
                        1e-12
                    )
                )
            )
        )

        result_metrics = {
            "round": round_idx,

            "mean_train_loss": (
                sum(
                    r["loss"]
                    for r in results
                )
                / len(results)
            ),

            **validation,

            "expert_utilization_entropy": (
                expert_entropy
            ),

            "round_seconds": (
                time.perf_counter()
                - start
            ),
        }

        for expert_id, value in enumerate(
            weighted_util.tolist()
        ):
            result_metrics[
                f"expert_util_{expert_id}"
            ] = float(value)

        return result_metrics

    def checkpoint_payload(
        self,
        round_idx,
        metrics,
    ):
        """
        Everything needed to reproduce the shared model and the known-client
        personalized validation state at a selected round.
        """
        return {
            "round": int(
                round_idx
            ),
            "metrics": dict(
                metrics
            ),
            "config": self.cfg,
            "global_model_state": {
                k: v.detach().cpu().clone()
                for k, v
                in self.global_model.state_dict().items()
            },
            "client_adapter_states": {
                int(client.client_id):
                    client.adapter_state_dict()
                for client in self.clients
            },
            "client_splits": [
                client.split_metadata()
                for client in self.clients
            ],
        }
