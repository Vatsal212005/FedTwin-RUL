import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from twinfed_moe.data.cmapss import load_processed_cmapss
from twinfed_moe.models.twinfed_moe import TwinFedMoE
from twinfed_moe.training.metrics import metric_dict
from twinfed_moe.training.trainer import predict
from twinfed_moe.utils.device import resolve_device


def resolve_checkpoint(
    outputs_root: Path,
    checkpoint_arg,
):
    if checkpoint_arg:
        checkpoint = Path(
            checkpoint_arg
        )

        if not checkpoint.exists():
            raise FileNotFoundError(
                checkpoint
            )

        return checkpoint

    latest_pointer = (
        outputs_root
        / "latest_run.txt"
    )

    if not latest_pointer.exists():
        raise FileNotFoundError(
            "No --checkpoint supplied and "
            "outputs/latest_run.txt does not exist."
        )

    latest_dir = Path(
        latest_pointer.read_text(
            encoding="utf-8"
        ).strip()
    )

    checkpoint = (
        latest_dir
        / "best_model.pt"
    )

    if not checkpoint.exists():
        raise FileNotFoundError(
            checkpoint
        )

    return checkpoint


def build_model(
    cfg,
    dataset,
):
    m = cfg["model"]

    return TwinFedMoE(
        num_sensors=cfg["data"]["num_sensors"],
        twin_aux_dim=int(
            dataset.twin_aux.shape[-1]
        ),
        seq_len=cfg["data"]["seq_len"],
        d_model=m["d_model"],
        nhead=m["nhead"],
        transformer_layers=m["transformer_layers"],
        dim_feedforward=m["dim_feedforward"],
        num_experts=m["num_experts"],
        top_k=m["top_k"],
        twin_latent_dim=m["twin_latent_dim"],
        expert_hidden_dim=m["expert_hidden_dim"],
        adapter_hidden_dim=m["adapter_hidden_dim"],
        dropout=m["dropout"],
    )


def safe_metric_dict(
    pred,
    target,
):
    metrics = metric_dict(
        pred,
        target,
    )

    return {
        k: (
            None
            if not np.isfinite(v)
            else float(v)
        )
        for k, v
        in metrics.items()
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Path to best_model.pt. "
            "If omitted, outputs/latest_run.txt is used."
        ),
    )

    parser.add_argument(
        "--outputs-root",
        default="outputs",
    )

    parser.add_argument(
        "--device",
        default="auto",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
    )

    args = parser.parse_args()

    output_root = Path(
        args.outputs_root
    )

    checkpoint_path = resolve_checkpoint(
        output_root,
        args.checkpoint,
    )

    run_dir = (
        checkpoint_path.parent
    )

    device = resolve_device(
        args.device
    )

    print(
        f"Checkpoint: "
        f"{checkpoint_path}"
    )

    print(
        f"Device: "
        f"{device}"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    cfg = checkpoint[
        "config"
    ]

    test_path = cfg[
        "data"
    ]["test_path"]

    dataset = load_processed_cmapss(
        test_path
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    model = build_model(
        cfg,
        dataset,
    )

    model.load_state_dict(
        checkpoint[
            "global_model_state"
        ],
        strict=True,
    )

    model = model.to(
        device
    )

    adapter_mode = cfg.get(
        "evaluation",
        {},
    ).get(
        "official_test_adapter_mode",
        "bypass",
    )

    if adapter_mode != "bypass":
        raise ValueError(
            "Official unseen-engine evaluation currently "
            "supports only adapter_mode='bypass'."
        )

    # New engines have no learned client-local adapter.
    result = predict(
        model,
        loader,
        device,
        cfg,
        use_personal_adapter=False,
        collect_routing=True,
    )

    pred = result[
        "pred"
    ]

    target = result[
        "target"
    ]

    overall = safe_metric_dict(
        pred,
        target,
    )

    dense_gate = result[
        "dense_gate"
    ]

    sparse_gate = result[
        "sparse_gate"
    ]

    utilization = (
        result[
            "expert_utilization"
        ]
        .numpy()
    )

    routing_entropy = float(
        result[
            "routing_entropy"
        ]
    )

    top_expert = (
        sparse_gate.argmax(
            dim=1
        ).numpy()
    )

    unit_ids = np.asarray(
        dataset.unit_ids
    )

    regime_ids = np.asarray(
        dataset.regime_ids
    )

    pred_np = pred.numpy()
    target_np = target.numpy()

    abs_error = np.abs(
        pred_np - target_np
    )

    error = (
        pred_np - target_np
    )

    # --------------------------------------------
    # Per-regime metrics and expert utilization.
    # --------------------------------------------
    regime_summary = {}

    for regime in sorted(
        np.unique(
            regime_ids
        ).tolist()
    ):
        mask = (
            regime_ids == regime
        )

        regime_pred = pred[
            torch.as_tensor(mask)
        ]

        regime_target = target[
            torch.as_tensor(mask)
        ]

        regime_metrics = (
            safe_metric_dict(
                regime_pred,
                regime_target,
            )
        )

        regime_sparse = (
            sparse_gate[
                torch.as_tensor(mask)
            ]
        )

        regime_dense = (
            dense_gate[
                torch.as_tensor(mask)
            ]
        )

        regime_summary[
            str(int(regime))
        ] = {
            "count": int(
                mask.sum()
            ),
            **regime_metrics,
            "expert_utilization": [
                float(x)
                for x in (
                    regime_sparse
                    .mean(dim=0)
                    .tolist()
                )
            ],
            "mean_dense_gate": [
                float(x)
                for x in (
                    regime_dense
                    .mean(dim=0)
                    .tolist()
                )
            ],
        }

    # --------------------------------------------
    # Save main official-test metrics.
    # --------------------------------------------
    metrics_payload = {
        "checkpoint": str(
            checkpoint_path
        ),
        "checkpoint_round": int(
            checkpoint["round"]
        ),
        "checkpoint_validation_metrics": (
            checkpoint["metrics"]
        ),
        "dataset": cfg[
            "data"
        ].get(
            "subset",
            "FD004",
        ),
        "test_engines": int(
            len(dataset)
        ),
        "adapter_mode": (
            "bypass_for_unseen_engines"
        ),
        "metrics": overall,
        "expert_utilization": [
            float(x)
            for x in utilization.tolist()
        ],
        "routing_entropy": (
            routing_entropy
        ),
        "regime_summary": (
            regime_summary
        ),
    }

    metrics_path = (
        run_dir
        / "official_test_metrics.json"
    )

    metrics_path.write_text(
        json.dumps(
            metrics_payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------
    # Save per-engine predictions.
    # --------------------------------------------
    predictions_path = (
        run_dir
        / "official_test_predictions.csv"
    )

    num_experts = int(
        cfg["model"]["num_experts"]
    )

    with predictions_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        fieldnames = [
            "engine_id",
            "regime_id",
            "true_rul",
            "pred_rul",
            "error",
            "abs_error",
            "top_expert",
            *[
                f"dense_gate_{i}"
                for i in range(
                    num_experts
                )
            ],
            *[
                f"sparse_gate_{i}"
                for i in range(
                    num_experts
                )
            ],
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        dense_np = (
            dense_gate.numpy()
        )

        sparse_np = (
            sparse_gate.numpy()
        )

        for i in range(
            len(dataset)
        ):
            row = {
                "engine_id": int(
                    unit_ids[i]
                ),
                "regime_id": int(
                    regime_ids[i]
                ),
                "true_rul": float(
                    target_np[i]
                ),
                "pred_rul": float(
                    pred_np[i]
                ),
                "error": float(
                    error[i]
                ),
                "abs_error": float(
                    abs_error[i]
                ),
                "top_expert": int(
                    top_expert[i]
                ),
            }

            for expert_id in range(
                num_experts
            ):
                row[
                    f"dense_gate_"
                    f"{expert_id}"
                ] = float(
                    dense_np[
                        i,
                        expert_id,
                    ]
                )

                row[
                    f"sparse_gate_"
                    f"{expert_id}"
                ] = float(
                    sparse_np[
                        i,
                        expert_id,
                    ]
                )

            writer.writerow(
                row
            )

    # --------------------------------------------
    # Save compact regime/expert table.
    # --------------------------------------------
    regime_path = (
        run_dir
        / "regime_expert_summary.csv"
    )

    with regime_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        fieldnames = [
            "regime_id",
            "count",
            "rmse",
            "mae",
            "r2",
            "nasa_score",
            *[
                f"expert_util_{i}"
                for i in range(
                    num_experts
                )
            ],
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for regime_id, summary in (
            regime_summary.items()
        ):
            row = {
                "regime_id": int(
                    regime_id
                ),
                "count": summary[
                    "count"
                ],
                "rmse": summary[
                    "rmse"
                ],
                "mae": summary[
                    "mae"
                ],
                "r2": summary[
                    "r2"
                ],
                "nasa_score": summary[
                    "nasa_score"
                ],
            }

            for expert_id, value in enumerate(
                summary[
                    "expert_utilization"
                ]
            ):
                row[
                    f"expert_util_"
                    f"{expert_id}"
                ] = value

            writer.writerow(
                row
            )

    # --------------------------------------------
    # Update run summary.
    # --------------------------------------------
    summary_path = (
        run_dir
        / "run_summary.json"
    )

    if summary_path.exists():
        summary = json.loads(
            summary_path.read_text(
                encoding="utf-8"
            )
        )

        summary[
            "official_test_evaluated"
        ] = True

        summary[
            "official_test_metrics"
        ] = overall

        summary[
            "official_test_metrics_json"
        ] = str(
            metrics_path
        )

        summary[
            "official_test_predictions_csv"
        ] = str(
            predictions_path
        )

        summary_path.write_text(
            json.dumps(
                summary,
                indent=2,
            ),
            encoding="utf-8",
        )

    print(
        "\nOFFICIAL FD004 TEST"
    )

    print(
        f"Engines: "
        f"{len(dataset)}"
    )

    print(
        f"Checkpoint round: "
        f"{checkpoint['round']}"
    )

    print(
        "Adapter mode: "
        "bypass (unseen engines)"
    )

    print(
        f"RMSE: "
        f"{overall['rmse']:.4f}"
    )

    print(
        f"MAE: "
        f"{overall['mae']:.4f}"
    )

    print(
        f"R2: "
        f"{overall['r2']:.4f}"
    )

    print(
        f"NASA score: "
        f"{overall['nasa_score']:.4f}"
    )

    print(
        f"Expert utilization: "
        f"{[round(float(x), 4) for x in utilization.tolist()]}"
    )

    print(
        f"Routing entropy: "
        f"{routing_entropy:.4f}"
    )

    print(
        f"\nSaved: "
        f"{metrics_path}"
    )

    print(
        f"Saved: "
        f"{predictions_path}"
    )

    print(
        f"Saved: "
        f"{regime_path}"
    )


if __name__ == "__main__":
    main()
