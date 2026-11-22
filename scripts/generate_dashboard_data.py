"""Generate browser-ready demo data from the real FedTwin-RUL checkpoint.

The output contains only real FD004 rows and model inference. It is intentionally
generated ahead of time so the frontend remains a fast, dependency-light Vite app.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from preprocess_cmapss import (  # noqa: E402
    OP_COLS,
    SELECTED_SENSORS,
    apply_standardizer,
    pad_left,
    read_cmapss_table,
    twin_aux_from_window,
)
from twinfed_moe.data.cmapss import load_processed_cmapss  # noqa: E402
from twinfed_moe.data.partition import balanced_regime_partition  # noqa: E402
from twinfed_moe.models.tcm_model import TwinFedTCM  # noqa: E402


DEFAULT_CHECKPOINT = ROOT / "outputs" / "fedtwin_final_FD004_20260827_211745" / "runs" / "alpha_0.1" / "seed_42" / "fedtwin_rul" / "best_model.pt"
DEFAULT_RESULTS = ROOT / "outputs" / "fedtwin_final_FD004_20260827_211745" / "final_with_published_baselines.csv"
DEFAULT_OUTPUT = ROOT / "app" / "frontend" / "public" / "demo-data.json"


def build_model(cfg: dict, aux_dim: int) -> TwinFedTCM:
    model_cfg = cfg["model"]
    return TwinFedTCM(
        num_sensors=cfg["data"]["num_sensors"],
        twin_aux_dim=aux_dim,
        seq_len=cfg["data"]["seq_len"],
        d_model=model_cfg["d_model"],
        nhead=model_cfg["nhead"],
        transformer_layers=model_cfg["transformer_layers"],
        dim_feedforward=model_cfg["dim_feedforward"],
        twin_latent_dim=model_cfg["twin_latent_dim"],
        fusion_hidden_dim=model_cfg["fusion_hidden_dim"],
        dropout=model_cfg["dropout"],
        fusion_mode="concat",
        use_dt_reconstruction=True,
    )


def round_list(values, digits=4):
    return [round(float(value), digits) for value in values]


def client_summary(cfg: dict) -> list[dict]:
    dataset = load_processed_cmapss(str(ROOT / cfg["data"]["train_path"]))
    clients = balanced_regime_partition(
        dataset,
        num_clients=cfg["data"]["num_clients"],
        alpha=0.1,
        seed=42,
        affinity_weight=cfg["data"].get("regime_affinity_weight", 0.20),
    )
    rows = []
    for client_id, subset in enumerate(clients, start=1):
        indices = np.asarray(subset.indices, dtype=np.int64)
        units = np.asarray(dataset.unit_ids)[indices]
        regimes = np.asarray(dataset.regime_ids)[indices]
        values, counts = np.unique(regimes, return_counts=True)
        rows.append({
            "id": client_id,
            "engines": int(len(np.unique(units))),
            "windows": int(len(indices)),
            "dominantRegime": int(values[np.argmax(counts)]) + 1,
            "status": "synchronized",
        })
    return rows


def result_summary() -> list[dict]:
    data = pd.read_csv(DEFAULT_RESULTS)
    order = ["fedtwin_rul", "fedavg_moe", "fedavg_transformer", "fedprox_transformer", "trans_lighter_repro"]
    names = {
        "fedtwin_rul": "FedTwin-RUL",
        "fedavg_moe": "FedAvg-MoE",
        "fedavg_transformer": "FedAvg Transformer",
        "fedprox_transformer": "FedProx Transformer",
        "trans_lighter_repro": "Trans-Lighter protocol-aligned reproduction",
    }
    rows = []
    for method in order:
        method_rows = data[data["method"] == method].sort_values("alpha", ascending=False)
        rows.append({
            "key": method,
            "name": names[method],
            "parameters": int(method_rows["parameters_mean"].iloc[0]),
            "scores": [
                {
                    "alpha": float(row.alpha),
                    "rmse": round(float(row.test_rmse_mean), 3),
                    "std": round(float(row.test_rmse_std), 3),
                }
                for row in method_rows.itertuples()
            ],
            "meanRmse": round(float(method_rows["test_rmse_mean"].mean()), 3),
        })
    return rows


def engine_replay(model, cfg, meta, normalized, rul_end, engine_id: int) -> dict:
    seq_len = int(cfg["data"]["seq_len"])
    rul_cap = float(cfg["data"]["rul_cap"])
    unit = normalized[normalized["unit"] == engine_id].sort_values("cycle")
    if unit.empty:
        raise ValueError(f"Engine {engine_id} was not found in FD004 test data")

    sensors = unit[SELECTED_SENSORS].to_numpy(np.float32)
    ops = unit[OP_COLS].to_numpy(np.float32)
    healthy_ref = np.asarray(meta["healthy_reference_standardized"], dtype=np.float32)
    centers = np.asarray(meta["regime_centers_standardized"], dtype=np.float32)

    windows, auxiliaries = [], []
    for end in range(1, len(unit) + 1):
        sensor_window = pad_left(sensors[:end], seq_len)
        op_window = pad_left(ops[:end], seq_len)
        windows.append(sensor_window)
        auxiliaries.append(twin_aux_from_window(sensor_window, op_window, healthy_ref))

    x = torch.as_tensor(np.asarray(windows), dtype=torch.float32)
    aux = torch.as_tensor(np.asarray(auxiliaries), dtype=torch.float32)
    predictions, twin_norms = [], []
    with torch.inference_mode():
        for start in range(0, len(x), 128):
            out = model(x[start:start + 128], aux[start:start + 128])
            predictions.extend((out["prediction"] * rul_cap).cpu().tolist())
            twin_norms.extend(torch.linalg.vector_norm(out["twin_state"], dim=-1).cpu().tolist())

    final_offset = float(rul_end[engine_id - 1])
    max_cycle = int(unit["cycle"].max())
    timeline = []
    for index, row in enumerate(unit.itertuples(index=False)):
        aux_row = auxiliaries[index]
        op_latest = ops[index]
        regime = int(np.argmin(np.linalg.norm(centers - op_latest, axis=1))) + 1
        true_rul = min(rul_cap, final_offset + max_cycle - int(row.cycle))
        timeline.append({
            "cycle": int(row.cycle),
            "predictedRul": round(float(predictions[index]), 2),
            "trueRul": round(float(true_rul), 2),
            "q": round(float(aux_row[3]), 4),
            "drift": round(float(aux_row[4]), 4),
            "ops": round_list(aux_row[:3]),
            "regime": regime,
            "twinNorm": round(float(twin_norms[index]), 4),
            "sensors": round_list(sensors[index], 4),
        })

    return {
        "id": engine_id,
        "cycles": len(timeline),
        "finalTrueRul": final_offset,
        "timeline": timeline,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--engines", type=int, nargs="+", default=[1, 25, 50, 100])
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    model = build_model(cfg, aux_dim=5)
    model.load_state_dict(checkpoint["global_model_state"], strict=True)
    model.eval()

    meta = json.loads((ROOT / "data" / "processed" / "FD004_meta.json").read_text(encoding="utf-8"))
    raw = read_cmapss_table(ROOT / "data" / "raw" / "cmapss" / "test_FD004.txt")
    norm_cols = SELECTED_SENSORS + OP_COLS
    mean = pd.Series(meta["normalization"]["mean"])
    std = pd.Series(meta["normalization"]["std"])
    normalized = apply_standardizer(raw, mean, std, norm_cols)
    rul_end = pd.read_csv(ROOT / "data" / "raw" / "cmapss" / "RUL_FD004.txt", sep=r"\s+", header=None).iloc[:, 0].to_numpy(np.float32)

    payload = {
        "generatedFrom": str(args.checkpoint.relative_to(ROOT)).replace("\\", "/"),
        "model": {
            "name": "FedTwin-RUL",
            "alpha": float(checkpoint["alpha"]),
            "seed": int(checkpoint["seed"]),
            "bestRound": int(checkpoint["best_round"]),
            "rounds": int(cfg["training"]["rounds"]),
            "parameters": 60159,
            "clients": int(cfg["data"]["num_clients"]),
        },
        "dataset": {
            "name": "NASA C-MAPSS FD004",
            "trainEngines": 249,
            "testEngines": 248,
            "conditions": 6,
            "faultModes": 2,
            "sequenceLength": 40,
            "sensorNames": SELECTED_SENSORS,
        },
        "clients": client_summary(cfg),
        "results": result_summary(),
        "engines": [engine_replay(model, cfg, meta, normalized, rul_end, engine_id) for engine_id in args.engines],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"Saved {args.output}")
    print(f"Engines: {args.engines}")
    print(f"Replay frames: {sum(engine['cycles'] for engine in payload['engines'])}")


if __name__ == "__main__":
    main()
