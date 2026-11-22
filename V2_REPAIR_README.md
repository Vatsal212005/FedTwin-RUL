# TwinFed-MoE v2 Repair Patch

This patch is intentionally separate from the completed v1 benchmark suite.

It fixes the structural personalization problem without overwriting the
existing baseline results.

## Structural change

The old prediction path was effectively:

```text
shared representation
    -> local adapter
    -> RUL regressor
```

The official test set contains unseen engines, so the local adapter had to be
bypassed even though the regressor was trained downstream of that adapter.

V2 instead uses:

```text
shared representation
    -> directly supervised shared RUL head -> global prediction

shared representation
    -> client-local adapter
    -> bounded residual correction
    -> personalized prediction
```

Mathematically:

```text
y_global = sigmoid(f_global(h))
delta_k  = s * tanh(f_local,k(A_k(h)))
y_local  = clip(y_global + delta_k, 0, 1)
```

The shared path is always directly supervised.

The local residual head is initialized to exactly zero, so personalization
starts from the global predictor rather than perturbing it randomly.

## Dual RUL objective

For personalized variants:

```text
L_RUL =
    (L_shared + lambda_personal * L_personal)
    / (1 + lambda_personal)
```

The default `lambda_personal` is 1.0.

For the no-personalization ablation:

```text
L_RUL = L_shared
```

## Residual bound

The local correction scale is frozen at:

```text
personal_residual_scale: 0.20
```

With RUL cap 125, this means the correction cannot exceed +/-25 cycles.

Do not change this after looking at FD004 test results.

## Model selection rule

V2 checkpoints are selected using:

```text
shared_pooled_rmse
```

on the held-out federated validation engines.

This is deliberate: the official NASA test contains unseen engines and uses
the shared global prediction, so checkpoint selection uses the matching
validation pathway.

The official test is never used to select the round.

## V2 experiment matrix

The patch runs only methods affected by the repair:

1. `fedper_residual_v2`
2. `twinfed_moe_v2`
3. `v2_no_dt`
4. `v2_no_moe`
5. `v2_no_personalization`
6. `v2_no_tgc`
7. `v2_fedavg_aggregation`

The completed centralized, local-only, FedAvg, FedProx, and FedAvg-MoE
results are reused from the existing benchmark suite.

## Run

No preprocessing is required.

```powershell
python scripts\run_v2_repair_suite.py --config configs\v2_fd004.yaml
```

The runner automatically reads the previous fixed benchmark from:

```text
outputs\latest_benchmark.txt
```

and creates:

```text
outputs\
└── v2_repair_FD004_YYYYMMDD_HHMMSS\
    ├── v2_results.csv
    ├── v2_vs_fixed_baselines.csv
    ├── suite_summary.json
    ├── fedper_residual_v2\
    ├── twinfed_moe_v2\
    ├── ...
    └── v2_fedavg_aggregation\
```

Each method directory contains:

```text
result.json
best_model.pt
round_metrics.csv
official_test_predictions.csv
```

## Resume

```powershell
python scripts\run_v2_repair_suite.py --resume --suite-dir "outputs\v2_repair_FD004_YYYYMMDD_HHMMSS"
```

Completed methods are skipped.

## Quick two-method check

If you want to verify the repair before running all seven:

```powershell
python scripts\run_v2_repair_suite.py --methods fedper_residual_v2 twinfed_moe_v2
```

## Summarize

```powershell
python scripts\summarize_v2.py
```

## Integrity check

```powershell
python scripts\check_v2.py
```

## Plots

```powershell
python scripts\plot_v2.py
```

## What to inspect

The key outputs are:

- `shared_val_rmse`
- `personal_val_rmse`
- `personalization_gain_rmse`
- `residual_abs_mean_cycles`
- `residual_abs_p95_cycles`
- `test_rmse`
- `test_mae`
- `test_r2`
- `test_nasa`

A positive personalization gain means the local residual improves known-client
validation while leaving the directly supervised global predictor intact for
unseen engines.
