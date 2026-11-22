# TwinFed-MoE Experiment Infrastructure Patch

This patch is designed to be copied directly over the current TwinFed-MoE
project. Choose **Replace existing files** when Windows prompts.

It does **not** modify or delete:

- `data/raw/cmapss/`
- `data/processed/FD004_train.npz`
- `data/processed/FD004_test.npz`
- existing output folders

No C-MAPSS preprocessing is required again.

## What this patch adds

### 1. Proper best-model checkpointing

Each run now creates a timestamped directory:

```text
outputs/
└── TwinFedMoE_FD004_YYYYMMDD_HHMMSS/
    ├── best_model.pt
    ├── last_model.pt
    ├── round_metrics.csv
    ├── run_summary.json
    └── config_snapshot.yaml
```

`best_model.pt` is selected **only using validation pooled RMSE**.

The official NASA test set is not used for model selection.

`last_model.pt` is updated every round as a crash-safe recovery checkpoint.

### 2. Reproducible client personalization

Each checkpoint stores:

- global/shared model parameters
- each client's personalized adapter
- train-engine IDs per client
- validation-engine IDs per client
- selected-round validation metrics
- the complete experiment configuration

### 3. Correct pooled validation

The runner now reports both:

- pooled metrics over all validation windows
- mean-client metrics
- worst-client RMSE

Pooled metrics are the main validation criterion.

### 4. Personalized vs shared-only validation

Every round evaluates:

```text
Personalized:
shared TwinFed-MoE + each client's local adapter

Shared-only:
shared TwinFed-MoE with adapter bypassed
```

The CSV includes:

```text
personalization_gain_rmse
```

defined as:

```text
shared-only pooled RMSE - personalized pooled RMSE
```

Positive values mean personalization helps.

### 5. Expert-routing diagnostics

Each round records:

- weighted utilization of every expert
- expert-utilization entropy

This lets the paper show whether MoE routing collapses onto one expert.

### 6. Official NASA FD004 evaluation

The official test evaluator loads the **best validation checkpoint** and
evaluates the 248 unseen FD004 test engines.

Because those engines were never federated clients, they do not possess a
trained client-local adapter. The default protocol therefore bypasses the
personal adapter and evaluates the learned global TwinFed-MoE representation.

The evaluator produces:

```text
official_test_metrics.json
official_test_predictions.csv
regime_expert_summary.csv
```

The prediction CSV includes per-engine:

- true RUL
- predicted RUL
- signed error
- absolute error
- operating regime
- selected/top expert
- dense router probabilities
- sparse Top-k router probabilities

### 7. Regime-wise expert specialization

`regime_expert_summary.csv` reports for each operating regime:

- count
- RMSE
- MAE
- R²
- NASA score
- utilization of every expert

This is directly useful for the Digital-Twin-guided MoE claim.

### 8. Paper-oriented plots

The plotting script generates:

- personalized vs shared validation RMSE
- personalized vs shared validation MAE
- personalized vs shared validation R²
- training loss
- expert utilization over communication rounds
- official predicted-vs-true RUL
- official absolute-error CDF
- regime-wise stacked expert utilization

## Run sequence

### Step 1: train a fresh checkpointed TwinFed-MoE run

```powershell
python scripts\run_federated.py --config configs\fd004.yaml
```

At the end it prints:

```text
Best round: ...
Best checkpoint: outputs\TwinFedMoE_FD004_...\best_model.pt
```

### Step 2: inspect the run

```powershell
python scripts\summarize_run.py
```

The script automatically reads:

```text
outputs\latest_run.txt
```

### Step 3: evaluate once on the official NASA test set

```powershell
python scripts\evaluate_official.py
```

This automatically evaluates `best_model.pt` from the latest run.

### Step 4: generate paper figures

```powershell
python scripts\plot_results.py
```

Figures are saved under:

```text
outputs\TwinFedMoE_FD004_...\figures\
```

## Important experimental rule

Do not repeatedly adjust the architecture based on official FD004 test results.

Use the federated validation split for:

- model selection
- training-round selection
- hyperparameter tuning

Use the official 248-engine FD004 test split for the final held-out evaluation.
