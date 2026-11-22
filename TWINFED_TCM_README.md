# TwinFed-TCM Patch

This patch is the clean successor to the TwinFed-MoE experiments.

It does not overwrite the completed MoE/v2 benchmark outputs. It creates a new
TCM experiment suite and reuses the already-completed fixed baselines.

No C-MAPSS preprocessing is required again.

## Core idea

The model computes:

```text
h = TemporalTransformer(X)
z = LightweightDigitalTwin(X, operating state)
```

Twin-Conditioned Modulation then generates a feature-wise affine transform:

```text
gamma(z), beta(z) = Conditioner(z)

h_tcm = gamma(z) * LayerNorm(h) + beta(z)
```

The RUL predictor operates on `h_tcm`.

The modulation block is initialized close to the identity transform:

```text
gamma ~= 1
beta  ~= 0
```

so training starts near the plain Transformer solution instead of forcing
arbitrary context modulation from round 1.

## Why this suite exists

The previous TwinFed-MoE study showed that model complexity alone did not
produce a reliable benefit. This suite isolates one specific mechanism:

> Does a learned Digital-Twin state improve federated RUL prediction when it
> conditions the interpretation of temporal features?

The suite therefore compares TCM against both simple feature concatenation and
a no-conditioning control.

## Methods

### 1. `dt_concat_fedavg`

Simple Digital-Twin fusion baseline:

```text
Transformer feature || Twin state
          -> fusion MLP
          -> RUL head
```

Uses FedAvg.

### 2. `tcm_fedavg`

Twin-Conditioned Modulation with FedAvg.

This tests whether TCM is better than simple concatenation under the same
federated optimizer.

### 3. `tcm_fedprox`

Twin-Conditioned Modulation with FedProx.

This tests whether proximal regularization helps under the existing
non-IID client partition.

### 4. `tcm_no_dt_reconstruction`

Same TCM + FedProx architecture, but removes the Digital-Twin reconstruction
auxiliary objective.

This isolates whether the auxiliary twin-state learning signal contributes.

### 5. `tcm_identity_modulation`

The Digital-Twin branch and reconstruction objective remain, but the Twin
state is not allowed to condition the Transformer representation.

This is the key mechanism control:

```text
Digital Twin auxiliary branch exists
but
prediction = plain normalized Transformer feature
```

If TCM beats this control on validation, the gain cannot be attributed merely
to adding an auxiliary Digital-Twin encoder.

## Fixed baselines reused

The runner automatically reads:

```text
outputs\latest_benchmark.txt
```

and merges the previously completed:

- Centralized Transformer
- Local-only Transformer
- FedAvg Transformer
- FedProx Transformer
- FedAvg-MoE

with the new TCM experiments.

No need to rerun those methods.

## Training protocol

The patch preserves the frozen FD004 setup:

- 14 selected sensors
- sequence length 40
- RUL cap 125
- stride-5 processed training windows
- 10 clients
- same regime-aware client partition
- same engine-wise validation split
- non-IID alpha 0.5
- batch size 256
- 40 federated rounds
- 2 local epochs
- validation-only checkpoint selection

## Main run

```powershell
python scripts\run_tcm_suite.py --config configs\tcm_fd004.yaml
```

It creates:

```text
outputs\
└── tcm_FD004_YYYYMMDD_HHMMSS\
    ├── tcm_results.csv
    ├── tcm_vs_fixed_baselines.csv
    ├── suite_summary.json
    ├── method_definitions.json
    ├── dt_concat_fedavg\
    ├── tcm_fedavg\
    ├── tcm_fedprox\
    ├── tcm_no_dt_reconstruction\
    └── tcm_identity_modulation\
```

Each method directory stores:

```text
best_model.pt
result.json
round_metrics.csv
official_test_predictions.csv
regime_diagnostics.csv
```

## Resume an interrupted suite

```powershell
python scripts\run_tcm_suite.py --resume --suite-dir "outputs\tcm_FD004_YYYYMMDD_HHMMSS"
```

Completed methods are skipped.

## Run a subset

Example:

```powershell
python scripts\run_tcm_suite.py --methods dt_concat_fedavg tcm_fedavg tcm_fedprox
```

## Summarize

```powershell
python scripts\summarize_tcm.py
```

This prints:

- validation ranking
- official-test ranking
- TCM vs concatenation delta
- TCM vs identity-control delta
- FedAvg vs FedProx under the same TCM architecture
- effect of removing DT reconstruction
- ranking against the existing fixed baselines

## Integrity check

```powershell
python scripts\check_tcm.py
```

## Generate figures

```powershell
python scripts\plot_tcm.py
```

The script generates:

- validation vs test RMSE comparison
- magnitude of learned gamma modulation
- TCM convergence curve
- regime-wise RMSE

## Export paper tables

```powershell
python scripts\export_tcm_tables.py
```

This creates:

```text
paper_tables\
    tcm_main_comparison.csv
    tcm_main_comparison.tex
```

## Diagnostic outputs

For TCM methods the official-test prediction CSV records:

- engine ID
- operating regime
- true RUL
- predicted RUL
- signed error
- absolute error
- mean absolute gamma shift from identity
- mean absolute beta modulation
- Twin-state norm

`regime_diagnostics.csv` summarizes prediction quality and modulation strength
for every FD004 operating regime.

## Decision rule

The mechanism is worth carrying into the paper only if validation supports it.

The key comparisons are:

```text
TCM + FedAvg
vs
DT concatenation + FedAvg
```

and:

```text
TCM + FedProx
vs
identity/no-conditioning control
```

A useful final architecture should show a repeatable validation advantage over
both simple concatenation and the no-conditioning control.

Because the FD004 official test set has already been inspected during earlier
development, final publication claims should later be confirmed with multiple
seeds and preferably additional C-MAPSS subsets or a second prognostics
dataset.
