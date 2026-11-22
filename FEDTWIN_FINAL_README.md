# FedTwin-RUL Final Experiment Patch

This is the final paper experiment harness.

It freezes the candidate architecture as:

**FedTwin-RUL = lightweight Digital-Twin state + temporal Transformer + FedAvg**

The Digital-Twin state is fused by simple concatenation because the previous
controlled experiments showed that the more complex TCM and MoE mechanisms
did not provide a reliable validation advantage.

## Research question

Does a lightweight operational Digital-Twin state improve federated RUL
prediction as IIoT client heterogeneity increases?

## Final comparison

Four methods are run:

1. FedAvg Transformer
2. FedProx Transformer
3. FedAvg-MoE
4. FedTwin-RUL

Across:

- alpha = 1.0
- alpha = 0.5
- alpha = 0.1

and seeds:

- 42
- 123
- 2027

Total:

```text
4 methods x 3 heterogeneity levels x 3 seeds = 36 runs
```

## Run

No preprocessing is required.

```powershell
python scripts\run_fedtwin_final_study.py --config configs\fedtwin_final.yaml
```

## Resume

```powershell
python scripts\run_fedtwin_final_study.py --resume --study-dir "outputs\fedtwin_final_FD004_YYYYMMDD_HHMMSS"
```

## Quick smoke test

Before the full 36-run study, you can run:

```powershell
python scripts\run_fedtwin_final_study.py --methods fedavg_transformer fedtwin_rul --alphas 0.5 --seeds 42
```

## Final outputs

The study directory contains:

```text
raw_results.csv
mean_std_results.csv
study_summary.json
runs\
```

Each individual run contains:

```text
result.json
best_model.pt
round_metrics.csv
```

## After completion

```powershell
python scripts\check_fedtwin_final.py
python scripts\summarize_fedtwin_final.py
python scripts\export_fedtwin_tables.py
```

The most important file is:

```text
mean_std_results.csv
```

That will be the basis for the paper's heterogeneity table.

## Interpretation rule

Do not tune the model against the official FD004 test set.

The principal development criterion is validation RMSE.

The final paper should report:

- mean ± std over seeds
- performance versus alpha
- mean-client RMSE
- worst-client RMSE
- official-test RMSE/MAE/R2/NASA
- parameter count
- runtime

The paper contribution is the lightweight Digital-Twin-assisted federated
IIoT prognostics framework and the controlled heterogeneity study, not the
claim that concatenation itself is novel.
