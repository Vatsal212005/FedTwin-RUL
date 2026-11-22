# TwinFed-MoE Benchmark + Ablation Suite

This is a drop-in patch for the current TwinFed-MoE project.

Extract it over the project root and choose **Replace existing files**.

It does not change the existing processed FD004 dataset, and you do not need
to preprocess again.

## Purpose

This patch freezes the current FD004 preprocessing and comparison protocol,
then executes all baselines and ablations under the same:

- train/validation engine split
- 10-client partition
- non-IID alpha
- sensor set
- sequence length
- RUL cap
- optimizer
- GPU precision mode
- model-selection rule

The official NASA test split is used only after validation-based model
selection for each method.

## Full experiment matrix

### References

1. `centralized_transformer`
   - non-federated Transformer reference
   - trained on the union of all client training engines
   - model selected by the pooled validation set

2. `local_only_transformer`
   - ten independent client models
   - no information exchange
   - validation metrics only
   - official test metrics intentionally left blank because an unseen engine
     has no uniquely defined local model

### Federated baselines

3. `fedavg_transformer`
   - Transformer + FedAvg

4. `fedprox_transformer`
   - Transformer + FedProx

5. `fedper_transformer`
   - FedPer-style personalized FL baseline using the same local adapter
     mechanism for a controlled comparison

6. `fedavg_moe`
   - temporal sparse MoE
   - Top-2 routing
   - no Digital Twin
   - ordinary FedAvg

### Proposed

7. `twinfed_moe`
   - Digital Twin state
   - temporal Transformer
   - Top-2 sparse MoE
   - DT-aware routing
   - local personalization
   - TGC loss
   - expert-utilization-aware aggregation

### Ablations

8. `ablation_no_dt`
   - removes Digital Twin state
   - temporal-only MoE router remains

9. `ablation_no_moe`
   - removes the expert bank and sparse routing

10. `ablation_no_personalization`
    - removes the local adapter

11. `ablation_no_tgc`
    - sets TGC weight to zero

12. `ablation_fedavg_aggregation`
    - keeps the full architecture
    - replaces expert-aware aggregation with ordinary FedAvg

## Runtime

The full proposed method previously took about 2 minutes on the RTX 4060
Laptop GPU. Plain Transformer baselines are cheaper.

The complete 12-method suite should therefore be practical on the same
machine, but the actual time depends on early stopping and CUDA behavior.

The runner writes a `result.json` after every completed method, so a crash or
manual interruption does not invalidate completed work.

## Run the full suite

```powershell
python scripts\run_benchmark_suite.py --config configs\benchmark_fd004.yaml
```

The runner creates:

```text
outputs\
└── benchmark_FD004_YYYYMMDD_HHMMSS\
    ├── benchmark_results.csv
    ├── suite_summary.json
    ├── method_definitions.json
    ├── centralized_transformer\
    ├── local_only_transformer\
    ├── fedavg_transformer\
    ├── ...
    └── ablation_fedavg_aggregation\
```

`outputs\latest_benchmark.txt` points to the newest suite.

## Run only a subset

Example:

```powershell
python scripts\run_benchmark_suite.py --methods fedavg_transformer fedprox_transformer twinfed_moe
```

This is useful for a quick sanity check.

## Resume an interrupted suite

Use the exact benchmark directory printed by the first run:

```powershell
python scripts\run_benchmark_suite.py --resume --suite-dir "outputs\benchmark_FD004_YYYYMMDD_HHMMSS"
```

Methods already containing `result.json` are skipped.

## Run a different seed

```powershell
python scripts\run_benchmark_suite.py --seed 123
```

Do not overwrite an existing benchmark directory when changing seed.

For the paper, run seeds 42, 123, and 2027 after the single-seed experiment
matrix is verified.

## Summarize rankings and ablation deltas

```powershell
python scripts\summarize_benchmark.py
```

It prints:

- validation ranking
- official FD004 ranking
- proposed-method metrics
- each ablation's RMSE delta from the full method
- runtime by method

## Integrity check

```powershell
python scripts\check_benchmark.py
```

This verifies:

- all 12 requested methods completed
- no duplicate methods
- required metrics are finite
- official test metrics exist for every method where they are meaningful

## Generate comparison figures

```powershell
python scripts\plot_benchmark.py
```

This produces:

```text
figures\
    main_test_rmse.png
    main_test_mae.png
    main_test_nasa.png
    ablation_test_rmse.png
    ablation_validation_rmse.png
    runtime_comparison.png
```

## Export paper-ready tables

```powershell
python scripts\export_paper_tables.py
```

This creates:

```text
paper_tables\
    main_comparison.csv
    ablation_comparison.csv
    main_comparison.tex
    ablation_comparison.tex
```

The `.tex` files can later be inserted into the Springer manuscript.

## Experimental interpretation

Do not decide whether to keep/remove a component based on its official FD004
test result. The ablation matrix is an analysis of the already frozen method,
not a new hyperparameter search.

The principal comparison metric should be official-test RMSE, supported by:

- MAE
- R²
- NASA asymmetric score
- validation mean-client RMSE
- validation worst-client RMSE
- runtime

The centralized model is a non-federated reference rather than a federated
competitor. The local-only method has no official unseen-engine test score by
design.
