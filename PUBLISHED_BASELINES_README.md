# FedTwin-RUL Published-Baseline Patch

This patch strengthens the final FedTwin-RUL paper against the objection that
all competing models were selected internally.

It adds two externally grounded baselines and merges them with the already
completed 36-run FedTwin-RUL study.

## Added baselines

### 1. Trans-Lighter (protocol-aligned reproduction)

Source:

Nguyen Huu Du, Nguyen Hoang Long, Kieu Ngan Ha, Nguyen Viet Hoang,
Truong Thu Huong, Kim Phuc Tran.

"Trans-Lighter: A light-weight federated learning-based architecture for
Remaining Useful Lifetime prediction."

Computers in Industry, 148, 103888, 2023.

DOI:

```text
10.1016/j.compind.2023.103888
```

The published architecture is based on a lightweight Transformer encoder and a
linear RUL prediction layer inside a federated-learning system.

This patch reproduces that publicly documented architecture family under the
same protocol as FedTwin-RUL:

- same processed FD004 data
- same RUL cap
- same 10-client construction
- same alpha values
- same seeds
- same 40 rounds
- same 2 local epochs
- same validation-only checkpoint selection

It is deliberately called **protocol-aligned** rather than an exact numerical
reproduction because the original paper used its own Bayesian hyperparameter
optimization and a different four-device testbed.

This distinction must remain in the paper.

### 2. SCAFFOLD Transformer

Source:

Karimireddy et al.

"SCAFFOLD: Stochastic Controlled Averaging for Federated Learning."

ICML / PMLR 119, 2020.

This gives the paper a well-established non-IID federated optimizer baseline
that explicitly addresses client drift.

The same non-Digital-Twin Transformer RUL backbone is used so that the
comparison focuses on the federated optimization strategy.

SCAFFOLD uses its standard control-variate update with SGD and fixed local
learning rate 0.01.

## Existing baselines that are already literature-grounded

The final paper should no longer describe every prior model as if it were
invented for this project.

The existing comparison contains:

- FedAvg Transformer
- FedProx Transformer
- FedAvg-MoE
- FedTwin-RUL

FedAvg and FedProx are standard published FL algorithms. Transformer-based
FedAvg/FedProx RUL evaluation is also directly represented in the published
turbofan RUL literature, including:

S. Kamei and S. Taghipour,
"A comparison study of centralized and decentralized federated learning
approaches utilizing the transformer architecture for estimating remaining
useful life,"
Reliability Engineering & System Safety, 233, 109130, 2023.

The FedAvg-MoE row should remain labeled as an **internal capacity control**,
not a published RUL baseline.

## Why GHDR-FL and FedRUL are not falsely reproduced here

Two very relevant published methods are:

- GHDR-FL, Chen et al., Reliability Engineering & System Safety, 239, 109511,
  2023. DOI 10.1016/j.ress.2023.109511.
- FedRUL, Guo et al., IEEE/ASME Transactions on Mechatronics, 28(1), 350-359,
  2023. DOI 10.1109/TMECH.2022.3195524.

They should be discussed in Related Work and can appear in a clearly marked
reported-literature table.

However, this patch does not label an approximation as an exact reproduction.
Their published training/deployment structures do not map cleanly onto the
current fixed 10-client unseen-engine protocol without additional assumptions.

If official source code or complete implementation details are later obtained,
they can be added separately.

## Number of new runs

The new controlled study is:

```text
2 baselines
x 3 heterogeneity levels
x 3 seeds
= 18 runs
```

The existing 36 FedTwin-RUL runs are not repeated.

## Install

Extract this ZIP over the project root and replace/add files.

No preprocessing is required.

## Run

```powershell
python scripts\run_published_baselines.py --config configs\published_baselines_fd004.yaml
```

## Resume after interruption

```powershell
$study = Get-Content outputs\latest_published_baselines.txt
python scripts\run_published_baselines.py --config configs\published_baselines_fd004.yaml --resume --study-dir "$study"
```

## Integrity check

```powershell
python scripts\check_published_baselines.py
```

Expected result:

```text
Completed runs: 18 / 18
Published-baseline integrity check passed.
```

## Merge with the completed FedTwin-RUL study

After all 18 runs:

```powershell
python scripts\merge_published_baselines.py
```

The script automatically reads:

```text
outputs\latest_fedtwin_final.txt
outputs\latest_published_baselines.txt
```

and writes into the existing final FedTwin-RUL study directory:

```text
final_with_published_baselines.csv
paper_tables\final_with_published_baselines.tex
```

The final comparison will contain:

- FedAvg Transformer
- FedProx Transformer
- FedAvg-MoE
- Trans-Lighter (protocol-aligned)
- SCAFFOLD Transformer
- FedTwin-RUL

## Paper wording

Use:

> We compare FedTwin-RUL against standard federated optimization baselines,
> a higher-capacity MoE control, and a protocol-aligned reproduction of the
> published Trans-Lighter federated RUL architecture.

Do not write:

> We exactly reproduce the numerical results of Trans-Lighter.

The preprocessing, client topology, RUL capping, and training protocol differ
from the original publication. The purpose here is a controlled reimplementation
under one common evaluation protocol.

## Main reviewer-defense point

This expanded experiment means the final method is not being compared only
against internally constructed alternatives.

The paper will have:

1. standard published FL baselines,
2. a canonical non-IID FL optimizer,
3. a published task-specific federated RUL architecture,
4. an internal capacity control,
5. the proposed Digital-Twin-assisted method.

That is a much stronger anti-cherry-picking baseline design.
