from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

latest_file = ROOT / "outputs" / "latest_fedtwin_final.txt"

study_dir = Path(
    latest_file.read_text(encoding="utf-8").strip()
)

results_path = study_dir / "mean_std_results.csv"

output_dir = study_dir / "paper_plots"
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Reading: {results_path}")
print(f"Saving plots to: {output_dir}")


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(results_path)

name_map = {
    "fedavg_transformer": "FedAvg",
    "fedprox_transformer": "FedProx",
    "fedavg_moe": "FedAvg-MoE",
    "fedtwin_rul": "FedTwin-RUL",
}

df = df[df["method"].isin(name_map)].copy()

df["Method"] = df["method"].map(name_map)


# ============================================================
# PUBLICATION STYLE
# ============================================================

sns.set_theme(
    context="paper",
    style="whitegrid",
    font_scale=1.05,
)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

palette = sns.color_palette(
    "colorblind",
    n_colors=4,
)

method_order = [
    "FedAvg",
    "FedProx",
    "FedAvg-MoE",
    "FedTwin-RUL",
]

color_map = dict(
    zip(method_order, palette)
)

marker_map = {
    "FedAvg": "o",
    "FedProx": "s",
    "FedAvg-MoE": "^",
    "FedTwin-RUL": "D",
}


# ============================================================
# FIGURE 1
# TEST RMSE VS CLIENT HETEROGENEITY
# ============================================================

fig, ax = plt.subplots(
    figsize=(7.16, 3.0)
)

# We deliberately display alpha from high -> low.
# Lower alpha corresponds to stronger non-IID heterogeneity.
alpha_order = [1.0, 0.5, 0.1]
x_positions = list(range(len(alpha_order)))

for method in method_order:

    subset = (
        df[df["Method"] == method]
        .set_index("alpha")
        .loc[alpha_order]
        .reset_index()
    )

    ax.errorbar(
        x_positions,
        subset["test_rmse_mean"],
        yerr=subset["test_rmse_std"],
        label=method,
        marker=marker_map[method],
        markersize=5.5,
        linewidth=1.6,
        capsize=3,
        capthick=1.0,
        elinewidth=1.0,
        color=color_map[method],
    )


ax.set_xticks(x_positions)

ax.set_xticklabels(
    [
        r"$\alpha=1.0$",
        r"$\alpha=0.5$",
        r"$\alpha=0.1$",
    ]
)

ax.set_xlabel(
    r"Dirichlet concentration $\alpha$  (lower = stronger heterogeneity)"
)

ax.set_ylabel(
    "Test RMSE (cycles)"
)

# Do NOT put a title inside the graph.
# The LaTeX figure caption will explain it.

ax.legend(
    ncol=4,
    frameon=True,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.04),
)

ax.grid(
    axis="y",
    alpha=0.25,
)

ax.grid(
    axis="x",
    visible=False,
)

sns.despine(
    ax=ax,
)

fig.tight_layout(
    pad=0.5
)

fig.savefig(
    output_dir / "rmse_vs_heterogeneity.pdf",
    bbox_inches="tight",
)

fig.savefig(
    output_dir / "rmse_vs_heterogeneity.png",
    bbox_inches="tight",
    dpi=600,
)

plt.close(fig)


# ============================================================
# FIGURE 2
# ACCURACY-COMPLEXITY TRADE-OFF
# ============================================================

# Average the test RMSE across the three alpha settings.
tradeoff = (
    df.groupby("Method", as_index=False)
    .agg(
        Test_RMSE=("test_rmse_mean", "mean"),
        Parameters=("parameters_mean", "mean"),
        Training_Time=("train_seconds_mean", "mean"),
    )
)

tradeoff["Parameters_k"] = (
    tradeoff["Parameters"] / 1000.0
)


fig, ax = plt.subplots(
    figsize=(3.45, 2.85)
)

for method in method_order:

    row = tradeoff[
        tradeoff["Method"] == method
    ].iloc[0]

    ax.scatter(
        row["Parameters_k"],
        row["Test_RMSE"],
        s=70 if method == "FedTwin-RUL" else 50,
        marker=marker_map[method],
        color=color_map[method],
        edgecolor="black",
        linewidth=0.55,
        zorder=3,
    )


# Manual annotation offsets keep labels readable.
annotation_offsets = {
    "FedAvg": (5, -12),
    "FedProx": (5, 7),
    "FedAvg-MoE": (-62, 7),
    "FedTwin-RUL": (6, -3),
}

for _, row in tradeoff.iterrows():

    method = row["Method"]

    dx, dy = annotation_offsets[method]

    ax.annotate(
        method,
        (
            row["Parameters_k"],
            row["Test_RMSE"],
        ),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=7.5,
        fontweight=(
            "bold"
            if method == "FedTwin-RUL"
            else "normal"
        ),
    )


ax.set_xlabel(
    "Trainable parameters (thousands)"
)

ax.set_ylabel(
    "Mean test RMSE (cycles)"
)

ax.grid(
    alpha=0.25,
)

sns.despine(
    ax=ax,
)

fig.tight_layout(
    pad=0.5
)

fig.savefig(
    output_dir / "accuracy_complexity_tradeoff.pdf",
    bbox_inches="tight",
)

fig.savefig(
    output_dir / "accuracy_complexity_tradeoff.png",
    bbox_inches="tight",
    dpi=600,
)

plt.close(fig)


# ============================================================
# PRINT DATA USED IN FIGURES
# ============================================================

print("\nGenerated:")
print("  rmse_vs_heterogeneity.pdf")
print("  rmse_vs_heterogeneity.png")
print("  accuracy_complexity_tradeoff.pdf")
print("  accuracy_complexity_tradeoff.png")

print("\nAccuracy-complexity data:")
print(
    tradeoff[
        [
            "Method",
            "Parameters",
            "Test_RMSE",
            "Training_Time",
        ]
    ].to_string(index=False)
)