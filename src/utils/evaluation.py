import numpy as np
import textwrap
from typing import List
import seaborn as sns
from utils.plotting import sns_colors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.axes as Axes
from typing import Literal


def barplot__cv_performance_comparison_of_pipelines(
    values: np.ndarray,
    pipeline_labels: List[str],
    pipeline_colors=None,
    metric: Literal["AUC 2yr", "AUC 3yr", "iAUC"] = "AUC 2yr",
    title="Cross validation performance comparison of gene selection pipelines",
    tick_labels_left_plot=None,
    tick_rotation=45,
    caption: str = None,
    plot_mean_of_benchmarks=True,
    axs=None,
    fig=None,
    verbose=0,
):
    """Row i of ``values`` corresponds to ``pipeline i``, and column j to
    ``CV-iteration j``. Row 1 corresponds to the ``candidate model``, for
    instance ``adaptive``."""
    # ---------- BARPLOT: AVERAGE PERFORMANCE FOR SELECTED METRIC -----------------------------------------------------------------------------------------
    if pipeline_colors is None:
        pipeline_colors = ["purple", "yellow", "orange", "red", "brown"][: len(pipeline_labels)]
        pipeline_colors = [sns_colors[c] for c in pipeline_colors]
    name_candidate_pipeline = pipeline_labels[0]
    n_folds = values.shape[1]
    df = pd.DataFrame(values, columns=[f"fold {i}" for i in range(n_folds)], index=pipeline_labels)
    long_df = df.reset_index(names="pipeline").melt(value_vars=[f"fold {i}" for i in range(n_folds)], id_vars="pipeline", var_name="fold")   # fmt: skip
    ylim = [min(0.5, long_df["value"].min()) * 0.95, long_df["value"].max() * 1.15]
    df.index = pipeline_labels
    mean_metric_vals = df.mean(axis=1).reset_index()
    mean_metric_vals.columns = ["Variant filter", "mean value"]
    display(long_df) if verbose > 0 else None

    if axs is None:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    else:
        assert fig is not None
    # --- PLOT AVERAGE PERFORMANCE
    ax = axs[0]
    ax.set_axisbelow(True)
    sns.barplot(long_df, x="pipeline", y="value", hue="pipeline", palette=pipeline_colors, edgecolor="k", width=1, ax=ax, capsize=0.2, errorbar=("pi", 100), legend=False)   # fmt: skip
    if tick_labels_left_plot is not None:
        ax.set_xticks(range(len(tick_labels_left_plot)))
        ax.set_xticklabels(tick_labels_left_plot)
    wrapped = [textwrap.fill(l.get_text(), width=12) for l in ax.get_xticklabels()]
    ax.set_xticks(range(len(wrapped)))
    ax.set_xticklabels(wrapped)
    ax.axhline(0.5, linestyle="--", c="k", alpha=0.3)
    ax.set_ylim(ylim)
    ax.set_xlabel("", fontsize=8)
    ax.set_ylabel(metric, fontsize=8)
    ax.set_title(f"Average performance of each pipeline")
    ax.tick_params(axis="x", rotation=tick_rotation)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.3, linestyle="--", axis="y")

    # --- PLOT PERFORMANCE BY FOLD
    ax: Axes = axs[1]
    ax.set_axisbelow(True)
    # ENSURE PREDICTABLE ORDERING OF PATCHES
    hue_order = pipeline_labels
    sns.barplot(long_df, x="fold", hue="pipeline", y="value", palette=pipeline_colors, edgecolor="k", hue_order=hue_order, ax=ax, capsize=0.15)   # fmt: skip
    ax.set_xlabel("")
    ax.set_xticks(range(n_folds))
    ax.set_xticklabels([f"fold {x}" for x in np.arange(1, n_folds + 1)])
    ax.set_ylabel("", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.set_title(f"Performance by fold and pipeline")
    ax.set_ylabel(metric)
    ax.set_ylim(ylim)
    ax.axhline(0.5, linestyle="--", c="k", alpha=0.3)
    for text in ax.get_legend().get_texts():
        text.set_text(textwrap.fill(text.get_text(), width=12))
    if caption is not None:
        ax.text(0.0, -0.15, caption, ha="left", va="top", transform=ax.transAxes, alpha=0.5, fontsize=8)   # fmt: skip

    # SORT ROWS SO THAT ORDER MATCHES THAT OF SEABORN PLOT
    if plot_mean_of_benchmarks:
        long_df["pipeline"] = pd.Categorical(long_df["pipeline"], categories=hue_order, ordered=True)   # fmt: skip
        long_df = long_df.sort_values(["pipeline", "fold"])
        long_df["x"] = np.nan
        long_df["y"] = np.nan
        for patch, (i, row) in zip(ax.patches, long_df.iterrows()):
            if patch.get_width() > 0:
                w = patch.get_width()
                long_df.loc[i, "x"] = patch.get_x()

        long_df["pipeline type"] = (long_df["pipeline"] == name_candidate_pipeline).map({True: name_candidate_pipeline, False: "benchmarks"})   # fmt: skip
        long_df["mean of pipeline type"] = long_df.groupby(["fold", "pipeline type"])["value"].transform("mean")   # fmt: skip

        for i in range(5):
            fold_i_df = long_df.query(f"fold=='fold {i}' & pipeline!='{name_candidate_pipeline}'")
            y = fold_i_df["value"].mean()
            x1, x2 = fold_i_df.iloc[[0, -1]]["x"]
            label = "benchmark average" if i == 0 else None
            ax.plot([x1, x2 + w], [y, y], c="k", linestyle="--", linewidth=2, label=label)

    # --- LEGEND
    ax.legend(fontsize=8, framealpha=0.5)
    # REMOVE NEWLINES
    text = ax.legend().get_texts()
    for text in ax.legend().get_texts():
        text.set_text(text.get_text().replace("\n", " "))
    # GRID
    ax.grid(alpha=0.3, linestyle="--", axis="y")

    fig.suptitle(title, fontsize=15)   # fmt: skip
