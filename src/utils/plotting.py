from utils.prelude import *
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import multivariate_logrank_test
from utils.prelude import *
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
# from utils.misc import stop 
from matplotlib_venn import venn3
from typing import List

sns_colors = {
    "blue": "#4C72B0",
    "orange": "#DD8452",
    "green": "#55A868",
    "red": "#C44E52",
    "dark_red": "#8C2D2D",
    "purple": "#8172B3",
    "magenta": "#8172B3",
    "brown": "#937860",
    "pink": "#DA8BC3",
    "grey": "#8C8C8C",
    "yellow": "#CCB974",
    "dark_orange": "#C76E00",
    "dark_brown": "#6B4C3B",
    "teal": "#64B5CD",
    "dark_green": "#3B7A57",
    "dark_blue": "#355C7D",
    "gold": "#E0B43B",
    "olive": "#8E9A48",
}


def get_risk_subsets(
    df,
    cutoffs=[1 / 3, 2 / 3],
    score_name="score",
    cutoff_type: Literal["percentile", "score"] = "percentile",
    method: Literal["manual", "pd.cut"] = "manual",
):
    """Splits dataframe into risk subsets corresponding to the risk cutoffs.
    Uses right-inclusive intervals: (-inf, c1], (c1, c2], ..., (c_n, inf).

    The ``manual`` method handles zero-inflated scores correctly (many tied
    values at a bin edge). The ``pd.cut`` method delegates to pandas but will
    raise if bin edges are not unique (e.g. when >46% of scores are 0)."""
    if cutoff_type == "percentile":
        score_cutoffs = [df[score_name].quantile(c) for c in cutoffs]
    else:
        score_cutoffs = cutoffs

    if method == "pd.cut":
        bins = [-np.inf] + score_cutoffs + [np.inf]
        labels = [f"group_{i}" for i in range(len(cutoffs) + 1)]
        group_col = pd.cut(df[score_name], bins=bins, labels=labels, right=True)
        return [df[group_col == label] for label in labels]

    n_groups = len(score_cutoffs) + 1
    risk_subsets = []
    for i in range(n_groups):
        if i == 0:
            risk_subset = df.query(f"{score_name}<={score_cutoffs[i]}")
        elif i == n_groups - 1:
            risk_subset = df.query(f"{score_name}>{score_cutoffs[i-1]}")
        else:
            risk_subset = df.query(f"{score_name}>{score_cutoffs[i-1]} and {score_name}<={score_cutoffs[i]}")
        risk_subsets.append(risk_subset)
    return risk_subsets


def plot_kaplan_meier_curves(
    modelfit_df,
    cutoffs=[1 / 3, 2 / 3],
    cutoff_type: Literal["percentile", "score"] = "percentile",
    score_name="score",
    risk_set_labels=["low risk", "medium risk", "high risk", "insane risk", "ultra risk"],
    risk_colors=["#3d525a", "#16a0d3", "#d98c3f", "darkred", "magenta"],
    ci_alpha=0.2,
    ax: Axes = None,
    fig: Figure = None,
    n_years=3,
    ylim=[0, 1],
    caption_type: Literal["detailed", "publication"] = "detailed",
    show_at_risk_labels: bool = True,
):
    if ax is None or fig is None:
        fig, ax = plt.subplots()
    risk_subsets = get_risk_subsets(modelfit_df, cutoffs, score_name=score_name, cutoff_type=cutoff_type)
    if len(risk_subsets) == 2:
        risk_set_labels = ["low risk", "high risk"]
        risk_colors = ["#3d525a", "#d98c3f"]

    km_fitters = []
    for df, label, color in zip(risk_subsets, risk_set_labels, risk_colors):
        kmf = KaplanMeierFitter()
        kmf.fit(df["time_to_event"] / 30.44, df["event_observed"], label=label)
        kmf.plot(ax=ax, c=color, ci_alpha=ci_alpha, show_censors=True, censor_styles={"ms": 4}, linewidth=3)
        km_fitters.append(kmf)
    # CALCULATE LOG RANK P-VALUE
    durations = []
    events = []
    groups = []
    for i, df in enumerate(risk_subsets):
        # Use extend to add iterable to list, like ["a"] += [1,2,3] -> ["a", 1, 2, 3], but for any iterable
        durations.extend(df["time_to_event"] / 30.44)
        events.extend(df["event_observed"])
        groups.extend([i] * len(df))  # this variable carries the group index

    result = multivariate_logrank_test(event_durations=durations, groups=groups, event_observed=events)

    ax.set_xticks(np.arange(0, 48, 6))
    ax.set_xlim([0, n_years * 12])
    ax.set_ylim(ylim)
    ax.grid(visible=True, which="major", axis="both", color="#d0d0d0", linestyle="--", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")

    at_risk_kwargs = {"xticks": np.arange(0, n_years * 12 + 1, 6)}
    if not show_at_risk_labels:
        at_risk_kwargs["labels"] = [""] * len(km_fitters)
    add_at_risk_counts(*km_fitters, ax=ax, rows_to_show=["At risk"], **at_risk_kwargs)
    ax.legend()
    ax.set_title("Kaplan meier curves stratified by risk-score")
    ax.set_xlabel("Time, months")
    ax.set_ylabel("Proportion alive")
    if caption_type == "detailed":
        ax.text(0.0, -0.15, f"Split percentiles: {', '.join([f'{x:.1%}' for x in cutoffs])}\nlog-rank p-value: {result.p_value:.4f}", ha="left", va="top", transform=ax.transAxes, alpha=0.5)   # fmt: skip
    else:
        p_txt = f"p={result.p_value:.3f}" if result.p_value >= 0.001 else "p<0.001"
        ax.text(0.05, 0.1, f"Log-rank {p_txt}", ha="left", va="top", transform=ax.transAxes)   # fmt: skip


def kaplan_meier_combined(
    modelfit_df,
    red_id_subsets: List[List[str]],
    risk_score_col="pooled_muta_feature",
    subset_names: List[List[str]] = ["Subset A", "Subset B"],
    risk_quantile_cutoff=0.5,
    stratification_varible="Stratification Var Name",
    fig=None,
    ax=None,
):
    assert "RED_ID" in modelfit_df.columns
    if ax is None or fig is None:
        fig, ax = plt.subplots(figsize=(7, 7.5))
    kmf = KaplanMeierFitter()

    # Define subsets: (label, RED_IDs, color)
    subsets = [(subset_names[0], red_id_subsets[0], "b"), (subset_names[1], red_id_subsets[1], "r")]

    # MARKERS TO VISUALLY SEPARATE THE HIGH AND LOW SCORE CLASSES
    markers = {"low": "o", "high": "^"}
    linestyles = {"low": "-", "high": ":"}

    for subset_label, red_ids, color in subsets:
        subset_df = modelfit_df.query(f"RED_ID in {red_ids}")
        cutoff = subset_df[risk_score_col].quantile(risk_quantile_cutoff)
        groups = [
            ("low", subset_df[subset_df[risk_score_col] <= cutoff]),
            ("high", subset_df[subset_df[risk_score_col] > cutoff]),
        ]
        for risk_label, group_df in groups:
            label = f"{subset_label} / {risk_label} risk"
            kmf.fit(group_df["time_to_event"], group_df["event_observed"], label=label)
            kmf.plot(
                ax=ax,
                color=color,
                linestyle=linestyles[risk_label],
                ci_show=False,
                linewidth=2.5,
                marker=markers[risk_label],
                markevery=0.005,  # add marker every ~10% of points
                markeredgecolor="k",
                markersize=6,
                alpha=0.8,
            )

    ax.set_title(f"Kaplan-Meier curves stratified by {stratification_varible} and risk scores")   # fmt: skip
    ax.set_xlabel("days")
    ax.set_ylabel("probability no event")
    ax.tick_params(labelsize=8)
    ax.set_ylim([0, 1.05])


def plot_intergene_correlations(mutations_df: pd.DataFrame):
    """Runs an analysis that studies the correlations between the genomic data."""
    # KEEP ONLY VAF COLUMNS
    mutations_df = mutations_df[[x for x in mutations_df.columns if x not in ["RED_ID", "time_to_event", "event_observed"]]].corr()   # fmt: skip
    mutations_df.columns = [x.split("_MUT")[0] for x in mutations_df.columns]
    genes = mutations_df.columns

    ticklabel_fontsize = 12
    fig, ax = plt.subplots(figsize=(10, 10))
    im0 = ax.imshow(mutations_df, cmap="magma")
    ax.tick_params(labelsize=6)
    ax.set_title("Correlation of mutation profiles across genes selected for external validation", fontsize=15)
    ax.set_xticklabels(genes)
    ax.set_yticklabels(genes)
    ax.set_xticks(range(len(genes)))
    ax.set_yticks(range(len(genes)))
    ax.tick_params(labelsize=ticklabel_fontsize)
    ax.tick_params(axis="x", rotation=90)
    fig.colorbar(im0, ax=ax, orientation="vertical", fraction=0.03)


def print_comparison_of_gene_selections(a, b, universe, color_a="blue", color_b="red"):
    """Compares two sets"""
    a, b, universe = set(a), set(b), set(universe)
    overlap = a.intersection(b)
    onlya = a.difference(b)
    onlyb = b.difference(a)
    unselected = universe.difference(a.union(b))
    showset = lambda s: ", ".join(list(s))
    white_print(f"N selected A: {len(a)} | N selected B: {len(b)} | N universe: {len(universe)}")
    color_print(f"Selected only by A: {showset(onlya)} (n={len(onlya)})", c=color_a)
    color_print(f"Selected only by B: {showset(onlyb)} (n={len(onlyb)})", c=color_b)
    color_print(f"Selected by A and B: {showset(onlya)} (n={len(overlap)})", c="green")
    color_print(f"Elements not selected by either: {showset(unselected)} (n={len(unselected)})", c="orange")


def plot_venn3_diagram_with_gene_labels(
    geneset_1,
    geneset_2,
    gene_selection_pool,
    set_labels=["TCGA", "Norway", "All genes"],
    set_colors=["brown", "blue", "black"],
    ax=None,
    fig=None,
    figsize=(25, 13),
    set_label_fontsize=20,
):
    geneset_1 = set(geneset_1)
    geneset_2 = set(geneset_2)
    all_genes_set = set(gene_selection_pool)
    sets = [geneset_1, geneset_2, all_genes_set]
    if ax is None and fig is None:
        fig, ax = plt.subplots(figsize=figsize)
    alpha = 0.4
    # Create Venn
    v = venn3(sets, set_labels=set_labels, set_colors=set_colors, ax=ax, alpha=alpha)
    # Map regions to the genes
    subset_ids = ["100", "010", "001", "110", "101", "011", "111"]  # standard order in venn3
    for label in v.set_labels:
        label.set_fontsize(set_label_fontsize)

    # Compute genes for each subset
    subset_genes = {}
    for i, subset_id in enumerate(subset_ids):
        # Determine which sets are included (1) or excluded (0)
        included = [sets[j] for j, bit in enumerate(subset_id) if bit == "1"]
        excluded = [sets[j] for j, bit in enumerate(subset_id) if bit == "0"]
        # Intersection minus excluded sets
        genes = set.intersection(*included) if included else set()
        if excluded:
            genes = genes - set.union(*excluded)
        subset_genes[subset_id] = genes

    # Add text labels
    for subset_id, genes in subset_genes.items():
        max_genes_per_line = 2
        genes = list(genes)
        lines = []

        for i in range(0, len(genes), max_genes_per_line):
            lines.append(", ".join(genes[i : i + max_genes_per_line]))
        gene_txt = "\n".join(lines)
        if v.get_label_by_id(subset_id) is not None:
            v.get_label_by_id(subset_id).set_text(gene_txt)

    # LEGEND
    patches = [Patch(facecolor=c, alpha=alpha, label=l) for c, l in zip(set_colors, set_labels)]   # fmt: skip
    ax.legend(handles=patches, fontsize=15, loc="lower left", framealpha=0.3)
