from utils.paths import Paths
from utils.prepare_data.prepare_data import DataBuilder
from utils.prepare_data.variant_data import GeneLevelFeatures
from utils.variablesearch.variable_searcher import VariableSearcher
from utils.display import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Literal
from statsmodels.stats.multitest import multipletests
from py_qvalue import qvalue
from matplotlib.ticker import PercentFormatter
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from scipy.stats import fisher_exact
from dataclasses import dataclass


# ---------- ANALYZING PERMUATION ADJUSTED P-VALUES  -----------------------------------------------------------------------------------------
def ana1__genTable__genes_selected_using_permutation_adjusted_pvals(variants_filtered_df):
    """Creates a table with an overview (p-values, cutoffs, and coefficients) of
    genes that are selected according to Benjamini-Hochberg method and
    permutation-based p-values."""
    # *** PAPER 1 TABLE (SUPPLEMENTARY) ***
    varsearcher = VariableSearcher(
        variants_df=variants_filtered_df,
        clinical_df=DataBuilder().get_clinical_data(),
    )
    varsearcher.load_results_from_json(compute_result_dfs=True)
    # Load permutation adjusted p-values
    pvalues_softadapted_df = pd.read_csv(Paths().permutated_pvals_esspenalized_adaptive_norway_csv, index_col=0)

    pvals = {}
    pvals["adaptive"] = pvalues_softadapted_df["pvalue_permuted_adaptive"]
    pvals["off"] = pvalues_softadapted_df["pvalue_permuted_off"]
    pvals["on"] = pvalues_softadapted_df["pvalue_permuted_on"]
    tso500_genes = DataBuilder.get_tso500_gene_list()
    non_tso500_genes = list(set(pvals["adaptive"].index).difference(tso500_genes))

    # ENSURE THAT P==NONE WHEN THERE ARE TOO FEW MUTATIONS
    VAF_df = GeneLevelFeatures.from_variants_using_max(
        variants_filtered_df=variants_filtered_df,
        create_zerocols_for_genes_with_no_oncovariants=True,
    ).genomic_features_df
    get_gene = lambda s: s.split("_MAX")[0]
    is_oncomaxcol, is_globalmaxcol = lambda x: "ONCO" in x, lambda x: "ONCO" not in x
    mutacount_all = (VAF_df.set_index("RED_ID") >= 0.03).sum().sort_values(ascending=False)
    untestable_genes = {}
    for oncosetting, is_coltype in zip(["on", "off"], [is_oncomaxcol, is_globalmaxcol]):
        mutacount = mutacount_all.loc[[x for x in mutacount_all.index if is_coltype(x)]]
        sparsly_mutated_genes = [get_gene(x) for x in mutacount[mutacount < 10].index]
        untestable_genes[oncosetting] = set(sparsly_mutated_genes)
        pvals[oncosetting] = pvals[oncosetting].drop(index=pvals[oncosetting].index.intersection(sparsly_mutated_genes)).dropna()   # fmt: skip
    # FILTER ALSO ON ADAPTIVE P FOR GOOD MEASURE
    untestable_in_both = untestable_genes["on"] & untestable_genes["off"]
    pvals["adaptive"] = pvals["adaptive"].drop(index=pvals["adaptive"].index.intersection(untestable_in_both)).dropna()

    # GET MPH ADJUSTED P-VALUES (Q-VALUES) FOR TSO 500 GENES
    dump_paths = [
        Paths().results_from_univariate_tso500_gene_analyses_csv,
        Paths().results_from_univariate_nontso500_gene_analyses_csv,
    ]
    fdr_level = 0.1
    for gene_list, dump_path, label in zip([tso500_genes, non_tso500_genes], dump_paths, ["TSO500", "NON TSO500"]):
        white_print(f"\n\n\n--- {label} GENES---")
        pvalues = {}
        qvalues = {}
        for oncosetting in ["adaptive", "off", "on"]:
            p = pvals[oncosetting]
            p = p.loc[p.index.intersection(gene_list)]
            h0_rejected, qvals, _, _ = multipletests(p.values, alpha=fdr_level, method="fdr_bh")   # fmt: skip
            teal_print(f"(onco-filter setting={oncosetting}) Genes selected by BH (FDR={fdr_level:.0%}): {h0_rejected.sum()} ({h0_rejected.mean():.1%})")   # fmt: skip
            pvalues[oncosetting] = p
            qvalues[oncosetting] = pd.Series(qvals, index=p.index, name=f"q_{oncosetting}")

        results_df = pd.concat(list(pvalues.values()), axis=1).rename_axis("gene")
        results_df = results_df.join(qvalues["adaptive"].rename("q"))
        results_df = results_df.join(
            varsearcher.summary_best_models_df[["vaf_cutoff", "hazard_ratio", "best_oncofiltersetting"]]
        )
        results_df = results_df.sort_values("pvalue_permuted_adaptive")
        p_cols = ["q", "pvalue_permuted_adaptive", "pvalue_permuted_off", "pvalue_permuted_on"]
        results_df = results_df[["vaf_cutoff", "best_oncofiltersetting", "hazard_ratio"] + p_cols]

        results_df = results_df.rename(
            columns={
                "vaf_cutoff": "VAF cutoff",
                "best_oncofiltersetting": "filter on oncogenicity",
                "hazard_ratio": "HR",
                "q": "q-value",
                "pvalue_permuted_adaptive": "p-value (adaptive oncogenicity filter)",
                "pvalue_permuted_off": "p-value (no oncogenicity filter)",
                "pvalue_permuted_on": "p-value (with oncogenicity filter)",
            }
        )
        results_df["filter on oncogenicity"] = results_df["filter on oncogenicity"].replace({"off": "no", "on": "yes"})
        results_df = results_df.dropna(subset="VAF cutoff")
        display(results_df)
        Paths.dump_to_csv(results_df, dump_path)


@dataclass
class PlotBHSelection:
    fig: Figure = None
    ax: Axes = None
    title: str = "Illustration of BH selection"
    n_to_plot: int = None

    def plot(self, pvals: pd.Series):
        """Plots pvalues along with BH decision thresholds, with p-values of
        selected genes highlighted to illustrate how the BH procedure works."""
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(7, 7))

        pvals = pvals.sort_values()
        pvals.index = np.arange(len(pvals))
        thresholds_list = []
        for a in [0.05, 0.1, 0.2]:
            thresholds = pd.Series(np.arange(1, len(pvals) + 1) / len(pvals) * a)
            thresholds_list.append(thresholds)

        def _bh_count(threshold_series):
            below = pvals[pvals <= threshold_series]
            if len(below) == 0:
                return 0
            return below.index[-1] + 1

        n_BH_detected = [_bh_count(thresholds_list[i]) for i in range(3)]
        selected_at_10prct_BH = pvals.iloc[: n_BH_detected[1]]

        n_to_plot = self.n_to_plot
        if self.n_to_plot is None:
            n_to_plot = n_BH_detected[1] + 20
        xx = pvals.index

        ax = self.ax
        ax.plot(xx, pvals, "-o", c="k", label="P-value")
        ax.plot(xx, thresholds_list[0], linestyle="--", color="red", label=f"BH rejection thresholds (alpha=0.05; {n_BH_detected[0]} detected)")   # fmt: skip
        ax.plot(xx, thresholds_list[1], linestyle="--", color="darkred", label=f"BH rejection thresholds (alpha=0.1; {n_BH_detected[1]} detected)")   # fmt: skip
        ax.plot(xx, thresholds_list[2], linestyle="--", color="orange", label=f"BH rejection thresholds (alpha=0.2; {n_BH_detected[2]} detected)")   # fmt: skip
        if len(selected_at_10prct_BH) > 0:
            ax.scatter(selected_at_10prct_BH.index, selected_at_10prct_BH.values, marker="s", s=300, edgecolors="k", alpha=0.3, color="teal", label=f"Detected gene (BH)")   # fmt: skip
        ax.axhline(y=0, linestyle="--", c="k", alpha=0.2)
        ax.set_xlim([-1, n_to_plot])
        ax.set_ylim([-0.001, pvals.iloc[min(n_to_plot, len(pvals) - 1)]])
        ax.set_ylabel("P-value")
        ax.set_xlabel("Gene rank")
        ax.set_xticklabels([int(ticklabel.get_position()[0] + 1) for ticklabel in ax.get_xticklabels()])
        ax.tick_params(rotation=0, labelsize=8 * (n_to_plot / 30))
        ax.legend()
        ax.set_title(self.title)


def ana2__analyze_permutation_pvalues():
    """Compare permutation-adjusted p-values: original vs ESS-penalized AdaPipe vs Stiff.

    Runs BH/Storey FDR selection per oncofilter setting. For onco-off, also plots
    a detection curve comparing ESS-penalized AdaPipe against Stiff (5%).
    """
    gene_set: Literal["all", "TSO500"] = "TSO500"
    if gene_set == "all":
        genes = DataBuilder.get_genes_in_variant_data()
    elif gene_set == "TSO500":
        genes = DataBuilder.get_tso500_gene_list()
    white_print("-" * 10 + f" USING {gene_set} GENES " + "-" * 100)

    oncofilter_strats = ["adaptive", "on", "off"]
    get_col = lambda df, c: df[c] if c in df.columns else pd.Series(np.nan, index=df.index)

    for oncofilter_strat in oncofilter_strats:
        white_print("\n" * 2 + "=" * 5 + f" Oncofilter setting: {oncofilter_strat} " + "=" * 60)
        pval_col = f"pvalue_permuted_{oncofilter_strat}"
        # Load results
        pvalues_ess_df = pd.read_csv(Paths().permutated_pvals_esspenalized_adaptive_norway_csv, index_col=0).reindex(genes).dropna(how="all")   # fmt: skip
        pvalues_fixed_df = pd.read_csv(Paths().permutated_pvals_fixedcut_norway_csv, index_col=0).reindex(genes).dropna(how="all")   # fmt: skip
        pvalues_softonco_df = pd.read_csv(Paths().permutation_adjusted_pvals_soft_onco_norway_csv, index_col=0).reindex(genes).dropna(how="all").rename(columns={"pvalue_permuted": "pvalue_permuted_adaptive"})   # fmt: skip

        # Assemble comparison
        @dataclass
        class PipeTracker:
            key: str
            pretty_label: str
            color: str

        pipeinfo = [
            PipeTracker("original_adapipe", "AdaPipe (no penalty)", "magenta"),
            PipeTracker("ESSpenalized_adapipe", "AdaPipe with ESS-penalty", "red"),
            PipeTracker("stiff", "Stiff (5% cutoff)", "blue"),
            PipeTracker("SoftOnco", "SoftOnco", "green"),
        ]
        merged_df = pd.concat(
            [
                pvalues_ess_df[pval_col],
                pvalues_fixed_df[pval_col],
                get_col(pvalues_softonco_df, pval_col),
            ],
            axis=1,
            keys=[f"{x.key}_p_permuted" for x in pipeinfo],
        )

        # --- FDR-based selection
        fdr_level = 0.1
        for info in pipeinfo:
            pvals = merged_df[f"{info.key}_p_permuted"].dropna().sort_values()
            if len(pvals) == 0:
                continue
            color_print(f"\n--- {info.pretty_label} ---", c=info.color)
            # Benjamini-Hochberg
            h0_rejected, _, _, _ = multipletests(pvals.values, alpha=fdr_level, method="fdr_bh")
            green_print(f"Benjamini-Hochberg (FDR={fdr_level:.0%}): {h0_rejected.sum()} {"":<30} genes selected: {list(pvals[h0_rejected].index)}")   # fmt: skip

        if oncofilter_strat == "off":
            pval_cutoffs = np.linspace(0, 0.01, 1000)
            fig, ax = plt.subplots(figsize=(7, 5))
            for info in [pipeinfo[1], pipeinfo[2], pipeinfo[3], pipeinfo[4]]:  # ESS AdaPipe vs Stiff
                p = merged_df[f"{info.key}_p_permuted"].dropna()
                prop = [(p <= c).mean() for c in pval_cutoffs]
                ax.plot(pval_cutoffs * 100, np.array(prop) * 100, label=info.pretty_label, c=info.color)
            ax.plot(pval_cutoffs * 100, pval_cutoffs * 100, linestyle="--", c="grey", label="Expected under H0")
            ax.set_xlabel("P-value threshold")
            ax.set_ylabel("Proportion of genes selected")
            ax.set_title("Detection curve — AdaPipe vs Stiff (onco off, Norway)")
            ax.xaxis.set_major_formatter(PercentFormatter(decimals=1))
            ax.yaxis.set_major_formatter(PercentFormatter(decimals=1))
            ax.legend(fontsize=8, framealpha=0.5)

        if oncofilter_strat == "adaptive":
            pval_cutoffs = np.linspace(0, 0.05, 1000)
            fig, ax = plt.subplots(figsize=(7, 5))
            for info in [pipeinfo[0], pipeinfo[1], pipeinfo[4]]:  # ESS AdaPipe vs Stiff
                p = merged_df[f"{info.key}_p_permuted"].dropna()
                prop = [(p <= c).mean() for c in pval_cutoffs]
                ax.plot(pval_cutoffs * 100, np.array(prop) * 100, label=info.pretty_label, c=info.color)
            ax.plot(pval_cutoffs * 100, pval_cutoffs * 100, linestyle="--", c="grey", label="Expected under H0")
            ax.set_xlabel("P-value threshold")
            ax.set_ylabel("Proportion of genes selected")
            ax.set_title("Detection curve — Adaptive oncofiltering")
            ax.xaxis.set_major_formatter(PercentFormatter(decimals=1))
            ax.yaxis.set_major_formatter(PercentFormatter(decimals=1))
            ax.legend(fontsize=8, framealpha=0.5)


def ana3__impact_of_variant_filters_on_univariate_gene_selection():
    """Paper 1 stats on how variant filters affect univariate gene selection.

    Counts genes lost under a fixed 5% cutoff or a universal oncogenicity filter.
    Run via ana3__permutationAdjusted_pvals.py.
    """
    gene_set: Literal["all", "TSO500", "driver"] = "TSO500"
    if gene_set == "all":
        genes = DataBuilder.get_genes_in_variant_data()
    elif gene_set == "TSO500":
        genes = DataBuilder.get_tso500_gene_list()
    else:
        genes = DataBuilder.get_driver_genes()
    white_print("-" * 10 + f" USING {gene_set} GENES " + "-" * 100)

    # LOAD RESULTS
    pvalues_fluid_df = (
        pd.read_csv(Paths().permutated_pvals_esspenalized_adaptive_norway_csv, index_col=0)
        .reindex(genes)
        .dropna(how="all")
    )
    pvalues_fixed_df = (
        pd.read_csv(Paths().permutated_pvals_fixedcut_norway_csv, index_col=0).reindex(genes).dropna(how="all")
    )

    # --- How many genes are lost when using fixed VAF cutoff?
    white_print(f"\n\n{"--"*5} How many genes are lost when using fixed VAF cutoff?")
    merged_pvals_df = pvalues_fluid_df.join(pvalues_fixed_df, lsuffix="_AdaPipe", rsuffix="_StiffPipe")[["pvalue_permuted_off_AdaPipe", "pvalue_permuted_off_StiffPipe"]]   # fmt: skip
    merged_pvals_df.columns = ["p_fluid", "p_stiff"]
    signif_under_ada_df = merged_pvals_df.query("p_fluid<0.001").copy()
    finding_lost_under_stiff = signif_under_ada_df["p_stiff"] > 0.01
    green_print(f"{finding_lost_under_stiff.sum()} / {len(signif_under_ada_df)} of genes with p<0.001 under AdaPipe have p>0.01 under StiffPipe.")   # fmt: skip

    # --- How many genes are lost due to onco-filtering?
    white_print(f"\n\n{"--"*5} How many genes are lost due to onco-filtering?")
    merged_pvals_df = pvalues_fluid_df[["pvalue_permuted_adaptive", "pvalue_permuted_off", "pvalue_permuted_on"]]
    merged_pvals_df.columns = ["p_ada", "p_off", "p_on"]

    fdr_level = 0.1
    for label, col in [("OFF", "p_off"), ("ON", "p_on"), ("ADA", "p_ada")]:
        p = merged_pvals_df[col].dropna()
        h0_rejected, _, _, _ = multipletests(p.values, alpha=fdr_level, method="fdr_bh")
        green_print(f"({label}) Benjamini-Hochberg (FDR={fdr_level:.0%}): {h0_rejected.sum()} genes selected ({h0_rejected.sum() / len(p):.1%})")   # fmt: skip

    # --- How many genes only became significant under onco-filtering?
    white_print(f"\n\n{"--"*5} How many genes only became significant under onco-filtering?")
    signif_under_oncoon_df = merged_pvals_df.query("p_on<0.001")
    green_print(f"{(signif_under_oncoon_df["p_off"]>0.1).sum()} / {len(signif_under_oncoon_df)} genes had p<0.001 under onco-filtering, but p>0.1 without oncofiltering.")   # fmt: skip

    # --- How many selected genes are drivers?
    white_print(f"\n\n{"--"*5} How many selected genes are drivers?")
    pvalues_fluid_df = (
        pd.read_csv(Paths().permutated_pvals_esspenalized_adaptive_norway_csv, index_col=0)
        .reindex(genes)
        .dropna(how="all")
    )

    pvals = merged_pvals_df["p_ada"]
    selected_genes = set(pvals.sort_values().index[:10])
    panel_genes = set(pvals.index)
    driver_genes = set(DataBuilder.get_driver_genes()) & panel_genes
    a = len(selected_genes & driver_genes)  # selected & driver
    b = len(selected_genes - driver_genes)  # selected & non-driver
    a / (b + a)
    c = len(driver_genes - selected_genes)  # not selected & driver
    d = len(panel_genes - selected_genes - driver_genes)  # not selected & non-driver
    c / (d + c)
    table = [[a, b], [c, d]]
    odds_ratio, p_value = fisher_exact(table, alternative="greater")
    green_print(f"Selected drivers: {a}/{len(selected_genes)}")
    green_print(f"Drivers in panel: {len(driver_genes)}/{len(panel_genes)}")
    green_print(f"Contingency table [[sel+drv, sel+non], [non+drv, non+non]]: {table}")
    green_print(f"OR={odds_ratio:.2f}, p={p_value:.4g}")
