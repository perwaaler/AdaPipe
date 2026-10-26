from __future__ import annotations
from typing import TYPE_CHECKING, Literal, List, Union

from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from matplotlib.ticker import PercentFormatter
from matplotlib.axes import Axes

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils.display import *
from utils.misc import add_text_to_legend
from utils.models.models import CPHModel
from utils.paths import Paths


if TYPE_CHECKING:
    from utils.variablesearch.variable_searcher import VariableSearcher


def evaluate_sensitivity_to_cutoff_selection(vaf_cutoffs, neg_logp):
    """How much the best -log10(p) stands out from the adjacent cutoffs."""
    idx_best_vaf = neg_logp.argmax()
    if isinstance(vaf_cutoffs, pd.Series):
        vaf_cutoffs = vaf_cutoffs.values
    if isinstance(neg_logp, pd.Series):
        neg_logp = neg_logp.values
    neg_logp_left = neg_logp[idx_best_vaf - 1] if idx_best_vaf > 0 else np.nan
    neg_logp_right = neg_logp[idx_best_vaf + 1] if idx_best_vaf < len(vaf_cutoffs) - 1 else np.nan
    mean_of_neighbours = np.nanmean([neg_logp_left, neg_logp_right])
    if mean_of_neighbours == 0:
        return np.nan
    return np.max(neg_logp) / mean_of_neighbours


def compute_vaf_center_of_significance(vaf_cutoffs: pd.Series, pvalues: pd.Series):
    """VAF cutoff weighted by -log10(p)."""
    neg_logp = -np.log10(pvalues)
    return (vaf_cutoffs * (neg_logp / neg_logp.sum())).sum()


class VarSearcherPlotter:
    def __init__(self, varsearcher: "VariableSearcher"):
        self.vs = varsearcher

    def plot_bootstrapped_vaf_cutoff_estimates(self):
        """Plots confidence intervals for gene-wise estimates of optimal VAF
        cutoffs. Used in Paper 1."""
        # *** PAPER 1 FIGURE (SUPPLEMENTARY) ***
        bs_results = Paths.load_json(Paths().bootstrapped_vaf_cutoffs_json)
        df = pd.DataFrame(bs_results)
        df = df[self.vs.summary_best_models_df.loc[df.columns].sort_values("p_softcut").index]
        genes = df.columns
        fig, ax = plt.subplots(figsize=(15, 4))
        melted_df = df.melt(var_name="gene")
        stats = melted_df.groupby("gene", sort=False)["value"].quantile([0.025, 0.5, 0.975]).unstack()
        stats.columns = ["low", "median", "high"]
        ax.errorbar(
            x=range(len(stats)),
            y=stats["median"],
            yerr=[stats["median"] - stats["low"], stats["high"] - stats["median"]],
            capsize=3,
            color="k",
            fmt="o",
        )
        ax.set_xticks(range(len(genes)))
        ax.set_xticklabels([x.replace("_", "-") for x in stats.index])
        ax.tick_params(axis="x", rotation=90)
        ax.set_ylabel("VAF")
        ax.grid(linestyle="--", alpha=0.3)
        ax.set_title("Estimates of optimal VAF cutoffs for top 30 prognostic genes", fontsize=15)
        ax.tick_params(rotation=90)

        genes_with_high_cutoff = stats[stats["low"] >= 0.05].index.to_list()
        values = stats.loc[genes_with_high_cutoff, "median"].values

        text = ", ".join([f"{g} ({v:.1%})" for g, v in zip(genes_with_high_cutoff, values)])
        green_print(f"Genes with lower bound over 5%: {text}")

    def plot_results(self, gene, ax=None, savefig_path=None):
        fig, axs = plt.subplots(4, 2, figsize=(12, 18), constrained_layout=True)
        get_signif = self.vs._calc_neglogp
        best_oncofilter_status = self.vs.merged_summary_df.loc[gene, "best_oncofiltersetting"]

        for oncofilter_status in ["on", "off"]:
            if oncofilter_status == "on":
                raw_results = self.vs.raw_results_df__on.loc[gene]
                ax = axs[:, 0]
            else:
                raw_results = self.vs.raw_results_df__off.loc[gene]
                ax = axs[:, 1]

            cutoffs = raw_results["vaf_cutoff"]
            pvals = raw_results["p"]
            cindexes = raw_results["cindex"]
            neglogp = raw_results["neglogp"]
            hazards = np.exp(raw_results["coef"])
            hazards_lower_ci = np.exp(raw_results["coef_lower_ci"])
            hazards_upper_ci = np.exp(raw_results["coef_upper_ci"])
            ci_array = np.stack([hazards_lower_ci, hazards_upper_ci]).transpose()

            if np.isnan(pvals).any():
                continue
            best_idx = self.vs.grid_cutoff_selector.get_index_of_best_cutoff(raw_results)
            best_p = pvals[best_idx]
            best_cutoff = cutoffs[best_idx]
            highest_neglogp = neglogp[best_idx]
            # --- P-VALUES
            ax[0].plot(cutoffs, neglogp, marker="s")
            ax[0].set_xlabel("Min VAF", alpha=0.7)
            ax[0].set_ylabel("-log10(p)")
            ax[0].set_ylim([0, 8])
            ax[0].set_title(f"p-values (best permissible={best_p:.1e})")
            ax[0].axhline(y=get_signif(0.05), linestyle="--", c="orange", alpha=0.5, label="p=0.05")
            ax[0].axhline(y=get_signif(0.01), linestyle="--", c="red", alpha=0.5, label="p=0.01")
            ax[0].grid(linestyle="--", alpha=0.5)
            ax[0].legend()
            sensitivity_of_p_estimate = evaluate_sensitivity_to_cutoff_selection(cutoffs, neglogp)
            vaf_center_of_significance = compute_vaf_center_of_significance(cutoffs, pvals)
            ax[0].axvline(x=vaf_center_of_significance, linestyle="--", label=f"VAF center of significance (x={vaf_center_of_significance:.2})")   # fmt: skip
            text = f"Mean -log(p): {neglogp.mean():.1f}" + f"\np* cutoff-sensitivity: {sensitivity_of_p_estimate:.2f}" + f"\nn cutoffs: {len(cutoffs):.0f}"   # fmt: skip
            handles = add_text_to_legend(text, ax[0])
            ax[0].legend(handles=handles, loc="upper left", framealpha=0.5)

            # --- HAZARD RATIOS
            univarfit_df = self.vs.get_univarfit_df(gene, oncofilter_status)
            ax[1].plot(cutoffs, hazards, marker="s")
            softcut_results = self.vs.fit_softcut_function(univarfit_df)
            vaf_squashed_grid = np.linspace(0, cutoffs[-1], 100)
            vaf_squashed = softcut_results.trainer.model.squash_vaf(vaf_squashed_grid).flatten()
            effect_sign = np.sign(softcut_results.trainer.model.effect.weight.detach().numpy()).flatten()[0]
            ax[1].plot(vaf_squashed_grid, vaf_squashed * effect_sign, label=f"Softcut p={softcut_results.p:.1e}")   # fmt: skip
            ax[1].plot(cutoffs, ci_array, c="grey", linestyle="--")
            ax[1].scatter(best_cutoff, hazards[best_idx], marker="*", edgecolor="g", s=300, facecolors="none", zorder=10)   # fmt: skip
            ax[1].set_xlabel("Min VAF", alpha=0.7)
            ax[1].set_ylabel("Hazard Ratio")
            ax[1].set_ylim([0, max(ci_array.max(axis=1)) * 1.1])
            ax[1].set_title(f"Hazard ratio")
            ax[1].axhline(y=1, linestyle="--", alpha=0.5, label="HR=1")
            ax[1].grid(linestyle="--", alpha=0.5)
            ax[1].legend()

            # --- KAPLAN MEIER CURVES FOR BEST CUTOFF
            best_vaf_cutoff = best_cutoff
            univarfit_df["MUTATED"] = univarfit_df["VAF"] >= best_vaf_cutoff
            cph = CPHModel(predictors="MUTATED")
            cph.fit(univarfit_df, precision=self.vs.cph_precision)
            kmf_neg = KaplanMeierFitter()
            neg_df = univarfit_df.query("MUTATED==False")
            kmf_neg.fit(durations=neg_df["time_to_event"], event_observed=neg_df["event_observed"])
            kmf_neg.plot(label=f"{gene} -", ci_show=True, ax=ax[2])
            kmf_pos = KaplanMeierFitter()
            pos_df = univarfit_df.query("MUTATED==True")
            kmf_pos.fit(durations=pos_df["time_to_event"], event_observed=pos_df["event_observed"])
            kmf_pos.plot(label=f"{gene} +", c="r", ci_show=True, ax=ax[2])

            add_at_risk_counts(kmf_neg, kmf_pos, ax=ax[2], labels=[f"{gene} -", f"{gene} +"])

            ax[2].grid(linestyle="--", alpha=0.5)
            ax[2].set_title(f"Kapplan meier curves (VAF>={best_vaf_cutoff:.2}, p={best_p:.1e})")
            ax[2].set_xlabel("days", alpha=0.7)

            # --- C-INDEX
            colors = np.select([pvals < 0.001, pvals < 0.01], ["red", "orange"], default="black")
            ax[3].scatter(cutoffs, cindexes, c=colors, marker="s", alpha=0.9)
            ax[3].plot(cutoffs, cindexes, alpha=0.7, c="k")
            ax[3].grid(alpha=0.5)
            ax[3].set_ylim([np.min([0.5, np.min(cindexes)]), np.max([0.65, np.max(cindexes)])])
            ax[3].set_ylabel("c-index")
            ax[3].set_xlabel("VAF cutoff", alpha=0.7)
            ax[3].scatter(best_cutoff, cindexes[best_idx], marker="*", edgecolor="g", s=300, facecolors="none", label=f"C at best cutoff where p={pvals[best_idx]:.1e}", zorder=10)   # fmt: skip
            ax[3].scatter(cutoffs[pvals.argmin()], cindexes[pvals.argmin()], marker="s", edgecolor="b", s=150, facecolors="none", label=f"lowest p ({pvals.min():.1e})", zorder=11)   # fmt: skip
            ax[3].legend()
            ax[3].set_title("C-index")

            # HIGHLIGHT TEXT IF ONCO-FILTER-SETTING IS OPTIMAL
            f_highlight_best = lambda s: f"*** {s} ***"
            star_color = "magenta"
            clip_x = lambda x: np.clip(x * 0.95, 0, 0.07)
            star_text = lambda ax, x, y: ax.text(clip_x(x * 0.95), y + 0.3, r"(VAF$^*$, -log(p$^*$))", color=star_color)
            plot_star = lambda ax, x, y: ax.plot(x, y, marker="*", color=star_color, markersize=14, markeredgecolor="k")

            if best_oncofilter_status == oncofilter_status:
                ax[0].set_title(f_highlight_best(ax[0].get_title()))
                ax[1].set_title(f_highlight_best(ax[1].get_title()))
                ax[3].set_title(f_highlight_best(ax[3].get_title()))
                star_text(ax[0], best_cutoff, highest_neglogp)
                plot_star(ax[0], best_cutoff, highest_neglogp)

        plt.suptitle(f"Results from single variable CPH fit: TTR vs {gene} mutation", fontsize=15)

        if savefig_path is not None:
            fig.savefig(savefig_path)

    def plot_results_light_version(self, gene, axs=None, savefig_path=None):
        assert not isinstance(self.vs.grid_cutoff_selector, grid_cutoff_selectors.MinPvalGloballyOverBestKGenes), "Current cutoff selector cant select optimal cutoff based only on local information"   # fmt: skip

        fig, axs0 = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
        best_oncofilter_status = self.vs.merged_summary_df.loc[gene, "best_oncofiltersetting"]
        add_star_if_best = lambda txt, oncofilter_status: "* " + txt if oncofilter_status == best_oncofilter_status else txt   # fmt: skip
        for oncofilter_status in ["on", "off"]:
            if oncofilter_status == "on":
                raw_results = self.vs.raw_results_df__on.loc[gene]
                axs = axs0[:, 0]
                hr_plot_title = add_star_if_best(f"Hazard ratio (only oncogenic variants)", oncofilter_status)
            else:
                raw_results = self.vs.raw_results_df__off.loc[gene]
                axs = axs0[:, 1]
                hr_plot_title = add_star_if_best(f"Hazard ratio (no oncogenicity filtering)", oncofilter_status)

            cutoffs = raw_results["vaf_cutoff"]
            n_mutas = raw_results["n_muta"]
            pvals = raw_results["p"]
            neglogp = raw_results["neglogp"]
            hazards = np.exp(raw_results["coef"])
            hazards_lower_ci = np.exp(raw_results["coef_lower_ci"])
            hazards_upper_ci = np.exp(raw_results["coef_upper_ci"])
            ci_array = np.stack([hazards_lower_ci, hazards_upper_ci]).transpose()

            if np.isnan(pvals).any():
                continue
            best_idx = self.vs.grid_cutoff_selector.get_index_of_best_cutoff(raw_results)
            best_p = pvals[best_idx]
            n_mutas_at_best = n_mutas[best_idx]
            best_cutoff = cutoffs[best_idx]
            highest_neglogp = neglogp[best_idx]

            # --- HAZARD RATIOS
            univarfit_df = self.vs.get_univarfit_df(gene, oncofilter_status)
            ax = axs[0]
            ax.plot(cutoffs, hazards, marker="s")
            ax.plot(cutoffs, ci_array, c="grey", linestyle="--")
            p_text = lambda p: f"p-value={best_p:.5f}" if p > 0.00001 else "<0.000001"
            label = f"{p_text(best_p)} (cutoff={best_cutoff:.1%}, {n_mutas_at_best} mutations)"
            ax.xaxis.set_major_formatter(PercentFormatter(1.0))
            ax.scatter(best_cutoff, hazards[best_idx], marker="*", edgecolor="k", facecolor="g", s=300, zorder=10, label=label)   # fmt: skip
            ax.set_xlabel("VAF cutoff")
            ax.set_ylabel("Hazard Ratio")
            ax.set_ylim([0, max(ci_array.max(axis=1)) * 1.1])
            ax.set_title(hr_plot_title)
            ax.axhline(y=1, linestyle="--", alpha=0.5, label="HR=1")
            ax.grid(linestyle="--", alpha=0.5)
            ax.legend()

            # --- KAPLAN MEIER CURVES FOR BEST CUTOFF
            ax = axs[1]
            best_vaf_cutoff = best_cutoff
            univarfit_df["MUTATED"] = univarfit_df["VAF"] >= best_vaf_cutoff
            cph = CPHModel(predictors="MUTATED")
            cph.fit(univarfit_df, precision=self.vs.cph_precision)
            kmf_neg = KaplanMeierFitter()
            neg_df = univarfit_df.query("MUTATED==False")
            kmf_neg.fit(durations=neg_df["time_to_event"], event_observed=neg_df["event_observed"])
            kmf_neg.plot(label=f"{gene} -", ci_show=True, ax=ax)
            kmf_pos = KaplanMeierFitter()
            pos_df = univarfit_df.query("MUTATED==True")
            kmf_pos.fit(durations=pos_df["time_to_event"], event_observed=pos_df["event_observed"])
            kmf_pos.plot(label=f"{gene} +", c="r", ci_show=True, ax=ax)

            ax.grid(linestyle="--", alpha=0.5)
            ax.set_title(f"Kapplan meier curves (VAF cutoff: {best_vaf_cutoff:.1%})")
            ax.set_xlabel("days", alpha=0.7)

        plt.suptitle(f"Univariate CPH analysis of VAF cutoffs for {gene}", fontsize=20, y=1.07)

        if savefig_path is not None:
            fig.savefig(savefig_path)

    def plot_results_from_squashing_vaf(
        self,
        gene,
        oncofilter_status="on",
        fit_params=None,
        ax_softcut=None,
        ax_loss=None,
        plot_losses=True,
        return_fitresults=True,
        kwargs_sigmoidplot=None,
    ):
        """Plots the squashed VAF along with loss history from model fitting."""
        if fit_params is None:
            fit_params = self.vs.SoftCutFitParams()
        if kwargs_sigmoidplot is None:
            kwargs_sigmoidplot = {}
        univarfit_df = self.vs.get_univarfit_df(gene, oncofilter_status)
        fitresults = self.vs.fit_softcut_function(univarfit_df, fit_params=fit_params)
        if ax_softcut is None:
            fig, ax_softcut = plt.subplots()

        fitresults.trainer.model.plot_vaf_curve(ax=ax_softcut, **kwargs_sigmoidplot)

        if plot_losses:
            if ax_loss is None:
                fig, ax_loss = plt.subplots()
            fitresults.trainer.plot_losses(ax=ax_loss)

        if return_fitresults:
            return fitresults

    def plot_fitted_softmuta_curves(
        self,
        nrows=4,
        ncols=5,
        sorting_variable: Literal["p_softcut", "p"] = "p_softcut",
        fit_params=None,
        genes: List[str] = None,
    ):
        if fit_params is None:
            fit_params = self.vs.softcut_fit_params
        if genes is None:
            df = self.vs.summary_best_models_df.sort_values(sorting_variable).head(40)
        else:
            df = self.vs.summary_best_models_df.loc[genes]

        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 3))
        axs: List[Axes] = axs.flatten()

        for gene, hard_cutoff, ax in zip(df.index, df["vaf_cutoff"], axs):
            fitresults = self.plot_results_from_squashing_vaf(
                gene,
                fit_params=fit_params,
                ax_softcut=ax,
                plot_losses=False,
                oncofilter_status=df.loc[gene, "best_oncofiltersetting"],
            )
            ax.axvline(hard_cutoff, linestyle="--", label=f"Hard cutoff: {hard_cutoff:.3f}")
            ax.legend(fontsize=8)
            ax.set_title(f"")
            ax.tick_params(labelsize=5, rotation=30)
            ax.set_ylabel("VAF squashed", fontsize=8)
            ax.set_xlabel("VAF", fontsize=8)
            p_soft = df.loc[gene]["p_softcut"]
            p_hard = df.loc[gene]["p"]
            txt = "\n".join(
                [
                    f"{gene}",
                    f"p-softcut: {p_soft:.3%}",
                    f"p-hardcut: {p_hard:.3%}",
                    f"n mutas: {df.loc[gene, 'n_muta']}",
                ]
            )
            ax.text(0.05, 0.75, txt, ha="left", va="top", transform=ax.transAxes, alpha=0.7 if p_soft < p_hard else 1.0, fontsize=7)   # fmt: skip

    def plot_nyear_relapse_rate_per_vaf_bin(
        self,
        variants_filtered_df=None,
        gene="TET2",
        oncofilter_setting: Literal["on", "off"] = "off",
        clinical_df=None,
        relapse_window=365 * 2,
        min_bin_size=15,
        bin_edges=None,
        ax=None,
        red_id_subset_to_analyze: List[str] = None,
    ):
        if variants_filtered_df is None:
            variants_filtered_df = self.vs.variants_df
        if red_id_subset_to_analyze is None:
            red_id_subset_to_analyze = self.vs.red_id_used_to_fit_filters

        clinical_df = self.vs.clinical_df if clinical_df is None else clinical_df
        if isinstance(self.vs.red_id_used_to_fit_filters, list):
            clinical_df = clinical_df[clinical_df["RED_ID"].isin(self.vs.red_id_used_to_fit_filters)].copy()

        if oncofilter_setting == "off":
            candidate_selector = lambda g: (g["SYMBOL"] == gene) & g["IS_ONCOGENIC"]
        else:
            candidate_selector = lambda g: (g["SYMBOL"] == gene) & (g["ONCOGENICITY_SCORE"] >= 3)

        vaf_extraction_func = lambda g: (
            g[candidate_selector(g)]["VAF_TUMOR"].max() if (candidate_selector(g)).any() else 0
        )
        df = (
            variants_filtered_df.groupby("RED_ID")
            .apply(vaf_extraction_func, include_groups=False)
            .reset_index(name="VAF")
        )
        df = df.merge(clinical_df[["RED_ID", "time_to_event", "event_observed"]], on="RED_ID", how="inner")
        df = df[df["RED_ID"].isin(red_id_subset_to_analyze)]
        df1 = df.query(f"event_observed==True or time_to_event > {relapse_window}").copy()
        df1["VAF"] = df["VAF"] * 100
        if bin_edges is not None:
            vaf_cuts = np.asarray(bin_edges)
        else:
            base_cuts = np.array([0, 0.0125, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.5, 1.0]) * 100
            vaf_cuts = base_cuts
            for i in range(7):
                df1["VAF bin"] = pd.cut(df1["VAF"], bins=vaf_cuts, right=False)
                bin_sizes = df1.groupby("VAF bin", observed=False).size()
                bin_too_small = (bin_sizes < min_bin_size).reset_index(name="bin_to_small")

                if bin_too_small["bin_to_small"].any():
                    idx_bin_to_expand = bin_too_small[bin_too_small["bin_to_small"]].index[0]
                    idx_right_edge = idx_bin_to_expand + 1
                    idx_rightmost_edge = len(vaf_cuts) - 1
                    if idx_right_edge < idx_rightmost_edge:
                        idx_edge_to_drop = idx_bin_to_expand + 1
                    else:
                        idx_edge_to_drop = idx_bin_to_expand
                    vaf_cuts = np.delete(vaf_cuts, idx_edge_to_drop)
                else:
                    break

        df1["VAF bin"] = pd.cut(df1["VAF"], bins=vaf_cuts, right=False)

        rates = df1.groupby("VAF bin", observed=False)["time_to_event"].agg([lambda x: (x < relapse_window).mean() * 100, lambda x: (x <= relapse_window).sum(), len])   # fmt: skip
        rates.columns = ["event rate", "event count", "bin size"]

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(rates, x="VAF bin", y="event rate", color="grey", ax=ax)
        ax.tick_params(rotation=30, labelsize=7)
        ax.set_title(f"Prevalence of 2-year relapse per VAF bin")
        ax.set_ylim([0, np.max([80, rates["event rate"].max() * 1.1])])
        for bar, event_count, total_count in zip(ax.patches, rates["event count"], rates["bin size"]):
            x_coord = bar.get_x() + bar.get_width() / 2
            y_coord = bar.get_height()
            rate = event_count / total_count
            se = np.sqrt(rate * (1 - rate) / total_count)
            ci_margin = 1.96 * se * 100
            ax.errorbar(x=x_coord, y=y_coord, yerr=[[ci_margin], [ci_margin]], color="k", capsize=3)

    def plot_nyear_relapse_rate_per_vaf_bin_for_topk_genes(
        self,
        sorting_variable: Literal["p", "p_softcut"] = "p",
        min_criteria: Literal["p<0.01", "p<0.001"] = "p<0.01",
        k: Union[20, 30, 40] = 40,
    ):
        df_top_genes = self.vs.summary_best_models_df.query(min_criteria).sort_values(sorting_variable)
        df_top_k_genes = df_top_genes[~df_top_genes.index.isin(["H1_2"])].iloc[:k]
        settings_map = {
            20: {"n_rows": 2, "n_cols": 10, "figsize": (45, 20), "fontsize_ticks": 11},
            30: {"n_rows": 3, "n_cols": 10, "figsize": (45, 25), "fontsize_ticks": 12},
            40: {"n_rows": 4, "n_cols": 10, "figsize": (45, 25), "fontsize_ticks": 13},
        }
        fig, axs = plt.subplots(
            settings_map[k]["n_rows"], settings_map[k]["n_cols"], figsize=settings_map[k]["figsize"]
        )
        axs = axs.flatten()
        for i, (gene, ax) in enumerate(zip(df_top_k_genes.index, axs)):
            self.plot_nyear_relapse_rate_per_vaf_bin(self.vs.variants_df, gene=gene, ax=ax)
            ax.set_title("")
            ax.tick_params(axis="x", labelsize=settings_map[k]["fontsize_ticks"])
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.text(0.05, 0.95, f"{gene}", ha="left", va="top", transform=ax.transAxes, alpha=0.5, fontsize=12)   # fmt: skip

    def scatterplot_variants(self, variants_filtered_df, gene="PIM1"):
        """Plots the mutations in scatterplot with x=VAF, y=SUPPORT, and markers
        indicating ONCOGENIC=T/F. 1 mutation per patient."""
        axs = plot_vaf_support_and_oncoscore_for_variants(variants_filtered_df, gene, return_axs=True)
        axs[0].axvline(self.vs.summary_best_models_df.loc[gene, "vaf_cutoff"], color="green", linestyle="-", label="Best cutoff")   # fmt: skip
        axs[0].legend()

    def plot_hazardcurves_for_genes(self, list_of_genes=None, order_by_signif=True, title=None):
        """Plots hazard curves in tile-plot for list of genes."""
        if list_of_genes is None:
            list_of_genes = self.vs.summary_best_models_df.sort_values("p_softcut").head(20).index

        n_cols = 4
        n_rows = int(np.ceil(len(list_of_genes) / n_cols))
        n_tiles = n_cols * n_rows
        fig, ax = plt.subplots(n_rows, n_cols, figsize=np.array((n_cols, n_rows)) * 3.5)
        ax = ax.flatten()

        info_all_genes_df = self.vs.summary_best_models_df.loc[list_of_genes]
        if order_by_signif:
            info_all_genes_df = info_all_genes_df.sort_values("p")
            list_of_genes = info_all_genes_df.index
        max_y = np.exp(info_all_genes_df["coef_upper_ci"]).max()
        max_x = info_all_genes_df["vaf_cutoff"].max()

        for i, gene in enumerate(list_of_genes):
            analysis_info = self.vs.summary_best_models_df.loc[gene]
            best_oncofiltersetting = self.vs.summary_best_models_df.loc[gene, "best_oncofiltersetting"]
            if best_oncofiltersetting == "on":
                grid_results = self.vs.raw_results_df__on.loc[gene]
            else:
                grid_results = self.vs.raw_results_df__off.loc[gene]

            hazards = np.exp(grid_results["coef"])
            hazards_lower_ci = np.exp(grid_results["coef_lower_ci"])
            hazards_upper_ci = np.exp(grid_results["coef_upper_ci"])
            vaf_grid = grid_results["vaf_cutoff"]
            vaf_cutoff = analysis_info["vaf_cutoff"]

            ax[i].plot(vaf_grid, hazards, c="k")
            ax[i].plot(vaf_grid, hazards_lower_ci, c="k", alpha=0.4, linestyle=":")
            ax[i].plot(vaf_grid, hazards_upper_ci, c="k", alpha=0.4, linestyle=":")
            ax[i].axhline(y=1, c="k", alpha=0.5, linestyle="-.")
            ax[i].axvline(x=vaf_cutoff, c="k", alpha=0.4, linestyle="-")
            ax[i].grid(alpha=0.6)
            ax[i].set_ylim([0, max_y * 1.01])
            ax[i].set_xlim([0, max_x * 1.1])
            ax[i].tick_params(labelsize=5)
            infotext = f"{gene}, pval: {analysis_info.p:.1e}\nSoftcut p: {analysis_info.p_softcut:.1e}\nc-index={analysis_info.cindex:.3f}\ncutoff={vaf_cutoff:.2f}\nn mutas={analysis_info.n_muta:.0f}"
            ax[i].text(0.005, max_y * 0.7, infotext, fontsize=7)

        for i in range(len(list_of_genes), n_tiles):
            ax[i].set_xticks([])
            ax[i].set_yticks([])

        if title is not None:
            fig.suptitle(title)
        fig.tight_layout()

    def plot_performancediff_between_softcut_and_hardcut(self):
        df = -np.log10(self.vs.summary_best_models_df[["p", "p_softcut"]]).rename(
            columns={"p": "neglogp", "p_softcut": "neglogp_softcut"}
        )
        df = df.rename_axis("gene").reset_index()
        df["neglogp_diff"] = df["neglogp_softcut"] - df["neglogp"]
        max_neglogp = df[["neglogp", "neglogp_softcut"]].max().max()
        df["neglogp_absdiff"] = df["neglogp_diff"].abs()
        genes_with_large_difference = df[df["neglogp_absdiff"] >= 2].sort_values("neglogp_diff", ascending=False)

        fig, axs = plt.subplots(1, 2, figsize=(25, 12))
        axs[0].scatter(df["neglogp"], df["neglogp_softcut"], marker="o", edgecolor="k", alpha=0.6)
        axs[0].plot(np.linspace(0, max_neglogp), np.linspace(0, max_neglogp), c="k", alpha=0.3)
        axs[0].set_xlabel(r"-log10($p_{hardcut}$)")
        axs[0].set_ylabel(r"-log10($p_{softcut}$)")
        axs[0].set_title("Performance of models")

        for i in range(len(genes_with_large_difference)):
            gene = genes_with_large_difference["gene"].iloc[i]
            x = df["neglogp"].loc[genes_with_large_difference.index[i]]
            y = df["neglogp_softcut"].loc[genes_with_large_difference.index[i]]
            axs[0].text(x, y, gene, fontsize=8)

        axs[1].scatter(range(len(df)), df["neglogp_absdiff"], marker="o", edgecolor="k", alpha=0.6)
        axs[1].set_xlabel(r"gene")
        axs[1].set_ylabel(r"[-log10($p_{softcut}$)] - [-log10($p_{hardcut}$)]")
        axs[1].set_title("Performance difference (softcut vs hardcut)")

        for i in range(len(genes_with_large_difference)):
            gene = genes_with_large_difference["gene"].iloc[i]
            x = genes_with_large_difference.index[i]
            y = genes_with_large_difference["neglogp_absdiff"].iloc[i]
            axs[1].text(x, y, gene, fontsize=10)

        plt.show()

        green_print("HR-cutoff plots for genes where softcut significantly outperforms hardcut model")
        self.plot_hazardcurves_for_genes(genes_with_large_difference["gene"].to_list())

    def scatterplots_of_pvalues__oncofilt1_vs_oncofilt2(self):
        fig, ax = plt.subplots()
        colors = self.vs.summary_best_models_df["p_diff_neglogp"]
        cmap = "YlOrRd"
        sc = ax.scatter(self.vs.summary_df__on["p"], self.vs.summary_df__off["p"], s=10, cmap=cmap, c=colors)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(r"$| \log_{10}(p_1) - \log_{10}(p_2) |$")
        ax.set_ylim([0.00001, 1])
        ax.set_xlabel("p-value from fit: oncofilter on")
        ax.set_ylabel("p-value from fit: oncofilter off")
        ax.set_title("P-values from cutoff-optimized univariate fits:\nwith oncofilter turned on and off")
        x = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 100)
        ax.plot(x, x, alpha=0.5, linestyle="--")
        ax.axvline(x=0.05, c="k", linestyle="--", alpha=0.3, label="p=0.05")
        ax.axhline(y=0.05, c="k", linestyle="--", alpha=0.3)
        ax.grid(alpha=0.5)
        ax.loglog()
        ax.fill_between(x=[-1, 0.01], y1=0.01, y2=1, color="red", alpha=0.2, label=r"$p_{on}<0.01, p_{off}\geq0.01$")   # fmt: skip
        ax.fill_between(x=[0.01, 1], y1=0, y2=0.01, color="magenta", alpha=0.2, label=r"$p_{off}<0.01, p_{on}\geq0.01$")   # fmt: skip
        ax.legend(framealpha=0.3, fontsize=8)
        ax.margins(x=0, y=0)

        fig, ax = plt.subplots()
        colors = self.vs.summary_best_models_df["p_diff_neglogp"]
        sc = ax.scatter(self.vs.summary_df__on["hazard_ratio"], self.vs.summary_best_models_df["hazard_ratio"], s=colors * 10, cmap=cmap, c=colors)   # fmt: skip
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(r"$| \log_{10}(p_1) - \log_{10}(p_2) |$")
        ax.grid(alpha=0.5)
        ax.set_xlabel("HR from fit: oncofilter on")
        ax.set_ylabel("HR from fit: oncofilter off")
        ax.set_title("Hazard ratios from cutoff-optimized univariate fits:\nwith oncofilter turned on and off")   # fmt: skip
        ax.axvline(x=1, c="k", linestyle="--", alpha=0.3, label="HR=1")
        ax.axhline(y=1, c="k", linestyle="--", alpha=0.3, label="HR=1")
        x = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 100)
        ax.plot(x, x, alpha=0.5, linestyle="--")

    def barplots_of_variant_type_distribution_stratified_by_oncogenicity(self, variants_filtered_df):
        genes_of_interest__on = self.vs.summary_best_models_df.query("best_oncofiltersetting=='on'")["p_diff_neglogp"].abs().sort_values(ascending=0).head(5).index   # fmt: skip
        genes_of_interest__off = self.vs.summary_best_models_df.query("best_oncofiltersetting=='off'")["p_diff_neglogp"].abs().sort_values(ascending=0).head(5).index   # fmt: skip
        genes_of_interest = list(genes_of_interest__on) + list(genes_of_interest__off)

        types = ["SNV", "insertion", "deletion"]
        gen_counts_series = lambda counts: pd.Series({t: counts.loc[t] if t in counts.index else 0 for t in types}, name="count").rename_axis("Variant type")   # fmt: skip

        fig, axs = plt.subplots(5, 2, figsize=(5, 13))
        axs = axs.flatten()
        counter = 0
        for gene, ax in zip(genes_of_interest, axs):
            df_mutas = variants_filtered_df.query(f"SYMBOL=='{gene}'")
            df__oncoyes = df_mutas.query("ONCOGENICITY_SCORE>=3")
            df__oncono = df_mutas.query("ONCOGENICITY_SCORE<3")
            counts_oncono = df__oncono["VARIANT_CLASS"].value_counts() / len(df__oncono["VARIANT_CLASS"])
            counts_oncoyes = df__oncoyes["VARIANT_CLASS"].value_counts() / len(df__oncoyes["VARIANT_CLASS"])
            counts_oncono = gen_counts_series(counts_oncono)
            counts_oncoyes = gen_counts_series(counts_oncoyes)

            plot_df = pd.concat([counts_oncono, counts_oncoyes], keys=["non oncogenic", "oncogenic"], names=["origin"]).reset_index()   # fmt: skip

            ax.grid(alpha=0.4, linestyle="--")
            sns.barplot(plot_df, x="origin", y="count", hue="Variant type", edgecolor="k", palette=["b", "m", "red"], ax=ax, alpha=0.8)   # fmt: skip
            ax.legend(fontsize=6, framealpha=0.6)
            ax.set_xticks([0, 1])
            ax.tick_params(axis="y", labelsize=6)
            ax.set_xticklabels(["non oncogenic", "oncogenic"], fontsize=6, alpha=0.6)
            ax.set_xlabel("")
            ax.set_ylabel("Proportion", fontsize=6)
            if counter % 2 == 1:
                ax.set_ylabel("")
            if counter == 0:
                ax.set_title(f"Best oncofilter setting: 'off'\n\n{gene}", fontsize=8)
            elif counter == 1:
                ax.set_title(f"Best oncofilter setting: 'on'\n\n{gene}", fontsize=8)
            else:
                ax.set_title(f"{gene}", fontsize=8)

            ax.set_ylim([0, 1])
            counter += 1

        kmf_pos.plot(label=f"{gene} deletion+", c="m", ci_show=True, ax=ax)

    def hazard_ratio_curves_for_diagram(self, gene):
        """Simplified HR curves for use in flowchart diagrams."""
        # *** PAPER 1 FIGURE 1A (FLOWCHART) ***
        fig, axs = plt.subplots(1, 2, figsize=(8, 3.5))
        titles = ["Oncogenic variants", "All variants"]
        gene = "TET2"

        for i, oncofilter_status in enumerate(["on", "off"]):
            if oncofilter_status == "on":
                raw_results = self.vs.raw_results_df__on.loc[gene]
            else:
                raw_results = self.vs.raw_results_df__off.loc[gene]

            cutoffs = raw_results["vaf_cutoff"]
            hazards = np.exp(raw_results["coef"])
            hazards_lower_ci = np.exp(raw_results["coef_lower_ci"])
            hazards_upper_ci = np.exp(raw_results["coef_upper_ci"])

            best_idx = self.vs.grid_cutoff_selector.get_index_of_best_cutoff(raw_results)
            best_cutoff = cutoffs[best_idx]

            ax = axs[i]
            ax.plot(cutoffs, hazards, "k-o", markersize=5, linewidth=2)
            ax.fill_between(cutoffs, hazards_lower_ci, hazards_upper_ci, alpha=0.15, color="k")
            ax.scatter(
                best_cutoff,
                hazards[best_idx],
                marker="o",
                edgecolor="k",
                facecolor="red",
                s=250,
                zorder=10,
                linewidths=2,
            )
            ax.axhline(y=1, linestyle="--", alpha=0.4, color="k")
            ax.set_title(titles[i], fontsize=15)
            ax.set_xlabel("VAF cutoff", fontsize=15)
            ax.set_xticks([])
            ax.set_yticks([])
            if oncofilter_status == "on":
                ax.set_yticks([1])
            ax.set_ylim([0.5, 7])
            if i == 0:
                ax.set_ylabel("Hazard Ratio", fontsize=15)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        fig.tight_layout()
        Paths.dump_fig(fig, Paths().hazard_ratio_curves_for_flowdiagram_pdf)
        return fig, axs

    def sigmoid_curve_for_diagram(self):
        """Simplified sigmoid curve for use in flowchart diagrams."""
        fig, ax = plt.subplots()
        vaf_transformer = lambda x, loc, scale: 1 / (1 + np.exp(-(x - loc) / scale))
        loc_true, scale_true = 0.135, 0.005
        n_estimation_points = 7
        locations = np.geomspace(0.09, loc_true, n_estimation_points)
        scales = np.geomspace(0.02, scale_true, n_estimation_points)
        alphas = np.linspace(0.1, 0.7, n_estimation_points)
        lw = 4
        xx = np.linspace(0, 0.3, 100)
        for loc, scale, alpha in zip(locations[:-1], scales[:-1], alphas):
            ax.plot(xx, vaf_transformer(xx, loc, scale), "k", linestyle="-", linewidth=lw, alpha=alpha)
        ax.plot(xx, vaf_transformer(xx, loc_true, scale_true), color="k", linewidth=lw + 3)
        ax.plot(xx, vaf_transformer(xx, loc_true, scale_true), color="red", linewidth=lw + 2)

        risks = np.array([0.12, 0.13, 0.32, 0.45, 0.47]) * 1.6
        n_bars = len(risks)
        bars = pd.DataFrame({"risk": risks, "VAF": np.linspace(0, 0.25, len(risks))})
        ax.bar(
            bars["VAF"],
            bars["risk"],
            width=0.25 / n_bars * 1.15,
            alpha=0.5,
        )
        ax.set_xlim([0, 0.25])
        ax.set_ylabel("Relapse Risk", fontsize=30)
        ax.set_xlabel("VAF", fontsize=30)
        ax.set_xticks([])
        ax.set_yticks([])
        Paths.dump_fig(fig, Paths().sigmoid_curves_showign_VAF_relapse_relationship_for_flowdiagram_pdf)


def plot_vaf_support_and_oncoscore_for_variants(gatk0_raw_df, gene="KRAS", min_onco_score=3, return_axs=False):
    """Scatterplot of VAF and SUPPORT for variants."""

    def get_muta_info(x):
        main_muta = x.iloc[x["SUPPORTING_READS"].argmax()]
        return pd.Series({"Support": main_muta["SUPPORTING_READS"], "VAF": main_muta["VAF_TUMOR"], "Consequence": main_muta["CONSEQUENCE"], "OncoScore": main_muta["ONCOGENICITY_SCORE"] >= 3})   # fmt: skip

    _, axs = plt.subplots(1, 2, figsize=(10, 4))

    plot_df = gatk0_raw_df.query(f"SYMBOL=='{gene}'").groupby("RED_ID").apply(get_muta_info, include_groups=0)
    axs[0].plot(plot_df["VAF"], plot_df["Support"], "o", markersize=3, markerfacecolor="none", markeredgecolor="blue", markeredgewidth=0.5)   # fmt: skip
    axs[0].scatter(plot_df.query("OncoScore==True")["VAF"], plot_df.query("OncoScore==True")["Support"], marker="s", facecolor="none", edgecolors="red", label=f"OncoScore>={min_onco_score}")    # fmt: skip
    axs[0].set_xlabel("VAF")
    axs[0].set_ylabel("Supporting reads")
    axs[0].axvline(x=0.03, color="magenta", linestyle="-.", alpha=0.5, label="VAF=0.03")
    axs[0].axvline(x=0.1, color="purple", linestyle="--", alpha=0.5, label="VAF=0.1")
    axs[0].axvline(x=0.2, color="blue", linestyle=":", alpha=0.5, label="VAF=0.2")
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    xticks = np.arange(0.01, 1.01, 0.02)
    axs[0].set_xticks(xticks)
    axs[0].get_xaxis().set_major_formatter(plt.ScalarFormatter())  # Show decimal labels
    axs[0].grid(True, which="major", axis="x", alpha=0.5)
    axs[0].tick_params(rotation=90, labelsize=6)
    axs[0].set_title(f"Frequency and support of {gene} variants\n(highest-VAF variant represents each patient)")
    axs[0].legend(framealpha=0.5)

    axs[1] = plot_df["VAF"].hist(bins=50, edgecolor="k")
    axs[1].set_xlabel("VAF")
    axs[1].set_ylabel("count")
    axs[1].set_title(f"VAF distribution for {gene}")

    if return_axs:
        return axs
