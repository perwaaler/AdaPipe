"""Permutation engine for discovery-cohort gene p-values.

Adaptive mode shuffles VAF once per draw and records off, on, and adaptive.
The off and on modes each record that one oncogenicity setting.
"""

from dataclasses import dataclass
from dataclasses import field
from typing import Literal

import numpy as np
import pandas as pd
import torch

from utils.display import *
from utils.models.models import CPHModel
from utils.models.torch_models import CoxPHSoftCut
from utils.models.torch_models import CoxPHUnivariateTrainer
from utils.paths import Path
from utils.paths import Paths
from utils.prepare_data.prepare_data import DataBuilder
from utils.variablesearch.variable_searcher import VariableSearcher


@dataclass
class VAFTransformer:
    """Maps gene VAF to the feature used in the univariate Cox model."""

    method: Literal["adapt_softcut", "fixed_cutoff"]
    stiff_cutoff: float = None
    softcut_fitparams: VariableSearcher.SoftCutFitParams = None
    transformer: callable = None

    def __post_init__(self):
        """Build the fixed-cutoff indicator, or a default softcut fit."""
        if self.method == "fixed_cutoff":
            assert self.stiff_cutoff is not None, "You have not set fixed VAF cutoff"
            self.transformer = lambda x: x >= self.stiff_cutoff
        elif self.method == "adapt_softcut":
            if self.softcut_fitparams is None:
                self.softcut_fitparams = VariableSearcher.SoftCutFitParams()


@dataclass
class PValuePermuter:
    """Bias-adjusted gene p-values from shuffled VAF in the Norwegian cohort."""

    vaf_transformer: VAFTransformer = None
    dump_path: str | Path = field(default_factory=lambda: Paths().permutated_pvals_norway_json)
    csv_dump_path: str | Path = field(default_factory=lambda: Paths().permuted_pvals_norway_csv)
    n_permutations_max: int = 5000
    n_nullhits_to_stop_sampling: int = 20
    # "adaptive" records off, on, and adaptive from the same draws. "off" and "on" record that setting only.
    oncofilter_settings: Literal["adaptive", "off", "on"] = "adaptive"
    eligible_genes: list[str] = field(init=False, default_factory=list)
    genes_to_analyze: list[str] = field(init=False, default_factory=list)

    def run_permutation_tests(self, variants_filtered_df):
        """Permute VAF for every gene with at least 10 mutations at VAF > 3%."""
        if self.vaf_transformer is None:
            orange_print("No VAF transformer provided; defaulting to SoftCut transform of VAF")
            self.vaf_transformer = VAFTransformer(method="adapt_softcut")

        red_id_subset = Paths.load_json(Paths().red_id_norway_path)
        varsearcher = VariableSearcher(
            variants_df=variants_filtered_df,
            clinical_df=DataBuilder().get_clinical_data(),
            red_id_used_to_fit_filters=red_id_subset,
        )
        varsearcher.load_results_from_json(compute_result_dfs=True)

        vaf_df = varsearcher.VAFmax_df_for_filterfit
        vaf_df = vaf_df[[x for x in vaf_df.columns if "MAX_VAF" in x]].copy()
        atleast_10muta = (vaf_df > 0.03).sum() >= 10
        self.eligible_genes = [x.replace("_MAX_VAF", "") for x in atleast_10muta[atleast_10muta].index]
        excluded_genes = [x.replace("_MAX_VAF", "") for x in atleast_10muta[~atleast_10muta].index]
        blue_print(f"N analyzable genes (>=10 mutations): {len(self.eligible_genes)} (excluded: {excluded_genes})")
        self.genes_to_analyze = self.eligible_genes
        green_print(f"N genes to analyze: {len(self.genes_to_analyze)}")

        self.null_pvalues_oncofilt_adaptive = {g: [] for g in self.genes_to_analyze}
        self.null_pvalues_oncofilt_off = {g: [] for g in self.genes_to_analyze}
        self.null_pvalues_oncofilt_on = {g: [] for g in self.genes_to_analyze}
        self.pvalues_orgl_adaptive = {}
        self.pvalues_orgl_oncofilt_off = {}
        self.pvalues_orgl_oncofilt_on = {}
        self._index_gene = 0

        white_print(f"Running permutation test for {len(self.genes_to_analyze)} genes (Thresholding method: {self.vaf_transformer.method}); Oncofilter='{self.oncofilter_settings}'")   # fmt: skip
        for self._index_gene, gene in enumerate(self.genes_to_analyze):
            white_print(f"\nPermutation analysis for {gene} (#{self._index_gene})")
            self._gene = gene
            univarfit_off_df = varsearcher.get_univarfit_df(gene, "off")
            pval_orgl_off = estimate_p_softcut(univarfit_off_df, self.vaf_transformer)
            univarfit_on_df = varsearcher.get_univarfit_df(gene, "on")
            if univarfit_on_df is None:
                univarfit_on_df = univarfit_off_df.copy()
                univarfit_on_df["VAF"] = 0.0
            if (univarfit_on_df["VAF"] > 0).sum() >= 10:
                pval_orgl_on = estimate_p_softcut(univarfit_on_df, self.vaf_transformer)
            else:
                pval_orgl_on = np.nan

            pval_orgl_adaptive = np.nanmin([pval_orgl_off, pval_orgl_on])

            self.print_original_pvalues(gene, pval_orgl_off, pval_orgl_on, pval_orgl_adaptive)
            self.pvalues_orgl_adaptive[gene] = pval_orgl_adaptive
            self.pvalues_orgl_oncofilt_off[gene] = pval_orgl_off
            self.pvalues_orgl_oncofilt_on[gene] = pval_orgl_on

            if self.oncofilter_settings == "adaptive":
                null_pvals_off, null_pvals_on, null_pvals_oncofilt_adaptive = (
                    self.simulate_pvalues_from_null_distribution__adaptive_onco(
                        gene,
                        univarfit_off_df,
                        univarfit_on_df,
                        pval_orgl_off,
                        pval_orgl_on,
                        pval_orgl_adaptive,
                    )
                )
                self.null_pvalues_oncofilt_off[gene] = null_pvals_off
                self.null_pvalues_oncofilt_on[gene] = null_pvals_on
                self.null_pvalues_oncofilt_adaptive[gene] = null_pvals_oncofilt_adaptive
            elif self.oncofilter_settings == "off":
                self.null_pvalues_oncofilt_off[gene] = self.simulate_pvalues_from_null_distribution__static_onco(
                    gene,
                    univarfit_off_df,
                    pval_orgl_off,
                )
            elif self.oncofilter_settings == "on":
                self.null_pvalues_oncofilt_on[gene] = self.simulate_pvalues_from_null_distribution__static_onco(
                    gene,
                    univarfit_on_df,
                    pval_orgl_on,
                )

    def simulate_pvalues_from_null_distribution__adaptive_onco(
        self,
        gene,
        univarfit_off_df,
        univarfit_on_df,
        pval_orgl_off,
        pval_orgl_on,
        pval_orgl_adaptive,
    ):
        """One shuffle loop. Stores a null p-value for off, on, and adaptive.

        Off and on are byproducts of the adaptive run: each draw fits both
        oncogenicity settings, and adaptive is the smaller of the two p-values.
        """
        n_null_hits_off = 0
        n_null_hits_on = 0 if not np.isnan(pval_orgl_on) else np.nan
        n_null_hits_adaptive = 0
        n_data = len(univarfit_off_df)
        n_mutas_on = (univarfit_on_df["VAF"] > 0).sum()

        null_pvals_off = []
        null_pvals_on = []
        null_pvals_adaptive = []

        for i_permute in range(self.n_permutations_max):
            permutation_i = np.random.permutation(n_data)
            null_p_off = self.draw_pvalue_from_null_distribution(univarfit_off_df, permutation_i)
            if n_mutas_on >= 10:
                null_p_on = self.draw_pvalue_from_null_distribution(univarfit_on_df, permutation_i)
                null_p_adaptive = min(null_p_off, null_p_on)
            else:
                null_p_on = np.nan
                null_p_adaptive = null_p_off

            null_pvals_on.append(null_p_on)
            null_pvals_off.append(null_p_off)
            null_pvals_adaptive.append(null_p_adaptive)

            if null_p_off < pval_orgl_off:
                n_null_hits_off += 1
            if null_p_on < pval_orgl_on:
                n_null_hits_on += 1
            if null_p_adaptive < pval_orgl_adaptive:
                n_null_hits_adaptive += 1
                if n_null_hits_adaptive % 5 == 0:
                    self.print_permutation_progress(
                        gene,
                        oncofilter_settings=["off", "on", "adaptive"],
                        null_ps=[null_p_off, null_p_on, null_p_adaptive],
                        n_null_hits=[n_null_hits_off, n_null_hits_on, n_null_hits_adaptive],
                        n_sim=len(null_pvals_adaptive),
                        p_orgls=[pval_orgl_off, pval_orgl_on, pval_orgl_adaptive],
                    )

            if i_permute % 500 == 0 and i_permute > 0:
                self.print_permutation_milestone(
                    i_permute,
                    n_null_hits=[n_null_hits_off, n_null_hits_on, n_null_hits_adaptive],
                    n_sims=[len(null_pvals_off), len(null_pvals_on), len(null_pvals_adaptive)],
                )
            enough_null_hits = self.check_for_early_stoppage(n_null_hits_off, n_null_hits_on, n_null_hits_adaptive)
            if enough_null_hits or i_permute >= self.n_permutations_max - 1:
                pval_unbiased_off = self.compute_rate_of_null_victories(null_pvals_off, pval_orgl_off)
                if n_mutas_on >= 10:
                    pval_unbiased_on = self.compute_rate_of_null_victories(null_pvals_on, pval_orgl_on)
                else:
                    pval_unbiased_on = np.nan
                pval_unbiased_adaptive = self.compute_rate_of_null_victories(null_pvals_adaptive, pval_orgl_adaptive)
                self.print_diagnostics_after_sampling(
                    gene=gene,
                    pvalues_unbiased=[pval_unbiased_off, pval_unbiased_on, pval_unbiased_adaptive],
                    pvalues_orgl=[pval_orgl_off, pval_orgl_on, pval_orgl_adaptive],
                    numbers_of_null_hits=[n_null_hits_off, n_null_hits_on, n_null_hits_adaptive],
                    numbers_of_trials=[len(null_pvals_off), len(null_pvals_on), len(null_pvals_adaptive)],
                )
                break

        return null_pvals_off, null_pvals_on, null_pvals_adaptive

    def simulate_pvalues_from_null_distribution__static_onco(self, gene, univarfit_df, pval_orgl):
        """Permutation null for one fixed oncogenicity setting."""
        if np.isnan(pval_orgl):
            return []
        n_null_hits = 0
        n_data = len(univarfit_df)
        n_mutas = (univarfit_df["VAF"] > 0).sum()
        null_pvals = []

        for i_permute in range(self.n_permutations_max):
            permutation_i = np.random.permutation(n_data)
            null_p = np.nan
            if n_mutas >= 10:
                null_p = self.draw_pvalue_from_null_distribution(univarfit_df, permutation_i)
                null_pvals.append(null_p)
            if null_p < pval_orgl:
                n_null_hits += 1
                if pval_orgl < 0.001 or (n_null_hits % 5 == 0 and n_null_hits > 0):
                    self.print_permutation_progress(
                        gene,
                        oncofilter_settings=[self.oncofilter_settings],
                        null_ps=[null_p],
                        n_null_hits=[n_null_hits],
                        n_sim=len(null_pvals),
                        p_orgls=[pval_orgl],
                        p_adjusted=[(n_null_hits + 1) / (len(null_pvals) + 1)],
                    )

            enough_null_hits = n_null_hits >= self.n_nullhits_to_stop_sampling
            if enough_null_hits or i_permute >= self.n_permutations_max - 1:
                if n_mutas >= 10:
                    pval_unbiased = self.compute_rate_of_null_victories(null_pvals, pval_orgl)
                else:
                    pval_unbiased = np.nan
                self.print_diagnostics_after_sampling(
                    gene=gene,
                    pvalues_unbiased=[pval_unbiased],
                    pvalues_orgl=[pval_orgl],
                    numbers_of_null_hits=[n_null_hits],
                    numbers_of_trials=[len(null_pvals)],
                    oncofilter_settings=[self.oncofilter_settings],
                )
                break

        return null_pvals

    def compute_rate_of_null_victories(self, null_pvals: list[float], pval_orgl: float):
        """Permutation p-value, (b + 1) / (m + 1)."""
        b = (np.array(null_pvals) < pval_orgl).sum()
        m = len(null_pvals)
        return (b + 1) / (m + 1)

    def draw_pvalue_from_null_distribution(self, univarfit_off_df, permuted_indexes: np.ndarray):
        """Shuffle VAF and refit the univariate model."""
        univarfit_off_shuffled_df = univarfit_off_df.copy()
        univarfit_off_shuffled_df["VAF"] = univarfit_off_df["VAF"].iloc[permuted_indexes].values
        return estimate_p_softcut(univarfit_off_shuffled_df, self.vaf_transformer)

    def check_for_early_stoppage(self, n_null_hits_off, n_null_hits_on, n_null_hits_adaptive) -> bool:
        """Stop once off, on, and adaptive each have enough null hits."""
        off_has_enough_hits = n_null_hits_off >= self.n_nullhits_to_stop_sampling
        on_has_enough_hits = n_null_hits_on >= self.n_nullhits_to_stop_sampling or np.isnan(n_null_hits_on)
        adaptive_has_enough_hits = n_null_hits_adaptive >= self.n_nullhits_to_stop_sampling
        return off_has_enough_hits and on_has_enough_hits and adaptive_has_enough_hits

    def dump_results(self):
        """Write the null p-values and the bias-adjusted table."""
        null_pvalues = {
            "oncofilter_off": self.null_pvalues_oncofilt_off,
            "oncofilter_on": self.null_pvalues_oncofilt_on,
            "oncofilter_adaptive": self.null_pvalues_oncofilt_adaptive,
        }
        Paths.dump_to_json(null_pvalues, self.dump_path)
        silent_print(f"Dumping simulated null p-values to {self.dump_path}")
        pvals_adaptive_df = self.get_biasadjusted_pvalues_for_all_genes(oncofilter_setting="adaptive")
        pvals_off_df = self.get_biasadjusted_pvalues_for_all_genes(oncofilter_setting="off")
        pvals_on_df = self.get_biasadjusted_pvalues_for_all_genes(oncofilter_setting="on")
        pvals_df = pd.concat([pvals_adaptive_df, pvals_off_df, pvals_on_df], keys=["adaptive", "off", "on"], axis=1)
        pvals_df.columns = [f"{x[1]}_{x[0]}" for x in pvals_df.columns]
        pvals_df.to_csv(self.csv_dump_path)
        silent_print(f"Dumping simulated null p-values to {self.csv_dump_path}")

    def get_biasadjusted_pvalues_for_all_genes(
        self, oncofilter_setting: Literal["adaptive", "off", "on"]
    ) -> pd.DataFrame:
        """Bias-adjusted and original p-values for one oncogenicity setting."""
        if oncofilter_setting == "adaptive":
            null_pvalues = self.null_pvalues_oncofilt_adaptive
            pvalues_biased = self.pvalues_orgl_adaptive
        elif oncofilter_setting == "off":
            null_pvalues = self.null_pvalues_oncofilt_off
            pvalues_biased = self.pvalues_orgl_oncofilt_off
        elif oncofilter_setting == "on":
            null_pvalues = self.null_pvalues_oncofilt_on
            pvalues_biased = self.pvalues_orgl_oncofilt_on

        bias_adjusted_pvals = {}
        genes_with_orglpval_calculated = [g for g in pvalues_biased if not np.isnan(pvalues_biased[g])]
        genes_with_nullpvals_simulated = [g for g in null_pvalues if null_pvalues[g] != []]
        genes_with_complete_results = list(set(genes_with_orglpval_calculated) & set(genes_with_nullpvals_simulated))
        for gene in genes_with_complete_results:
            b = (np.array(null_pvalues[gene]) < pvalues_biased[gene]).sum()
            m = len(null_pvalues[gene])
            bias_adjusted_pvals[gene] = (b + 1) / (m + 1)
        pvals_df = pd.DataFrame({"pvalue_permuted": bias_adjusted_pvals, "pvalue_orgl": pvalues_biased})
        return pvals_df.sort_values("pvalue_orgl")

    def _format_p_pct(self, p: float) -> str:
        """Format a p-value as a percentage, or n/a."""
        if p is None or (isinstance(p, float) and np.isnan(p)):
            return f"{'n/a':<8}"
        return f"{p:<8.1%}"

    def _format_null_hits(self, n_null_hits) -> str:
        """Format a null-hit count, or n/a."""
        if n_null_hits is None or (isinstance(n_null_hits, float) and np.isnan(n_null_hits)):
            return f"{'n/a':<4}"
        return f"{int(n_null_hits):<4}"

    def format_permutation_diagnostic_line(
        self,
        oncofilter_setting: str,
        p_orgl: float,
        p_unbiased: float = None,
        n_null_hits=None,
        n_sim: int = None,
    ) -> str:
        """One result line for an oncogenicity setting."""
        if p_unbiased is None:
            return f"OncoFilter: {oncofilter_setting:<8}  p_orgl:     {self._format_p_pct(p_orgl)}"
        return (
            f"OncoFilter: {oncofilter_setting:<8}  "
            f"p_permuted: {self._format_p_pct(p_unbiased)}  "
            f"n_null_hits: {self._format_null_hits(n_null_hits)} / {n_sim:<15}  "
            f"(p_orgl={self._format_p_pct(p_orgl).strip()})"
        )

    def format_permutation_progress_line(
        self,
        oncofilter_setting: str,
        null_p: float,
        n_null_hits,
        n_sim: int,
        p_orgl: float,
        p_adjusted: float = None,
    ) -> str:
        """One progress line during sampling."""
        adj_txt = f"  p_adj={self._format_p_pct(p_adjusted).strip()}" if p_adjusted is not None else ""
        return (
            f"OncoFilter: {oncofilter_setting:<8}  "
            f"p_sim:      {self._format_p_pct(null_p)}  "
            f"n_null_hits: {self._format_null_hits(n_null_hits)} / {n_sim:<15}  "
            f"(p_orgl={self._format_p_pct(p_orgl).strip()}{adj_txt})"
        )

    def print_original_pvalues(self, gene: str, pval_orgl_off: float, pval_orgl_on: float, pval_orgl_adaptive: float):
        """Print the unpermuted p-values for the active settings."""
        settings_pvals = {
            "off": pval_orgl_off,
            "on": pval_orgl_on,
            "adaptive": pval_orgl_adaptive,
        }
        if self.oncofilter_settings == "adaptive":
            active_settings = ["off", "on", "adaptive"]
        else:
            active_settings = [self.oncofilter_settings]

        teal_print(f"Original p-values for gene {self._index_gene}: {gene}")
        for setting in active_settings:
            txt = self.format_permutation_diagnostic_line(setting, settings_pvals[setting])
            color_print(txt, c=self.color_permutation_result_text(settings_pvals[setting]))

    def print_permutation_progress(
        self,
        gene: str,
        oncofilter_settings: list[str],
        null_ps: list[float],
        n_null_hits: list,
        n_sim: int,
        p_orgls: list[float],
        p_adjusted: list[float] = None,
    ):
        """Print the latest simulated p-values."""
        orange_print(f"  {gene}  sim #{n_sim}")
        for i, setting in enumerate(oncofilter_settings):
            adj = p_adjusted[i] if p_adjusted is not None else None
            orange_print(
                self.format_permutation_progress_line(
                    setting, null_ps[i], n_null_hits[i], n_sim, p_orgls[i], p_adjusted=adj
                )
            )

    def print_permutation_milestone(self, i_permute: int, n_null_hits: list, n_sims: list, oncofilter_settings: list[str] = None):
        """Print null-hit counts every 500 permutations."""
        if oncofilter_settings is None:
            oncofilter_settings = ["off", "on", "adaptive"]
        orange_print(f"Permutation milestone @ sim {i_permute}:")
        for setting, n_hits, n_sim in zip(oncofilter_settings, n_null_hits, n_sims):
            orange_print(f"OncoFilter: {setting:<8}  n_null_hits: {self._format_null_hits(n_hits)} / {n_sim:<15}")

    def print_diagnostics_after_sampling(
        self,
        gene,
        pvalues_unbiased: list,
        pvalues_orgl: list,
        numbers_of_null_hits: list,
        numbers_of_trials: list,
        oncofilter_settings: list[str] = None,
    ):
        """Print the bias-adjusted p-value when sampling stops. Default order is off, on, adaptive."""
        if oncofilter_settings is None:
            oncofilter_settings = ["off", "on", "adaptive"]
        teal_print(f"Results for gene {self._index_gene}: {gene}")
        for setting, p_unbiased, p_orgl, n_null_hits, n_sim in zip(
            oncofilter_settings, pvalues_unbiased, pvalues_orgl, numbers_of_null_hits, numbers_of_trials
        ):
            txt = self.format_permutation_diagnostic_line(setting, p_orgl, p_unbiased, n_null_hits, n_sim)
            color_print(txt, c=self.color_permutation_result_text(p_unbiased))

    def color_permutation_result_text(self, pval_biasadjusted):
        """Red below 0.001, orange below 0.01, otherwise blue."""
        if pval_biasadjusted is None or (isinstance(pval_biasadjusted, float) and np.isnan(pval_biasadjusted)):
            return "blue"
        return "red" if pval_biasadjusted < 0.001 else "orange" if pval_biasadjusted < 0.01 else "blue"


def estimate_p_softcut(univarfit_df, vaf_transformer: VAFTransformer):
    """Univariate Cox p-value after the softcut or fixed-cutoff transform."""
    univarfit_df = univarfit_df.copy()
    if vaf_transformer.method == "adapt_softcut":
        fit_params = vaf_transformer.softcut_fitparams
        vaf_transformer.transformer = fit_sigmoid_function(univarfit_df, fit_params)
    elif vaf_transformer.method == "fixed_cutoff":
        pass
    else:
        raise ValueError(f"Unknown VAFTransformer method: {vaf_transformer.method}")

    univarfit_df["VAF_TRANSFORMED"] = vaf_transformer.transformer(univarfit_df["VAF"])
    cph = CPHModel(predictors="VAF_TRANSFORMED", penalizer=0.1)
    try:
        cph.fit(univarfit_df, suppress_warnings=True)
        p_softcut = cph.pvalues.values[0]
    except Exception:
        silent_print("CPH model failed to converge; defaulting to p=0.5")
        p_softcut = 0.5
    return p_softcut


def fit_sigmoid_function(univarfit_df, fit_params: VariableSearcher.SoftCutFitParams) -> callable:
    """Fit the logistic VAF transform and return it as a function of VAF."""
    df_sorted = univarfit_df.sort_values(by="time_to_event", ascending=False).reset_index(drop=True)
    X_vaf = torch.tensor(df_sorted["VAF"].values, dtype=torch.float32).unsqueeze(1)
    time_to_event = torch.tensor(df_sorted["time_to_event"].values, dtype=torch.float32)
    event = torch.tensor(df_sorted["event_observed"].values, dtype=torch.float32)
    model = CoxPHSoftCut(vaf_scale_initval=fit_params.scale_initval, vaf_location_initval=fit_params.location_initval)
    sigmoid_trainer = CoxPHUnivariateTrainer(
        model=model,
        lambda_l2loss=fit_params.l2_penalty,
        n_epochs=fit_params.n_epochs,
        learning_rate=fit_params.learning_rate,
        min_n_epochs=fit_params.min_n_epochs,
        effective_sample_regularizer=fit_params.effective_samplesize_regularizer,
        verbose=fit_params.verbose,
        display=fit_params.display,
    )
    sigmoid_trainer.fit(X_muta=None, X_vaf=X_vaf, X_onco=None, time=time_to_event, event=event)
    loc, scale = sigmoid_trainer.model.get_location_and_scale_params()
    return lambda x: 1 / (1 + np.exp(-(x - loc) / scale))
