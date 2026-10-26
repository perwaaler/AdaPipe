"""Compare AdaPipe with the fixed VAF-cutoff benchmarks in the discovery cohort.

Each pipeline is cross-validated over a range of selected genes. The paper
reports 2-year AUC, the curve against the number of genes, and the average of
models with 10–20 genes.
"""

from scipy.stats import ttest_1samp
from typing import List
from typing import Literal
from dataclasses import dataclass
from dataclasses import field
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from utils.plotting import sns_colors
import textwrap

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils.analysis_and_calculations import calc_auc_for_predicting_relapse_from_df


from utils.display import *
from utils.crossvalidation import CrossValidator
from utils.variablesearch.grid_cutoff_selectors import ManuallySelectedCutoff
from utils.variablesearch.grid_cutoff_selectors import MinPvalOverEligible
from utils.variablesearch.cutoff_choosers import VAFCutoffSelector
from utils.datafactory.data_factory import DataFactory
from utils.datafactory.gene_selectors import TopKBestGenes
import utils.datafactory.design_matrixes as design_matrixes
from utils.prepare_data.prepare_data import DataBuilder
from utils.evaluation import barplot__cv_performance_comparison_of_pipelines
from utils.paths import Paths
from utils.models.models import CPHModel


@dataclass
class CaptionConfig:
    basetxt: str = None
    extratxt: str = ""
    alpha: float = 0.5
    x: float = 0.0
    y: float = -0.15
    fontsize: int = 8
    wrapwidth: int = 60


@dataclass
class FoldMetricStats:
    label: str
    fold_values: np.ndarray

    @property
    def mean(self) -> float:
        return float(np.asarray(self.fold_values).mean())

    @property
    def std(self) -> float:
        return float(np.asarray(self.fold_values).std())

    @property
    def p_vs_chance(self) -> float:
        return float(ttest_1samp(self.fold_values, popmean=0.5, alternative="two-sided").pvalue)

    def variance_str(self) -> str:
        v = np.asarray(self.fold_values)
        return f"std {v.std():.2%}  range ({v.min():.2%}, {v.max():.2%})"


# Print labels: "ensemble" = avg predictions, then AUC. Two ensemble levels when both apply.
METRIC_N_SELECTED_ENSEMBLE = "n_selected ensemble"
METRIC_MEAN_AUC_N_SELECTED = "mean of per-k AUCs over n_selected"
LABEL_BENCHMARK_ENSEMBLE = "benchmark ensemble"

# Default CV grid for ensemble/barplot analyses (ensemble_range [10, 20] ⊆ this grid).
N_GENES_SELECTED = np.arange(10, 21, 2)
# Wider grid for AdaPipe-vs-stiff line plots (AUC vs n_selected figures).
N_GENES_SELECTED_LINEPLOTS = np.arange(10, 30, 2)


@dataclass
class PairedComparison:
    candidate: FoldMetricStats
    reference: FoldMetricStats

    @property
    def delta_per_fold(self) -> np.ndarray:
        return np.asarray(self.candidate.fold_values) - np.asarray(self.reference.fold_values)

    @property
    def delta_mean(self) -> float:
        return float(self.delta_per_fold.mean())

    @property
    def p_paired(self) -> float:
        return float(ttest_1samp(self.delta_per_fold, popmean=0).pvalue)


@dataclass
class PipelineComparisonSummary:
    candidate: FoldMetricStats
    benchmarks: List[FoldMetricStats]
    benchmark_ensemble: FoldMetricStats | None = None
    pipeline_ensembles: dict = field(default_factory=dict)
    ensemble_range: tuple[int, int] | None = None
    metric_description: str = METRIC_N_SELECTED_ENSEMBLE

    @property
    def best_benchmark(self) -> FoldMetricStats:
        return max(self.benchmarks, key=lambda b: b.mean)

    @property
    def worst_benchmark(self) -> FoldMetricStats:
        return min(self.benchmarks, key=lambda b: b.mean)

    @property
    def benchmark_mean(self) -> float:
        return float(np.mean([b.mean for b in self.benchmarks]))

    def paired_vs(self, reference: FoldMetricStats) -> PairedComparison:
        return PairedComparison(candidate=self.candidate, reference=reference)

    @property
    def vs_benchmark_ensemble(self) -> PairedComparison | None:
        if self.benchmark_ensemble is None:
            return None
        return self.paired_vs(self.benchmark_ensemble)

    @property
    def vs_best_benchmark(self) -> PairedComparison:
        return self.paired_vs(self.best_benchmark)


def _format_metric_row(label: str, stats: FoldMetricStats, label_width: int) -> str:
    v = np.asarray(stats.fold_values)
    return (
        f"{label:<{label_width}}  "
        f"{stats.mean:>6.1%}  "
        f"{stats.p_vs_chance:>7.4f}  "
        f"{stats.std:>6.2%}  "
        f"({v.min():.2%}, {v.max():.2%})"
    )


def _format_mean_auc_row(label: str, mean_auc: float, label_width: int) -> str:
    return f"{label:<{label_width}}  {mean_auc:>6.1%}"


def _format_delta_row(description: str, delta_mean: float, p_paired: float, desc_width: int) -> str:
    return f"{description:<{desc_width}}  {delta_mean:>+6.1%}  {p_paired:>7.4f}"


def _comparison_row_labels(summary: PipelineComparisonSummary) -> list[str]:
    labels = [f"Candidate  {summary.candidate.label}"]
    if summary.pipeline_ensembles:
        labels.extend(summary.pipeline_ensembles.keys())
    elif summary.benchmark_ensemble is not None:
        labels.append(summary.benchmark_ensemble.label)
    labels.extend(b.label for b in summary.benchmarks)
    labels.extend(
        [
            f"Best benchmark  {summary.best_benchmark.label}",
            f"Worst benchmark  {summary.worst_benchmark.label}",
            "Benchmark average",
        ]
    )
    return labels


@dataclass
class ListOfDataFactories:
    """Label → DataFactory map for pipeline comparison."""

    pipelines: dict[str, DataFactory] = field(default_factory=dict)
    cutoffs: dict[str, float] = field(default_factory=dict)

    @property
    def labels(self) -> list[str]:
        return list(self.pipelines.keys())

    @property
    def datafactories(self) -> list[DataFactory]:
        return list(self.pipelines.values())

    def add(self, label: str, datafactory: DataFactory, cutoff: float | None = None) -> None:
        if label in self.pipelines:
            raise ValueError(f"Pipeline label already exists: {label!r}")
        self.pipelines[label] = datafactory
        if cutoff is not None:
            self.cutoffs[label] = cutoff

    def __add__(self, other: "ListOfDataFactories") -> "ListOfDataFactories":
        overlap = set(self.pipelines) & set(other.pipelines)
        if overlap:
            raise ValueError(f"Duplicate pipeline labels: {sorted(overlap)}")
        return ListOfDataFactories(
            pipelines={**self.pipelines, **other.pipelines},
            cutoffs={**self.cutoffs, **other.cutoffs},
        )

    def refresh_minimal_criteria(self):
        for label, datafactory in self.pipelines.items():
            cutoff = self.cutoffs.get(label)
            if cutoff is None:
                continue
            datafactory.gene_selector.minimal_criteria = lambda r, c=cutoff: r["vaf_cutoff"] > c - 0.01

    def update_oncofilter_strategy(
        self,
        new_strategy: Literal["pick_on", "pick_off", "highest_neglogp", "lowest_p_softcut"],
        pipeline_specifications: list[str] = None,
    ):
        pipelines_updated = pipeline_specifications if pipeline_specifications is not None else self.pipelines.values()
        for datafactory in pipelines_updated:
            datafactory.grid_cutoff_selector.oncofilter_selection = new_strategy
            datafactory.update_oncofilter_strategy()


@dataclass
class PipelineComparisonAnalysis:
    """Cross-validate every pipeline in ``pipelines``.

    ``candidate_label`` is AdaPipe. The other labels are the benchmarks. ``eval_histology`` keeps only LUAD or LUSC patients in the test folds.
    """

    pipelines: ListOfDataFactories
    nyears_in_roc_calc: float = 2
    penalizer: float = 0.1
    train_endpoint: Literal["TTR", "DSS", "OS", "DFS"] = None
    eval_endpoint: Literal["TTR", "DSS", "OS", "DFS"] = None
    n_genes_selected_list: np.ndarray = field(default_factory=lambda: np.arange(10, 21, 2))
    # OTHER
    colors_benchmarks: List[str] = None
    color_candidate: str = field(default_factory=lambda: sns_colors["purple"])
    benchmark_labels: List[str] = None
    candidate_label: str = "AdaPipe"
    eval_histology: Literal["NSCLC", "LUAD", "LUSC"] = None
    verbose: int = 1  # 0=quiet; 1=pipeline AUC summaries; 2=+per-k AUC; 3=+per-fold Cox coefs
    # Pipeline ensemble: None = pool all benchmarks; [] = off; list = pool these labels.
    benchmark_ensemble_labels: list[str] | None = None
    benchmark_ensemble_label: str | None = None
    # Named pipeline ensembles (which models to ensemble; overrides benchmark_ensemble_labels when set).
    # Values are pipeline labels; may include candidate_label.
    pipeline_ensemble_groups: dict[str, list[str]] | None = None

    def __post_init__(self):
        self.model_factory = lambda: CPHModel(penalizer=self.penalizer)
        if self.colors_benchmarks is None:
            self.colors_benchmarks = [sns_colors[c] for c in ["yellow", "orange", "red", "dark_red"]]
        self.pipeline_colors = self.colors_benchmarks + [self.color_candidate]

        if self.benchmark_labels is None:
            if self.candidate_label in self.pipelines.pipelines:
                self.benchmark_labels = [l for l in self.pipelines.labels if l != self.candidate_label]
            else:
                self.benchmark_labels = list(self.pipelines.labels)

        self.figs_lineplots: dict[str, Figure] = {}
        self.axes_lineplots: dict[str, Axes] = {}
        self.fig_barplot: Figure = None
        self.axes_barplot: Axes = None

    def _crossvalidator_verbose(self) -> int:
        """Map PCA verbosity to CrossValidator (2 = print fitted coefficients per fold)."""
        return 2 if self.verbose >= 3 else 0

    def _model_factory_for(self, datafactory: DataFactory):
        if getattr(datafactory, "model_factory", None) is not None:
            return datafactory.model_factory
        return self.model_factory

    def _benchmark_ensemble_display_label(self) -> str:
        return self.benchmark_ensemble_label or LABEL_BENCHMARK_ENSEMBLE

    def _cv_results_for_pipeline_label(self, label: str) -> list:
        return self.cv_results[label]

    def _resolved_benchmark_ensemble_labels(self) -> list[str] | None:
        if self.pipeline_ensemble_groups:
            return None
        if self.benchmark_ensemble_labels is not None:
            if len(self.benchmark_ensemble_labels) < 2:
                return None
            return list(self.benchmark_ensemble_labels)
        if len(self.benchmark_labels) <= 1:
            return None
        return list(self.benchmark_labels)

    def _resolved_pipeline_ensemble_groups(self) -> dict[str, list[str]]:
        if self.pipeline_ensemble_groups:
            return {name: labels for name, labels in self.pipeline_ensemble_groups.items() if len(labels) >= 2}
        labels = self._resolved_benchmark_ensemble_labels()
        if labels is None:
            return {}
        return {self._benchmark_ensemble_display_label(): labels}

    def _pipeline_ensemble_auc_per_fold(self, labels: list[str], ensemble_range: list) -> np.ndarray:
        cv_results_lists = [self._cv_results_for_pipeline_label(label) for label in labels]
        pooled_df = self.get_pooled_ensemble_predictions(cv_results_lists, ensemble_range)
        return (
            pooled_df.reset_index()
            .groupby("cv_idx")
            .apply(lambda g: calc_auc_for_predicting_relapse_from_df(g, g["ypred"]), include_groups=False)
            .values
        )

    def _apply_eval_histology_to_datafactories(self):
        if self.eval_histology is None:
            return
        for datafactory in self.pipelines.datafactories:
            DataFactoryUpdater(datafactory).update_test_histology(self.eval_histology)


    def generate_results(self):
        """Fit every pipeline and store per-fold AUC, c-index, and predictions."""
        for datafactory in self.pipelines.datafactories:
            datafactory.red_id_restriction_test_sets = None
        self._apply_eval_histology_to_datafactories()
        self.run_analysis()
        self.collect_results_in_dataframes()

    def collect_results_in_dataframes(self):
        self.metric_dfs_2yr = self._make_metric_dfs(self.aucs_2yr)
        self.metric_dfs_3yr = self._make_metric_dfs(self.aucs_3yr)
        self.metric_dfs_iauc = self._make_metric_dfs(self.iaucs)
        self.metric_dfs_cindex = self._make_metric_dfs(self.cindexes)

    def _make_metric_dfs(self, metrics: dict[str, list]) -> dict[str, pd.DataFrame]:
        return {label: self._make_metric_df(values, self.n_genes_selected_list) for label, values in metrics.items()}

    def _make_metric_df(self, metric_values: list, index):
        df = pd.DataFrame(metric_values)
        df.columns = [f"fold{i}" for i in range(5)]
        df.index = index
        df.index.name = "n genes selected"
        return df

    def _benchmark_metric_dfs(self, metric_dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        return {label: metric_dfs[label] for label in self.benchmark_labels}


    @staticmethod
    def _summarize_2yr_auc_over_n_selected(aucs_per_k: list) -> str:
        mean_per_k = [float(np.mean(a)) for a in aucs_per_k]
        overall = float(np.mean(mean_per_k))
        return f"2yrAUC={overall:.1%} (over n_selected: {min(mean_per_k):.1%}–{max(mean_per_k):.1%})"

    def run_analysis(self):
        labels = list(self.pipelines.pipelines.keys())
        self.aucs_2yr = {l: [] for l in labels}
        self.aucs_3yr = {l: [] for l in labels}
        self.cindexes = {l: [] for l in labels}
        self.iaucs = {l: [] for l in labels}
        self.cv_results = {l: [] for l in labels}

        for label, datafactory_i in self.pipelines.pipelines.items():
            if self.verbose > 0:
                orange_print(f"\n{label}")
            for k in self.n_genes_selected_list:
                if self.verbose > 1:
                    white_print(f"  n_selected={k}")
                datafactory_i.gene_selector.k = k
                crossval = CrossValidator(
                    data_factory=datafactory_i.build_data_for_ith_fold,
                    model_factory=self._model_factory_for(datafactory_i),
                    train_endpoint=self.train_endpoint,
                    eval_endpoint=self.eval_endpoint,
                    eval_histology=self.eval_histology,
                    verbose=self._crossvalidator_verbose(),
                )
                crossval.prepare_data()
                crossval.train_models_in_crossvalidation()
                crossval.calc_auc_stats(n_years=self.nyears_in_roc_calc)
                crossval.calc_iauc_stats()
                if self.verbose > 1:
                    brown_print(f"    2yrAUC: {crossval.auc_mean:.3f} (p={crossval.auc_pvalue:.3f})")
                self.aucs_2yr[label].append(crossval.calc_auc_stats(n_years=2, return_auc_values=True))
                self.aucs_3yr[label].append(crossval.calc_auc_stats(n_years=3, return_auc_values=True))
                self.cindexes[label].append(crossval.cindex_values)
                self.iaucs[label].append(crossval.iauc_values)
                self.cv_results[label].append(crossval.cv_results_df)

            if self.verbose == 1:
                silent_print(f"  {label}: {self._summarize_2yr_auc_over_n_selected(self.aucs_2yr[label])}")


    def plot_roc_curve_for_specific_k_for_candidate(self, k=20, label: str = None):
        """ROC for one pipeline at a single number of genes."""
        if self.verbose > 1:
            white_print(f"  n_selected={k} (ROC)")
        if label is None:
            label = self.candidate_label
        datafactory = self.pipelines.pipelines[label]
        datafactory.gene_selector.k = k
        crossval = CrossValidator(
            data_factory=datafactory.build_data_for_ith_fold,
            model_factory=self._model_factory_for(datafactory),
            train_endpoint=self.train_endpoint,
            eval_endpoint=self.eval_endpoint,
            eval_histology=self.eval_histology,
            verbose=self._crossvalidator_verbose(),
        )
        crossval.prepare_data()
        crossval.train_models_in_crossvalidation()
        crossval.run_roc_curve_analysis(n_years=self.nyears_in_roc_calc)
        if self.verbose > 1:
            white_print(f"    2yrAUC: {crossval.auc_mean:.3f} (p={crossval.auc_pvalue:.3f})")

    def plot_auc_for_different_n_select(self, metrics_to_plot=["AUC (2 yr)"]):
        """AUC against the number of selected genes."""
        alpha = 0.8
        metric_fixedGS_dataframes = [
            self._benchmark_metric_dfs(self.metric_dfs_2yr),
            self._benchmark_metric_dfs(self.metric_dfs_3yr),
            self._benchmark_metric_dfs(self.metric_dfs_cindex),
        ]
        metrics_fluidGS_dataframes = [
            self.metric_dfs_2yr[self.candidate_label],
            self.metric_dfs_3yr[self.candidate_label],
            self.metric_dfs_cindex[self.candidate_label],
        ]
        labels = ["AUC (2 yr)", "AUC (3 yr)", "c-index"]

        # --- 2-YEAR AUC
        # Get average AUC (across global cutoffs) per n_selected_genes
        for metrics_fixedGS, metrics_fluidGS, label in zip(
            metric_fixedGS_dataframes, metrics_fluidGS_dataframes, labels
        ):
            # COMPUTE AVERAGE VALUE ACROSS BENCHMARKS
            if label not in metrics_to_plot:
                continue
            max_yvals = []  # Used to decide y-limits
            metrics_averaged = 0
            for key, metrics_df in metrics_fixedGS.items():
                metrics_averaged += metrics_df
            metrics_averaged = metrics_averaged / len(metrics_fixedGS)
            # PLOT RESULTS FOR BENCHMARK PIPELINES
            fig, ax = plt.subplots(figsize=(7, 5))
            for bench_label, color in zip(self.benchmark_labels, self.colors_benchmarks):
                yvals = metrics_fixedGS[bench_label].mean(axis=1)
                ax.plot(
                    self.n_genes_selected_list,
                    yvals,
                    "-o",
                    markeredgecolor="k",
                    color=color,
                    label=bench_label,
                    alpha=alpha,
                )
                max_yvals.append(yvals.max())
            # PLOT RESULTS FOR CANDIDATE PIPELINE
            yvals = metrics_fluidGS.mean(axis=1)
            max_yvals.append(yvals.max())
            ax.plot(
                self.n_genes_selected_list,
                yvals,
                "-s",
                c=self.color_candidate,
                markeredgecolor="k",
                alpha=alpha,
                label=self.candidate_label,
            )
            ax.plot(self.n_genes_selected_list, metrics_averaged.mean(axis=1), "-s", c="k", alpha=0.5, linewidth=5, label=f"benchmark average")   # fmt: skip
            ax.set_ylim([0.5, max(0.75, max(max_yvals) * 1.1)])
            ax.set_xlabel(r"$n_{selected}$")
            ax.set_ylabel(label)

            ax.legend(fontsize=9)
            ax.grid(alpha=0.5, linestyle="--")
            ax.set_title(f"Predictive Performance of Stiff vs Fluid Pipelines by Number of Selected Genes", y=1.05)
            fig.tight_layout()

            self.figs_lineplots[label] = fig
            self.axes_lineplots[label] = ax

    def plot_barplot(
        self,
        n_selected_range=[10, 50],
        metric: Literal["auc_2yr", "auc_3yr", "cindex"] = "auc_2yr",
        captionconfig: CaptionConfig = None,
    ):
        """Barplot of performance averaged over ``n_selected_range``."""
        if captionconfig is None:
            captionconfig = CaptionConfig()
        # RESHAPE RESULTS TO EXPECTED FORMAT
        sample_n = [int(n) for n in self.n_genes_selected_list if n >= n_selected_range[0] and n <= n_selected_range[1]]
        assert sample_n.__len__() > 0, f"Range must include samples in {n_selected_range}"
        if metric == "cindex":
            benchmark_metric_values = [self.metric_dfs_cindex[l] for l in self.benchmark_labels]
            candidate_vals = self.metric_dfs_cindex[self.candidate_label].loc[sample_n].mean().values
        elif metric == "auc_3yr":
            benchmark_metric_values = [self.metric_dfs_3yr[l] for l in self.benchmark_labels]
            candidate_vals = self.metric_dfs_3yr[self.candidate_label].loc[sample_n].mean().values
        else:
            benchmark_metric_values = [self.metric_dfs_2yr[l] for l in self.benchmark_labels]
            candidate_vals = self.metric_dfs_2yr[self.candidate_label].loc[sample_n].mean().values

        bencmark_vals = [df.loc[sample_n].mean().values for df in benchmark_metric_values]
        metric_array = np.vstack((candidate_vals, bencmark_vals))
        # --- PLOT
        if captionconfig.basetxt is None:
            captionconfig.basetxt = f"Metrics are averaged across 'N selected' in {sample_n}"
        caption = "\n".join(textwrap.wrap(captionconfig.basetxt, width=captionconfig.wrapwidth)) + captionconfig.extratxt   # fmt: skip
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        barplot__cv_performance_comparison_of_pipelines(
            values=metric_array,
            pipeline_labels=[self.candidate_label] + self.benchmark_labels,
            pipeline_colors=[self.color_candidate] + self.colors_benchmarks,
            title=f"Cross validation comparison: {METRIC_MEAN_AUC_N_SELECTED}",
            caption=caption,
            fig=fig,
            axs=axs,
        )
        axs[1].text(captionconfig.x, captionconfig.y, caption, ha="left", va="top", transform=axs[1].transAxes, alpha=captionconfig.alpha, fontsize=captionconfig.fontsize)   # fmt: skip
        self.fig_barplot = fig
        self.axes_barplot = axs
        self.barplot_metric_array = metric_array
        self.mean_pipeline_performance = np.mean(metric_array, axis=1).mean()

    def get_n_selected_ensemble_predictions(
        self, result_dataframes: List[pd.DataFrame], ensemble_range=[10, 20]
    ) -> pd.DataFrame:
        """result_dataframes contains CV results from each value of
        ``n_selected``, where each ``results_df`` comes from CrossValidator
        class and contains scores, fold index, time-to-event data, and
        RED_ID."""
        # EXTRACT PREDICTIONS (LIST OF SCORES IN LONG FORMAT)
        scores = [df.set_index("RED_ID")["ypred"] for df in result_dataframes]
        # EXTRACT OTHER COLUMNS NEEDED FOR GROUPING AND CALCULATIONS
        # This info does not depend on experiment parameter, so just use first
        info_df = result_dataframes[0].set_index("RED_ID")[["time_to_event", "event_observed", "cv_idx"]]
        # GROUP PREDICTIONS IN LONG DATAFRAME
        index_names = ["n_genes", "RED_ID"]
        predictions_df = pd.concat(scores, keys=self.n_genes_selected_list, names=index_names)   # fmt: skip
        # MERGE SCORES WITH OTHER COLUMNS INTO SINGLE DATAFRAME
        predictions_df = predictions_df.reset_index(level="n_genes")
        predictions_df = predictions_df.merge(info_df, on="RED_ID", how="left")
        # GROUP DATA AND AVERAGE SCORES ACROSS SAMPLES
        subset_df = predictions_df.query(f"n_genes>={ensemble_range[0]} & n_genes<={ensemble_range[1]}")   # fmt: skip
        ensemble_predictions_df = subset_df.groupby(["RED_ID", "cv_idx"])[["ypred", "time_to_event", "event_observed"]].mean()   # fmt: skip
        return ensemble_predictions_df

    def get_pooled_ensemble_predictions(
        self,
        cv_results_lists: list,
        ensemble_range: list | tuple | None = None,
    ) -> pd.DataFrame:
        if ensemble_range is None:
            ensemble_range = [10, 20]
        ensemble_prediction_dfs = [
            self.get_n_selected_ensemble_predictions(result_dataframes, ensemble_range)
            for result_dataframes in cv_results_lists
        ]
        df = pd.concat(
            ensemble_prediction_dfs,
            keys=range(len(ensemble_prediction_dfs)),
            names=["pipeline", "RED_ID", "cv_idx"],
        )
        return df.groupby(["RED_ID", "cv_idx"])[["ypred", "time_to_event", "event_observed"]].mean()

    def get_benchmark_ensemble_predictions(self, ensemble_range=[10, 20]) -> pd.DataFrame | None:
        labels = self._resolved_benchmark_ensemble_labels()
        if labels is None:
            return None
        cv_results_lists = [self._cv_results_for_pipeline_label(label) for label in labels]
        return self.get_pooled_ensemble_predictions(cv_results_lists, ensemble_range)

    def calc_auc_for_ensemble_predictions_df(self, ensemble_predictions_df):
        # CALCULATE AUC PER FOLD
        ensemble_auc_per_fold = ensemble_predictions_df.reset_index().groupby("cv_idx").apply(lambda g: calc_auc_for_predicting_relapse_from_df(g, g["ypred"]), include_groups=False)  # fmt: skip
        return ensemble_auc_per_fold.values

    def calc_auc_of_ensemble_models(self, ensemble_range=[10, 20]):
        auc_vals_benchmarks = []
        for label in self.benchmark_labels:
            ensemble_scores_df = self.get_n_selected_ensemble_predictions(self.cv_results[label], ensemble_range)
            auc_vals_benchmarks.append(self.calc_auc_for_ensemble_predictions_df(ensemble_scores_df))

        ensemble_predictions_candidate_df = self.get_n_selected_ensemble_predictions(
            self.cv_results[self.candidate_label], ensemble_range
        )
        auc_vals_candidate = self.calc_auc_for_ensemble_predictions_df(ensemble_predictions_candidate_df)

        full_ensemble_rows = [auc_vals_candidate]
        self.pipeline_ensemble_auc_per_fold = {}
        for group_name, group_labels in self._resolved_pipeline_ensemble_groups().items():
            group_auc = self._pipeline_ensemble_auc_per_fold(group_labels, ensemble_range)
            self.pipeline_ensemble_auc_per_fold[group_name] = group_auc
            full_ensemble_rows.append(group_auc)

        # STACK
        self.ensemble_auc_per_fold_and_pipeline = np.vstack([auc_vals_candidate, auc_vals_benchmarks])
        self.ensemble_auc_per_pipeline = self.ensemble_auc_per_fold_and_pipeline.mean(axis=1)

        self.full_ensemble_auc_per_fold_and_pipeline = np.vstack(full_ensemble_rows)
        self.full_ensemble_auc_per_pipeline = self.full_ensemble_auc_per_fold_and_pipeline.mean(axis=1)

    def plot_barplot_for_ensemble_predictions(
        self, ensemble_range=[10, 20], tick_rotation=45, plot_mean_of_benchmarks=True
    ):
        # --- ENSEMBLE PREDICTIONS
        self.calc_auc_of_ensemble_models(ensemble_range)

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        barplot__cv_performance_comparison_of_pipelines(
            values=self.ensemble_auc_per_fold_and_pipeline,
            pipeline_labels=[self.candidate_label] + self.benchmark_labels,
            pipeline_colors=[self.color_candidate] + self.colors_benchmarks,
            title=f"Cross validation comparison: {METRIC_N_SELECTED_ENSEMBLE}",
            fig=fig,
            axs=axs,
            tick_rotation=tick_rotation,
            plot_mean_of_benchmarks=plot_mean_of_benchmarks,
        )
        axs[0].set_ylabel("AUC")
        axs[1].set_ylabel("AUC")
        fig.suptitle("Cross validation (n_selected ensemble)")
        self.axes_barplot_ensemble = axs
        self.fig_barplot_ensemble = fig

    def get_ensemble_summary_dfs(self, ensemble_range=[10, 20]):
        """Return two DataFrames summarising ensemble and average-AUC results.

        ``ensemble_df``  – per-fold AUC when predictions are *ensembled* (averaged
        across n-genes in ``ensemble_range``).

        ``average_df``   – per-fold AUC when the AUC values themselves are *averaged*
        across n-genes in ``ensemble_range`` (i.e. mean of individual-model AUCs).
        """
        self.calc_auc_of_ensemble_models(ensemble_range)
        # --- ENSEMBLE AUC (ensemble predictions, then compute AUC)
        ensemble_row_labels = [self.candidate_label]
        ensemble_row_labels.extend(self._resolved_pipeline_ensemble_groups().keys())
        values = np.vstack((self.full_ensemble_auc_per_fold_and_pipeline, self.ensemble_auc_per_fold_and_pipeline[1:]))
        labels = ensemble_row_labels + self.benchmark_labels
        ensemble_df = pd.DataFrame(values * 100, index=labels)
        ensemble_df.columns = [f"fold{i+1}" for i in range(ensemble_df.shape[1])]

        # --- AVERAGE AUC (compute AUC per n-genes, then average)
        sample_n = [int(n) for n in self.n_genes_selected_list if ensemble_range[0] <= n <= ensemble_range[1]]
        candidate_avg = self.metric_dfs_2yr[self.candidate_label].loc[sample_n].mean().values
        benchmark_avgs = [self.metric_dfs_2yr[l].loc[sample_n].mean().values for l in self.benchmark_labels]
        avg_values = np.vstack([candidate_avg] + benchmark_avgs)
        avg_labels = [self.candidate_label] + self.benchmark_labels
        average_df = pd.DataFrame(avg_values * 100, index=avg_labels)
        average_df.columns = [f"fold{i+1}" for i in range(average_df.shape[1])]

        self.ensemble_summary_df = ensemble_df
        self.average_auc_summary_df = average_df
        return ensemble_df, average_df

    def _barplot_colors_for_full_ensemble(self, n_labels: int) -> list[str]:
        ensemble_group_colors = ["k"] * len(self._resolved_pipeline_ensemble_groups())
        colors = [self.color_candidate] + ensemble_group_colors + list(self.colors_benchmarks)
        if len(colors) < n_labels:
            colors = colors + ["k"] * (n_labels - len(colors))
        return colors[:n_labels]

    def plot_barplot_for_full_ensemble_models(self, ensemble_range=[10, 20], tick_rotation=45):
        """Barplot of the ensemble over ``ensemble_range`` (10–20 genes in the paper)."""
        ensemble_df, _average_df = self.get_ensemble_summary_dfs(ensemble_range)
        values = ensemble_df.values / 100
        labels = ensemble_df.index.tolist()
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        barplot__cv_performance_comparison_of_pipelines(
            values=values,
            pipeline_labels=labels,
            pipeline_colors=self._barplot_colors_for_full_ensemble(len(labels)),
            title=f"Cross validation comparison: {METRIC_N_SELECTED_ENSEMBLE}",
            plot_mean_of_benchmarks=False,
            tick_rotation=tick_rotation,
            fig=fig,
            axs=axs,
        )
        axs[0].set_ylabel("AUC")
        axs[1].legend(fontsize=7)
        axs[1].set_ylabel("AUC")
        fig.suptitle("Cross validation (n_selected + benchmark ensemble)", fontsize=15)
        self.axes_barplot_full_ensemble = axs
        self.fig_barplot_full_ensemble = fig

    def summarize_ensemble_comparison(
        self,
        ensemble_range: list | tuple | None = None,
    ) -> PipelineComparisonSummary:
        if ensemble_range is None:
            ensemble_range = [10, 20]
        self.calc_auc_of_ensemble_models(ensemble_range)
        candidate = FoldMetricStats(
            self.candidate_label,
            self.full_ensemble_auc_per_fold_and_pipeline[0],
        )
        pipeline_ensembles = {
            name: FoldMetricStats(name, auc_per_fold)
            for name, auc_per_fold in self.pipeline_ensemble_auc_per_fold.items()
        }
        benchmark_ensemble = pipeline_ensembles.get(self._benchmark_ensemble_display_label())
        if benchmark_ensemble is None and len(pipeline_ensembles) == 1:
            benchmark_ensemble = next(iter(pipeline_ensembles.values()))
        benchmarks = [
            FoldMetricStats(label, self.ensemble_auc_per_fold_and_pipeline[i + 1])
            for i, label in enumerate(self.benchmark_labels)
        ]
        return PipelineComparisonSummary(
            candidate=candidate,
            benchmarks=benchmarks,
            benchmark_ensemble=benchmark_ensemble,
            pipeline_ensembles=pipeline_ensembles,
            ensemble_range=(ensemble_range[0], ensemble_range[1]),
            metric_description=METRIC_N_SELECTED_ENSEMBLE,
        )

    def summarize_from_fold_values(
        self,
        values: np.ndarray,
        benchmark_labels: list | None = None,
    ) -> PipelineComparisonSummary:
        labels = benchmark_labels if benchmark_labels is not None else self.benchmark_labels
        candidate = FoldMetricStats(self.candidate_label, values[0])
        benchmarks = [FoldMetricStats(label, values[i + 1]) for i, label in enumerate(labels)]
        return PipelineComparisonSummary(
            candidate=candidate,
            benchmarks=benchmarks,
            benchmark_ensemble=None,
            ensemble_range=None,
            metric_description=METRIC_MEAN_AUC_N_SELECTED,
        )

    def print_comparison_summary(self, summary: PipelineComparisonSummary) -> None:
        if self.verbose <= 0:
            return

        if summary.ensemble_range is not None:
            lo, hi = summary.ensemble_range
            white_print(f"\n=== Comparison ({summary.metric_description}={lo}–{hi}) ===")
        else:
            white_print(f"\n=== Comparison ({summary.metric_description}) ===")

        label_width = max(len(label) for label in _comparison_row_labels(summary))
        header = f"{'':<{label_width}}  {'AUC':>6}  {'p':>7}  {'std':>6}  range"
        white_print(header)

        green_print(_format_metric_row(f"Candidate  {summary.candidate.label}", summary.candidate, label_width))

        if summary.pipeline_ensembles:
            for name, stats in summary.pipeline_ensembles.items():
                orange_print(_format_metric_row(name, stats, label_width))
            if len(summary.pipeline_ensembles) == 2:
                ada_name, stiff_name = summary.pipeline_ensembles.keys()
                comp = PairedComparison(
                    summary.pipeline_ensembles[ada_name],
                    summary.pipeline_ensembles[stiff_name],
                )
                green_print(
                    _format_delta_row(
                        f"{ada_name} vs {stiff_name}",
                        comp.delta_mean,
                        comp.p_paired,
                        label_width,
                    )
                )
            elif summary.vs_benchmark_ensemble is not None:
                ens = summary.benchmark_ensemble
                comp = summary.vs_benchmark_ensemble
                green_print(_format_delta_row(f"vs {ens.label.lower()}", comp.delta_mean, comp.p_paired, label_width))
        elif summary.benchmark_ensemble is not None:
            ens = summary.benchmark_ensemble
            orange_print(_format_metric_row(ens.label, ens, label_width))
            comp = summary.vs_benchmark_ensemble
            green_print(_format_delta_row(f"vs {ens.label.lower()}", comp.delta_mean, comp.p_paired, label_width))

        cutoff_header = "\n--- Detailed results ---"
        white_print(cutoff_header)
        for benchmark in summary.benchmarks:
            orange_print(_format_metric_row(benchmark.label, benchmark, label_width))

        best = summary.best_benchmark
        worst = summary.worst_benchmark
        green_print(_format_metric_row(f"Best benchmark  {best.label}", best, label_width))
        green_print(_format_metric_row(f"Worst benchmark  {worst.label}", worst, label_width))
        orange_print(_format_mean_auc_row("Benchmark average", summary.benchmark_mean, label_width))

        comp_best = summary.vs_best_benchmark
        green_print(_format_delta_row("vs Best benchmark", comp_best.delta_mean, comp_best.p_paired, label_width))

    def compare_candidate_vs_benchmark_stats(
        self,
        values=None,
        labels=None,
        ensemble_range=None,
    ) -> PipelineComparisonSummary | None:
        """Print AdaPipe against the benchmarks on the ensemble AUC."""
        if self.verbose <= 0 and values is None:
            return None
        if values is None:
            summary = self.summarize_ensemble_comparison(ensemble_range)
        else:
            summary = self.summarize_from_fold_values(values, labels)
        self.print_comparison_summary(summary)
        return summary


@dataclass
class DataFactoryUpdater:
    """Change settings on a factory that is reused across discovery analyses."""

    datafactory: DataFactory

    def update_test_histology(self, histology_restriction: Literal["LUAD", "LUSC", "NSCLC"] = "LUSC"):
        """Keep only LUAD or LUSC patients in the test fold. NSCLC clears the restriction."""
        teal_print(f"Histology restriction: {histology_restriction}")
        if histology_restriction == "LUAD":
            self.datafactory.red_id_restriction_test_sets = DataBuilder.get_red_id_luad_data()
        elif histology_restriction == "LUSC":
            self.datafactory.red_id_restriction_test_sets = DataBuilder.get_red_id_lusc_data()
        elif histology_restriction == "NSCLC":
            self.datafactory.red_id_restriction_test_sets = None

    def update_gene_pool(self, gene_pool_restriction: Literal["TCGA LUAD", "TCGA NSCLC", "TSO500"] = None) -> list:
        """Limit gene selection to a TCGA pool, or to TSO500 when no pool is named."""
        if gene_pool_restriction == "TCGA LUAD":
            gene_selection_pool = DataBuilder.get_genepool_for_TCGA_validation("LUAD")
            red_print(f"Gene selection pool: TCGA LUAD (>=5 oncogenic mutas, n={len(gene_selection_pool)})")   # fmt: skip
        elif gene_pool_restriction == "TCGA NSCLC":
            gene_selection_pool = DataBuilder.get_genepool_for_TCGA_validation("NSCLC")
            red_print(f"Gene selection pool: TCGA NSCLC (>=5 oncogenic mutas, n={len(gene_selection_pool)})")   # fmt: skip
        else:
            red_print("Gene selection pool: TSO500")
            gene_selection_pool = DataBuilder.get_tso500_gene_list()

        self.datafactory.gene_selector.gene_selection_pool = gene_selection_pool

    def update_oncofilter_strategy(
        self,
        new_oncofilter_strategy: Literal[
            "pick_on",
            "pick_off",
            "highest_neglogp",
            "lowest_p_softcut",
        ] = "highest_neglogp",
    ):
        """Re-pick the oncogenicity side and rebuild the gene summary."""
        self.datafactory.grid_cutoff_selector.oncofilter_selection = new_oncofilter_strategy
        self.datafactory.update_oncofilter_strategy()

    def update_gene_selector(
        self,
        n_genes: int = 15,
        gene_rater: str = "p_softcut",
        gene_selection_pool: list = None,
        minimal_criteria: callable = None,
    ):
        """Replace the selector with top-k ranking by ``gene_rater``."""
        if minimal_criteria is None:
            minimal_criteria = lambda row: row["p"] < 0.05
        red_print(f"Gene selection: TopK (k={n_genes}, rater={gene_rater})")
        self.datafactory.gene_selector = TopKBestGenes(
            k=n_genes,
            gene_rater=gene_rater,
            minimal_criteria=minimal_criteria,
            gene_selection_pool=gene_selection_pool,
        )


def build_paper1_discovery_fluid_datafactory(variants_filtered_df) -> DataFactory:
    """AdaPipe (fluid) pipeline with Paper 1 discovery-cohort default settings."""
    green_print("Loading default fluid datafactory")
    datafactory_fluid = DataFactory(
        variants_filtered_df=variants_filtered_df,
        grid_cutoff_selector=MinPvalOverEligible(oncofilter_selection="highest_neglogp"),
        vaf_cutoff_selector=VAFCutoffSelector(vaf_cutoff_type="grid_optimized"),
        path_varsearcher_dumpdir=Paths().crossval_varselection_dump_flexigrid_norway_dir,
        gene_selector=TopKBestGenes(k=20, gene_rater="p_softcut", minimal_criteria=lambda row: row["p"] < 0.05),
        path_crossval_split=Paths().cv_splits_norway_path,
    )
    datafactory_fluid.load_varsearchers()
    return datafactory_fluid


def build_paper1_discovery_stiff_datafactories(variants_filtered_df) -> ListOfDataFactories:
    """Fixed-cutoff benchmark pipelines (3–6% VAF) for Paper 1 discovery cohort."""
    return load_datafactories_for_various_fixedcuts(
        variants_filtered_df,
        oncofilter_selection="pick_off",
        path_varsearcher_dumpdir=Paths().crossval_varselection_dump_fixedgrid_norway_dir,
        path_crossval_split=Paths().cv_splits_norway_path,
        cutoffs=[0.03, 0.04, 0.05, 0.06],
    )


def reset_defaults(
    datafactory_fluid: DataFactory,
    stiff_datafactories: ListOfDataFactories | None = None,
) -> None:
    """Restore shallow Paper 1 discovery defaults on shared DataFactories (no
    varsearcher reload).

    Idempotent on a freshly built fluid factory: skips oncofilter re-merge when
    the strategy is already ``highest_neglogp`` (re-merge can shift tied genes).
    """
    target_onco = "highest_neglogp"
    if datafactory_fluid.grid_cutoff_selector.oncofilter_selection != target_onco:
        DataFactoryUpdater(datafactory_fluid).update_oncofilter_strategy(target_onco)

    gs = datafactory_fluid.gene_selector
    if isinstance(gs, TopKBestGenes):
        gs.k = 20
        gs.gene_rater = lambda row: row["p_softcut"]
        gs.minimal_criteria = lambda row: row["p"] < 0.05
        gs.gene_selection_pool = None
    else:
        datafactory_fluid.gene_selector = TopKBestGenes(
            k=20,
            gene_rater="p_softcut",
            minimal_criteria=lambda row: row["p"] < 0.05,
        )

    datafactory_fluid.feature_type = "MUTAS"
    datafactory_fluid.design_matrix_preparator = design_matrixes.Genomic()
    datafactory_fluid.red_id_restriction_test_sets = None
    datafactory_fluid.red_id_restriction = None
    datafactory_fluid.model_factory = None
    datafactory_fluid.verbose = 0
    datafactory_fluid.info = DataFactory.VarselectionInfo()

    if stiff_datafactories is not None:
        for datafactory in stiff_datafactories.datafactories:
            datafactory.gene_selector.gene_selection_pool = None
            if hasattr(datafactory.gene_selector, "k"):
                datafactory.gene_selector.k = 20
            datafactory.red_id_restriction_test_sets = None


def load_datafactories_for_various_fixedcuts(
    variants_filtered_df,
    data_builder: DataBuilder = None,
    path_varsearcher_dumpdir=None,
    path_crossval_split=None,
    cutoffs=None,
    oncofilter_selection="pick_off",
) -> ListOfDataFactories:
    if path_varsearcher_dumpdir is None:
        path_varsearcher_dumpdir = Paths().crossval_varselection_dump_fixedgrid_norway_dir
    if path_crossval_split is None:
        path_crossval_split = Paths().cv_splits_norway_path
    if cutoffs is None:
        cutoffs = [0.03, 0.04, 0.05, 0.06]

    stiff_datafactories = ListOfDataFactories()

    for cutoff in cutoffs:
        green_print(f"cutoff: {cutoff:.0%}")
        # Note: cutoff=cutoff freezes the current loop value; otherwise the lambda would capture the variable and all instances would use the final cutoff after the loop (late binding).
        gene_selector = TopKBestGenes(
            gene_rater="p", minimal_criteria=lambda r, cutoff=cutoff: r["vaf_cutoff"] > cutoff - 0.005
        )
        datafactory_i = DataFactory(
            variants_filtered_df=variants_filtered_df,
            data_builder=data_builder,
            path_varsearcher_dumpdir=path_varsearcher_dumpdir,
            grid_cutoff_selector=ManuallySelectedCutoff(
                manually_chosen_cutoff=cutoff,
                oncofilter_selection=oncofilter_selection,
            ),
            load_varsearchers_at_init=True,
            gene_selector=gene_selector,
            path_crossval_split=path_crossval_split,
        )
        stiff_datafactories.add(f"Cutoff: {cutoff:.0%}", datafactory_i, cutoff=cutoff)

    return stiff_datafactories

