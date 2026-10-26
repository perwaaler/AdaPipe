"""Helper code for analyzing models fitted in the discovery cohort it in
TCGA."""

from __future__ import annotations
from utils.prelude import *
from utils.paths import Paths
from utils.prepare_data.variant_data import GeneLevelFeatures
from dataclasses import dataclass
from dataclasses import field
import seaborn as sns
from matplotlib.ticker import PercentFormatter
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from utils.models.models import CPHModel
from utils.analysis_and_calculations import ROCAnalysis
from utils.TCGA.prepare_data import PrepareTCGAClinData
from utils.plotting import plot_kaplan_meier_curves
from utils.plotting import sns_colors
from utils.plotting import get_risk_subsets
from typing import Literal
from lifelines.utils import concordance_index
from pathlib import Path
from utils.build_models_for_TCGAvalidation import map_experiment_settings_to_label
from utils.build_models_for_TCGAvalidation import EXPERIMENT_DICTS
from utils.analysis_and_calculations import get_rows_where_nyear_relapse_is_defined
from sklearn.metrics import roc_auc_score


@dataclass
class MainTCGA_Analysis:
    """An instance of this class is responsible for a analyzing a particular
    model/experiment, such as 'Train AdaPipe on NSCLC and TTR using AdaPipe;
    evaluate on TCGA LUAD'."""

    tcga_filtered_df: pd.DataFrame
    experiment: Literal["E1: train on NSCLC", "E2: train on LUAD", "E3: train on NSCLC and DSS"] = None
    sub_experiment: Literal["LUADeval", "NSCLCeval", "LUSCeval"] = None
    pipeline_type: Literal["StiffCutGS", "FluidCutGS"] = "FluidCutGS"  # AdaPipe or benchmark
    stiff_cutoff: float = 0.05  # Only used when fixed-cutoff model is used
    publication_mode: bool = False
    drop_stage4: bool = True

    def __post_init__(self):
        """Load the experiment settings and build the analyser."""
        white_print(f"SETTINGS FOR EXPERIMENT: '{self.experiment}' (SUB EXPERIMENT: '{self.sub_experiment}')")
        self.experiment_config = EXPERIMENT_DICTS[self.experiment]
        self.subexperiment_config = self.experiment_config[self.sub_experiment]

        # Select fluid or stiff paths based on pipeline_type
        if self.pipeline_type == "FluidCutGS":
            path_hp = self.subexperiment_config["hyperparam_dump_path__fluidGS"]
            stiff_cutoff = None
        else:
            hp_key = self.subexperiment_config.get("hyperparam_dump_path__stiffGS_lookup", {})
            path_hp = hp_key.get(self.stiff_cutoff, self.subexperiment_config["hyperparam_dump_path__stiffGS"])
            stiff_cutoff = self.stiff_cutoff

        self.analyzer = TCGANorway_MutaModelAnalyzer(
            self.tcga_filtered_df,
            train_histology=self.experiment_config["train_histology"],
            eval_histology=self.subexperiment_config["eval_histology"],
            path_hyperparams=path_hp,
            stiff_cutoff=stiff_cutoff,
            train_endpoint=self.experiment_config["endpoint"],
        )

    def model_preparation(self):
        """Apply the exported discovery model to TCGA."""
        TCGA_features_df, selected_genes_final = self.analyzer.build_tcga_features_from_export()
        coef = pd.Series(self.analyzer.model_artifact["coef"], dtype=float).loc[selected_genes_final]
        TCGA_features_df["pooled_muta_feature"] = self.analyzer.get_risk_scores(TCGA_features_df, coef, selected_genes_final)   # fmt: skip

        self.TCGA_features_df = TCGA_features_df
        self.percentile_cutoffs = self.analyzer.model_artifact["percentile_cutoffs"]
        self.coefficients_from_norway_fit = coef

        self._genes_selected = [x for x in TCGA_features_df.columns if x not in ["RED_ID", "pooled_muta_feature"]]
        self._n_selected = len(self._genes_selected)
        self._oncofilter_setting = self.analyzer.oncofilter_setting
        self._penalizer = self.analyzer.penalizer
        self.analyzer.plot_fitparameters(coef, selected_genes_final)

    def incremental_gene_auc(self, endpoint: Literal["DSS", "OS", "PFI", "PFS"] = "DSS", n_years=2):
        """AUC in TCGA as genes are added by coefficient size (Figure S13)."""
        coef = self.coefficients_from_norway_fit
        genes_ranked = coef.abs().sort_values(ascending=False).index.tolist()

        # Get TCGA clinical data for AUC evaluation
        histology = self.analyzer.eval_histology
        clinical_tcga_df = PrepareTCGAClinData(
            endpoint=endpoint,
            infer_eventobserved_to_censored_if_timetoevent_exists=True,
            histology=histology,
        ).get_clinical_data()

        features_df = self.TCGA_features_df.set_index("RED_ID")
        common_ids = features_df.index.intersection(clinical_tcga_df["RED_ID"])
        clinical_tcga_df = clinical_tcga_df[clinical_tcga_df["RED_ID"].isin(common_ids)].set_index("RED_ID")
        features_df = features_df.loc[clinical_tcga_df.index]
        clinical_tcga_df = clinical_tcga_df.dropna(subset=["pstage"])
        features_df = features_df.loc[clinical_tcga_df.index]

        time = clinical_tcga_df["time_to_event"]
        event = clinical_tcga_df["event_observed"]
        defined = get_rows_where_nyear_relapse_is_defined(time, event, n_years=n_years)
        time_def, event_def = time[defined], event[defined]
        target = time_def <= n_years * 365

        results = []
        for k, gene in enumerate(genes_ranked, 1):
            genes_k = genes_ranked[:k]
            scores = features_df[genes_k].dot(coef[genes_k])
            scores_def = scores[defined]
            auc = roc_auc_score(target, scores_def)
            results.append({"gene": gene, "n_genes": k, "auc": auc})

        results_df = pd.DataFrame(results)

        # Plot
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(results_df["n_genes"], results_df["auc"], "-o", markersize=4)
        ax.set_xlabel("Number of genes (ranked by |coefficient|)")
        ax.set_ylabel(f"{n_years}-year AUC")
        ax.set_title(f"Incremental gene contribution to TCGA {endpoint} prediction")
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        y_min, y_max = 0.50, max(0.68, np.ceil(results_df["auc"].max() / 0.02) * 0.02)
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(np.arange(y_min, y_max + 0.001, 0.02))
        ax.set_xticks(results_df["n_genes"])
        ax.set_xticklabels(results_df["gene"], rotation=90, fontsize=8)
        ax.grid(alpha=0.3, linestyle="--")
        fig.tight_layout()
        self.fig_incremental_auc = fig

        green_print(f"The 11 first genes are: {", ".join(results_df["gene"].iloc[:11])}, which achieve an AUC of {results_df.loc[10, "auc"]:.1%}")   # fmt: skip

        return results_df

    def load_varsearchers_and_hyperparams(self):
        """Load hyperparameters and the exported discovery model artifact."""
        self.analyzer.load_hyperparams()
        self.analyzer.apply_hyperparams()
        self.analyzer.load_exported_model()

    def get_performance_evaluation_metrics(self, endpoint: Literal["DSS", "OS", "PFI", "PFS"] = "DSS", n_years=2):
        """ROC, Kaplan–Meier, and Cox statistics for one TCGA endpoint."""
        # --- ROC CURVES + CPH REGRESSION ANALYSIS
        histology = self.analyzer.eval_histology
        # --- ANALYZE WITH CPH MODEL + ROC ANALYSIS
        self.roc_output = plot_roc_curves_for_norway_features_on_tcga(
            TCGA_features_df=self.TCGA_features_df,
            n_years=n_years,
            endpoint=endpoint,
            histology=histology,
            pathomuta_coeff=self.analyzer.pathomuta_coeff,
        )
        ax = self.roc_output.ax
        ax.set_title(f"Predictive power of Norway model in TCGA {endpoint} ({n_years}-year AUC)")
        if not self.publication_mode:
            txt = "\n".join(
                [
                    f"Norway subset (developement set): {self.analyzer.train_histology}",
                    f"TCGA subset (test set): {histology}",
                ]
            )
            ax.text(0.0, -0.15, txt, ha="left", va="top", transform=ax.transAxes, alpha=0.5)
        # --- KAPLAN MEIER
        self.km_analyzer = KaplanMeierAnalysisTCGA(
            pipeline_tag=self.analyzer.experiment_label,
            eval_histology=self.analyzer.eval_histology,
            publication_mode=self.publication_mode,
        )
        self.km_analyzer.run_kaplan_meier_analysis(
            TCGA_features_df=self.TCGA_features_df,
            endpoint=endpoint,
            percentile_cutoffs=self.percentile_cutoffs,
        )
        self.km_analyzer.savefig(pipeline_type=self.pipeline_type)
        # --- HAZARD RATIO: LOWEST VS HIGHEST RISK GROUP
        modelfit_df = self.get_modelfit_df(self.TCGA_features_df, endpoint=endpoint, histology=histology)
        self.hr_results_df = compute_hazard_ratio_between_risk_groups(modelfit_df, self.percentile_cutoffs)
        white_print("Hazard ratio: lowest vs highest risk group")
        display(self.hr_results_df)

        # --- SUMMARY OF REPORTABLE STATISTICS
        n_obs = len(modelfit_df)
        n_events = int(modelfit_df["event_observed"].sum())
        median_tte_months = modelfit_df["time_to_event"].median() / 30.44
        defined_2yr = get_rows_where_nyear_relapse_is_defined(
            modelfit_df["time_to_event"], modelfit_df["event_observed"], n_years=n_years
        )
        n_2yr_defined = int(defined_2yr.sum())
        n_2yr_events = int((modelfit_df.loc[defined_2yr, "time_to_event"] <= n_years * 365).sum())
        roc = self.roc_output
        hr_row = self.hr_results_df.iloc[0]
        white_print("\n" + "=" * 60)
        white_print("REPORTABLE STATISTICS SUMMARY")
        white_print("=" * 60)
        white_print(f"Experiment:        {self.analyzer.experiment_label}")
        white_print(f"Pipeline:          {self.pipeline_type}")
        white_print(f"Endpoint:          {endpoint}")
        white_print(f"Train histology:   {self.analyzer.train_histology}")
        white_print(f"Eval histology:    {histology}")
        white_print(f"N observations:    {n_obs}")
        white_print(f"N events:          {n_events}")
        white_print(f"Median TTE:        {median_tte_months:.1f} months")
        white_print(f"N {n_years}yr-defined:  {n_2yr_defined}")
        white_print(f"N {n_years}yr-events:   {n_2yr_events}")
        white_print(f"N genes selected:  {self._n_selected}")
        green_print(f"AUC (muta-score):  {roc.auc:.1%}")
        green_print(f"AUC (pStage):      {roc.auc_pstage:.1%}")
        green_print(f"AUC (pStage+muta; TNM-I-fitted): {roc.auc_pathgenomic_honest:.1%}")
        green_print(f"C-index (pStage+muta; TNM-I-fitted): {roc.cindex_pathgenomic_honest:.1%}")
        green_print(f"C-index (muta):    {roc.cindex:.1%}")
        green_print(f"p (unadjusted):    {roc.pval:.4f}")
        green_print(f"p (stage-adjusted):{roc.pval_adjusted:.4f}")
        if roc.pval_stage23 is not None:
            green_print(f"p (stage 2-3 only):{roc.pval_stage23:.4f}")
            green_print(f"C-idx (stage 2-3): {roc.cindex_stage23:.1%}")
        green_print(f"HR (low vs high):  {hr_row['exp(coef)']:.2f} (p={hr_row['p']:.4f})")
        white_print("=" * 60)

    def compare_auc_using_bootstrap(
        self,
        other_scores: pd.Series | MainTCGA_Analysis,
        endpoint: str = "DSS",
        n_years: int = 2,
        n_bootstrap: int = 1000,
        random_state: int = 42,
    ) -> dict:
        """Paired bootstrap test of the AUC difference against another model."""
        if isinstance(other_scores, MainTCGA_Analysis):
            other_scores = other_scores.TCGA_features_df.set_index("RED_ID")["pooled_muta_feature"]
        histology = self.analyzer.eval_histology
        clinical_df = PrepareTCGAClinData(endpoint=endpoint, histology=histology, verbose=0).get_clinical_data()
        self_scores = self.TCGA_features_df.set_index("RED_ID")["pooled_muta_feature"]
        common_ids = self_scores.index.intersection(other_scores.index).intersection(clinical_df["RED_ID"])
        clinical_df = clinical_df[clinical_df["RED_ID"].isin(common_ids)].set_index("RED_ID")

        defined = get_rows_where_nyear_relapse_is_defined(
            clinical_df["time_to_event"], clinical_df["event_observed"], n_years=n_years
        )
        clinical_df = clinical_df[defined]
        self_scores = self_scores.loc[clinical_df.index].values
        other_scores = other_scores.loc[clinical_df.index].values
        y_true = (clinical_df["time_to_event"].values <= 365 * n_years).astype(int)

        rng = np.random.RandomState(random_state)
        auc_self_full = roc_auc_score(y_true, self_scores)
        auc_other_full = roc_auc_score(y_true, other_scores)
        observed_diff = auc_self_full - auc_other_full

        idx_pos = np.where(y_true == 1)[0]
        idx_neg = np.where(y_true == 0)[0]

        boot_diffs = np.empty(n_bootstrap)
        boot_aucs_self = np.empty(n_bootstrap)
        boot_aucs_other = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            boot_pos = rng.choice(idx_pos, size=len(idx_pos), replace=True)
            boot_neg = rng.choice(idx_neg, size=len(idx_neg), replace=True)
            idx = np.concatenate([boot_pos, boot_neg])
            auc_s = roc_auc_score(y_true[idx], self_scores[idx])
            auc_o = roc_auc_score(y_true[idx], other_scores[idx])
            boot_aucs_self[i] = auc_s
            boot_aucs_other[i] = auc_o
            boot_diffs[i] = auc_s - auc_o

        # Two-sided p-value: proportion of bootstrap diffs on the wrong side of zero
        if observed_diff >= 0:
            p_value = (boot_diffs <= 0).mean() * 2
        else:
            p_value = (boot_diffs >= 0).mean() * 2
        p_value = min(p_value, 1.0)

        ci_lower, ci_upper = np.percentile(boot_diffs, [2.5, 97.5])

        green_print(f"Bootstrap AUC comparison ({n_bootstrap} iterations, {n_years}-year {endpoint})")
        green_print(
            f"Self AUC:  {auc_self_full:.3f} (bootstrap mean: {boot_aucs_self.mean():.3f} ± {boot_aucs_self.std():.3f})"
        )
        green_print(
            f"Other AUC: {auc_other_full:.3f} (bootstrap mean: {boot_aucs_other.mean():.3f} ± {boot_aucs_other.std():.3f})"
        )
        green_print(f"Δ AUC: {observed_diff:.3f} (95% CI: [{ci_lower:.3f}, {ci_upper:.3f}], p={p_value:.4f})")

        return {
            "auc_self": auc_self_full,
            "auc_other": auc_other_full,
            "delta_auc": observed_diff,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "p_value": p_value,
            "boot_diffs": boot_diffs,
        }

    def get_modelfit_df(self, TCGA_features_df, endpoint="DSS", histology="LUAD"):
        """Merge TCGA outcome data with the mutation risk score."""
        clinical_tcga_df = PrepareTCGAClinData(endpoint=endpoint, histology=histology, verbose=0, drop_stage4=self.drop_stage4).get_clinical_data()   # fmt: skip
        modelfit_df = clinical_tcga_df[["RED_ID", "time_to_event", "event_observed"]].merge(TCGA_features_df, on="RED_ID", how="left").dropna()   # fmt: skip
        return modelfit_df

    def plot_heatmap_for_pStage_vs_mutaRisk_crosstable(
        self, endpoint="DSS", eval_histology: Literal["LUSC", "LUAD", "NSCLC"] = "LUAD", nyears=2
    ):
        """Two-year survival by mutation-risk group and pathological stage."""
        # --- PREPARATION
        clinical_tcga_df = PrepareTCGAClinData(
            endpoint=endpoint, histology=eval_histology, verbose=0
        ).get_clinical_data()
        modelfit_df = (
            self.TCGA_features_df[["RED_ID", "pooled_muta_feature"]]
            .merge(
                clinical_tcga_df[["RED_ID", "time_to_event", "event_observed", "pstage"]],
                on="RED_ID",
                how="left",
            )
            .dropna()
        )

        # GET ROWS WHERE FOLLOW UP IS SUFFICIENT TO DETERMINE SURVIVAL STATUS
        analyzable_rows = get_rows_where_nyear_relapse_is_defined(modelfit_df["time_to_event"], modelfit_df["event_observed"], n_years=nyears)   # fmt: skip
        df = modelfit_df[analyzable_rows].copy()
        orange_print(f"N ppl with known {nyears} DSS survival status: {analyzable_rows.sum()} ({analyzable_rows.mean():.1%})")   # fmt: skip
        # GET SURVIVAL STATUS AT N YEARS POST SURGERY
        red_id_alive_at_2yr = df.query(f"time_to_event>{nyears}*365")["RED_ID"]
        df["alive_2y"] = df["RED_ID"].isin(red_id_alive_at_2yr)
        # CUT RISK SCORES INTO RISK CATEGORIES
        risk_labels = ["low risk", "medium risk", "high risk"]
        risk_subsets = get_risk_subsets(
            df, self.percentile_cutoffs, score_name="pooled_muta_feature", cutoff_type="percentile"
        )
        for subset, label in zip(risk_subsets, risk_labels):
            df.loc[subset.index, "mutarisk_lvl"] = label
        df["mutarisk_lvl"] = pd.Categorical(df["mutarisk_lvl"], categories=risk_labels, ordered=True)

        heatmap_df = pd.crosstab(index=df["mutarisk_lvl"], columns=df["pstage"], values=df["alive_2y"], aggfunc="mean")   # fmt: skip
        alive_df = pd.crosstab(index=df["mutarisk_lvl"], columns=df["pstage"], values=df["alive_2y"], aggfunc="sum")   # fmt: skip
        total_df = pd.crosstab(index=df["mutarisk_lvl"], columns=df["pstage"], values=df["alive_2y"], aggfunc="count")   # fmt: skip
        # ANNOTATION DATAFRAMES (STRINGS DISPLAYED IN CELLS)
        anno_detailed_df = heatmap_df.map(lambda x: f"{x:.0%}").astype(str) + " (" + alive_df.astype(str) + "/" + total_df.astype(str) + ")"   # fmt: skip
        anno_percentage_df = heatmap_df.map(lambda x: f"{x:.0%}").astype(str)

        fig, ax = plt.subplots(figsize=(5, 5))
        hm = sns.heatmap(heatmap_df, cmap="viridis", fmt="", linewidths=0.5, annot=anno_detailed_df)
        ax.set_title(f"Patients with no DSS event after {nyears} years")
        ax.set_xticklabels(["stage I", "stage II", "stage III"])
        ax.set_xlabel("")
        ax.set_ylabel("Mutation risk score")
        ax.tick_params(labelsize=9)

        cbar = hm.collections[0].colorbar
        cbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        cbar.ax.tick_params(labelsize=8)

        self.heatmap_fig = fig
        self.heatmap_ax = ax


def compute_hazard_ratio_between_risk_groups(
    modelfit_df: pd.DataFrame,
    percentile_cutoffs: list[float],
    score_name: str = "pooled_muta_feature",
) -> pd.DataFrame:
    """Cox hazard ratio for the highest versus lowest mutation-risk group."""
    risk_subsets = get_risk_subsets(modelfit_df, cutoffs=percentile_cutoffs, score_name=score_name, cutoff_type="percentile")   # fmt: skip
    comparison_df = pd.concat(
        [
            risk_subsets[0].assign(risk_group=0),
            risk_subsets[-1].assign(risk_group=1),
        ]
    )
    cph = CPHModel(predictors=["risk_group"])
    cph.fit(comparison_df)
    return cph.results_df


@dataclass
class KaplanMeierAnalysisTCGA:
    """Kaplan–Meier curves for the mutation score and the combined score."""

    # *** PAPER 1 FIGURE ***
    fig: Figure = None
    ax: Axes = None
    figsize: tuple = field(default_factory=lambda: (12, 5))
    pipeline_tag: str = None
    publication_mode: bool = False

    eval_histology: str = "LUAD"

    def savefig(self, pipeline_type: Literal["FluidCutGS", "StiffCutGS"], fig_path=None):
        """Write the Kaplan–Meier figure."""
        if pipeline_type == "StiffCutGS" and fig_path is None:
            red_print("No default fig paths for Stiff pipelines; figure not saved")
        Paths.dump_fig(self.fig, Paths().get_tcga_kaplanmeier_fig_dump_path(self.pipeline_tag, self.endpoint), dpi=300)   # fmt: skip

    def run_kaplan_meier_analysis(self, TCGA_features_df, endpoint, percentile_cutoffs=[0.456, 0.736]):
        """Plot Kaplan–Meier curves for the mutation score and the combined score."""
        self.endpoint = endpoint
        clinical_tcga_df = PrepareTCGAClinData(
            endpoint=endpoint, verbose=0, histology=self.eval_histology
        ).get_clinical_data()
        modelfit_df = (
            TCGA_features_df[["RED_ID", "pooled_muta_feature"]]
            .merge(
                clinical_tcga_df[["RED_ID", "time_to_event", "event_observed", "pstage2", "pstage3"]],
                on="RED_ID",
                how="left",
            )
            .dropna()
        )
        # MEAN-VAF ADJUSTED FOR PSTAGE
        cph_adj = CPHModel()
        cph_adj.fit(modelfit_df)

        try:
            caption_type = "publication" if self.publication_mode else "detailed"
            self.fig, self.axs = plt.subplots(1, 2, figsize=self.figsize)
            ci_alpha = 0.1
            ax = self.axs[0]
            plot_kaplan_meier_curves(
                modelfit_df,
                score_name="pooled_muta_feature",
                cutoffs=percentile_cutoffs,
                ci_alpha=ci_alpha,
                fig=self.fig,
                ax=ax,
                caption_type=caption_type,
            )
            ax.set_title("Risk-stratified by mutation-score")

            # --- KAPLAN MEIER CURVES; FUSED MODEL
            ax = self.axs[1]
            scores = cph_adj.predict(modelfit_df)
            scores.name = "score"
            modelfit_adj_df = pd.concat([scores, modelfit_df["time_to_event"], modelfit_df["event_observed"]], axis=1)
            plot_kaplan_meier_curves(
                modelfit_adj_df,
                score_name="score",
                cutoffs=percentile_cutoffs,
                fig=self.fig,
                ax=ax,
                ci_alpha=ci_alpha,
                caption_type=caption_type,
                show_at_risk_labels=False,
            )
            ax.set_title("Risk-stratified by mutation-score + pStage")

        except Exception as e:
            red_print(f"Could not plot KM curves: {e}")   # fmt: skip


@dataclass
class TCGANorway_MutaModelAnalyzer:
    """Apply an exported discovery model to TCGA mutation data."""

    tcga_filtered_df: pd.DataFrame
    tcga_vaf_features_df: pd.DataFrame = None
    train_histology: Literal["NSCLC", "LUAD"] = "LUAD"
    eval_histology: Literal["NSCLC", "LUAD", "LUSC"] = "LUAD"
    train_endpoint: Literal["TTR", "DSS"] = "TTR"
    path_hyperparams: Path = None
    oncofilter_setting: Literal["pick_on", "pick_off", "highest_neglogp", "lowest_p_softcut"] = "pick_on"
    stiff_cutoff: float = None
    p_cutoff: float = 0.1
    penalizer: float = 1.0
    verbose: int = 1
    filterinfo_selected_final_df: pd.DataFrame = None
    model_artifact: dict = None
    pathomuta_coeff: pd.Series = None

    def __post_init__(self):
        """Restrict TCGA eval cohort and set experiment label."""
        self.experiment_label = map_experiment_settings_to_label(self.train_histology, self.eval_histology, self.train_endpoint)   # fmt: skip
        self.log(f"Evaluating model trained on TNM-I {self.train_histology} ({self.train_endpoint})...", "blue")

        if self.eval_histology == "LUAD":
            self.tcga_filtered_df = self.tcga_filtered_df.query("histology=='LUAD'").copy()
        elif self.eval_histology == "NSCLC":
            self.tcga_filtered_df = self.tcga_filtered_df.copy()

        orange_print(f"N samples in TCGA mutation data: {self.tcga_filtered_df["RED_ID"].nunique()}")

    def load_hyperparams(self):
        """Read the hyperparameter JSON for this experiment."""
        self.hyperparams = Paths.load_json(self.path_hyperparams)

    def load_exported_model(self):
        """Load coef/cutoffs/KM settings produced by ``build_models_for_TCGAvalidation``."""
        self.model_artifact = Paths.load_json(self.path_hyperparams)
        pm = self.model_artifact["pathomuta_coeff"]
        self.pathomuta_coeff = pd.Series(
            {
                "pooled_muta_feature": pm["mutascore"],
                "pstage2": pm["stage2"],
                "pstage3": pm["stage3"],
            },
            dtype=float,
        )

    def get_TCGA_VAF_df(self):
        """Max oncogenic VAF per gene in TCGA."""
        vaf_df = GeneLevelFeatures.from_variants_using_max(
            self.tcga_filtered_df,
            create_zerocols_for_genes_with_no_oncovariants=True,
        ).genomic_features_df
        return vaf_df

    def get_names_of_vaf_columns(self, filterinfo_selected_df):
        """Column name for each gene, with or without the oncogenicity filter."""
        get_representative_vaf_col = lambda row: f"{row.gene}_MAX_ONCO_VAF" if row["best_oncofiltersetting"] == "on" else f"{row.gene}_MAX_VAF"   # fmt: skip
        return filterinfo_selected_df.reset_index(names="gene").apply(get_representative_vaf_col, axis=1).to_list()

    def extract_selected_vaf_columns(self, tcga_vaf_features_df, filterinfo_selected_df):
        """Keep the VAF columns for the selected genes."""
        vaf_columns_to_extract = self.get_names_of_vaf_columns(filterinfo_selected_df)
        vaf_columns_found = [x for x in vaf_columns_to_extract if x in tcga_vaf_features_df.columns]
        vaf_columns_not_found = [x for x in vaf_columns_to_extract if x not in tcga_vaf_features_df.columns]
        self.log(f"VAF columns not found (no variants in TCGA; dropping from analysis): {vaf_columns_not_found}", c="orange")   # fmt: skip
        TCGA_VAF_selected_df = tcga_vaf_features_df[["RED_ID"] + vaf_columns_found].copy()
        return TCGA_VAF_selected_df

    def apply_hyperparams(self):
        """Copy n_selected, the penalty, and the oncogenicity filter from the saved search."""
        hp = self.hyperparams
        self.oncofilter_setting = hp["best_oncofilter_setting"]
        self.p_cutoff = hp["best_p_cutoff"]
        self.penalizer = hp["best_ridge_penalty"]
        self.selected_genes = list(hp["pvals_selected_genes"].keys())
        self.n_selected = hp["best_n_selected"]
        pipeline_label = "stiff" if self.stiff_cutoff is not None else "fluid"
        orange_print(f"--- Applying hyperparameters from Norway training ({pipeline_label}) ---")
        orange_print(f"P-cutoff: {self.p_cutoff:.1%} ({self.n_selected} selected) | OncoFilter: {self.oncofilter_setting} | Penalizer: {self.penalizer:.2f}")   # fmt: skip

    @staticmethod
    def apply_genewise_thresholds(X_df: pd.DataFrame, vaf_cutoffs: pd.Series) -> pd.DataFrame:
        """Mark a gene mutated when its VAF meets the discovery cutoff."""
        vaf_cutoffs_series = pd.Series(vaf_cutoffs.values, index=X_df.columns)
        X_df = X_df.ge(vaf_cutoffs_series)
        return X_df

    def build_tcga_features_from_export(self) -> tuple[pd.DataFrame, list[str]]:
        """Binary mutation matrix from exported gene cutoffs and oncofilter flags."""
        assert self.model_artifact is not None, "Call load_exported_model() first"
        tcga_vaf_features_df = self.get_TCGA_VAF_df()

        genes = self.model_artifact["genes"]
        filterinfo_selected_df = pd.DataFrame(
            {
                "vaf_cutoff": self.model_artifact["vaf_cutoff"],
                "best_oncofiltersetting": self.model_artifact["gene_oncofilter"],
            }
        ).loc[genes]

        TCGA_VAF_selected_df = self.extract_selected_vaf_columns(tcga_vaf_features_df, filterinfo_selected_df)
        TCGA_VAF_selected_df.columns = [x.split("_MAX")[0] for x in TCGA_VAF_selected_df.columns]
        selected_genes_final = TCGA_VAF_selected_df.drop(columns="RED_ID").columns.to_list()
        self.filterinfo_selected_final_df = filterinfo_selected_df.loc[selected_genes_final]
        vaf_cutoffs = self.filterinfo_selected_final_df["vaf_cutoff"]

        TCGA_features_df = TCGA_VAF_selected_df.copy()
        TCGA_features_df[selected_genes_final] = self.apply_genewise_thresholds(
            TCGA_VAF_selected_df[selected_genes_final], vaf_cutoffs
        )
        self.TCGA_features_df = TCGA_features_df
        green_print(f"N genes with TCGA VAF data: {len(selected_genes_final)} (of {len(genes)} in export)")
        return TCGA_features_df, selected_genes_final

    def get_risk_scores(self, TCGA_features_df, coef: pd.Series, selected_genes_final) -> pd.Series:
        """Linear predictor from the discovery-cohort coefficients."""
        return TCGA_features_df[selected_genes_final].dot(coef.loc[selected_genes_final]).astype(float)   # fmt: skip

    def plot_fitparameters(self, coefficients, genes):
        """Coefficient plot for the fitted multivariate model."""
        tick_labelsize = 9
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_axisbelow(True)
        colors = [sns_colors["blue"] if x < 0 else sns_colors["red"] for x in coefficients]
        sns.barplot(x=coefficients, y=genes, hue=genes, palette=colors, edgecolor="k", ax=ax)
        ax.tick_params(rotation=0, labelsize=tick_labelsize)
        ax.set_ylabel("")
        ax.set_title("Coefficients in multivariate model")
        ax.grid(alpha=0.3, linestyle="--")

        self.fig_coefplot = fig
        self.ax_coefplot = ax

    def log(self, txt: str, c="grey", verbose_lvl=1):
        """Print a message when verbose is high enough."""
        if self.verbose >= verbose_lvl:
            color_print(txt, c=c)


def plot_roc_curves_for_norway_features_on_tcga(
    TCGA_features_df,
    n_years=2,
    endpoint: Literal["OS", "PFI", "PFS", "DSS"] = "DSS",
    histology: Literal["LUAD", "LUSC", "NSCLC"] = "LUAD",
    pathomuta_coeff: pd.Series = None,
    ax=None,
):
    """ROC curves in TCGA for the mutation score, stage, and their combination."""

    @dataclass
    class ROCAnalysisOut:
        fig: Figure = None
        ax: Axes = None
        pval: float = None
        pval_adjusted: float = None
        pval_stage23: float = None
        auc: float = None
        auc_pstage: float = None
        auc_pathgenomic_honest: float = None
        cindex_pathgenomic_honest: float = None
        cindex: float = None
        cindex_stage23: float = None
        endpoint: str = None
        eval_histology: str = None
        gene_pvalues_df: pd.DataFrame = None

    out = ROCAnalysisOut(endpoint=endpoint, eval_histology=histology)

    clinical_tcga_df = PrepareTCGAClinData(endpoint=endpoint, histology=histology, verbose=0).get_clinical_data()
    modelfit_df = clinical_tcga_df[["RED_ID", "time_to_event", "event_observed"]].merge(TCGA_features_df, on="RED_ID", how="left").dropna()   # fmt: skip
    # --- FIT MODELS ---
    # UNIVARIATE: MUTATION-RISK SCORE
    cph_genomic = CPHModel(predictors=["pooled_muta_feature"])
    cph_genomic.fit(modelfit_df)
    white_print("Fit results for AdaPipe risk scores")
    display(cph_genomic.results_df)
    green_print(f"C-index: {cph_genomic.cindex:.1%}")
    out.cindex = cph_genomic.cindex
    out.pval = cph_genomic.results_df.loc["pooled_muta_feature", "p"]

    # --- PER-GENE UNIVARIATE COX ANALYSIS ---
    gene_cols = [c for c in TCGA_features_df.columns if c not in ("RED_ID", "pooled_muta_feature")]
    gene_results = []
    for gene in gene_cols:
        try:
            cph_gene = CPHModel(predictors=[gene])
            cph_gene.fit(modelfit_df)
            row = cph_gene.results_df.loc[gene]
            gene_results.append({"gene": gene, "coef": row["coef"], "exp(coef)": row["exp(coef)"], "p": row["p"]})
        except Exception:
            gene_results.append({"gene": gene, "coef": np.nan, "exp(coef)": np.nan, "p": np.nan})
    out.gene_pvalues_df = pd.DataFrame(gene_results).sort_values("p").reset_index(drop=True)
    white_print("Per-gene univariate Cox regression (sorted by p-value)")
    display(out.gene_pvalues_df)

    modelfit_df = (
        TCGA_features_df[["RED_ID", "pooled_muta_feature"]]
        .merge(
            clinical_tcga_df[["RED_ID", "time_to_event", "event_observed", "pstage2", "pstage3"]],
            on="RED_ID",
            how="left",
        )
        .dropna()
    )
    # MUTATION-RISK SCORE ADJUSTED FOR PSTAGE
    cph_adj = CPHModel()
    cph_adj.fit(modelfit_df)
    white_print("Fit results for pStage + mutas model")
    display(cph_adj.results_df)
    out.pval_adjusted = cph_adj.results_df.loc["pooled_muta_feature", "p"]
    cph_pstage = CPHModel(predictors=["pstage2", "pstage3"])
    cph_pstage.fit(modelfit_df)
    green_print(f"C-index (pStage): {cph_pstage.cindex:.1%}")

    # MUTATION-RISK SCORE IN STAGE 2-3 ONLY
    modelfit_stage23_df = modelfit_df.query("pstage2 == 1 or pstage3 == 1")
    cph_stage23 = CPHModel(predictors=["pooled_muta_feature"])
    cph_stage23.fit(modelfit_stage23_df)
    white_print(f"Fit results for mutas model in stage 2-3 only (n={len(modelfit_stage23_df)})")
    display(cph_stage23.results_df)
    out.pval_stage23 = cph_stage23.results_df.loc["pooled_muta_feature", "p"]
    out.cindex_stage23 = cph_stage23.cindex

    # --- PLOT ROC CURVES
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure
    roc_analyzer = ROCAnalysis(modelfit_df=modelfit_df)

    # MUTATION SCORE
    roc_analyzer.scores = cph_genomic.predict(modelfit_df)
    out.auc = roc_analyzer.get_auc(n_years=n_years)
    x, y = roc_analyzer.get_roc_curve(n_years=n_years)
    ax.plot(x, y, c=sns_colors["red"], label=f"AdaPipe: AUC={out.auc:.1%}")

    # PSTAGE
    roc_analyzer.scores = cph_pstage.predict(modelfit_df)
    out.auc_pstage = roc_analyzer.get_auc(n_years=n_years)
    x, y = roc_analyzer.get_roc_curve(n_years=n_years)
    ax.plot(x, y, c="k", label=f"pStage: AUC={out.auc_pstage:.1%}")

    # STAGE + MUTASCORE (combined patho-genomic model)
    pathomuta_cols = ["pooled_muta_feature", "pstage2", "pstage3"]
    if pathomuta_coeff is None:
        raise ValueError("pathomuta_coeff is required")
    combined_scores = (modelfit_df[pathomuta_cols] * pathomuta_coeff.values).sum(axis=1)

    roc_analyzer.scores = combined_scores
    combined_auc = roc_analyzer.get_auc(n_years=n_years)
    combined_cindex = concordance_index(modelfit_df["time_to_event"], -combined_scores, modelfit_df["event_observed"])
    out.auc_pathgenomic_honest = combined_auc
    out.cindex_pathgenomic_honest = combined_cindex

    x, y = roc_analyzer.get_roc_curve(n_years=n_years)
    ax.plot(
        x,
        y,
        c=sns_colors["purple"],
        label=f"pStage+AdaPipe: AUC={combined_auc:.1%}",
    )
    # 1-1 line
    ax.plot(x, x, linestyle="--", c="grey")
    ax.legend()
    ax.set_title("Predicting 2-yr survival based on VAF of genes selected in Norway cohort")
    ax.set_xlabel("1 - specificity")
    ax.set_ylabel("sensitivity")
    out.ax = ax
    out.fig = fig

    return out
