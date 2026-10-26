from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.space import Real
from skopt.space import Integer
from skopt.space import Categorical
from skopt.plots import plot_convergence
from typing import Literal
from dataclasses import dataclass
from dataclasses import field
from matplotlib.axes import Axes

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.cm as cm

from utils.prelude import *
from utils.crossvalidation import CrossValidator
from utils.variablesearch.variable_searcher import VariableSearcher
from utils.variablesearch.grid_cutoff_selectors import ManuallySelectedCutoff
from utils.variablesearch.grid_cutoff_selectors import MinPvalOverEligible
from utils.datafactory.data_factory import DataFactory
from utils.datafactory.gene_selectors import ManualGeneSelection
from utils.prepare_data.prepare_data import DataBuilder
from utils.models.models import CPHModel
from utils.models.models import SurvivalTreeModel
import hashlib
from utils.maps_and_lists.TCGAexperiments import EXPERIMENT_DICTS


def history_hash(history: list[dict], key="auc") -> str:
    values = [h[key] for h in history]
    raw = ",".join(f"{v:.15f}" for v in values)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


@dataclass
class BaysianOptimizerSettings:
    n_calls: int = 50
    seed: int = 1
    n_initial_points: int = 10
    n_selected_min: int = 5
    n_selected_max: int = 30
    ridge_penalty_min: float = 1.0
    ridge_penalty_max: float = 10.0
    oncofilter_settings: list[str] = field(default_factory=lambda: ["pick_on"])


def map_experiment_settings_to_label(
    train_hist="NSCLC", eval_hist="LUAD", train_endpoint="TTR", other_info: str = None
):
    label = f"{train_hist}train{train_endpoint}_{eval_hist}eval"
    if other_info is not None:
        label = f"{label}_{other_info}"
    return label


# Maps oncofilter strategy to wich p-value column to use to rankorder genes
ONCOFILTERSETTING_TO_PVAL_MAP = {"pick_on": "p_on", "pick_off": "p_off", "highest_neglogp": "p_best"}


@dataclass
class CrossValHyperParOptimizer:
    """Performs baysian hyperparameter optimization based on cross-validated
    performance."""
    experiment: Literal["E1: train on NSCLC", "E2: train on LUAD", "E3: train on NSCLC and DSS"] = "E1: train on NSCLC"
    sub_experiment: Literal["NSCLCeval", "LUSCeval", "LUADeval"] = "LUADeval"
    stiff_cutoff: float = None
    freeze_selections: bool = True
    optim_settings: BaysianOptimizerSettings = field(default_factory=BaysianOptimizerSettings)
    verbose: int = 1

    def __post_init__(self):
        white_print(f"MAIN EXPERIMENT:  '{self.experiment}'\nSUB EXPERIMENT:  '{self.sub_experiment}'")
        self.args_dict = EXPERIMENT_DICTS[self.experiment]
        self.endpoint = self.args_dict["endpoint"]
        subexperiment_settings = self.args_dict[self.sub_experiment]

        if self.stiff_cutoff is None:
            white_print(f"Using fluid pipeline")
            self.grid_cutoff_selector = MinPvalOverEligible()
            self.varsearcher_path = self.args_dict["path_to_varsearcher_fluidGS"]
            self.cv_results_path = self.args_dict["path_to_CVresults__fluidGS"]
            self.hyperparam_dump_path = subexperiment_settings["hyperparam_dump_path__fluidGS"]
            self.gene_ranking_metric = "p_softcut"
        else:
            white_print(f"Using stiff pipeline with VAF cutoff={ self.stiff_cutoff:.1%}")
            self.grid_cutoff_selector = ManuallySelectedCutoff(self.stiff_cutoff)
            self.varsearcher_path = self.args_dict["path_to_varsearcher_stiffGS"]
            self.cv_results_path = self.args_dict["path_to_CVresults__stiffGS"]
            hp_lookup = subexperiment_settings.get("hyperparam_dump_path__stiffGS_lookup", {})
            self.hyperparam_dump_path = hp_lookup.get(
                self.stiff_cutoff, subexperiment_settings["hyperparam_dump_path__stiffGS"]
            )
            self.gene_ranking_metric = "p"

        self.path_to_gene_pool = subexperiment_settings["gene_selection_pool_path"]

    def setup(self, variants_filtered_df):
        # --- LOAD VARSEARCHER (USED ONLY FOR GENE SELECTION)
        white_print(f"Loading varsearcher...")
        varsearcher = VariableSearcher(
            variants_df=variants_filtered_df,
            clinical_df=DataBuilder(endpoint=self.args_dict["endpoint"]).get_clinical_data(),
            endpoint=self.args_dict["endpoint"],
            grid_cutoff_selector=self.grid_cutoff_selector,
        )
        varsearcher.load_results_from_json(dump_dir_path=self.varsearcher_path)
        varsearcher.compute_result_summary_dfs()
        self.varsearcher = varsearcher

        # GET GENES ELIGIBLE FOR TCGA VALIDATION MODEL
        overlapping_genes = DataBuilder.load_gene_selection_pool(self.path_to_gene_pool)
        green_print(f"Size of gene selection pool: {len(overlapping_genes)}")
        # GET PVALUES USED FOR GLOBAL (PRIOR TO CV) SELECTION
        self.pvalues_df = get_pvalue_dataframe(self.varsearcher, self.gene_ranking_metric)
        self.pvalues_df = self.pvalues_df.loc[[x for x in overlapping_genes if x in self.pvalues_df.index]].sort_values("p_best")   # fmt: skip
        nan_rows = self.pvalues_df[self.pvalues_df.isna().any(axis=1)]
        if len(nan_rows) > 0:
            display(nan_rows)
            self.pvalues_df = self.pvalues_df.dropna()
            assert self.pvalues_df.__len__() > 0, "There are no rows with"
            red_print(f"Dropping {list(nan_rows.index)} from gene-selection pool due to nans...")   # fmt: skip
            red_print(f"Size of gene selection pool: {len(self.pvalues_df)}")
        self.overlapping_genes = overlapping_genes

        if self.verbose > 1:
            green_print("P-values for pool of selectable genes")
            display(self.pvalues_df)

        # ---------- FluidCut GS -----------------------------------------------------------------------------------------
        # --- FLUID SELECTION
        # LOAD DATAFACTORY
        self.datafactory = DataFactory(
            variants_filtered_df=variants_filtered_df,
            grid_cutoff_selector=self.grid_cutoff_selector,
            path_varsearcher_dumpdir=self.cv_results_path,
            path_crossval_split=self.args_dict["crossval_path"],
            data_builder=DataBuilder(endpoint=self.args_dict["endpoint"]),
        )
        self.datafactory.load_varsearchers()

        if self.freeze_selections:
            red_print("Basing gene and filter selection on global optimization...")
            for i in range(5):
                self.datafactory.varsearchers[i] = varsearcher
        else:
            green_print("Filter optimization nested inside of cross validation...")

    def run_bayesian_optimization(self, optim_settings: BaysianOptimizerSettings = None):
        if optim_settings is not None:
            self.optim_settings = optim_settings
        # --- HYPER PARAMETER OPTIMIZATION
        white_print("--- RUNNING BAYSIAN HYPERPARAMETER OPTIMIZATION ---")
        self.optim_settings.n_selected_max = min(len(self.pvalues_df), self.optim_settings.n_selected_max)
        self.results, self.history = run_bayesian_CrossVal_hyperparameter_optimization(
            datafactory=self.datafactory,
            pvalues_df=self.pvalues_df,
            optim_settings=self.optim_settings,
        )
        historyhash = history_hash(self.history)
        teal_print(f"Hash based on AUC history: {historyhash}")

    def plot_histogram_of_hyperpar_training_history(self, metric_name: Literal["auc", "cindex"] = "auc"):
        metric_namemap = {"auc": "AUC", "cindex": "c-index"}
        std_col = f"{metric_name}_std"
        metric_name_nice = metric_namemap[metric_name]
        oncostrat_namemap = {
            "pick_on": "oncofilter: on",
            "pick_off": "oncofilter: off",
            "highest_neglogp": "oncofilter: adaptive",
        }
        results_df = pd.DataFrame(self.history)
        results_df["auc_neglogp"] = -np.log10(results_df["auc_p"])
        results_df["cindex_neglogp"] = -np.log10(results_df["cindex_p"])

        # STABILITY-BASED TIE-BREAKER: SELECT BEST FROM AUC'S THAT ARE WITHIN TOLERANCE OF BEST AUC
        best_idx = results_df[metric_name].idxmax()
        best_score = results_df[metric_name].loc[best_idx]
        margin = 0.005
        std_of_candidates_df = results_df[best_score - results_df[metric_name] < margin][std_col]
        idx_best_stable_val = std_of_candidates_df.idxmin()
        results_stable_winner = results_df.loc[idx_best_stable_val]

        x_limits = results_df[metric_name].agg(["min", "max"]).values
        onco_colormap = {"pick_on": "m", "pick_off": "k", "highest_neglogp": "teal"}

        # --- HISTOGRAM BY ONCOFILTER STRATEGY
        oncofilter_strats = sorted(results_df["oncofilter_setting"].unique())
        fig, axs = plt.subplots(1, 1, figsize=(10, 5))
        for oncosetting in oncofilter_strats:
            ax = axs
            ax: Axes
            ax.set_axisbelow(True)
            ax.grid(alpha=0.3, linestyle="--")
            df = results_df.query(f"oncofilter_setting=='{oncosetting}'")
            median_val = df[metric_name].mean()
            std_val = df[metric_name].var()
            hist_label = f"{oncosetting}: Highest {metric_name_nice}={df[metric_name].max():.2%}"
            sns.histplot(df[metric_name], bins=15, color=onco_colormap[oncosetting], ax=ax, alpha=0.6, label=hist_label)
            ax.axvline(median_val, label=f"Mean={median_val:.2%} (STD={std_val:.2%})", c=onco_colormap[oncosetting], linestyle="--")   # fmt: skip
            ax.set_xlim(x_limits)
            ax.legend()
            ax.set_xlabel(metric_name_nice)
            ax.set_title(f"Oncofilter-setting: '{oncosetting.replace("_", " ")}'")
            ax.legend()
        fig.subplots_adjust(hspace=0.45)

        # --- PLOT PERFORMANCE BY RIDGE PENALTY
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))
        ax = axs[0]
        markermap = {"pick_on": "s", "pick_off": "s", "highest_neglogp": "*"}
        df = results_df.sort_values("ridge_penalty").reset_index(drop=True)
        ax.plot(df["ridge_penalty"], df[metric_name], "--", c="k", alpha=0.5)
        # Plot markers
        for setting, subdf in df.groupby("oncofilter_setting"):
            ax.scatter(
                subdf["ridge_penalty"],
                subdf[metric_name],
                c=onco_colormap[setting],
                marker=markermap[setting],
                edgecolor="k",
                label=setting,
                alpha=0.4,
                s=100,
            )
        ax.legend()
        ax.set_xlabel("Ridge penalty")
        ax.set_ylabel(metric_name)
        ax.set_title(f"Performance ({metric_name}) as function of ridge penalty")

        # --- PLOT PERFORMANCE BY N SELECTED
        ax = axs[1]
        onco_colormap = {"pick_on": "m", "pick_off": "k", "highest_neglogp": "m"}
        markermap = {"pick_on": "s", "pick_off": "s", "highest_neglogp": "*"}
        df = results_df.sort_values("n_selected").reset_index(drop=True)
        ax.plot(df["n_selected"], df[metric_name], "--", c="k", alpha=0.5)
        # Plot markers
        for setting, subdf in df.groupby("oncofilter_setting"):
            ax.scatter(
                subdf["n_selected"],
                subdf[metric_name],
                c=onco_colormap[setting],
                marker=markermap[setting],
                edgecolor="k",
                alpha=0.4,
                label=oncostrat_namemap[setting],
                s=100,
            )
        ax.legend()
        ax.set_xlabel("N genes selected")
        ax.set_ylabel(metric_name)
        ax.set_title(f"Performance ({metric_name}) as function of N selected genes")

        # --- PLOT PERFORMANCE BY N SELECTED
        fig, ax = plt.subplots()
        plt.get_cmap("viridis")
        norm = mcolors.Normalize(vmin=df[std_col].min(), vmax=df[std_col].max())
        cmap = plt.get_cmap("viridis", 7)
        colors = cmap(norm(df[std_col]))   # fmt: skip
        markermap = {"pick_on": "s", "pick_off": "s", "highest_neglogp": "*"}
        df = results_df.sort_values("n_selected").reset_index(drop=True)
        ax.plot(df["n_selected"], df[metric_name], "--", c="k", alpha=0.5)
        # Plot markers
        for setting, subdf in df.groupby("oncofilter_setting"):
            ax.scatter(
                subdf["n_selected"],
                subdf[metric_name],
                c=colors[subdf.index],
                marker=markermap[setting],
                edgecolor="k",
                alpha=0.8,
                s=100,
            )
        ax.axhspan(
            best_score - margin,
            best_score + margin,
            alpha=0.3,
            color="grey",
            label="Band containing ties (margin=1%)",
        )
        ax.scatter(results_stable_winner.n_selected, results_stable_winner[metric_name], color="m", marker="*", edgecolor="k", s=150, alpha=1, label="Most stable high performer")   # fmt: skip
        ax.legend()
        ax.set_xlabel("N genes selected")
        ax.set_ylabel(metric_name)
        ax.set_title("Performance as function of N selected genes")
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])  # required for older matplotlib versions
        fig.colorbar(sm, ax=ax, label=f"{metric_name} significance (-log10 p)")

        plt.show()

        idx_max = results_df[metric_name].idxmax()
        best = results_df.loc[idx_max]
        white_print(f"--- BEST RESULTS (SETTINGS THAT MAXIMIZE {metric_name.upper()}) ---")
        lime_print(f"Best: choose top {best.n_selected}, oncofilter='{best.oncofilter_setting}', and lambda={best.ridge_penalty:.2f}")   # fmt: skip
        lime_print(f"AUC: {best.auc:.2%} ({best.auc_p:.3f})\nC-index: {best.cindex:.1%} ({best.cindex_p:.3f})\n\n")   # fmt: skip

        best_stable = df.loc[idx_best_stable_val]
        white_print(f"--- BEST IN STABILITY SELECTION ---")
        lime_print(f"Best: choose top {best_stable.n_selected}, oncofilter='{best_stable.oncofilter_setting}', and lambda={best_stable.ridge_penalty:.2f}")   # fmt: skip
        lime_print(f"AUC: {best_stable.auc:.2%} ({best_stable.auc_p:.3f})\nC-index: {best_stable.cindex:.1%} ({best_stable.cindex_p:.3f})\n\n")   # fmt: skip

    def fit_final_model(
        self,
        min_samples_per_km_strata: int = 40,
        n_strata: int = 3,
    ):
        """Fit the final discovery Cox model and KM stratification artifacts for TCGA eval."""
        selected_genes = list(self.results.pvals_selected_genes.keys())
        varsearcher = self.varsearcher
        oncofilter = self.results.best_oncofilter_setting
        if varsearcher.grid_cutoff_selector.oncofilter_selection != oncofilter:
            varsearcher.grid_cutoff_selector.oncofilter_selection = oncofilter
            varsearcher.merge_oncogenicity_filtering_results()

        # --- FIT MODEL
        # Get dataframe with mutation status for selected genes
        mutation_df = varsearcher.get_modelfit_df_for_subset_of_genes(
            genes_to_include_in_model=selected_genes, restrict_to_red_id_used_to_fit_filters=True
        )
        mutation_df.columns = [c.replace("_MUTATED", "") for c in mutation_df.columns]
        # Fit model with selected ridge penalty
        cph = CPHModel(penalizer=self.results.best_ridge_penalty)
        cph.fit(mutation_df)
        cph.results_df.index = selected_genes
        # Collect info on fitted filters
        filterinfo = varsearcher.summary_best_models_df.loc[selected_genes, ["vaf_cutoff", "best_oncofiltersetting"]]
        vaf_cutoffs = filterinfo["vaf_cutoff"]
        gene_oncofilter = filterinfo["best_oncofiltersetting"].to_dict()
        
        # --- GET GUTOFFS FOR SPLITTING COHORT INTO RISK GROUPS BY FITTING SURVIVAL TREE TO RISK SCORES
        scores = cph.predict(mutation_df)
        scores.name = "score"
        df = pd.concat([mutation_df[["time_to_event", "event_observed"]], scores], axis=1)
        tree = SurvivalTreeModel(
            predictors="score",
            max_leaf_nodes=n_strata,
            min_samples_leaf=min_samples_per_km_strata,
            max_depth=2,
        )
        tree.fit(model_df=df)
        tree_ = tree.model.tree_
        score_cutoffs = sorted(tree_.threshold[tree_.feature != -2])
        percentile_cutoffs = [float((df["score"] < cutoff).mean()) for cutoff in score_cutoffs]

        # --- FIT MUTA-RISK + STAGE COX MODEL
        clinical_df = DataBuilder(endpoint=self.endpoint).get_clinical_data()
        clinical_df["stage2"] = clinical_df["pstage"] == "2"
        clinical_df["stage3"] = clinical_df["pstage"] == "3"
        modelfit_mixed_df = mutation_df.merge(clinical_df[["RED_ID", "stage2", "stage3"]], on="RED_ID")
        modelfit_mixed_df["mutascore"] = scores.values
        cph_mixed = CPHModel(penalizer=0.1, predictors=["mutascore", "stage2", "stage3"])
        cph_mixed.fit(modelfit_mixed_df)
        pathomuta_coeff = cph_mixed.results_df["coef"].to_dict()

        return {
            "coef": cph.results_df["coef"],
            "vaf_cutoff": vaf_cutoffs,
            "gene_oncofilter": gene_oncofilter,
            "pathomuta_coeff": pathomuta_coeff,
            "percentile_cutoffs": percentile_cutoffs,
        }

    def dump_results(self):
        self.results.vaf_cutoff = self.stiff_cutoff
        dump_dict = self.results.__dict__
        final_fit = self.fit_final_model()
        dump_dict["coef"] = final_fit["coef"].to_dict()
        dump_dict["vaf_cutoff"] = final_fit["vaf_cutoff"].to_dict()
        dump_dict["gene_oncofilter"] = final_fit["gene_oncofilter"]
        dump_dict["pathomuta_coeff"] = final_fit["pathomuta_coeff"]
        dump_dict["percentile_cutoffs"] = final_fit["percentile_cutoffs"]
        dump_dict["endpoint"] = self.endpoint
        dump_dict["train_histology"] = self.args_dict["train_histology"]
        dump_dict["genes"] = list(dump_dict["coef"].keys())
        Paths.dump_to_json(dump_dict, self.hyperparam_dump_path, print_confirm=True)   # fmt: skip

    def plot_roc_curve_for_best_hyperparams(
        self,
        eval_set: Literal["LUAD", "LUSC", "NSCLC"] = None,
        eval_endpoint: Literal["TTR", "DSS", "OS", "RFS"] = None,
    ):
        if eval_set == "LUAD":
            blue_print(f"Evaluating on {eval_set}...")
            self.datafactory.red_id_restriction_test_sets = DataBuilder.get_red_id_luad_data()
        elif eval_set is not None:
            blue_print(f"Evaluating on {eval_set}...")
            self.datafactory.red_id_restriction_test_sets = DataBuilder.get_red_id_luad_data()
        cv = run_crossvalidation(
            datafactory=self.datafactory,
            pvalues_df=self.pvalues_df,
            ridge_penalty=self.results.best_ridge_penalty,
            n_selected=self.results.best_n_selected,
            oncofilter_setting=self.results.best_oncofilter_setting,
            eval_endpoint=eval_endpoint,
        )
        self.datafactory.red_id_restriction_test_sets = None
        cv.verbose = 1
        cv.run_roc_curve_analysis()
        plt.show()


def get_pvalue_dataframe(varsearcher: VariableSearcher, pvalue: Literal["p_softcut", "p"] = "p_softcut"):
    """Get dataframe with pvalues compute with oncogenicity filter set to 'off',
    'on', and 'pick best' respectively."""
    df = varsearcher.merged_summary_df[[f"{pvalue}__off", f"{pvalue}__on"]]
    pvals_df = df.merge(varsearcher.summary_best_models_df[pvalue], left_index=True, right_index=True)
    # STANDARDIZE NAMES
    pvals_df.columns = ["p_off", "p_on", "p_best"]
    return pvals_df


def run_crossvalidation(
    datafactory: DataFactory,
    pvalues_df: pd.Series,
    ridge_penalty: float,
    n_selected: int,
    oncofilter_setting: Literal["pick_on", "pick_off", "highest_neglogp"],
    eval_endpoint: Literal["TTR", "DSS", "OS", "RFS"] = None,
    verbose=1,
) -> CrossValidator:
    """Objective function for baysian hyperparameter optimization based on c-index."""
    datafactory.grid_cutoff_selector.oncofilter_selection = oncofilter_setting
    datafactory.update_oncofilter_strategy()
    model_factory = lambda: CPHModel(penalizer=ridge_penalty)
    pvals = pvalues_df[ONCOFILTERSETTING_TO_PVAL_MAP[oncofilter_setting]]
    selected_genes = pvals.sort_values()[:n_selected].index.to_list()
    datafactory.gene_selector = ManualGeneSelection(selection=selected_genes)
    # CROSS VALIDATION
    cv = CrossValidator(data_factory=datafactory.build_data_for_ith_fold, model_factory=model_factory, eval_endpoint=eval_endpoint, verbose=0)   # fmt: skip
    cv.prepare_data()
    cv.train_models_in_crossvalidation()
    cv.calc_all_performance_stats()
    if verbose > 0:
        info_strings = " | ".join(
            [
                f"C-index: {cv.cindex_mean:.2%} (p={cv.cindex_pvalue:.1%})",
                f"AUC: {cv.auc_mean:.2%} (p={cv.auc_pvalue:.1%})",
                f"penalty: {ridge_penalty:.2f}",
                f"N selected: {n_selected}",
                f"Oncofilter: {oncofilter_setting}",
            ]
        )
        orange_print(info_strings)
    return cv


@dataclass
class BaysianOptimizationResults:
    best_ridge_penalty: float
    best_n_selected: float
    best_cindex: float
    best_oncofilter_setting: str
    best_auc: float
    best_cindex_p: float
    best_auc_p: float
    best_p_cutoff: float
    vaf_cutoff: float = None
    pvals_selected_genes: list[float] = None


def run_bayesian_CrossVal_hyperparameter_optimization(
    datafactory: DataFactory,
    pvalues_df: pd.DataFrame,
    optim_settings: BaysianOptimizerSettings = BaysianOptimizerSettings(),
    metric: Literal["auc", "cindex"] = "auc",
):
    brown_print(f"METRIC TO OPTIMIZE: {metric.upper()}")
    # DEFINE SEARCH RANGES FOR HYPER PAREMETERS
    space = [
        Real(optim_settings.ridge_penalty_min, optim_settings.ridge_penalty_max, prior="uniform", name="ridge_penalty"),
        Integer(optim_settings.n_selected_min, optim_settings.n_selected_max, prior="uniform", name="n_selected"),
        Categorical(optim_settings.oncofilter_settings, name="oncofilter_setting"),
    ]
    history = []

    # DEFINE OBJECTIVE FUNCTION TO OPTIMIZE
    @use_named_args(space)
    def objective(ridge_penalty, n_selected, oncofilter_setting):
        result = run_crossvalidation(
            datafactory,
            pvalues_df=pvalues_df,
            ridge_penalty=ridge_penalty,
            n_selected=n_selected,
            oncofilter_setting=oncofilter_setting,
        )
        history.append(
            {
                "ridge_penalty": ridge_penalty,
                "n_selected": n_selected,
                "oncofilter_setting": oncofilter_setting,
                "auc": result.auc_mean,
                "auc_p": result.auc_pvalue,
                "cindex": result.cindex_mean,
                "cindex_p": result.cindex_pvalue,
                "auc_std": np.std(result.auc_values),
                "cindex_std": np.std(result.auc_values),
            }
        )
        if metric == "auc":
            metric_value = result.auc_mean
        elif metric == "cindex":
            metric_value = result.cindex_mean
        return -metric_value

    # RUN BAYSIAN OPTIMIZATION
    res = gp_minimize(
        func=objective,
        dimensions=space,
        n_calls=optim_settings.n_calls,  # total evaluations
        n_initial_points=optim_settings.n_initial_points,  # random starts before GP kicks in
        random_state=optim_settings.seed,
    )

    best_ridge_penalty, best_n_selected, best_oncofilter_setting = res.x
    best_metric = -res.fun
    plot_convergence(res)
    plt.show()

    # RUN ONE LAST TIME AT BEST RESULTS TO GET AUC AT BEST
    cv = run_crossvalidation(datafactory, pvalues_df, best_ridge_penalty, best_n_selected, best_oncofilter_setting)
    cv.run_roc_curve_analysis()
    plt.show()

    pvals_selected = pvalues_df[ONCOFILTERSETTING_TO_PVAL_MAP[best_oncofilter_setting]].sort_values()[:best_n_selected]
    p_cutoff = pvals_selected.max()
    white_print(f"--- RESULTS FOR CHOICE OF HYPERPARAMETERS THAT MAXIMIX {metric.upper()} ---")
    lime_print(f"Best: choose top {best_n_selected} genes (p<={p_cutoff:.3f}), oncofilter='{best_oncofilter_setting}', and lambda={best_ridge_penalty:.2f}")   # fmt: skip
    lime_print(f"Selected genes: {", ".join(pvals_selected.index.to_list())}")   # fmt: skip
    lime_print(f"AUC: {cv.auc_mean:.2%} ({cv.auc_pvalue:.3f})\nC-index: {cv.cindex_mean:.1%} ({cv.cindex_pvalue:.3f})\n\n")   # fmt: skip

    results_collected = BaysianOptimizationResults(
        best_ridge_penalty=best_ridge_penalty,
        best_n_selected=best_n_selected,
        best_oncofilter_setting=best_oncofilter_setting,
        best_cindex=cv.cindex_mean,
        best_auc=cv.auc_mean,
        best_cindex_p=cv.cindex_pvalue,
        best_auc_p=cv.auc_pvalue,
        best_p_cutoff=p_cutoff,
        pvals_selected_genes=pvals_selected[:best_n_selected].to_dict(),
    )
    return results_collected, history
