"""5-fold cross-validation for the discovery-cohort Cox models.

Folds are the saved TNM-I splits. Each fold is fit on the training endpoint
and scored on the test fold, which can be restricted to LUAD or LUSC.
"""

from lifelines.utils import concordance_index
from scipy.stats import ttest_1samp
from typing import Literal
from typing import Callable
from utils.plotting import sns_colors
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from utils.analysis_and_calculations import calc_iauc_for_predicting_relapse
from utils.analysis_and_calculations import confidence_interval_of_mean
from utils.display import *
from utils.models.models import CPHModel
from utils.prepare_data.prepare_data import DataBuilder

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class CrossValidator:
    """Fit and score one Cox model on each saved fold."""

    def __init__(
        self,
        data_factory: Callable,
        model_factory: Callable | None = None,
        train_endpoint: Literal["TTR", "DSS", "OS", "DFS"] = None,
        eval_endpoint: Literal["TTR", "DSS", "OS", "DFS"] = None,
        eval_histology: Literal["NSCLC", "LUAD", "LUSC"] = None,
        verbose=1,
    ):
        """``data_factory(fold)`` returns the train and test frames for that fold.
        
        ``train_endpoint`` and ``eval_endpoint`` replace the outcome columns when set.
        ``eval_histology`` keeps only LUAD or LUSC patients in the test fold.
        """
        self.verbose = verbose
        self.model_factory = (lambda: CPHModel()) if model_factory is None else model_factory
        self.data_factory = data_factory
        self.n_splits = 5
        self.train_endpoint = train_endpoint
        self.eval_endpoint = eval_endpoint
        self.eval_histology = eval_histology
        if self.verbose > 0:
            if train_endpoint is not None:
                orange_print(f"Training models on {train_endpoint} endpoint")
            if eval_endpoint is not None:
                orange_print(f"Evaluating models on {eval_endpoint} endpoint")
            if eval_histology is not None:
                orange_print(f"Evaluating models on {eval_histology}")
        self.red_id_eval_histology = self._get_red_id_eval_histology()
        self.train_data = []
        self.test_data = []

    def _get_red_id_eval_histology(self):
        if self.eval_histology == "LUAD":
            return DataBuilder.get_red_id_luad_data()
        if self.eval_histology == "LUSC":
            return DataBuilder.get_red_id_lusc_data()
        return None

    def prepare_data(self):
        """Build the train and test frame for each fold."""
        orange_print("Constructing train and test dataframes...") if self.verbose > 0 else None
        self.train_data = []
        self.test_data = []

        for i in range(self.n_splits):
            modelfit_df_train, modelfit_df_test = self.data_factory(i)
            if self.train_endpoint is not None:
                modelfit_df_train = self._swap_endpoints(modelfit_df_train, self.train_endpoint)
            if self.eval_endpoint is not None:
                modelfit_df_test = self._swap_endpoints(modelfit_df_test, self.eval_endpoint)
            if self.eval_histology is not None:
                modelfit_df_test = self._restrict_to_eval_histology(modelfit_df_test)

            if self.verbose > 1:
                display(modelfit_df_test.head(3))
                pink_print(f"Features selected in iteration {i}: {sorted(modelfit_df_train.columns[3:])}\n")

            self.train_data.append(modelfit_df_train)
            self.test_data.append(modelfit_df_test)

            if self.verbose > 1:
                green_print(f"N events in split {i}: n train={modelfit_df_train['event_observed'].sum()}, n test={modelfit_df_test['event_observed'].sum()}")  # fmt: skip
                blue_print(f"N events in 2 years in split{i}: n test={(modelfit_df_test.query('event_observed==1')['time_to_event'] < 2 * 365).sum()}")   # fmt: skip

    def train_models_in_crossvalidation(self):
        """Fit one model per fold and store the out-of-fold risk scores."""
        if self.train_data == []:
            self.prepare_data()

        self.cindex_values = np.zeros(self.n_splits)
        result_dfs = []

        for i in range(self.n_splits):
            modelfit_df_train = self.train_data[i]
            modelfit_df_test = self.test_data[i]

            model_i: CPHModel = self.model_factory()
            if self.verbose > 1:
                pink_print(modelfit_df_train.drop(columns=["RED_ID", "time_to_event"]).sum())

            variances = modelfit_df_train.drop(columns=["RED_ID", "time_to_event"]).var()
            assert (variances != 0).all(), f"Zero-variance columns encountered: {variances}"

            model_i.fit(modelfit_df_train)

            if hasattr(model_i, "results_df") and self.verbose >= 2:
                white_print(f"  fold {i} — fitted coefficients:")
                display(model_i.get_light_summary())
                if getattr(model_i, "convergence_failed", False):
                    red_print("  (Newton-Raphson convergence warning for this fold)")

            scores_train = model_i.predict(modelfit_df_train)
            scores_test = model_i.predict(modelfit_df_test)
            c_index_test = self._get_cindex(modelfit_df_test, scores_test)
            white_print(f"{c_index_test=:.3f}, {self._get_cindex(modelfit_df_train, scores_train)=:.3f}") if self.verbose > 1 else None   # fmt: skip
            self.cindex_values[i] = c_index_test
            result_dfs.append(
                pd.DataFrame(
                    {
                        "ypred": scores_test,
                        "time_to_event": modelfit_df_test["time_to_event"],
                        "event_observed": modelfit_df_test["event_observed"],
                        "RED_ID": modelfit_df_test["RED_ID"],
                    }
                )
            )

        self.cv_results_df = pd.concat(result_dfs, keys=range(self.n_splits), names=["cv_idx", "lvl1"]).reset_index(level="cv_idx").reset_index(drop=True)   # fmt: skip
        self.calc_cindex_stats()

    def _get_cindex(self, modelfit_df_test, scores_test):
        """C-index on this frame. Scores are risks, so higher is worse."""
        return concordance_index(
            event_times=modelfit_df_test["time_to_event"],
            event_observed=modelfit_df_test["event_observed"],
            predicted_scores=-scores_test,
        )

    def calc_cindex_stats(self):
        """Mean c-index across folds, tested against 0.5."""
        if (self.cindex_values == 0).any():
            red_print("C-index not calculated yet!")
        else:
            self.cindex_mean = np.mean(self.cindex_values)
            self.cindex_confint = confidence_interval_of_mean(self.cindex_values)
            self.cindex_pvalue = ttest_1samp(self.cindex_values, popmean=0.5).pvalue

    def calc_auc_stats(self, n_years=2, return_auc_values=False):
        """AUC for relapse within ``n_years``, one value per fold."""
        df = self.cv_results_df.query(f"event_observed==1 or time_to_event>={n_years} * 365").copy()
        df["target"] = df["time_to_event"] <= 365 * n_years

        green_print(f'N cases where {n_years}-year relapse is defined per fold: {df.groupby("cv_idx").size().values}') if self.verbose > 1 else None   # fmt: skip
        green_print(f'N {n_years}-year relapses observed per fold: {df.groupby("cv_idx")["target"].sum().values}') if self.verbose > 1 else None   # fmt: skip

        auc_vals = df.groupby("cv_idx").apply(lambda df: roc_auc_score(df["target"], df["ypred"]), include_groups=False)   # fmt: skip
        roc_curves = df.groupby("cv_idx").apply(lambda df: roc_curve(df["target"], df["ypred"])[:2], include_groups=False).values   # fmt: skip
        self.auc_values = auc_vals
        self.auc_nyear_roc_curves = roc_curves
        self.auc_mean = np.mean(auc_vals)
        self.auc_pvalue = ttest_1samp(self.auc_values, popmean=0.5).pvalue
        self.auc_nyear_relapse_confint = confidence_interval_of_mean(self.auc_values)
        return auc_vals if return_auc_values else None

    def calc_iauc_stats(self, time_cutoffs=np.linspace(2, 3, 10)):
        """Integrated AUC from 2 to 3 years, one value per fold."""
        group_f = lambda df: calc_iauc_for_predicting_relapse(time_to_event=df["time_to_event"], event_observed=df["event_observed"], scores=df["ypred"], time_cutoffs=time_cutoffs)   # fmt: skip
        auc_vals = self.cv_results_df.groupby("cv_idx").apply(group_f, include_groups=False)   # fmt: skip
        self.iauc_values = auc_vals
        self.iauc_mean = np.mean(auc_vals)
        self.iauc_pvalue = ttest_1samp(auc_vals, popmean=0.5).pvalue
        self.iauc_confint = confidence_interval_of_mean(self.auc_values)

    def calc_all_performance_stats(self, time_cutoffs=np.linspace(2, 3, 10), n_years=2):
        """2-year AUC, integrated AUC, and c-index."""
        self.calc_auc_stats(n_years)
        self.calc_iauc_stats(time_cutoffs)
        self.calc_cindex_stats()

    def run_roc_curve_analysis(
        self,
        n_years=2,
        time_cutoffs=np.linspace(2, 3, 10),
        ax=None,
        fig=None,
        verbose=1,
    ):
        """Per-fold ROC curves and their mean, for relapse within ``n_years``."""
        assert self.cv_results_df is not None, "No cross validation results; first run train_models_in_crossvalidation"   # fmt: skip
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))

        self.calc_all_performance_stats(n_years=n_years, time_cutoffs=time_cutoffs)

        colors = [sns_colors[c] for c in ["green", "blue", "purple", "orange", "red"]]
        for counter, (auc, (x, y), color) in enumerate(zip(self.auc_values, self.auc_nyear_roc_curves, colors), start=1):
            ax.plot(x, y, alpha=0.3, color=color, label=f"CV fold {counter}: AUC={auc:.1%}")

        ax.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), c="grey", linestyle="--", alpha=0.5)
        ax.set_xlabel("1 - specificity")
        ax.set_ylabel("sensitivity")

        endpoint_label = self.eval_endpoint or "TTR"
        self.interpolate_roc_curves()
        auc_sd = np.std(self.auc_values)
        ax.plot(self.interp_mean_fpr, self.interp_mean_tpr, c="k", label=f"{endpoint_label} mean ROC (AUC={self.auc_mean:.1%} ± {auc_sd:.1%})")   # fmt: skip
        ax.fill_between(
            self.interp_mean_fpr,
            np.clip(self.interp_mean_tpr - self.interp_std_tpr, 0, 1),
            np.clip(self.interp_mean_tpr + self.interp_std_tpr, 0, 1),
            color="grey",
            alpha=0.2,
            label=f"{endpoint_label} ± 1 SD",
        )

        ax.legend(fontsize=9)
        info_strings = [f"mean AUC: {self.auc_mean:.1%}, p={self.auc_pvalue:.4f}"]
        if verbose > 0:
            info_strings += [
                f"mean iAUC: {self.iauc_mean:.1%}, p={self.iauc_pvalue:.5f}",
                f"mean c-index: {self.cindex_mean:.1%}, p={self.cindex_pvalue:.4f}",
            ]
        if self.verbose > 0:
            ax.text(0.57, 0.55, "\n".join(info_strings), color="grey", ha="left", va="top", fontsize=9)
        ax.set_title(f"CV ROC — {endpoint_label} within {n_years} years", fontsize=13)
        self.fig_roc_curve = fig
        self.ax_roc_curve = ax

    def interpolate_roc_curves(self, n_points=1000):
        """Average the fold ROC curves on a common false-positive grid."""
        mean_fpr = np.linspace(0, 1, n_points)
        tprs = []
        for fpr, tpr in self.auc_nyear_roc_curves:
            tpr_interp = np.interp(mean_fpr, fpr, tpr)
            tpr_interp[0] = 0.0
            tprs.append(tpr_interp)
        tprs = np.array(tprs)
        self.interp_mean_fpr = mean_fpr
        self.interp_mean_tpr = np.mean(tprs, axis=0)
        self.interp_std_tpr = np.std(tprs, axis=0)
        self.interp_tprs = tprs
        self.auc_nyear_interpolated_roc_curves = self.interp_mean_tpr

    def _swap_endpoints(self, df, endpoint):
        """Replace the outcome columns with another clinical endpoint."""
        clinical_df = DataBuilder(endpoint=endpoint).get_clinical_data()
        red_ids = df["RED_ID"].to_list()
        endpoint_cols = ["time_to_event", "event_observed"]
        df[endpoint_cols] = clinical_df.set_index("RED_ID").loc[red_ids, endpoint_cols].values
        return df.dropna()

    def _restrict_to_eval_histology(self, df):
        """Keep test patients of the requested histology."""
        return df[df["RED_ID"].isin(self.red_id_eval_histology)]
