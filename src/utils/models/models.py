from typing import Literal
from typing import Union
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceWarning
from lifelines.utils import concordance_index
from sklearn.linear_model import LogisticRegression
from sksurv.tree import SurvivalTree
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
import statsmodels.api as sm
from utils.analysis_and_calculations import calc_auc_for_predicting_relapse_from_df
from utils.display import *
from utils.misc import get_rows_where_nyear_relapse_is_defined
import numpy as np
import pandas as pd
import warnings
from utils.models.base_model import BaseTimeToEventModel


# ---------- CLASSIFICATION MODELS -----------------------------------------------------------------------------------------
class LogRegModel(BaseTimeToEventModel):
    """Internally drops rows that where 'relapsed_within_3years' is nan."""

    model: LogisticRegression

    def __init__(
        self,
        relapse_window_in_years: int = 2,
        predictors: Union[list[str], Literal["inferr_from_column_names"]] = "inferr_from_column_names",
    ):
        super().__init__()
        self.relapse_window_in_years = relapse_window_in_years
        assert predictors == "inferr_from_column_names" or isinstance(predictors, list), "predictors must be 'inferr_from_column_names' or list of strings"   # fmt: skip
        self.predictors = predictors
        self.result = None  # will hold the fitted statsmodels result

    def fit(self, model_df, **kwarg):
        """Set ``disp=1`` to see convergence info."""
        X, y = self._prepare_data(model_df)
        self.n_rows_used_to_fit_model = len(X)

        # Add intercept column (statsmodels doesn’t do this automatically)
        X = sm.add_constant(X)
        self.model = sm.Logit(y, X)
        self.result = self.model.fit(disp=0)
        self.global_pvalue = self.result.llr_pvalue
        self.results_df = self._get_fit_results_df()

    def predict(self, model_df):
        X, _ = self._prepare_data(model_df)
        X = sm.add_constant(X, has_constant="add")
        return self.result.predict(X)

    def _get_fit_results_df(self) -> pd.DataFrame:
        """Internal: build ``results_df`` indexed by covariate (``coef``, ``p``, ``std``)."""
        results_df = pd.DataFrame(
            {
                "p": self.result.pvalues,
                "coef": self.result.params,
                "std": self.result.bse,
            },
        )
        results_df.index.name = "covariate"
        return results_df.drop(index="const", errors="ignore")

    def summary(self):
        if self.result is not None:
            return self.result.summary()
        else:
            raise RuntimeError("Model not yet fitted.")

    def _prepare_data(self, model_df):
        model_df = model_df[get_rows_where_nyear_relapse_is_defined(model_df, n_years=self.relapse_window_in_years)]

        if self.predictors == "inferr_from_column_names":
            X = model_df[inferr_x_columns(model_df)]
        else:
            X = model_df[self.predictors]
        # Convert bool to 1.0 and 0.0 (statsmodels does not accept bool type)
        X = X.astype(float)

        y = model_df["time_to_event"] <= 365 * self.relapse_window_in_years

        return X, y


# ---------- SURVIVAL-MODELS -----------------------------------------------------------------------------------------
class CPHModel(BaseTimeToEventModel):
    def __init__(
        self,
        predictors: list | None = None,
        penalizer=0,
        l1_ratio=0.0,
        print_df_before_fit=False,
        **kwargs,
    ):
        """event_col indicates if recurrence was censored or not."""
        super().__init__()
        predictors = [] if predictors is None else predictors
        # Ensure predictors is a list of strings or empty list
        self.predictors = [predictors] if isinstance(predictors, str) else predictors
        self.model = CoxPHFitter(**kwargs, penalizer=penalizer, l1_ratio=l1_ratio)
        self.pvalues = None
        self.cindex = None
        self.auc = None
        self.global_pvalue

        self.print_df_before_fit = print_df_before_fit

    def identify_predictors(self):
        return list().difference()

    def fit(self, model_df, precision=1e-07, suppress_warnings=False, **kwargs):
        prepared_df = self._prepare_data(model_df)
        if self.print_df_before_fit:
            green_print("Head of dataframe used in model.fit()")
            display(prepared_df.head(3))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")  # capture all warnings

            self.model.fit(
                prepared_df,
                duration_col=self._duration_col,
                event_col=self._event_col,
                fit_options={"precision": precision},
            )

            # Flag convergence failure
            self.convergence_failed = any(issubclass(warning.category, ConvergenceWarning) for warning in w)

        if not suppress_warnings:
            # Re-emit the warnings so they are printed normally
            for warning in w:
                warnings.warn(warning.message, warning.category)

        self.pvalues = self.model.summary["p"]
        self.n_rows_used_to_fit_model = len(model_df)
        self.results_df = self._get_fit_results_df()
        # PERFORMANCE METRICS
        self.cindex = self.model.concordance_index_
        self.aic = self.model.AIC_partial_
        self.global_pvalue = self.model.log_likelihood_ratio_test().p_value

    def predict(self, model_df):
        y_pred = self.model.predict_partial_hazard(model_df)
        return y_pred

    def get_cindex(self, model_df):
        return concordance_index(model_df["time_to_event"], -self.predict(model_df), model_df["event_observed"])

    def get_auc(self, model_df, n_years=2, time_to_event_unit: Literal["days", "months"] = "days"):
        if time_to_event_unit == "months":
            model_df = model_df.copy()
            # Convert to days
            model_df["time_to_event"] = model_df["time_to_event"] * 30.44
        return calc_auc_for_predicting_relapse_from_df(model_df, self.predict(model_df), n_years=n_years)

    def print_roc_curve(self, model_df, n_years=2, time_to_event_unit: Literal["days", "months"] = "days"):
        if time_to_event_unit == "months":
            model_df = model_df.copy()
            # Convert to days
            model_df["time_to_event"] = model_df["time_to_event"] * 30.44
        return calc_auc_for_predicting_relapse_from_df(model_df, self.predict(model_df), n_years=n_years)

    def _prepare_data(self, model_df):
        """Extracts the predictors + CPH columns if predictors has been
        provided."""
        if self.predictors != [] and self.predictors is not None:
            # predictors have been provided: keep only essential columns
            model_df = model_df[self.predictors + [self._duration_col, self._event_col]]
        # Remove RED_ID column if it exists
        model_df = model_df[[c for c in model_df.columns if c != "RED_ID"]]
        return model_df

    def get_summary(self):
        return self.model.summary

    def get_light_summary(self):
        return self.model.summary[["coef", "exp(coef)", "p"]]

    def _get_fit_results_df(self):
        """Internal: build ``results_df`` indexed by covariate (``coef``, ``p``, CIs, ``exp(coef)``)."""
        coefs = self.model.summary[["coef", "p", "exp(coef)", "coef lower 95%", "coef upper 95%"]].copy()
        coefs = coefs.rename(
            columns={
                "coef lower 95%": "ci_coef_lower",
                "coef upper 95%": "ci_coef_upper",
            }
        )
        coefs.index.name = "covariate"
        return coefs


class SurvivalTreeModel(BaseTimeToEventModel):
    """Survival tree wrapper. Mostly used for cutoff optimization."""

    def __init__(self, predictors: list | None = None, max_depth=1, min_samples_leaf=10, max_leaf_nodes=None):
        predictors = [] if predictors is None else predictors
        # Ensure predictors is list of strings
        self.predictors = [predictors] if isinstance(predictors, str) else predictors
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.model = SurvivalTree(max_depth=self.max_depth, min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes)   # fmt: skip

        self.thresholds = None
        self.threshold_first_split = None
        self.cindex = None

    def fit(self, model_df):
        """If using only 1 cutoff (max_depth=1), ``self.thresholds[0]`` returns
        the the threshold."""
        X, y = self._prepare_data(model_df)

        self.model.fit(X, y)

        # Extract threshold
        tree_ = self.model.tree_
        self.thresholds = tree_.threshold
        self.threshold_first_split = tree_.threshold[0]

        self.cindex = self.get_cindex(model_df)

    def _prepare_data(self, model_df):
        """Returns ``X``, ``y`` ready for ``.fit()``"""
        if self.predictors == [] or self.predictors is None:
            # Assume all columns that are not event_observed, time_to_event, or RED_ID are predictors
            self.predictors = [c for c in model_df.columns if c not in self._mandatory_columns]
        # Keep only essential variables (also removes RED_ID if it exists)
        model_df = model_df[self.predictors + [self._duration_col, self._event_col]]
        y = Surv.from_dataframe(event=self._event_col, time=self._duration_col, data=model_df)
        X = model_df[self.predictors]

        return X, y

    def predict(self, model_df):
        X, _ = self._prepare_data(model_df)
        return self.model.predict(X)

    def get_thresholds(self):
        """If using only 1 cutoff (max_depth=1), use get_thresholds()"""

    def fit_cutoff(self, modelfit_df):
        self.fit(modelfit_df)
        return self.thresholds[0]


class RandomSurvivalForestModel(BaseTimeToEventModel):
    """Wrapper for random forest survival model."""

    def __init__(self, predictors: list | None = None, n_estimators=100, max_depth=None, random_state=0):
        predictors = [] if predictors is None else predictors
        # Ensure predictors is list of strings
        self.predictors = [predictors] if isinstance(predictors, str) else predictors
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = RandomSurvivalForest(n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=self.random_state, n_jobs=-1)   # fmt: skip
        self.cindex = None

    def fit(self, model_df):
        """If using only 1 cutoff (max_depth=1), ``self.thresholds[0]`` returns
        the the threshold."""
        X, y = self._prepare_data(model_df)

        self.model.fit(X, y)
        self.cindex = self.get_cindex(model_df)

    def _prepare_data(self, model_df):
        """Returns ``X``, ``y`` ready for ``.fit()``"""
        if self.predictors == [] or self.predictors is None:
            # Assume all columns that are not event_observed, time_to_event, or RED_ID are predictors
            self.predictors = [c for c in model_df.columns if c not in self._mandatory_columns]
        # Keep only essential variables (also removes RED_ID if it exists)
        model_df = model_df[self.predictors + [self._duration_col, self._event_col]]
        y = Surv.from_dataframe(event=self._event_col, time=self._duration_col, data=model_df)
        X = model_df[self.predictors]

        return X, y

    def predict(self, model_df):
        X, _ = self._prepare_data(model_df)
        return self.model.predict(X)


# ---------- UTILS -----------------------------------------------------------------------------------------
def inferr_x_columns(modelfit_df) -> list[str]:
    """Returns list of variables which are not ``RED_ID``, ``time_to_event`` or
    ``event_observed``."""
    non_x_columns = ["RED_ID", "time_to_event", "event_observed"]
    x_columns = [c for c in modelfit_df if c not in non_x_columns]
    assert len(x_columns) != 0, "There are no predictors in modelfit_df"
    return x_columns
