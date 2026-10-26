"""Contains base model class which outlines the methods that each model wrapper
must implement."""

from lifelines.utils import concordance_index
from sklearn.metrics import roc_curve, roc_auc_score
from utils.display import *
from utils.misc import add_text_to_legend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------- BASE MODEL -----------------------------------------------------------------------------------------
class BaseTimeToEventModel:
    """Base class for all model wrappers that are concerned with predicting
    relapse from a dataframe with columns ``time_to_event``, ``event_observed``,
    and ``RED_ID``, plus covariates."""

    n_rows_used_to_fit_model: int
    model: any
    results_df: pd.DataFrame  # will hold dataframe with results (standardized names)
    global_pvalue: float = np.nan
    _duration_col: str = "time_to_event"
    _event_col: str = "event_observed"
    _mandatory_columns: list[str] = ["time_to_event", "event_observed", "RED_ID"]

    def __init__(self):
        self.n_rows_used_to_fit_model = None
        self.model = None
        self.results_df = None

    def fit(self, model_df, **kwargs):
        raise NotImplementedError

    def predict(self, model_df):
        raise NotImplementedError

    def _get_fit_results_df(self) -> pd.DataFrame:
        """Internal: build standardized ``results_df`` after ``fit()``."""
        raise NotImplementedError

    def _prepare_data(self, model_df):
        raise NotImplementedError

    @classmethod
    def simulate_test_data(cls, n_samples=400):
        model_df = pd.DataFrame(
            {
                "age": np.random.normal(60, 10, n_samples),
                "sex": np.random.choice([0, 1], size=n_samples),
                "biomarker": np.random.normal(0, 1, n_samples),
            }
        )

        linear_term = (1 * model_df["age"] + 2 * model_df["sex"] + 1 * model_df["biomarker"]) * 0.0001
        event_times = np.random.exponential(scale=np.exp(-linear_term), size=n_samples)
        plt.plot(event_times)
        censoring_times = np.random.exponential(scale=event_times, size=n_samples)

        model_df["time_to_event"] = np.minimum(event_times, censoring_times)
        model_df["event_observed"] = (event_times <= censoring_times).astype(int)

        model_df["relapsed_within_3years"] = model_df["time_to_event"] < model_df["time_to_event"].quantile(0.5)

        return model_df

    def get_attributes(self):
        attrs = {k: v for k, v in self.__dict__.items() if not k.startswith("__") and not callable(v)}
        return attrs

    def get_methods(self):
        methods = [name for name in dir(self) if not name.startswith("__") and callable(getattr(self, name))]
        return methods

    def get_cindex(self, model_df):
        """Predicts scores and computes the c-index for the survival data."""
        return concordance_index(
            event_times=model_df["time_to_event"],
            predicted_scores=-self.predict(model_df),
            event_observed=model_df["event_observed"],
        )

    def print_performance_metrics(self, model_df, n_years=3, ax=None):
        """Plots roc curve with AUC for predicting relapse within 3 years."""
        # COMPUTE C-INDEX
        cindex = self.get_cindex(model_df)
        # ROC-CURVE CALCULATIONS
        test_df = model_df.query(f"event_observed==1 or time_to_event>={n_years}")
        pred = self.predict(test_df)
        true = test_df["time_to_event"] <= n_years * 365
        x, y, _ = roc_curve(true, pred)
        auc = roc_auc_score(true, pred)

        # PLOT RESULTS
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), linestyle="--", alpha=0.5)
        ax.grid(alpha=0.5, linestyle="--")
        ax.set_title(f"ROC curve: predicting {n_years}-year relapse")
        add_text_to_legend(f"AUC={auc:.3f}\nc-index={cindex:.3f}\nAIC={auc:.1f}", ax=ax)
