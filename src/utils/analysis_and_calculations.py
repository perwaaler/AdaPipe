"""AUC, c-index intervals, and a few small summaries used by the paper.

Relapse within n years counts an event in that window, or a patient still
event-free at the cutoff. Everyone else is left out of that AUC.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import scipy.stats as stats
from fancyimpute import IterativeImputer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


def pearson_correlation(x, y):
    """Pearson r and a two-sided p-value. Rows with a missing value are dropped."""

    @dataclass
    class PearsonResult:
        value: float
        p: float

    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Inputs must be 1D arrays")
    if len(x) != len(y):
        raise ValueError("Arrays must have the same length")
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]

    n = len(x)
    if n < 3:
        raise ValueError("At least 3 observations are required")

    r = np.corrcoef(x, y)[0, 1]
    t = r * np.sqrt((n - 2) / (1 - r**2))
    p = 2 * stats.t.sf(np.abs(t), df=n - 2)
    return PearsonResult(value=r, p=p)


def confidence_interval_of_mean(data, confidence=0.95):
    """Normal-theory interval for the mean. Returns the lower and upper ends."""
    data = np.array(data)
    n = len(data)
    mean = np.mean(data)
    margin = stats.sem(data) * stats.t.ppf((1 + confidence) / 2, df=n - 1)
    return float(mean - margin), float(mean + margin)


def impute_missing_variables(predictors_df):
    """Fill missing predictor values with iterative imputation. Dtypes are kept."""
    imputer = IterativeImputer()
    imputed_values = imputer.fit_transform(predictors_df)
    predictors_imputed_df = pd.DataFrame(imputed_values, columns=predictors_df.columns)
    for col in predictors_df.columns:
        predictors_imputed_df[col] = predictors_imputed_df[col].astype(predictors_df[col].dtype)
    return predictors_imputed_df


def get_rows_where_nyear_relapse_is_defined(
    time_to_event: pd.Series, event_observed: pd.Series, n_years=2
) -> pd.Series:
    """Patients with an event, or still free of the event at n years."""
    return (event_observed == 1) | (time_to_event >= n_years * 365)


def calc_iauc_for_predicting_relapse(
    time_to_event: pd.Series,
    event_observed: pd.Series,
    scores: pd.Series,
    time_cutoffs=np.linspace(1, 3, 10),
):
    """Mean AUC across the time cutoffs, in years."""
    aucs = []
    for n_year in time_cutoffs:
        nyear_relapse_is_defined = get_rows_where_nyear_relapse_is_defined(time_to_event, event_observed, n_years=2)
        y_true = time_to_event[nyear_relapse_is_defined] <= 365 * n_year
        y_pred = scores[nyear_relapse_is_defined]
        aucs.append(roc_auc_score(y_true, y_pred))
    return np.mean(aucs)


def calc_auc_for_predicting_relapse_from_df(df: pd.DataFrame, scores: pd.Series, n_years=2):
    """AUC for relapse within n years, from a frame with time and event columns."""
    nyear_relapse_is_defined = get_rows_where_nyear_relapse_is_defined(
        df["time_to_event"], df["event_observed"], n_years=n_years
    )
    y_true = df["time_to_event"][nyear_relapse_is_defined] <= 365 * n_years
    y_pred = scores[nyear_relapse_is_defined]
    return roc_auc_score(y_true, y_pred)


def get_roc_curve_for_predicting_nyear_relapse(
    time_to_event: pd.Series,
    event_observed: pd.Series,
    scores: pd.Series,
    n_years=2,
    time_to_event_unit: Literal["days", "months"] = "days",
):
    """False-positive and true-positive rates for relapse within n years."""
    if time_to_event_unit == "months":
        time_to_event = time_to_event * 30.44
    nyear_relapse_is_defined = get_rows_where_nyear_relapse_is_defined(time_to_event, event_observed, n_years=2)
    y_true = time_to_event[nyear_relapse_is_defined] <= 365 * n_years
    y_pred = scores[nyear_relapse_is_defined]
    x, y, _ = roc_curve(y_true, y_pred)
    return x, y


@dataclass
class ROCAnalysis:
    """2-year ROC for one score vector and one outcome frame."""

    scores: pd.Series = None
    modelfit_df: pd.DataFrame | None = None
    time_to_event: pd.Series = None
    event_observed: pd.Series = None

    def __post_init__(self):
        if self.modelfit_df is None:
            assert self.time_to_event is not None and self.event_observed is not None, "Pass either modelfit_df or time_to_event and event_observed"   # fmt: skip
        else:
            self.time_to_event = self.modelfit_df["time_to_event"]
            self.event_observed = self.modelfit_df["event_observed"]

    def get_auc(self, n_years=2):
        """AUC for relapse within n years."""
        known = self.get_rows_where_outcome_is_known(n_years)
        y_true = self.time_to_event[known] <= 365 * n_years
        return roc_auc_score(y_true, self.scores[known])

    def get_roc_curve(self, n_years=2):
        """False-positive and true-positive rates for relapse within n years."""
        known = self.get_rows_where_outcome_is_known(n_years)
        y_true = self.time_to_event[known] <= 365 * n_years
        x, y, _ = roc_curve(y_true, self.scores[known])
        return x, y

    def get_rows_where_outcome_is_known(self, n_years):
        return (self.event_observed == 1) | (self.time_to_event >= n_years * 365)
