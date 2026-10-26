"""Picks the winning VAF cutoff from the per-gene grid search.

Each gene is tested at many cutoffs, with the oncogenicity filter on and off.
MinPvalOverEligible keeps the cutoff with the best p-value (AdaPipe).
ManuallySelectedCutoff keeps the grid point nearest a fixed percentage (exact
match if grid is setup right), which is the stiff benchmark. The chosen number
is stored as vaf_cutoff.

Applying that number to patients happens later, in cutoff_choosers.py.
"""

from dataclasses import dataclass

from typing import Literal
from utils.display import *

import numpy as np
import pandas as pd


@dataclass
class BaseCutoffSelector:
    oncofilter_selection: Literal[
        "pick_on",
        "pick_off",
        "highest_neglogp",
        "lowest_p_softcut",
    ] = "highest_neglogp"
    min_gain_for_oncoswitch: float = 0.0  # Minimim required gain in the selected performance metric to set "oncogenicity filtering" as the best strategy     # fmt: skip

    def __post_init__(self):
        allowed = {
            "pick_on",
            "pick_off",
            "highest_neglogp",
            "lowest_p_softcut",
        }
        if self.oncofilter_selection not in allowed:
            red_print(f"`how_to_choose_best_oncofilter` '{self.oncofilter_selection}' is not in {allowed}")   # fmt: skip

    def extract_values_at_the_best_cutoff(self, raw_df_cleaned: pd.DataFrame) -> pd.DataFrame:
        return raw_df_cleaned.apply(self._get_value_at_best_idx, axis=1)

    def _get_value_at_best_idx(self, row) -> pd.Series:
        """Extract the values at the best cutoff index for this row.
        Returns a Series with the same columns as the row."""
        idx_best = self.get_index_of_best_cutoff(row)
        return row.apply(lambda x: x[idx_best])

    def get_index_of_best_cutoff(self, row) -> int:
        """To be implemented in child classes. Should return the index of
        the best cutoff for the row."""
        raise NotImplementedError


@dataclass
class MinPvalOverEligible(BaseCutoffSelector):
    min_n_mutas: int = 0

    def get_index_of_best_cutoff(self, row) -> int:
        """Extract cutoff with best p-value over cutoffs that yield atleast
        ``min_n_muta_to_allow_vaf_cutoff`` mutations."""
        mask = row["n_muta"] >= self.min_n_mutas
        return np.argmax(row["neglogp"] * mask) if mask.any() else np.argmin(row["p"])


@dataclass
class ManuallySelectedCutoff(BaseCutoffSelector):

    def __init__(self, manually_chosen_cutoff=0.03, oncofilter_selection="pick_off"):
        self.manually_chosen_cutoff = manually_chosen_cutoff
        self.oncofilter_selection = oncofilter_selection

    def get_index_of_best_cutoff(self, row) -> int:
        """Index of the grid point nearest the chosen fixed cutoff."""
        return np.argmin(np.abs(row["vaf_cutoff"] - self.manually_chosen_cutoff))
