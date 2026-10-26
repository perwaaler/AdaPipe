"""Chooses which stored cutoff to apply when marking a patient mutated.

This runs after the grid search. grid_optimized, from_treefit, and
from_softcutfit each read one stored column: the grid-search cutoff, the
survival-tree cutoff, or the fitted softcut location.
"""

from dataclasses import dataclass
from typing import Literal, List

import pandas as pd


@dataclass
class VAFCutoffSelector:
    """Pick which grid-search summary column supplies the VAF threshold per gene."""

    vaf_cutoff_type: Literal[
        "grid_optimized",
        "from_treefit",
        "from_softcutfit",
    ] = "grid_optimized"

    def select_cutoffs(
        self,
        univariate_analysis_info_df: pd.DataFrame = None,
        vaf_cols: List[str] = None,
    ) -> pd.Series:
        """One cutoff per gene (aligned with ``vaf_cols``)."""
        column_for_cutoff = {
            "grid_optimized": "vaf_cutoff",
            "from_treefit": "vaf_cutoff_treefit",
            "from_softcutfit": "loc_softcut",
        }
        if self.vaf_cutoff_type not in column_for_cutoff:
            raise ValueError(f"Unknown vaf_cutoff_type={self.vaf_cutoff_type!r}")
        column = column_for_cutoff[self.vaf_cutoff_type]
        return pd.Series(univariate_analysis_info_df[column].values, index=vaf_cols)
