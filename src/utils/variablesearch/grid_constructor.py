from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class GridConstructor:
    """Encapsulates VAF grid construction parameters and methods.

    Controls how the grid of VAF cutoffs is built for the univariate
    filter-optimization analysis in VariableSearcher.
    """

    vaf_grid: Union[np.ndarray, Literal["adaptive", "fixed"]] = "adaptive"
    n_vaf_gridpoints: int = 10
    adaptive_grid_cap: float = None
    min_cases_per_mutaclass: int = 10

    def __post_init__(self):
        if isinstance(self.vaf_grid, str):
            if self.vaf_grid == "fixed":
                self.vaf_grid = np.linspace(1e-10, 0.1, 11)
                self.n_vaf_gridpoints = len(self.vaf_grid)
        else:
            self.n_vaf_gridpoints = len(self.vaf_grid)

    def prepare_vaf_grid(self, largest_allowed_cutoff) -> np.ndarray:
        if isinstance(self.vaf_grid, (list, np.ndarray)):
            vaf_grid = self.vaf_grid[self.vaf_grid <= largest_allowed_cutoff]
        elif self.vaf_grid == "adaptive":
            vaf_grid = np.linspace(1e-9, largest_allowed_cutoff, self.n_vaf_gridpoints)
        return vaf_grid

    def get_largest_allowed_vaf_cutoff(self, univarfit_df) -> float:
        """Finds the largest VAF cutoff such that there are at least
        ``min_cases_per_mutaclass`` mutations in the dataset."""
        max_vaf_cutoff = univarfit_df["VAF"].sort_values().iloc[-self.min_cases_per_mutaclass]
        if self.adaptive_grid_cap is not None:
            max_vaf_cutoff = np.min([max_vaf_cutoff, self.adaptive_grid_cap])
        n_mutas = (univarfit_df["VAF"] >= max_vaf_cutoff).sum()
        if n_mutas < self.min_cases_per_mutaclass:
            raise ValueError("Insufficient number of non-zero VAFs")
        if max_vaf_cutoff == 0:
            raise ValueError(f"max_vaf_cutoff == 0 not allowed! {(univarfit_df.VAF>0).sum()=}")
        return max_vaf_cutoff

    def classes_are_too_unbalanced(self, univarfit_df) -> bool:
        """Checks if muta-positive or muta-negative classes have enough hits
        to run CPH analysis."""
        n_mutas = univarfit_df["MUTATED"].sum()
        n_non_mutas = len(univarfit_df) - n_mutas
        return (n_mutas < self.min_cases_per_mutaclass) | (n_non_mutas < self.min_cases_per_mutaclass)


    def construct_adaptive_grid(self, vafs: pd.Series, plot_grid=False):
        """NOT CURRENTLY USED: Constructs a grid so that the fraction of VAF
        values that exceed cutoffs increases linearly. Also determines an
        appropriate grid-resolution based on settings."""
        nonzero_vafs = vafs[vafs > 0]
        n_nonzero_vafs = len(nonzero_vafs)
        n_actual_cutoffs = np.min(
            [np.floor(n_nonzero_vafs / self.min_cases_per_mutaclass), self.n_vaf_gridpoints]
        )

        fractions = np.linspace(0, 1, int(n_actual_cutoffs) + 1)[:-1]
        vaf_grid = nonzero_vafs.quantile(fractions).values

        if plot_grid:
            _, ax = plt.subplots(figsize=(4, 4))
            ax.plot(nonzero_vafs, "o", alpha=0.5)
            ax.grid()
            for u in vaf_grid:
                ax.axhline(
                    y=u, linestyle="--", alpha=0.5, label=f"N exceedences: {(vafs>=u).sum()} ({(vafs>=u).mean():.1%})"
                )
            ax.legend(fontsize=7)
            ax.loglog()

        return vaf_grid
