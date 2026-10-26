from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from abc import abstractmethod
import torch
from utils.display import *


class BasePenalizer:

    @abstractmethod
    def calc_loss_contribution(self, x, location, scale):
        pass

    @abstractmethod
    def plot_penalty_function(self, ax=None, xmax=None, location=None, scale=None):
        pass


@dataclass
class EffectiveSampleSizePenalizer(BasePenalizer):
    """Class responsible for calculating loss contribution that
    discourages extreme solutions in the form of low effective sample size and
    unbalanced distribution in the transformed data.
    """

    min_n_mutas_soft: float = 10
    penalizer: float = 2

    def calc_loss_contribution(self, x: torch.tensor, location: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        x_transf = self.transform(x, location, scale)
        effective_samplesize = self.calc_effective_samplesize(x_transf)
        self.effective_samplesize = effective_samplesize
        return self.calc_penalty_for_effective_samplesize(effective_samplesize)

    def calc_penalty_for_effective_samplesize(self, effective_samplesize) -> torch.Tensor:
        return torch.relu(self.min_n_mutas_soft - effective_samplesize) ** 2 * self.penalizer

    def calc_effective_samplesize(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum() ** 2 / (x**2).sum()

    def transform(self, x: torch.Tensor, location, scale) -> torch.Tensor:
        return 1 / (1 + torch.exp(-(x - location) / scale))

    def plot_penalty_function(
        self,
        x_vaf: torch.tensor,
        ax=None,
        xmax=None,
        location=None,
        scale=None,
    ):
        if ax is None:
            fig, ax = plt.subplots()
        if not isinstance(x_vaf, torch.Tensor):
            x_vaf = torch.Tensor(x_vaf)
        # CONVERT INPUTS TO TO FLOATS IF NEED BE
        if isinstance(location, torch.Tensor):
            location = location.item()
        if isinstance(scale, torch.Tensor):
            scale = scale.item()

        # PLOT LOSS CONTRIBUTION FOR DIFFERENT EFFECTIVE SAMPLE SIZES
        ess_vals = torch.tensor(np.linspace(0, 50, 500))
        x_transf = self.transform(x_vaf, location, scale)
        ess_at_solution = self.calc_effective_samplesize(x_transf)
        loss_at_solution = self.calc_penalty_for_effective_samplesize(ess_at_solution)
        ax.axvline(self.min_n_mutas_soft, c="r", label="Soft minimum for ESS", linestyle="--")   # fmt: skip
        ax.scatter(
            ess_at_solution.item(), loss_at_solution.item(), edgecolors="k", label="ESS and penalty for solution"
        )
        ax.set_title("Penalizer for low Effective Sample Size", fontsize=10)   # fmt: skip
        ax.plot(ess_vals, self.calc_penalty_for_effective_samplesize(ess_vals), c="r", label="ESS Loss contribution")
        ax.set_xlabel("effective sample size")
        ax.set_ylabel("loss contribution")
        ax.legend(fontsize=8)
