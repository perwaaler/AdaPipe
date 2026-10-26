"""Logistic VAF transform used to rank genes.

A two-parameter sigmoid squashes max VAF. The published softcut p-value is a
Cox model on that squashed VAF, fit outside this module.
"""

from utils.display import *
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from utils.models.penalizers import EffectiveSampleSizePenalizer


@dataclass
class TorchSurvivalDataPrepper:
    """Sort a one-column frame by time and return feature, time, and event tensors."""

    feature_column: str = None

    def get_torch_data(self, modelfit_df):
        if self.feature_column is None:
            features = [x for x in modelfit_df.columns if not x in ["RED_ID", "time_to_event", "event_observed"]]
            assert features.__len__() == 1, f"Feature column ambiguous: found: {features}"
            feature = features[0]
        else:
            feature = self.feature_column

        df_sorted = modelfit_df.sort_values(by="time_to_event", ascending=False).reset_index(drop=True)
        X = torch.tensor(df_sorted[feature].values, dtype=torch.float32).unsqueeze(1)
        time = torch.tensor(df_sorted["time_to_event"].values, dtype=torch.float32)
        event = torch.tensor(df_sorted["event_observed"].values, dtype=torch.float32)
        return X, time, event


class CoxPHSoftCut(nn.Module):
    """Sigmoid of VAF. Location is the cutoff; scale is the width."""

    def __init__(
        self,
        trainable_vaf_scale=True,
        vaf_scale_initval=0.08,
        vaf_location_initval=0.1,
        effect_initval=0.1,
    ):
        super().__init__()
        self.loc_scale = nn.Linear(1, 1, bias=True)
        self.effect = nn.Linear(1, 1, bias=False)
        self.trainable_vaf_scale = trainable_vaf_scale

        if not self.trainable_vaf_scale:
            self.loc_scale.weight.requires_grad = False

        with torch.no_grad():
            b0, w0 = bias_weight_parametrization(vaf_location_initval, vaf_scale_initval)
            self.loc_scale.bias.fill_(b0)
            self.loc_scale.weight.fill_(w0)
            self.effect.weight.fill_(effect_initval)

        self.input_type = "VAF"
        self.vaf_activation_f = torch.sigmoid

    def forward(self, X_vaf):
        """Log hazard at each VAF."""
        x1 = self.loc_scale(X_vaf)
        x2 = self.vaf_activation_f(x1)
        return self.effect(x2)

    def find_good_initvals_with_gridsearch(self, X_vaf, event):
        """Pick a starting location and scale by a small grid. Off unless the trainer asks for it."""
        scale_initvals = np.linspace(0.001**2, 0.05**2, 10) ** 0.5
        loc_initvals = np.linspace(0.001**2, 0.2**2, 10) ** 0.5

        best_loss = np.inf
        best_bias = None
        best_weight = None

        for loc in loc_initvals:
            for scale in scale_initvals:
                bias, weight = bias_weight_parametrization(loc, scale)
                with torch.no_grad():
                    self.loc_scale.weight.fill_(weight)
                    self.loc_scale.bias.fill_(bias)
                    loss = cox_ph_loss(self.forward(X_vaf), event)

                if loss < best_loss:
                    teal_print(f"Best loss improved: {location_scale_parametrization(bias, weight)}")
                    best_loss = loss
                    best_bias = bias
                    best_weight = weight

        with torch.no_grad():
            self.loc_scale.bias.fill_(best_bias)
            self.loc_scale.weight.fill_(best_weight)

        location, scale = self.get_location_and_scale_params()
        silent_print("Model: sigmoid((VAF - location) / scale), then a log-hazard coefficient")
        blue_print(f"   bias: {self.loc_scale.bias.item():.5}, weight: {self.loc_scale.weight.item():.5}")
        blue_print(f"   effect: {self.effect.weight.item():.6}")
        blue_print(f"   VAF location: {location:.5f}, VAF scale: {scale:.5f}")

    def get_location_and_scale_params(self):
        """Location and scale of the sigmoid, from the linear layer's bias and weight."""
        return location_scale_parametrization(bias=self.loc_scale.bias.item(), weight=self.loc_scale.weight.item())

    def plot_vaf_curve(
        self, ax=None, max_vaf=0.25, textbox_string=None, print_legend=True, alpha_sigmoid=1, x_vaf=None
    ):
        """The fitted sigmoid, with the location marked."""
        if ax is None:
            _, ax = plt.subplots(figsize=(3, 3))

        vaf_grid = np.linspace(0, max_vaf, 100)
        squashed_vaf = self.squash_vaf(vaf_grid)
        ax.plot(vaf_grid, squashed_vaf, c="k", alpha=alpha_sigmoid)
        ax.set_xlim([-0.01, max_vaf])
        ax.grid(alpha=0.5)
        ax.axvline(
            x=self.get_location_and_scale_params()[0],
            c="m",
            linestyle="--",
            label=f"VAF location: {self.get_location_and_scale_params()[0]:.3f}",
        )
        if x_vaf is not None:
            if isinstance(x_vaf, torch.Tensor):
                vaf_vals = x_vaf[x_vaf > 0].detach().numpy().flatten()
            else:
                vaf_vals = np.asarray(x_vaf)
                vaf_vals = vaf_vals[vaf_vals > 0]
            ax.plot(vaf_vals, np.zeros_like(vaf_vals), "|", color="steelblue", alpha=0.5, markersize=8)

        ax.set_xlabel("VAF")
        ax.set_ylabel("VAF sigmoided")
        ax.tick_params("x", labelsize=7)
        ax.tick_params("y", labelsize=7)
        if textbox_string is None:
            textbox_string = f"VAF location: {self.get_location_and_scale_params()[0]:.3f}"
        if print_legend:
            ax.legend()

    def squash_vaf(self, vaf_vals):
        """Sigmoid of these VAF values. Expects 100 values."""
        vaf_grid = torch.tensor(vaf_vals, dtype=torch.float32)
        vaf_grid = torch.reshape(vaf_grid, shape=(100, 1))
        return torch.sigmoid(self.loc_scale(vaf_grid)).detach().numpy()


@dataclass
class PrintConfig:
    """Debug print and plot toggles for the softcut trainer."""

    loss_summary: bool = False
    loss_summary_every: int = 100
    ess: bool = False
    ess_every: int = 50
    plot_curve_evolution: bool = False
    plot_ess_penalty: bool = False


class CoxPHUnivariateTrainer:
    """Train the sigmoid by Cox partial likelihood, plus L2, plus the ESS penalty when set.

    The L2 term is on the network weights, so it keeps the sigmoid from becoming a step.
    It is separate from the Cox penalizer applied later to the squashed VAF.
    ``find_initvals_with_gridsearch`` and ``penalize_negative_loc_param`` are off unless set.
    """

    PrintConfig = PrintConfig

    def __init__(
        self,
        gene=None,
        model: CoxPHSoftCut = None,
        n_epochs=2000,
        learning_rate=0.5,
        epochs_to_plot_diagnostics=3,
        lambda_l2loss=0.00001,
        effective_sample_regularizer: EffectiveSampleSizePenalizer = None,
        penalize_negative_loc_param=False,
        min_n_epochs=250,
        find_initvals_with_gridsearch=False,
        verbose=1,
        display: PrintConfig | None = None,
    ):
        self.model = CoxPHSoftCut() if model is None else model

        self.gene = gene
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.display = PrintConfig() if display is None else display
        self.find_initvals_with_gridsearch = find_initvals_with_gridsearch
        self.effective_sample_regularizer = effective_sample_regularizer
        self.lambda_l2loss = lambda_l2loss
        self.min_n_epochs = min_n_epochs
        self.penalize_negative_loc_param = penalize_negative_loc_param

        self.losses = []
        self.losses_cph = []
        self.losses_l2 = []
        self.losses_mutacount = []
        self.location_vals = []
        self.scale_vals = []
        self.verbose = verbose
        self.epochs_to_plot_diagnostics = epochs_to_plot_diagnostics
        if self.display.plot_curve_evolution:
            self.fig_curve_evolution, self.ax_curve_evolution = plt.subplots(figsize=(3, 2))
            self.ax_curve_evolution.set_title(f"Evolution of fitted curves (gene={self.gene})")

    def fit(self, X_muta, X_vaf, X_onco, time, event):
        """Fit location, scale, and the log-hazard coefficient. ``X_muta`` and ``X_onco`` are ignored."""
        failure_to_improve_counter = 0
        self.X_vaf = X_vaf
        self.event = event

        if self.find_initvals_with_gridsearch:
            self.model.find_good_initvals_with_gridsearch(X_vaf, event)

        for epoch in range(self.n_epochs):
            self._epoch = epoch
            self.optimizer.zero_grad()
            log_hazard = self.model(X_vaf)

            loss_cph = cox_ph_loss(log_hazard, event)
            l2_loss = sum(p.pow(2).sum() for p in self.model.parameters()) * self.lambda_l2loss
            loc, scale = location_scale_parametrization(self.model.loc_scale.bias, self.model.loc_scale.weight)
            mutacount_loss = self.get_ESS_penalty(X_vaf, loc, scale)
            loss_total = loss_cph + l2_loss + mutacount_loss

            if not torch.isfinite(loss_total):
                break
            loss_total.backward()
            self.optimizer.step()
            if not model_params_are_finite(self.model):
                break
            self.losses.append(loss_total.item())

            green_print(f"LOSSES:  total = {loss_total.item():<10.3f} ESS = {mutacount_loss.item():<10.5f} L2 = {l2_loss.item():<10.3f}    FITTED PARAMS:  loc = {loc.item():<10.3f} scale = {scale.item():<10.3f}") if self.display.loss_summary and self._is_print_epoch() else None  # fmt: skip

            if epoch >= self.min_n_epochs:
                total_improvement = self.losses[0] - self.losses[-1]
                if total_improvement <= 0:
                    failure_to_improve_counter += 1
                else:
                    relative_improvement_over_last_n_epochs = (self.losses[-self.min_n_epochs] - self.losses[-1]) / total_improvement * 100   # fmt: skip
                    if total_improvement > 0 and relative_improvement_over_last_n_epochs < 1:
                        failure_to_improve_counter += 1

            self.losses_cph.append(loss_cph.item())
            self.losses_l2.append(l2_loss.item())
            self.losses_mutacount.append(mutacount_loss.item())

            self.location_vals.append(loc)
            self.scale_vals.append(scale)

            if self.display.plot_curve_evolution:
                self.plot_evolution_of_fitted_curve(epoch)

            if failure_to_improve_counter >= 10:
                break

        if self.display.plot_ess_penalty and self.effective_sample_regularizer is not None:
            self.effective_sample_regularizer.plot_penalty_function(X_vaf, location=loc, scale=scale)

    def plot_losses(self, ax=None):
        """Cox, L2, and ESS contributions over epochs."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(3, 3))
        total_losses = np.array(self.losses)
        cph_losses = np.array(self.losses_cph)
        losses_l2 = np.array(self.losses_l2)
        losses_mutacount = np.array(self.losses_mutacount)
        ax.plot(total_losses, label="total loss", c="k")
        ax.plot(cph_losses, alpha=0.5, linestyle="--", label=f"CPH loss (mean={total_losses.mean():.3f})")   # fmt: skip
        ax.plot(cph_losses + losses_mutacount, alpha=0.5, linestyle="--", label=f"CPH loss + Muta-count loss (mean={losses_mutacount.mean():.3f})")   # fmt: skip
        ax.plot(cph_losses + losses_l2, alpha=0.5, linestyle="--", label=f"CPH loss + L2 loss (mean={losses_l2.mean():.3f})")   # fmt: skip
        ax.grid(alpha=0.5)
        ax.set_xlabel("epoch", fontsize=7)
        ax.set_ylabel("loss", fontsize=7)
        ax.tick_params(axis="x", labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_title(f"Training loss (input={self.model.input_type})")
        ax.legend()

    def plot_evolution_of_fitted_curve(self, epoch):
        """Draw the sigmoid every few early epochs. Used when ``plot_curve_evolution`` is on."""
        max_epoch_to_plot = 100
        plot_epoch = epoch % self.epochs_to_plot_diagnostics == 0 and epoch < max_epoch_to_plot
        if plot_epoch or epoch == max_epoch_to_plot:
            max_vaf = 0.25
            vaf_grid = np.linspace(0, max_vaf, max_epoch_to_plot)
            squashed_vaf = self.model.squash_vaf(vaf_grid)
            ax = self.ax_curve_evolution
            if epoch == max_epoch_to_plot:
                c = "r"
                alpha = 1
            else:
                c = "k"
                alpha = 0.5
            ax.plot(vaf_grid, squashed_vaf, c=c, alpha=alpha)
            ax.set_xlim([-0.01, max_vaf])
            ax.grid(alpha=0.5)
            ax.tick_params(labelsize=7)

    def get_ESS_penalty(self, X_vaf, loc, scale):
        """Effective-sample-size penalty, or zero when no penalizer is set."""
        if self.effective_sample_regularizer is not None:
            ESS_loss = self.effective_sample_regularizer.calc_loss_contribution(X_vaf, loc, scale)
            red_print(f"ESS loss:      {ESS_loss.item():<15.3f} ESS = {self.effective_sample_regularizer.effective_samplesize.item():<10.4f} (epoch {self._epoch})") if self.display.ess and self._epoch % self.display.ess_every == 0 else None  # fmt: skip
        else:
            ESS_loss = torch.tensor([0.0], dtype=loc.dtype, device=loc.device)
        return ESS_loss

    def _is_print_epoch(self):
        return (self._epoch + 1) % self.display.loss_summary_every == 0


def model_params_are_finite(model: nn.Module) -> bool:
    return all(torch.isfinite(p).all().item() for p in model.parameters())


def cox_ph_loss(log_hazard, event):
    """Negative Cox partial log-likelihood. Rows must already be sorted by time, descending."""
    hazard = log_hazard.view(-1)
    exp_hazard = torch.exp(hazard)
    cum_sum = torch.cumsum(exp_hazard, dim=0)
    log_cum_sum = torch.log(cum_sum)
    partial_log_likelihood = hazard - log_cum_sum
    return -torch.sum(partial_log_likelihood * event)


def bias_weight_parametrization(location, scale):
    """Bias and weight of ``bias + weight * VAF`` for a given location and scale."""
    bias = -location / scale
    weight = 1 / scale
    return bias, weight


def location_scale_parametrization(bias, weight):
    """Location and scale from the linear layer's bias and weight."""
    location = -bias / weight
    scale = 1 / weight
    return location, scale
