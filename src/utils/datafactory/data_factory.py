"""DataFactory is an orchestrator class that builds train and test datasets for
each cross-validation split."""

from dataclasses import dataclass
from dataclasses import field
from sklearn.preprocessing import MinMaxScaler
from typing import Literal
from typing import Tuple
from typing import List
from typing import Union
from utils.paths import Paths
from utils.display import *
from utils.prepare_data.prepare_data import DataBuilder
from utils.variablesearch.variable_searcher import VariableSearcher
from utils.variablesearch.cutoff_choosers import VAFCutoffSelector
from utils.variablesearch import grid_cutoff_selectors

import utils.datafactory.gene_selectors as gene_selectors
import utils.datafactory.design_matrixes as design_matrixes

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class DataNormalizer:
    """Min–max scale feature columns to [0, 1] using statistics from the train fold only."""

    xvars: List[str] = None
    scaler: List[str] = None

    def apply_transform(self, modelfit_df: pd.DataFrame) -> pd.DataFrame:
        """Apply the train-fitted scaler to the test fold (same ``xvars`` as fit)."""
        modelfit_df[self.xvars] = self.scaler.transform(modelfit_df[self.xvars])
        return modelfit_df

    def fit_transform(self, modelfit_df: pd.DataFrame) -> pd.DataFrame:
        """Fit on all columns except ``RED_ID``, ``time_to_event``, and ``event_observed``."""
        scaler = MinMaxScaler()
        xvars = [c for c in modelfit_df.columns if c not in ["RED_ID", "time_to_event", "event_observed"]]
        modelfit_df[xvars] = scaler.fit_transform(modelfit_df[xvars])
        self.scaler = scaler
        self.xvars = xvars
        return modelfit_df


class DataFactory:
    """Prepares data for cross-validation testing of end-to-end modeling
    pipelines that include variant-filter fitting and gene selection.
    build_data_for_ith_fold creates train and validation data for split i."""

    def __init__(
        self,
        variants_filtered_df,
        data_builder: DataBuilder | None = None,
        # Cutoff selection
        grid_cutoff_selector: (
            Union[
                grid_cutoff_selectors.MinPvalOverEligible,
                grid_cutoff_selectors.ManuallySelectedCutoff,
            ]
            | None
        ) = None,
        feature_type: Literal["MUTAS", "VAF", "SOFTCUT"] = "MUTAS",
        # Variable selection
        path_varsearcher_dumpdir=None,
        path_crossval_split=None,
        gene_selector: gene_selectors.GeneSelectorBase | None = None,
        vaf_cutoff_selector: VAFCutoffSelector | None = None,
        # Data fusion
        design_matrix_preparator: design_matrixes.BaseDesignMatrix | None = None,
        # Misc
        load_varsearchers_at_init=False,
        red_id_restriction=None,
        red_id_restriction_test_sets: list[str] = None,  # Example usecase: want to train on NSCLC and test only on LUAD
        print_dataset_sizes=False,
        verbose=0,
    ):
        """See class docstring for roles of ``grid_cutoff_selector`` and ``feature_type``.

        ``red_id_restriction`` — intersect train and test ``RED_ID`` lists with this set
        (e.g. Norway-only or LUAD-only discovery).

        ``red_id_restriction_test_sets`` — after building the test matrix, keep only
        these ``RED_ID``s (used when training on NSCLC but scoring LUAD in the test fold).
        """
        self.verbose = verbose
        # GET CROSS-VALIDATION SPLIT FOR SPLIT-INDEX
        if path_crossval_split is None:
            red_print("No cross-validation split path given: defaulting to Norway split")
            path_crossval_split = Paths().cv_splits_norway_path
        self._cv_splits: List[str] = Paths.load_json(path_crossval_split)
        self._variants_filtered_df = variants_filtered_df
        self._grid_cutoff_selector = grid_cutoff_selectors.MinPvalOverEligible() if grid_cutoff_selector is None else grid_cutoff_selector  # fmt: skip
        self.feature_type = feature_type
        # Check that both arguments are provided as a safety feature
        self.design_matrix_preparator = design_matrixes.Genomic() if design_matrix_preparator is None else design_matrix_preparator  # fmt: skip
        self.data_normalizer = DataNormalizer()
        if path_varsearcher_dumpdir is None:
            path_varsearcher_dumpdir = Paths().crossval_varselection_dump_flexigrid_norway_dir
        self.path_to_varsearcher_dumpdir = path_varsearcher_dumpdir
        self.gene_selector = gene_selectors.TopKBestGenes() if gene_selector is None else gene_selector  # fmt: skip
        self.red_id_restriction = red_id_restriction
        self.red_id_restriction_test_sets = red_id_restriction_test_sets
        self.data_builder = DataBuilder() if data_builder is None else data_builder
        self.clinical_df = self.data_builder.get_clinical_data()

        # Container class that stores results and metadata from running variable selection
        self.info = self.VarselectionInfo()
        self.vaf_cutoff_selector = VAFCutoffSelector("grid_optimized") if vaf_cutoff_selector is None else vaf_cutoff_selector  # fmt: skip

        self.varsearchers: List[VariableSearcher] = []
        if load_varsearchers_at_init:
            self.load_varsearchers()

        self._print_dataset_sizes = print_dataset_sizes
        self.verbose = verbose
        # Argument checks
        assert feature_type in ["MUTAS", "VAF", "SOFTCUT"], f"Feature type {feature_type} not recognized; use MUTAS, VAF, or SOFTCUT"   # fmt: skip

    # METHODS THAT UPDATES VARIABLE SEARCHERS WHEN CUTOFF SELECTOR IS CHANGED
    @property
    def grid_cutoff_selector(self):
        return self._grid_cutoff_selector

    @grid_cutoff_selector.setter
    def grid_cutoff_selector(self, value):
        """Re-executes cutoff-selection if the cutoff selection strategy is updated."""
        if value != self._grid_cutoff_selector:
            blue_print(f"Cutoff selection strat updated to {value.__class__.__name__}; reloading varsearchers...")   # fmt: skip
            self._grid_cutoff_selector = value
            self.load_varsearchers()

    @dataclass
    class VarselectionInfo:
        """Per-fold bookkeeping filled by :meth:`build_data_for_ith_fold`."""

        selected_genes: list = field(default_factory=lambda: list(range(5)))
        stats_for_selected: list = field(default_factory=lambda: list(range(5)))
        n_selected: list = field(default_factory=lambda: list(range(5)))

    def load_varsearchers(self):
        """Load five :class:`~utils.variablesearch.variable_searcher.VariableSearcher`
        instances from ``path_to_varsearcher_dumpdir/fold{i}/``.

        Each searcher is restricted to that fold's **training** ``RED_ID``s when
        reading variants; grid results were computed on the same split during
        ``run_gridsearch_analyses`` (or TCGA tuning via :class:`~utils.build_models_for_TCGAvalidation.CrossValHyperParOptimizer`).
        """
        self.varsearchers: List[VariableSearcher] = []
        for i, red_id_fold_i in enumerate(self._cv_splits["train"]):
            # LOAD VARIABLE-SELECTION RESULTS
            varsearcher_i = VariableSearcher(
                variants_df=self._variants_filtered_df,
                clinical_df=self.clinical_df,
                endpoint=self.data_builder.endpoint,
                grid_cutoff_selector=self._grid_cutoff_selector,
                red_id_used_to_fit_filters=red_id_fold_i,
                vaf_cutoff_selector=self.vaf_cutoff_selector,
            )
            # Load CV variable-selection results from fold i
            varsearcher_i.load_results_from_json(self.path_to_varsearcher_dumpdir, cv_idx=i)
            # Get cutoffs and statistics
            varsearcher_i.compute_result_summary_dfs()
            self.varsearchers.append(varsearcher_i)

    def update_oncofilter_strategy(
        self,
        new_strategy: Literal[
            "pick_on",
            "pick_off",
            "highest_neglogp",
            "lowest_p_softcut",
        ] = "highest_neglogp",
    ):
        """Switch per-gene oncogenicity branch (on / off / best p) and refresh
        ``summary_best_models_df`` on every loaded searcher (required before
        building features if the hyperparameter JSON specifies e.g. ``pick_on``)."""
        assert self.varsearchers != []
        blue_print("Re-merging summary_df__off and summary_df__on (updates summary_best_models_df after changing grid_cutoff_selector.oncofilter_selection)") if self.verbose > 0 else None # fmt: skip
        for varsearcher_i in self.varsearchers:
            varsearcher_i.grid_cutoff_selector.oncofilter_selection = new_strategy
            varsearcher_i.merge_oncogenicity_filtering_results()

    def build_data_for_ith_fold(self, cv_idx=0) -> Tuple[pd.DataFrame]:
        """Return ``(modelfit_train, modelfit_test)`` for fold ``cv_idx``.

        Pipeline: split ``RED_ID``s → ``gene_selector`` → binary/VAF/softcut matrix via
        the fold's varsearcher → optional clinical columns from
        ``design_matrix_preparator`` → align dummy columns between train/test →
        min–max normalize gene (and clinical) features using train statistics only.

        Both frames share the same column order. Outcome columns follow
        ``data_builder.endpoint`` (TTR, DSS, etc.).
        """
        red_id_train, red_id_test = self._split_red_id(cv_idx)
        varsearcher_i = self.varsearchers[cv_idx]
        selected_genes = self.gene_selector.select_genes(varsearcher_i=varsearcher_i, cv_idx=cv_idx)
        orange_print(f"({cv_idx}) Genes selected: {", ".join(selected_genes)}") if self.verbose > 0 else None   # fmt: skip

        # --- FEATURIZE SELECTED GENES
        genomic_modelfit_df = varsearcher_i.get_modelfit_df_for_subset_of_genes(
            selected_genes,
            feature_type=self.feature_type,
            vaf_cutoff_selector=self.vaf_cutoff_selector,
        )
        self.info.selected_genes[cv_idx] = list(selected_genes)

        # ---- EXTRACT TRAIN AND TEST DATA
        genomic_modelfit_df_train = genomic_modelfit_df.query(f"RED_ID in {red_id_train}").reset_index(drop=True).copy()   # fmt: skip
        genomic_modelfit_df_test = genomic_modelfit_df.query(f"RED_ID in {red_id_test}").reset_index(drop=True).copy()  # fmt: skip
        modelfit_df_train = self.design_matrix_preparator.prepare_modelfit_df(genomic_modelfit_df_train, fit_new_data_transforms=True, clinical_df=self.clinical_df)   # fmt: skip
        modelfit_df_test = self.design_matrix_preparator.prepare_modelfit_df(genomic_modelfit_df_test, fit_new_data_transforms=False, clinical_df=self.clinical_df)   # fmt: skip
        if self.red_id_restriction_test_sets is not None:
            # Drop samples in test set that do not lie in the specified subset (e.g., histology=LUAD)
            modelfit_df_test = modelfit_df_test.query(f"RED_ID in {self.red_id_restriction_test_sets}").copy()
        # ENSURE ALL CATEGORIES ARE REPRESENTED IN EACH DATASET
        modelfit_df_train, modelfit_df_test = self.ensure_all_categories_are_represented_in_each_set(
            modelfit_df_train, modelfit_df_test
        )
        # CHECK TRAIN & VALIDATION SETS ARE INDEPENDENT
        assert set(modelfit_df_train["RED_ID"]).intersection(modelfit_df_test["RED_ID"]) == set(), f"Train and test set contain overlapping IDs"   # fmt: skip
        n_cols_train, n_cols_test, n_train, n_test = len(modelfit_df_train.columns), len(modelfit_df_test.columns), len(modelfit_df_train), len(modelfit_df_test)   # fmt: skip
        orange_print(f"({cv_idx}) N: total={n_train+n_test}, train={n_train}, test={n_test}  |  N features: train={n_cols_train - 3}, test={n_cols_test - 3}") if self.verbose > 0 else None   # fmt: skip

        # --- NORMALIZE DATA
        modelfit_df_train = self.data_normalizer.fit_transform(modelfit_df_train)
        modelfit_df_test = self.data_normalizer.apply_transform(modelfit_df_test)

        # Check that dataframes have the same columns
        cols_in_train_but_not_in_test = set(modelfit_df_train.columns).difference(modelfit_df_test.columns)   # fmt: skip
        assert len(set(modelfit_df_train.columns).difference(modelfit_df_test.columns)) == 0, f"Columns in train but not in test {cols_in_train_but_not_in_test}"   # fmt: skip

        # ---------- STORE SOME STATS (SANITY CHECKS) -----------------------------------------------------------------------------------------
        # Store statistics about the selected genes
        self.info.stats_for_selected[cv_idx] = varsearcher_i.summary_best_models_df.loc[self.info.selected_genes[cv_idx]]
        self.info.n_selected[cv_idx] = self.info.selected_genes[cv_idx].__len__()

        return modelfit_df_train, modelfit_df_test

    def _split_red_id(self, cv_idx=0):
        """Lists of ``RED_ID`` in train and test for ``cv_idx`` from ``path_crossval_split`` JSON."""
        red_id_train, red_id_test = self._cv_splits["train"][cv_idx], self._cv_splits["test"][cv_idx]
        if self.red_id_restriction:
            restrict_red_id = lambda id_list: list(set(self.red_id_restriction).intersection(id_list))
            red_id_train = restrict_red_id(red_id_train)
            red_id_test = restrict_red_id(red_id_test)
        return red_id_train, red_id_test

    @staticmethod
    def ensure_all_categories_are_represented_in_each_set(modelfit_df_train, modelfit_df_test):
        """Align train/test columns when one-hot encoding clinical categories.

        If a dummy column appears in only one split, fill the missing split with
        that column's mode (expects binary dummies).
        """
        vars_only_in_train = set(modelfit_df_train.columns).difference(modelfit_df_test.columns)
        vars_only_in_test = set(modelfit_df_test.columns).difference(modelfit_df_train.columns)

        for var in vars_only_in_train:
            fill_value = modelfit_df_train[var].mode().iloc[0]
            modelfit_df_test[var] = fill_value
            n_values = modelfit_df_train[var].nunique()
            assert modelfit_df_train[var].nunique() == 2, f"{var} appears only in train data and has {n_values} values"

        for var in vars_only_in_test:
            fill_value = modelfit_df_test[var].mode().iloc[0]
            modelfit_df_train[var] = fill_value
            n_values = modelfit_df_train[var].nunique()
            assert modelfit_df_train[var].nunique() == 2, f"{var} appears only in test data and has {n_values} values"

        return modelfit_df_train, modelfit_df_test

    def print_variable_consistency_in_heatmap(
        self,
        variable_to_colorcode: Literal["neglogp", "p", "p_softcut"] = "neglogp",
        ax=None,
        fig=None,
        verbose=1,
    ):
        """Heatmap of genes picked each fold (stars) with univariate significance (color).

        Uses ``self.info.stats_for_selected`` populated by prior calls to
        :meth:`build_data_for_ith_fold`. Figure S4 in the paper supplements.
        """
        # GET DATAFRAME THAT INDICATES GENE SELECTION FOR EACH ITERATION
        selected_genes_per_fold = [pd.DataFrame({f"selection {i}": self.info.stats_for_selected[i].index}, index=self.info.stats_for_selected[i].index) for i in range(5)]   # fmt: skip
        gene_selection_df = pd.concat(selected_genes_per_fold, axis=1)
        gene_selection_df = gene_selection_df.map(lambda x: "*" if isinstance(x, str) else "")
        # Order genes by the number of folds in which they were selected
        genes_ordered_by_n_selections = gene_selection_df.apply(lambda row: (row == "*").sum(), axis=1).sort_values(ascending=False).index   # fmt: skip
        gene_selection_df = gene_selection_df.loc[genes_ordered_by_n_selections]
        index_all_selected = gene_selection_df.index

        # HEATMAP OF SIGNIFICANCE PER CV FOLD
        # Get -log10p for each gene in each iteration
        pvals_of_selected_df = pd.concat([self.varsearchers[i].summary_best_models_df.loc[index_all_selected, variable_to_colorcode] for i in range(5)], axis=1)   # fmt: skip

        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color="grey")

        scatter_long_df = (
            (gene_selection_df == "*").reset_index().melt(id_vars="index", var_name="cv", value_name="selected")
        )
        scatter_long_df = scatter_long_df.rename(columns={"index": "gene"})
        # Relabel from strings like 'iteration 1' to integers
        scatter_long_df["cv"] = scatter_long_df["cv"].str[-1].astype(int)
        # Keep only rows where selected == True
        scatter_long_df = scatter_long_df[scatter_long_df["selected"]]

        imshow_df = pvals_of_selected_df.transpose()
        gene_to_pos_map = {g: i for i, g in enumerate(imshow_df.columns)}
        scatter_long_df["x_pos"] = scatter_long_df["gene"].apply(lambda x: gene_to_pos_map[x])
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(15, 5))
        im = ax.imshow(imshow_df, cmap=cmap)
        # Add colorbar with smaller size and some space from the plot
        ax.scatter(scatter_long_df["x_pos"], scatter_long_df["cv"], c="k")
        ax.set_ylabel("CV iteration")
        ax.set_xticks(np.arange(len(pvals_of_selected_df)))
        ax.set_yticks(range(5))
        ax.set_xticklabels(pvals_of_selected_df.index)
        ax.tick_params(labelsize=8, rotation=45)
        ax.set_title("Genes Selected in Cross Validation", y=1.3, fontsize=15)
        infotxt = "\n".join(
            [
                f"*Significance measured by {variable_to_colorcode.replace('_', ' ')}",
                f"*Grey color indicates too few mutations in training split to model gene-mutation effect",
                f"*Genes ordered by the number of folds in which they were selected",
            ]
        )
        if verbose > 0:
            ax.text(0.0, -2, infotxt, ha="left", va="top", transform=ax.transAxes, alpha=0.8)   # fmt: skip

        cbar = fig.colorbar(im, ax=ax, pad=0.2, fraction=0.1, orientation="horizontal")
        string_map = lambda s: r"-log$_{10}$(p)" if s == "neglogp" else s
        cbar.set_label(f"Statistical significance ({string_map(variable_to_colorcode)}) at the selected cutoff")

        plt.tight_layout()
        self.fig_geneselection_heatmap = fig
        self.ax_geneselection_heatmap = ax
