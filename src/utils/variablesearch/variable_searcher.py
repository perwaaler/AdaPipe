from dataclasses import dataclass, field
from collections.abc import Iterable
from datetime import datetime

from typing import Literal
from typing import Union
from utils.paths import Paths
from utils.display import *
from utils.misc import replace_hyphen
from utils.models.torch_models import CoxPHUnivariateTrainer
from utils.models.torch_models import CoxPHSoftCut
from utils.models.torch_models import PrintConfig
from utils.models.torch_models import EffectiveSampleSizePenalizer
from utils.models.models import CPHModel
from utils.models.models import SurvivalTreeModel
from utils.prepare_data.variant_data import GeneLevelFeatures
from utils.variablesearch import grid_cutoff_selectors
from utils.variablesearch import cutoff_choosers
from utils.models.torch_models import TorchSurvivalDataPrepper
from utils.variablesearch.grid_constructor import GridConstructor

import numpy as np
import os
import pandas as pd


class VariableSearcher:
    """Per-gene VAF cutoff search.

    The following is expected for ``clinical_df`` and ``variants_df``:

    - ``clinical_df``: one row per patient. Required: ``RED_ID`` (patient
      identifier), ``time_to_event``, ``event_observed`` (0 or 1).
    - ``variants_df``: one row per variant. Required: ``RED_ID``, ``SYMBOL``,
      ``VAF_TUMOR``, ``ID`` (variant identifier), ``IS_ONCOGENIC`` (boolean
      variable indicating if variant is oncogenic).

    For each gene, a survival tree screens whether a Cox grid over VAF cutoffs
    is worth running, with the oncogenicity filter on and off. A logistic
    softcut fit supplies ``p_softcut``. ``grid_cutoff_selector`` stores the
    winning cutoff. ``get_modelfit_df_for_subset_of_genes`` later turns the VAF
    table into a mutation matrix at those cutoffs.

    ``VAFmax_df`` is the max VAF per gene for every patient in the variant table
    who has a survival time, plus ``RED_ID``, ``time_to_event``, and
    ``event_observed``. ``VAFmax_df_for_filterfit`` is that table restricted to
    ``red_id_used_to_fit_filters``, and is the cohort the cutoffs are estimated
    on.
    
    Model matrices are drawn from ``VAFmax_df`` unless a caller asks to keep
    only the filter-fit cohort. ``restrict_to_red_id_used_to_fit_filters`` at
    the time of fitting the filters. When using VariableSearcher objects in
    cross-validation, the fitted filters are loaded, so
    ``red_id_used_to_fit_filters`` is only true if not overwritten or if it
    matches the ID list that was originally used to fit the filters.
    """

    @dataclass
    class SoftCutFitParams:
        """Settings for the logistic VAF fit that produces p_softcut."""

        n_epochs: int = 2000
        learning_rate: float = 0.5
        l2_penalty: float = 0.00001
        min_n_epochs: int = 250
        effective_samplesize_regularizer: EffectiveSampleSizePenalizer = None
        trainable_vaf_scale: bool = True
        scale_initval: float = 0.08
        location_initval: float = 0.1
        verbose: int = 1
        display: PrintConfig = field(default_factory=PrintConfig)

    @dataclass
    class SoftcutResult:
        """Location, scale, and Cox p-value from one softcut fit."""

        p: float
        loc: float
        scale: float
        modeltrainer: CoxPHUnivariateTrainer
        convergence_failed: bool

    # ---------- *** CLASS INITIATION *** -----------------------------------------------------------------------------------------
    def __init__(
        self,
        genes_to_analyze: Union[list[str], Literal["all_genes"]] = "all_genes",
        genes_to_force: list[str] | None = None,
        clinical_df: pd.DataFrame | None = None,
        endpoint: str | None = None,
        grid: GridConstructor | None = None,
        min_cases_per_mutaclass: int = None,
        gene_selection_var: Literal["p_softcut", "p"] = "p_softcut",
        prelim_pval_cutoff=0.1,
        compute_p_vaf_in_prelim=True,
        variants_df=None,
        red_id_used_to_fit_filters: Union[list[str], Literal["no_restriction"]] = None,
        grid_cutoff_selector: grid_cutoff_selectors.MinPvalOverEligible | None = None,
        fit_softcut_function_in_gridanalysis=True,
        softcut_fit_params: "VariableSearcher.SoftCutFitParams" = None,
        vaf_cutoff_selector: cutoff_choosers.VAFCutoffSelector | None = None,
        penalizer_cph: float = 0.1,
        suppress_cph_warnings: bool = True,
        verbose=1,
    ):
        """Pass ``variants_df`` and ``clinical_df`` to build ``VAFmax_df``.
        Omit both only when loading a saved search to inspect summaries; then
        pass ``genes_to_analyze`` as a list, because ``"all_genes"`` reads gene
        symbols from the variant table.

        ``red_id_used_to_fit_filters`` is the cohort the cutoffs are fit on. ``None``
        loads the Norway RED_ID list. ``"no_restriction"`` keeps every patient
        in ``VAFmax_df``. A list keeps those IDs. Any other string is rejected,
        because a string is iterable and would be read character by character.

        ``endpoint`` is stored in the dump metadata only. ``grid``,
        ``grid_cutoff_selector``, ``vaf_cutoff_selector``, and
        ``softcut_fit_params`` are built fresh when ``None``.

        ``suppress_cph_warnings`` silences lifelines ``ConvergenceWarning`` on
        softcut / linear-VAF Cox probes (low-variance ``VAF_SQUASHED`` is common).
        Grid threshold fits always suppress warnings. ``convergence_failed`` is
        still set on the model when a warning was captured.
        """
        # GENE SELECTION
        if genes_to_analyze == "all_genes":
            assert variants_df is not None, "Need to provide variants df if you do not specify genes"
            self.genes_to_analyze = variants_df.dropna(subset="RED_ID")["SYMBOL"].unique()
        elif isinstance(genes_to_analyze, Iterable):
            self.genes_to_analyze = genes_to_analyze
        else:
            raise ValueError(f"{genes_to_analyze=} not valid input")

        # Store original gene labels and remove hyphens (causes errors in some cases)
        self._genes_orgl = self.genes_to_analyze
        self.genes_to_analyze = [replace_hyphen(g) for g in self.genes_to_analyze]
        self.genes_to_force = [] if genes_to_force is None else genes_to_force
        # CLINICAL DATA AND ENDPOINTS
        if variants_df is not None:
            assert clinical_df is not None, "clinical_df is required when variants_df is provided"
            assert "IS_ONCOGENIC" in variants_df.columns, "variants_df must have IS_ONCOGENIC"
        if clinical_df is not None:
            missing = {"RED_ID", "time_to_event", "event_observed"} - set(clinical_df.columns)
            assert not missing, f"clinical_df is missing {sorted(missing)}"
        self.clinical_df = clinical_df
        self.endpoint = endpoint

        # VARIANT FILTER GRID
        self.grid = GridConstructor() if grid is None else grid
        self.min_cases_per_mutaclass = min_cases_per_mutaclass if min_cases_per_mutaclass is not None else self.grid.min_cases_per_mutaclass

        self.gene_selection_var = gene_selection_var
        # SOFTCUT PARAMETERS
        self.softcut_fit_params = self.SoftCutFitParams() if softcut_fit_params is None else softcut_fit_params  # fmt: skip

        self.penalizer_cph = penalizer_cph

        # Patients the cutoff search is fit on. None loads the Norway RED_ID list.
        # "no_restriction" keeps every patient in VAFmax_df. A list keeps those
        # IDs. Other strings are rejected: str is iterable, so one would otherwise
        # be read character by character.
        if red_id_used_to_fit_filters is None:
            orange_print("No RED_ID restriction specified: defaulting to Norway cohort")
            self.red_id_used_to_fit_filters = Paths.load_json(Paths().red_id_norway_path)
        elif red_id_used_to_fit_filters == "no_restriction" or (
            not isinstance(red_id_used_to_fit_filters, str) and isinstance(red_id_used_to_fit_filters, Iterable)
        ):
            self.red_id_used_to_fit_filters = red_id_used_to_fit_filters
        else:
            raise ValueError("red_id_used_to_fit_filters must be 'no_restriction' or a list of IDs")

        # CUTOFF/SETTING SELECTION STRATEGIES
        # Dict that maps strategy label to its associated function
        self.grid_cutoff_selector = grid_cutoff_selectors.MinPvalOverEligible() if grid_cutoff_selector is None else grid_cutoff_selector  # fmt: skip
        self.vaf_cutoff_selector = cutoff_choosers.VAFCutoffSelector(vaf_cutoff_type="grid_optimized") if vaf_cutoff_selector is None else vaf_cutoff_selector  # fmt: skip
        self._check_if_cutoff_selector_is_valid()

        # --- CONTAINER OBJECTS (GETS FILLED IN LOOP OVER GENES)
        prelim_result_collector = lambda: {
            "p": np.nan,
            "p_vaf": np.nan,
            "passed_prelim": True,
            "n_nonzero_vaf": 0,
            "cindex_treefit": np.nan,
            "cutoff_treefit": np.nan,
            "cutoff_prelim": np.nan,
            "p_cphfit": np.nan,
            "p_softcut": np.nan,
            "n_mutas_cphfit": np.nan,
        }
        self.prelim_results = {g: {"oncofilter_off": prelim_result_collector(), "oncofilter_on": prelim_result_collector()} for g in self.genes_to_analyze}   # fmt: skip
        result_collector_dict = lambda: {
            "p": [],
            "vaf_cutoff": [],
            "coef_lower_ci": [],
            "coef_upper_ci": [],
            "cindex": [],
            "n_muta": [],
        }
        self.grid_results = {g: {"oncofilter_off": result_collector_dict(), "oncofilter_on": result_collector_dict()} for g in self.genes_to_analyze}   # fmt: skip
        self.softcut_analysis_results = {
            g: {
                "oncofilter_off": {
                    "loc": np.nan,
                    "scale": np.nan,
                    "p_softcut": np.nan,
                    "convergence_failed": np.nan,
                },
                "oncofilter_on": {
                    "loc": np.nan,
                    "scale": np.nan,
                    "p_softcut": np.nan,
                    "convergence_failed": np.nan,
                },
            }
            for g in self.genes_to_analyze
        }

        self.VAFmax_df_for_filterfit = None
        self.VAFmax_df = None
        self.variant_info: GeneLevelFeatures = None
        # Result containers
        self.prelim_results_df = None
        self.softcut_analysis_df = None
        self.raw_results_df__on = None
        self.raw_results_df__off = None
        self.summary_df__on = None
        self.summary_df__off = None
        self.merged_summary_df = None
        self.summary_best_models_df = None

        # --- PRELIMINARY SCREENING
        self.prelim_p_cutoff = prelim_pval_cutoff
        self.compute_p_vaf_in_prelim = compute_p_vaf_in_prelim

        # --- MISC
        self._best_global_vaf_cutoff = None
        self.fit_softcut_in_gridanalysis = fit_softcut_function_in_gridanalysis
        self.cph_precision = 1e-3  # Used in CPH fit: lowering can speed up performance a bit
        self._current_iteration = None

        # VERBOSITY/PRINTING SETTINGS
        self.print_failed_prelim = False
        self.print_iteration_start = True
        self.print_muta_filter_analysis_start = True
        self.suppress_cph_warnings = suppress_cph_warnings
        self.verbose = verbose

        self.variants_df = variants_df
        if variants_df is not None:
            self.prepare_analysis_dataframe(variants_df)

        from utils.variablesearch.plotting import VarSearcherPlotter
        self.plotting = VarSearcherPlotter(self)

    def _check_if_cutoff_selector_is_valid(self):
        """Require an instance that can pick a grid point."""
        assert not isinstance(self.grid_cutoff_selector, type), "Cutoff selector is a class, but a class instance is expected"   # fmt: skip
        assert hasattr(self.grid_cutoff_selector, "extract_values_at_the_best_cutoff"), "Method extract_values_at_the_best_cutoff was not found in cutoff_selector (check the class methods)"   # fmt: skip

    # --- METHODS THAT UPDATES VARIABLE SEARCHERS WHEN CUTOFF SELECTOR IS CHANGED
    @property
    def best_global_vaf_cutoff(self):
        return self._best_global_vaf_cutoff

    @best_global_vaf_cutoff.setter
    def best_global_vaf_cutoff(self, value):
        """Cutoff recorded in ``meta.json``. Not used to choose a cutoff."""
        self._best_global_vaf_cutoff = value

    # ---------- *** VAF GENE-BY-SAMPLE DATAFRAME PREPARATION *** -----------------------------------------------------------------------------------------
    def prepare_analysis_dataframe(self, variants_df=None):
        """Build ``VAFmax_df`` and ``VAFmax_df_for_filterfit``.

        ``VAFmax_df`` is max VAF per gene plus survival columns, for patients in
        the variant table who have a survival time. ``VAFmax_df_for_filterfit``
        is that table restricted to ``red_id_used_to_fit_filters``.
        """
        if variants_df is None:
            variants_df = self.variants_df
        self.variant_info = GeneLevelFeatures.from_variants_using_max(
            variants_filtered_df=variants_df,
            create_zerocols_for_genes_with_no_oncovariants=True,
        )
        VAFmax_df = self.variant_info.genomic_features_df.merge(
            self.clinical_df[["RED_ID", "time_to_event", "event_observed"]],
            on="RED_ID",
            how="left",
        )
        VAFmax_df = VAFmax_df.dropna(subset=["time_to_event"]).reset_index(drop=True)

        self.VAFmax_df = VAFmax_df

        # DATAFRAME USED TO FIT VARIANT FILTERS: RESTRICT TO SPECIFIED SAMPLES IF RESTRICTED TO SUBSET
        if self.red_id_used_to_fit_filters == "no_restriction":
            self.VAFmax_df_for_filterfit = self.VAFmax_df.copy()
        else:
            self.VAFmax_df_for_filterfit = self.VAFmax_df.query(f"RED_ID in {list(self.red_id_used_to_fit_filters)}").reset_index(drop=True).copy()   # fmt: skip

    # ---------- ¤¤¤ MAIN METHOD: LOOP OVER ALL GENES ¤¤¤ -----------------------------------------------------------------------------------------
    def run_gridsearch_analysis_for_genes(self, genes=None):
        """Run the cutoff search for each gene, using ``VAFmax_df_for_filterfit``.

        Stores the grid in ``grid_results`` and the preliminary screen in
        ``prelim_results``. Genes that fail the screen are skipped unless they
        are in ``genes_to_force``.
        """
        if self.VAFmax_df is None:
            raise ValueError("VAFmax_df not created yet: run prepare_analysis_dataframe()")
        if genes is None:
            genes = self.genes_to_analyze

        magenta_print("PERFORMING VARIANT-FILTER GRID ANALYSIS...") if self.print_muta_filter_analysis_start else None

        for i, gene in enumerate(genes):
            if gene not in self.grid_results:
                continue
            self._current_iteration = i
            if self.verbose > 1:
                blue_print(f"{i}: Analyzing {gene}")
            else:
                blue_print(f"{i}: Analyzing {gene}") if self.print_iteration_start and i % 50 == 0 else None   # fmt: skip
            self.run_gridsearch_analysis_for_gene(gene)
            pvals = [self.prelim_results[gene][key]["p_cphfit"] for key in ["oncofilter_on", "oncofilter_off"]]
            pink_print(f"PRELIM RESULTS, {gene}: oncofilter on p={pvals[0]:.5}; oncofilter off p={pvals[1]:.5}") if min(pvals) < 0.0001 else None   # fmt: skip

    def run_gridsearch_analysis_for_gene(self, gene, preliminary_test=True):
        """Screen one gene, then fit the VAF grid and, if requested, the softcut.

        Both the oncogenicity-on and oncogenicity-off columns are tried. A side
        with too few non-zero VAFs is skipped.
        """
        for oncofilter_status in ["oncofilter_on", "oncofilter_off"]:
            univarfit_df = self.get_univarfit_df(gene, oncofilter_status)

            # Check how many {VAF>0} cases there are
            n_mutas = self.prelim_results[gene][oncofilter_status]["n_nonzero_vaf"]
            if n_mutas < self.min_cases_per_mutaclass or univarfit_df is None:
                # Too few: continue to next experiment
                continue

            # PRELIMINARY ANALYSIS
            prelim_analysis = self.run_preliminary_analysis(univarfit_df, oncofilter_status, gene)
            if prelim_analysis.test_result == "fail" and preliminary_test:
                continue

            # CONSTRUCT VAF GRID
            vaf_grid = self.grid.prepare_vaf_grid(prelim_analysis.largest_allowed_cutoff)

            if self.fit_softcut_in_gridanalysis:
                # SOFTCUT ANALYSIS
                softcut_results = self.fit_softcut_function(univarfit_df)
                self.softcut_analysis_results[gene][oncofilter_status]["loc"] = softcut_results.loc
                self.softcut_analysis_results[gene][oncofilter_status]["scale"] = softcut_results.scale
                self.softcut_analysis_results[gene][oncofilter_status]["p_softcut"] = softcut_results.p
                self.softcut_analysis_results[gene][oncofilter_status]["convergence_failed"] = softcut_results.convergence_failed   # fmt: skip

            # COMPUTE MODELS ON VAF GRID
            grid_results = self.fit_models_over_range_of_thresholds(univarfit_df, vaf_grid)
            # Store results from grid analysis
            self.grid_results[gene][oncofilter_status]["vaf_cutoff"] = grid_results["vaf_cutoff"]
            self.grid_results[gene][oncofilter_status]["p"] = grid_results["pval"]
            self.grid_results[gene][oncofilter_status]["cindex"] = grid_results["cindex"]
            self.grid_results[gene][oncofilter_status]["coef_lower_ci"] = grid_results["coef_lower_ci"]
            self.grid_results[gene][oncofilter_status]["coef_upper_ci"] = grid_results["coef_upper_ci"]
            self.grid_results[gene][oncofilter_status]["n_muta"] = grid_results["n_muta"]

    def run_preliminary_analysis(
        self,
        univarfit_df,
        oncofilter_status,
        gene,
        prelim_p_cutoff: float = None,
        store_results: bool = True,
    ):
        """Survival-tree gate for one gene and one oncofilter side.

        The tree proposes a VAF cutoff. A Cox model at that cutoff supplies the
        preliminary p-value. The gene fails the screen when that p-value is
        above ``prelim_p_cutoff``, unless the gene is in ``genes_to_force``.
        Returns whether the screen passed and the largest cutoff the grid may use.
        """
        if prelim_p_cutoff is None:
            prelim_p_cutoff = self.prelim_p_cutoff
        # FIT SURVIVAL TREE TO GET VAF-CUTOFF ESTIMATE AND FIT CPH TO GET P-VALUE
        largest_allowed_cutoff = self.grid.get_largest_allowed_vaf_cutoff(univarfit_df)
        tree_model = self.get_cutoff_by_fitting_survival_tree(univarfit_df)
        tree_cutoff = tree_model.threshold_first_split
        # Clamp cutoff in case it results in too few mutations
        prelim_cutoff = np.min([tree_cutoff, largest_allowed_cutoff])
        # Fit CPH model using the tree-model VAF-cutoff
        cph_model = self.fit_cph(univarfit_df, vaf_cutoff=prelim_cutoff)
        p_prelimtest = cph_model.pvalues["MUTATED"]

        # STORE PRELIM-ANALYSIS RESULTS
        if store_results:
            self.prelim_results[gene][oncofilter_status]["cindex_treefit"] = tree_model.get_cindex(univarfit_df)   # fmt: skip
            self.prelim_results[gene][oncofilter_status]["cutoff_treefit"] = tree_cutoff
            self.prelim_results[gene][oncofilter_status]["cutoff_prelim"] = prelim_cutoff
            self.prelim_results[gene][oncofilter_status]["p_cphfit"] = p_prelimtest
            self.prelim_results[gene][oncofilter_status]["n_mutas_cphfit"] = (univarfit_df["VAF"] >= prelim_cutoff).sum()   # fmt: skip
            if self.compute_p_vaf_in_prelim:
                self.fit_cph_as_linear_function_of_vaf_and_store_pvalue(univarfit_df, gene, oncofilter_status)

        if p_prelimtest > prelim_p_cutoff and gene not in self.genes_to_force:   # fmt: skip
            # PRELIMINARY TEST FAILED
            if store_results:
                self.prelim_results[gene][oncofilter_status]["passed_prelim"] = False
            red_print(f"{self._current_iteration=}, {gene}, {oncofilter_status}: failed screening: p={p_prelimtest:.3f}, {tree_cutoff=:.2f}") if self.print_failed_prelim else None   # fmt: skip
            test_result = "fail"
        else:
            test_result = "pass"

        @dataclass
        class PrelimTestResult:
            test_result: float
            largest_allowed_cutoff: float

        return PrelimTestResult(test_result=test_result, largest_allowed_cutoff=largest_allowed_cutoff)

    def get_univarfit_df(self, gene, oncofilter_status="on", vaf_df=None):
        """One gene and one oncofilter side, taken from ``VAFmax_df_for_filterfit``.

        The VAF column is renamed to ``VAF``. Also records how many patients
        have a non-zero VAF.
        """
        self._check_variant_data_is_loaded()
        if vaf_df is None:
            vaf_df = self.VAFmax_df_for_filterfit
        assert oncofilter_status.endswith("on") or oncofilter_status.endswith("off")

        if oncofilter_status.endswith("on"):
            oncofilter_status = "oncofilter_on"
            vaf_column = f"{gene}_MAX_ONCO_VAF"
        if oncofilter_status.endswith("off"):
            oncofilter_status = "oncofilter_off"
            vaf_column = f"{gene}_MAX_VAF"

        if vaf_column in vaf_df.columns:
            univarfit_df = vaf_df[[vaf_column, "RED_ID", "event_observed", "time_to_event"]].copy()   # fmt: skip
            univarfit_df = univarfit_df.rename(columns={vaf_column: "VAF"})
            self.prelim_results[gene][oncofilter_status]["n_nonzero_vaf"] = (univarfit_df["VAF"] > 0).sum()
        else:
            univarfit_df = None

        return univarfit_df

    def _check_variant_data_is_loaded(self):
        """Require ``prepare_analysis_dataframe`` before a per-gene fit."""
        if self.VAFmax_df_for_filterfit is None:
            raise ValueError("VAFmax_df not created yet: run prepare_analysis_dataframe()")

    def fit_softcut_function(
        self, univarfit_df, fit_params: SoftCutFitParams = None
    ) -> "VariableSearcher.SoftcutResult":
        """Fit the logistic VAF transform and return its location, scale, and p-value."""
        univarfit_df = univarfit_df.copy()

        if fit_params is None:
            fit_params = self.softcut_fit_params
        X_vaf, time, event = TorchSurvivalDataPrepper().get_torch_data(univarfit_df)

        model = CoxPHSoftCut(vaf_scale_initval=fit_params.scale_initval, vaf_location_initval=fit_params.location_initval, trainable_vaf_scale=fit_params.trainable_vaf_scale)   # fmt: skip
        sigmoid_trainer = CoxPHUnivariateTrainer(
            model=model,
            lambda_l2loss=fit_params.l2_penalty,
            n_epochs=fit_params.n_epochs,
            learning_rate=fit_params.learning_rate,
            min_n_epochs=fit_params.min_n_epochs,
            effective_sample_regularizer=fit_params.effective_samplesize_regularizer,
            verbose=fit_params.verbose,
            display=fit_params.display,
        )
        sigmoid_trainer.fit(X_muta=None, X_vaf=X_vaf, X_onco=None, time=time, event=event)

        # EXTRACT PARAMETERS AND CREATE SOFT MUTATION TRANSFORMER
        loc, scale = sigmoid_trainer.model.get_location_and_scale_params()
        vaf_transformer = lambda x: 1 / (1 + np.exp(-(x - loc) / scale))

        # FIT CPH USING SQUASHED VALUES TO GET OPTIMISTIC P-VALUE
        univarfit_df["VAF_SQUASHED"] = vaf_transformer(univarfit_df["VAF"])
        cph = CPHModel(predictors="VAF_SQUASHED", penalizer=self.penalizer_cph)
        cph.fit(univarfit_df, suppress_warnings=self.suppress_cph_warnings)
        p_softcut = cph.pvalues.values[0]

        return self.SoftcutResult(
            modeltrainer=sigmoid_trainer,
            p=p_softcut,
            scale=scale,
            loc=loc,
            convergence_failed=cph.convergence_failed,
        )

    def fit_cph_as_linear_function_of_vaf_and_store_pvalue(self, univarfit_df, gene, oncofilter_status):
        """Fit Cox model with raw VAF as the predictor. Stores ``p_vaf``."""
        # FIT CPH MODEL WITH VAF AS X-VARIABLE
        cph = CPHModel(predictors=["VAF"], penalizer=self.penalizer_cph)
        cph.fit(univarfit_df, precision=self.cph_precision, suppress_warnings=self.suppress_cph_warnings)
        p = cph.pvalues.values[0]
        self.prelim_results[gene][oncofilter_status]["p_vaf"] = p

    def get_cutoff_by_fitting_survival_tree(self, univarfit_df, min_samples_leaf=None) -> float:
        """Fit a survival tree on VAF and return it. The split is ``threshold_first_split``."""
        # FIT CPH MODEL WITH VAF AS X-VARIABLE
        if min_samples_leaf is None:
            min_samples_leaf = self.min_cases_per_mutaclass
        tree = SurvivalTreeModel(predictors=["VAF"], min_samples_leaf=min_samples_leaf)
        tree.fit(univarfit_df)
        return tree

    def fit_cph(self, univarfit_df, vaf_cutoff):
        """Cox model with a binary mutation indicator at ``vaf_cutoff``."""
        # FIT CPH MODEL USING CUTOFF OBTAINED WITH SURVIVAL TREE
        modelfit_df = univarfit_df.copy()
        modelfit_df["MUTATED"] = modelfit_df["VAF"] >= vaf_cutoff
        cph = CPHModel(predictors=["MUTATED"], penalizer=self.penalizer_cph)
        cph.fit(modelfit_df, suppress_warnings=True)
        return cph

    def fit_models_over_range_of_thresholds(self, univarfit_df, vaf_grid) -> tuple:
        """Fit Cox model at each cutoff on ``vaf_grid``. Skips cutoffs with too
        few mutations."""
        grid_results = {
            "vaf_cutoff": [],
            "pval": [],
            "cindex": [],
            "coef_lower_ci": [],
            "coef_upper_ci": [],
            "n_muta": [],
        }

        for vaf_cutoff in vaf_grid:
            univarfit_df["MUTATED"] = univarfit_df["VAF"] >= vaf_cutoff
            if self.grid.classes_are_too_unbalanced(univarfit_df):
                continue
            cph = CPHModel(predictors=["MUTATED"], penalizer=self.penalizer_cph)
            cph.fit(univarfit_df, precision=self.cph_precision, suppress_warnings=True)

            grid_results["vaf_cutoff"].append(vaf_cutoff)
            grid_results["pval"].append(cph.pvalues.values[0])
            grid_results["cindex"].append(cph.cindex)
            grid_results["coef_lower_ci"].append(cph.model.summary["coef lower 95%"].values[0])
            grid_results["coef_upper_ci"].append(cph.model.summary["coef upper 95%"].values[0])
            grid_results["n_muta"].append(univarfit_df["MUTATED"].sum())

        return grid_results

    # ---------- ORGANIZING FILTER FIT-RESULTS IN DATAFRAMES -----------------------------------------------------------------------------------------
    def merge_result_dicts(self, results_dict) -> pd.DataFrame:
        """Join the on and off result dicts into one frame, with ``__on`` and ``__off`` suffixes."""
        # SEPARATE RESULTS INTO TWO DATAFRAMES
        d_on = {g: d["oncofilter_on"] for g, d in results_dict.items()}
        d_off = {g: d["oncofilter_off"] for g, d in results_dict.items()}
        df_on = pd.DataFrame(d_on).transpose()
        df_off = pd.DataFrame(d_off).transpose()
        # Merge and add suffixes
        merged_df = pd.merge(df_on, df_off, left_index=True, right_index=True, suffixes=["__on", "__off"])
        return merged_df

    def compute_result_summary_dfs(self) -> pd.DataFrame:
        """Build the summary tables from ``grid_results``, ``prelim_results``, and the softcut fit.

        ``raw_results_df__on`` and ``raw_results_df__off`` hold the grid arrays.
        ``summary_df__on`` and ``summary_df__off`` hold the chosen cutoff on each
        side. ``summary_best_models_df`` keeps the winning side for each gene.
        """
        assert self.grid_results is not None, "Grid analysis not executed yet: nothing to summarize"

        # TRANSFER RESULTS FROM DICTS TO DATAFRAMES
        self.prelim_results_df = self.merge_result_dicts(self.prelim_results)
        self.softcut_analysis_df = self.merge_result_dicts(self.softcut_analysis_results)

        # COLLECT STATS IN DATAFRAMES
        for oncofilter_status in ["oncofilter_on", "oncofilter_off"]:
            # Extract results for oncofilter_status
            results_ofs_dict = {g: d[oncofilter_status] for g, d in self.grid_results.items()}
            # Convert to dataframe and map empty lists to nan
            raw_df = pd.DataFrame(results_ofs_dict).transpose()
            # Map empty lists to nan
            raw_df = raw_df.map(lambda x: np.nan if isinstance(x, list) and len(x) == 0 else x)
            # Order values by cutoff
            raw_df = raw_df.apply(self._order_values_by_increasing_cutoff, axis=1, result_type="expand")
            raw_df["coef"] = raw_df.apply(self._compute_coef_from_ci, axis=1)
            raw_df["hazard_ratio"] = raw_df["coef"].apply(lambda x: np.exp(x) if isinstance(x, np.ndarray) else np.nan)
            raw_df["neglogp"] = raw_df["p"].apply(lambda x: self._calc_neglogp(x) if isinstance(x, np.ndarray) else np.nan)   # fmt: skip

            # AGGRAGATE AND ANALYZE GRID RESULTS
            # Collect the best p-value and VAF cutoff in a new dataframe
            raw_df_cleaned = raw_df.dropna(how="all").copy()
            values_at_best_cutoffs_df = self.grid_cutoff_selector.extract_values_at_the_best_cutoff(raw_df_cleaned)
            values_at_best_cutoffs_df["neglogp"] = self._calc_neglogp(values_at_best_cutoffs_df["p"].astype(float))
            aggragation_metrics_df = raw_df.apply(self._aggragate_gridsearch_metrics, axis=1)
            summary_df = pd.concat([values_at_best_cutoffs_df, aggragation_metrics_df], axis=1)

            # INCLUDE SOME RESULTS FROM PRE-GRID-ANALYSIS TESTS
            suffix = "on" if oncofilter_status == "oncofilter_on" else "off"
            # Treefit results
            summary_df["p_treefit"] = self.prelim_results_df[f"p_cphfit__{suffix}"]
            summary_df["vaf_cutoff_treefit"] = self.prelim_results_df[f"cutoff_treefit__{suffix}"]
            summary_df["p_vaf"] = self._safe_copy_column(df_donor=self.prelim_results_df, var=f"p_vaf__{suffix}")  # fmt: skip
            summary_df["n_muta_treefit"] = self.prelim_results_df[f"n_mutas_cphfit__{suffix}"]
            # Softcut results
            summary_df["p_softcut"] = self.softcut_analysis_df[f"p_softcut__{suffix}"]
            summary_df["loc_softcut"] = self.softcut_analysis_df[f"loc__{suffix}"]
            summary_df["scale_softcut"] = self.softcut_analysis_df[f"scale__{suffix}"]
            summary_df["softcut_fit_failed"] = self.softcut_analysis_df[f"convergence_failed__{suffix}"]

            if oncofilter_status == "oncofilter_on":
                self.raw_results_df__on = raw_df
                self.summary_df__on = summary_df
            else:
                self.raw_results_df__off = raw_df
                self.summary_df__off = summary_df

        # MERGE SUMMARY STATS INTO SINGLE DATAFRAME
        self.merge_oncogenicity_filtering_results()

    def merge_oncogenicity_filtering_results(self):
        """Pick the winning oncofilter side for each gene and store it in ``summary_best_models_df``."""
        # Merge dataframes in wide dataframe and add suffixes
        self.merged_summary_df = self.summary_df__on.merge(self.summary_df__off, left_index=True, right_index=True, suffixes=["__on", "__off"])   # fmt: skip

        # FIND BEST ONCOGENICITY-BASED-FILTERING STRATEGY FOR EACH GENE
        self.merged_summary_df["best_oncofiltersetting"] = self.merged_summary_df.apply(self._get_best_oncofilter_setting, axis=1)  # fmt: skip
        # Get the values corresponding to the best onco-filter setting
        self.summary_best_models_df = self.merged_summary_df.apply(self._get_values_for_best_oncofilter_setting, axis=1)   # fmt: skip
        self.summary_best_models_df["neglogp"] = self.summary_best_models_df["p"].apply(self._calc_neglogp)   # fmt: skip
        self.summary_best_models_df["p_diff_neglogp"] = self.summary_df__on["neglogp"] - self.summary_df__off["neglogp"]

        # SORT COLUMN AND ROW ORDER FOR CONVENIENCE
        # Columns
        most_used_cols = [
            "p",
            "vaf_cutoff",
            "coef",
            "hazard_ratio",
            "best_oncofiltersetting",
            "p_softcut",
            "loc_softcut",
            "scale_softcut",
            "n_muta",
            "cindex",
        ]   # fmt: skip
        least_used_cols = [x for x in self.summary_best_models_df.columns if x not in most_used_cols]
        self.summary_best_models_df = self.summary_best_models_df[most_used_cols + least_used_cols]
        # Rows
        self.summary_best_models_df = self.summary_best_models_df.sort_values("p_softcut")

    # --- HELPER METHODS USED TO CONSTRUCT SUMMARY DATAFRAMES
    @staticmethod
    def _order_values_by_increasing_cutoff(row):
        """Sort each grid-result list by increasing VAF cutoff."""
        if row.isna().any():
            return row
        else:
            correct_order = np.argsort(row["vaf_cutoff"])
            return {key: np.array(val)[correct_order] for key, val in row.items()}

    @staticmethod
    def _compute_coef_from_ci(row):
        """Coefficient as the midpoint of the stored confidence interval."""
        return np.stack([row["coef_lower_ci"], row["coef_upper_ci"]]).mean(axis=0)

    @staticmethod
    def _calc_neglogp(x):
        """Minus log10 of a p-value."""
        return -np.log10(x)

    def _aggragate_gridsearch_metrics(self, row) -> pd.Series:
        """Number of VAF cutoffs evaluated for this gene and oncofilter side."""
        n_cutoffs = len(row["vaf_cutoff"]) if isinstance(row["p"], np.ndarray) else np.nan
        return pd.Series({"n_cutoffs": n_cutoffs})

    @staticmethod
    def _safe_copy_column(df_donor, var: str) -> pd.Series:
        """Returns ``np.nan`` if columns does not exist in donor df."""
        if var in df_donor.columns:
            return df_donor[var]
        else:
            return np.nan

    def _get_best_oncofilter_setting(self, row, grid_cutoff_selector: grid_cutoff_selectors.BaseCutoffSelector = None):
        """Choose on or off for one gene.

        ``pick_on`` and ``pick_off`` force a side. ``highest_neglogp`` and
        ``lowest_p_softcut`` compare the ``__on`` and ``__off`` columns.
        """
        grid_cutoff_selector = self.grid_cutoff_selector if grid_cutoff_selector is None else grid_cutoff_selector

        if grid_cutoff_selector.oncofilter_selection == "pick_on":
            return "on"
        if grid_cutoff_selector.oncofilter_selection == "pick_off":
            return "off"

        # --- EXTRACT NAME OF DECISION VARIABLE
        strategy = grid_cutoff_selector.oncofilter_selection
        if "highest" in strategy:
            higher_is_better = True
            decisionvar_basename = grid_cutoff_selector.oncofilter_selection.split("highest_")[1]
        elif "lowest" in strategy:
            higher_is_better = False
            decisionvar_basename = grid_cutoff_selector.oncofilter_selection.split("lowest_")[1]

        decisionvar__on = f"{decisionvar_basename}__on"
        decisionvar__off = f"{decisionvar_basename}__off"

        # --- HANDLE COMBINATIONS OF MISSING VALUES
        if pd.isna(row[decisionvar__on]) and pd.isna(row[decisionvar__off]):
            return np.nan
        elif pd.isna(row[decisionvar__on]):
            return "off"
        elif pd.isna(row[decisionvar__off]):
            return "on"

        # --- NO MISSING VALUES: COMPARE DECISION VARIABLE
        tolerance = grid_cutoff_selector.min_gain_for_oncoswitch
        if higher_is_better:
            return "on" if row[decisionvar__on] - tolerance > row[decisionvar__off] else "off"
        else:
            return "on" if row[decisionvar__on] + tolerance < row[decisionvar__off] else "off"

    @staticmethod
    def _get_values_for_best_oncofilter_setting(
        row,
        vars_to_extract=[
            "p",
            "vaf_cutoff",
            "coef",
            "coef_lower_ci",
            "coef_upper_ci",
            "hazard_ratio",
            "p_treefit",
            "vaf_cutoff_treefit",
            "p_vaf",
            "n_muta_treefit",
            "cindex",
            "n_muta",
            # SOFTCUT
            "p_softcut",
            "loc_softcut",
            "softcut_fit_failed",
            "scale_softcut",
            # OTHER
            "n_cutoffs",
        ],
        best_setting_column_name="best_oncofiltersetting",
    ):
        """Keep the columns that belong to the winning oncofilter side."""
        available_vars = [v for v in vars_to_extract if f"{v}__on" in row.index or f"{v}__off" in row.index]
        if row[best_setting_column_name] == "on":
            data = {var: row[f"{var}__on"] for var in available_vars}
        elif row[best_setting_column_name] == "off":
            data = {var: row[f"{var}__off"] for var in available_vars}
        else:
            data = {var: np.nan for var in available_vars}

        # Add the best-setting-info to the output
        data["best_oncofiltersetting"] = row[best_setting_column_name]

        return pd.Series(data)

    # ---------- VARIABLE SELECTION -----------------------------------------------------------------------------------------
    def get_filter_info_for_gene_selection(self, p_cutoff=None, n_select=None, selection_var=None, gene_selection_pool: list[str] = None):
        """Cutoff and oncofilter side for the selected genes.

        Pass ``p_cutoff`` or ``n_select``, not both. Ranking uses
        ``selection_var``, or ``gene_selection_var`` when that is omitted.
        """
        if (p_cutoff is None) == (n_select is None):
            red_print(f"Pass either p_cutoff OR n_select (you have passed {p_cutoff=}, {n_select=})")   # fmt: skip
        if selection_var is None:
            selection_var = self.gene_selection_var

        df = self.summary_best_models_df
        if gene_selection_pool is not None:
            df = df.loc[gene_selection_pool].copy()
        
        if p_cutoff is not None:
            selection_info_df = df.query(f"{self.gene_selection_var} <= {p_cutoff}")[["vaf_cutoff", "best_oncofiltersetting"]]   # fmt: skip
        else:
            selection_info_df = df.sort_values(selection_var)[["vaf_cutoff", "best_oncofiltersetting"]].head(n_select)   # fmt: skip

        return selection_info_df

    def get_modelfit_df_for_subset_of_genes(
        self,
        genes_to_include_in_model: list[str],
        feature_type: Literal["MUTAS", "VAF", "SOFTCUT"] = "MUTAS",
        vaf_cutoff_selector: cutoff_choosers.VAFCutoffSelector = None,
        restrict_to_red_id_used_to_fit_filters=False,
        verbose=0,
    ):
        """Feature matrix for a Cox fit, drawn from ``VAFmax_df``.

        ``MUTAS`` thresholds VAF at the stored cutoff. ``VAF`` keeps the raw
        value. ``SOFTCUT`` applies the fitted logistic. Rows are every patient
        in ``VAFmax_df`` unless ``restrict_to_red_id_used_to_fit_filters`` is
        set.
        """
        vaf_cutoff_selector = self.vaf_cutoff_selector if vaf_cutoff_selector is None else vaf_cutoff_selector
        assert feature_type in ["MUTAS", "VAF", "SOFTCUT"], "feature_type must be 'MUTAS', 'VAF', or 'SOFTCUT'"

        # RESTRICT REQUESTED GENES TO THOSE THAT ARE REPRESENTED IN THE DATA (TO GET RID OF SINGLE NUISSANCE GENE...)
        genes_in_modelfit_df = [x.split("_MAX")[0] for x in self.VAFmax_df.columns]
        genes_to_include_in_model = [g for g in genes_to_include_in_model if g in genes_in_modelfit_df]

        # CHECKS
        if self.summary_best_models_df is None:
            self.compute_result_summary_dfs()
        self._check_if_genes_are_in_results_df(genes_to_include_in_model)

        if feature_type == "SOFTCUT":
            self._check_if_softcut_params_have_been_fitted()

        # GET DATAFRAME WITH RESULTS FROM GRID-SEARCH ANALYSIS (OPTIMAL CUTOFFS, P-VALUES, ETC...)
        filter_fitinfo_df = self.summary_best_models_df.loc[genes_to_include_in_model].copy()
        assert len(filter_fitinfo_df) == len(genes_to_include_in_model), "list of genes should have same length as summary_best_models_df; may be caused by duplicate rows"   # fmt: skip

        # CREATE FEATURE DATAFRAME USING OPTIMIZED VARIANT FILTERS
        optimal_oncofiltersettings = filter_fitinfo_df["best_oncofiltersetting"]
        vaf_columns = [
            f"{gene}_MAX_ONCO_VAF" if best_oncosetting == "on" else f"{gene}_MAX_VAF"
            for gene, best_oncosetting in zip(genes_to_include_in_model, optimal_oncofiltersettings)
        ]
        X_df = self.VAFmax_df[vaf_columns]

        # ---------- THRESHOLDING (HARD, SOFT, OR NONE) -----------------------------------------------------------------------------------------
        if feature_type == "MUTAS":
            # THRESHOLD TO GET MUTATIONS
            vaf_cutoffs = vaf_cutoff_selector.select_cutoffs(filter_fitinfo_df, vaf_cols=vaf_columns)
            red_print(vaf_cutoffs) if verbose > 0 else None
            X_df = X_df.ge(vaf_cutoffs)  # Apply column-wise cutoffs using 'ge' (greater or equal) method
            # Rename columns
            X_df.columns = [f"{gene}_MUTATED" for gene in genes_to_include_in_model]

        elif feature_type == "VAF":
            X_df.columns = [f"{gene}_VAF" for gene in genes_to_include_in_model]

        elif feature_type == "SOFTCUT":
            X_df = X_df.copy()
            # Convert to series to simplify column-wise operation
            loc = pd.Series(filter_fitinfo_df["loc_softcut"].values, index=vaf_columns)
            scale = pd.Series(filter_fitinfo_df["scale_softcut"].values, index=vaf_columns)
            # Define the softcut function that operates on the df column-wise
            softcut_func = lambda df: 1 / (1 + np.exp(-(df - loc) / scale))
            # Apply softcut
            X_df.loc[:, vaf_columns] = softcut_func(X_df[vaf_columns])
            # Rename columns
            X_df.columns = [f"{gene}_VAF" for gene in genes_to_include_in_model]

        self.x_variable_names = X_df.columns

        # GLUE TOGETHER EXTRACTED X-VARIABLES WITH OTHER CPH VARIABLES
        modelfit_df = pd.concat([self.VAFmax_df[["RED_ID", "time_to_event", "event_observed"]], X_df], axis=1)   # fmt: skip
        # RESTRICT TO GRIDSEARCH ANALYSIS SUBSET IF REQUESTED
        if restrict_to_red_id_used_to_fit_filters:
            modelfit_df = modelfit_df.query(f"RED_ID in {self.red_id_used_to_fit_filters}").reset_index(drop=True).copy()

        return modelfit_df

    def _check_if_genes_are_in_results_df(self, genes_requested):
        """Raise if a requested gene is missing from ``summary_best_models_df``."""
        genes_requested_not_in_results_df = list(set(genes_requested).difference(self.summary_best_models_df.index))
        if len(genes_requested_not_in_results_df) > 0:
            raise ValueError(f"The following genes are not in summary_best_models_df: {genes_requested_not_in_results_df}")   # fmt: skip

    def _check_if_softcut_params_have_been_fitted(self):
        """Raise if the softcut location and scale have not been summarized."""
        if "loc_softcut" not in self.summary_best_models_df.columns:
            raise ValueError(f"'loc_softcut' and 'scale_softcut' have not been computed; cannot transform using softcut function")   # fmt: skip

    # ---------- FILE MANAGEMENT (STORING & DUMPING) -----------------------------------------------------------------------------------------
    def dump_results_to_json(self, dump_dir_path=None, cv_idx=None):
        """Write grid, preliminary-screen, and softcut results, plus metadata and the filter-fit patient IDs."""
        if dump_dir_path is None:
            dump_dir_path = Paths().varselection_dump_flexigrid_norway_dir

        dump_paths = self._get_resultdump_paths(dump_dir_path, cv_idx)

        Paths.dump_to_json(self.grid_results, path=dump_paths.grid_results, print_confirm=True)   # fmt: skip
        Paths.dump_to_json(self.prelim_results, path=dump_paths.prelim_results, print_confirm=True)  # fmt: skip
        Paths.dump_to_json(self.softcut_analysis_results, path=dump_paths.softcut_results, print_confirm=True)  # fmt: skip
        Paths.dump_to_json(self.get_metadata(), path=dump_paths.meta, print_confirm=True)  # fmt: skip
        Paths.dump_to_json(self.VAFmax_df_for_filterfit["RED_ID"].to_list(), path=dump_paths.red_id, print_confirm=True)  # fmt: skip

    def load_results_from_json(self, dump_dir_path=None, compute_result_dfs=False, cv_idx=None):
        """Load a saved grid search. Pass ``cv_idx`` for one cross-validation fold.

        ``red_id.json`` replaces ``red_id_used_to_fit_filters`` with the patient list
        that was actually used. A search run with ``"no_restriction"`` therefore
        reloads as a list.
        """
        if dump_dir_path is None:
            dump_dir_path = Paths().varselection_dump_flexigrid_norway_dir

        dump_paths = self._get_resultdump_paths(dump_dir_path, cv_idx)

        self.grid_results = Paths.load_json(path=dump_paths.grid_results, print_confirm=True)   # fmt: skip
        self.prelim_results = Paths.load_json(path=dump_paths.prelim_results, print_confirm=False)  # fmt: skip
        if os.path.exists(dump_paths.meta):
            metadata = Paths.load_json(path=dump_paths.meta, print_confirm=False)
            self.best_global_vaf_cutoff = metadata.get("best_global_vaf_cutoff")
        if os.path.exists(dump_paths.softcut_results):
            self.softcut_analysis_results = Paths.load_json(path=dump_paths.softcut_results, print_confirm=False)  # fmt: skip
        if os.path.exists(dump_paths.red_id):
            self.red_id_used_to_fit_filters = Paths.load_json(path=dump_paths.red_id, print_confirm=False)  # fmt: skip

        if compute_result_dfs:
            self.compute_result_summary_dfs()

    def _get_resultdump_paths(self, dump_dir_path, cv_idx=None):
        """JSON paths for one search. ``cv_idx`` selects ``fold{cv_idx}/``."""
        if cv_idx is not None:
            dump_dir_path = dump_dir_path / f"fold{cv_idx}/"

        @dataclass
        class DumpPaths:
            grid_results: str; prelim_results: str; softcut_results: str; meta: str; red_id: list[str]   # fmt: skip

        return DumpPaths(
            grid_results=dump_dir_path / "grid_results.json",
            prelim_results=dump_dir_path / f"prelim_results.json",
            softcut_results=dump_dir_path / f"softcut_results.json",
            meta=dump_dir_path / "meta.json",
            red_id=dump_dir_path / "red_id.json",
        )

    def get_metadata(self):
        """Grid settings, endpoint, and number of patients in the filter-fit cohort."""
        metadata = {
            "best_global_vaf_cutoff": self.best_global_vaf_cutoff,
            "vaf_grid": self.grid.vaf_grid,
            "n_vaf_gridpoints": self.grid.n_vaf_gridpoints,
            "vaf_grid_min_cases_per_mutaclass": self.grid.min_cases_per_mutaclass,
            "n_red_id_used_in_varsearch": self.VAFmax_df_for_filterfit["RED_ID"].nunique(),
            "endpoint": self.endpoint,
            "date": datetime.now().strftime("%Y-%m-%d"),
        }
        return metadata

