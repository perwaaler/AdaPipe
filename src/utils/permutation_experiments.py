"""Permutation runs that produce the reported discovery-cohort p-values."""

import pandas as pd

from utils.display import *
from utils.gen_permutation_adjusted_pvalues import PValuePermuter, VAFTransformer
from utils.models.penalizers import EffectiveSampleSizePenalizer
from utils.paths import Paths
from utils.variablesearch.variable_searcher import VariableSearcher


def run_permutation__stiff_5pct(variants_filtered_df: pd.DataFrame):
    """Fixed 5% VAF cutoff, oncogenicity filter off."""
    white_print("\n\n--- PERMUTATION: Stiff pipeline (5% cutoff) ---")
    permuter = PValuePermuter(
        vaf_transformer=VAFTransformer(method="fixed_cutoff", stiff_cutoff=0.05),
        dump_path=Paths().permutated_pvals_fixedcut_norway_json,
        csv_dump_path=Paths().permutated_pvals_fixedcut_norway_csv,
        n_permutations_max=5000,
        oncofilter_settings="off",
    )
    permuter.run_permutation_tests(variants_filtered_df)
    permuter.dump_results()
    return permuter


def run_permutation__fluid_ess_penalized_adaptive(variants_filtered_df: pd.DataFrame):
    """ESS-penalized softcut with oncofilter_settings="adaptive".

    That setting runs the adaptive oncogenicity search and stores a permutation
    p-value for each strategy separately: off, on, and adaptive. The off and on
    results are byproducts of the adaptive run, not separate experiments.
    """
    white_print("\n\n--- PERMUTATION: Fluid pipeline (ESS-penalized, adaptive oncofilter) ---")
    permuter = PValuePermuter(
        vaf_transformer=VAFTransformer(
            method="adapt_softcut",
            softcut_fitparams=VariableSearcher.SoftCutFitParams(
                effective_samplesize_regularizer=EffectiveSampleSizePenalizer(),
            ),
        ),
        dump_path=Paths().permutated_pvals_esspenalized_adaptive_norway_json,
        csv_dump_path=Paths().permutated_pvals_esspenalized_adaptive_norway_csv,
        oncofilter_settings="adaptive",
        n_permutations_max=5000,
    )
    permuter.run_permutation_tests(variants_filtered_df)
    permuter.dump_results()
    return permuter
