"""Whole-cohort per-gene VAF grid search used by run_gridsearch_analyses.py."""

from utils.display import *
from utils.prepare_data.prepare_data import DataBuilder
from utils.variablesearch.variable_searcher import VariableSearcher
from utils.variablesearch.grid_constructor import GridConstructor
from typing import Literal


def run_gridsearch_analysis_on_subset(
    variants_df,
    red_id_subset: list[str],
    vaf_grid: Literal["adaptive", "fixed"] = "adaptive",
    prelim_pval_required_for_cutoff_analysis: float = 0.1,
    endpoint: Literal["TTR", "DSS"] = "TTR",
    softcut_fit_params: VariableSearcher.SoftCutFitParams = None,
) -> VariableSearcher:
    """Fit one VAF cutoff per gene on a fixed set of patients.

    Genes in the TCGA LUAD pool are always searched, even when the preliminary
    p-value is above the cutoff.
    """
    assert isinstance(red_id_subset, list), "Provide a list of RED_IDs"
    white_print(f"Grid-type: {vaf_grid}")
    varsearcher = VariableSearcher(
        variants_df=variants_df,
        clinical_df=DataBuilder(endpoint=endpoint).get_clinical_data(),
        endpoint=endpoint,
        prelim_pval_cutoff=prelim_pval_required_for_cutoff_analysis,
        grid=GridConstructor(vaf_grid=vaf_grid),
        red_id_used_to_fit_filters=list(red_id_subset),
        genes_to_force=DataBuilder.get_geneselection_pool("LUAD"),
        fit_softcut_function_in_gridanalysis=True,
        softcut_fit_params=softcut_fit_params,
    )
    varsearcher.run_gridsearch_analysis_for_genes()
    varsearcher.compute_result_summary_dfs()
    return varsearcher
