"""Cross-validation per-gene VAF grid search used by run_gridsearch_analyses.py."""

from utils.display import *
from utils.prepare_data.prepare_data import DataBuilder
from utils.paths import Paths
from utils.variablesearch.variable_searcher import VariableSearcher
from utils.variablesearch.grid_constructor import GridConstructor
from typing import Literal


def run_gridsearch_analysis_on_each_crossval_trainset(
    variants_df,
    subset: Literal["Norway", "Norway, LUAD", "Norway, DSS", "Norway, OS"] = "Norway",
):
    """Fit adaptive and fixed VAF cutoffs on each cross-validation training fold."""
    if subset == "Norway":
        path_to_cv_splits = Paths().cv_splits_norway_path
        path_to_dump_fixedgrid_results = Paths().crossval_varselection_dump_fixedgrid_norway_dir
        path_to_dump_flexigrid_results = Paths().crossval_varselection_dump_flexigrid_norway_dir
        endpoint = "TTR"
    elif subset == "Norway, LUAD":
        path_to_cv_splits = Paths().cv_splits_norway_luad_path
        path_to_dump_fixedgrid_results = Paths().crossval_varselection_dump_fixedgrid_luad_norway_dir
        path_to_dump_flexigrid_results = Paths().crossval_varselection_dump_flexigrid_luad_norway_dir
        endpoint = "TTR"
    elif subset == "Norway, DSS":
        path_to_cv_splits = Paths().cv_splits_norway_path
        path_to_dump_fixedgrid_results = Paths().crossval_varselection_dump_fixedgrid_norway_DSS_dir
        path_to_dump_flexigrid_results = Paths().crossval_varselection_dump_flexigrid_norway_DSS_dir
        endpoint = "DSS"
    elif subset == "Norway, OS":
        path_to_cv_splits = Paths().cv_splits_norway_path
        path_to_dump_fixedgrid_results = Paths().crossval_varselection_dump_fixedgrid_norway_OS_dir
        path_to_dump_flexigrid_results = Paths().crossval_varselection_dump_flexigrid_norway_OS_dir
        endpoint = "OS"

    cv_splits = Paths.load_json(path_to_cv_splits)
    white_print(f"\n\nRunning CV grid-search analysis for {subset}\nDumping results to: {path_to_dump_fixedgrid_results} and {path_to_dump_flexigrid_results}\n\n")   # fmt: skip

    blue_print("COLLECTING CV RESULTS: FIXED GRID FROM 0-0.1")
    for i in range(5):
        orange_print(f"Iteration {i} of variable selection...")
        varsearcher_i = VariableSearcher(
            variants_df=variants_df,
            clinical_df=DataBuilder(endpoint=endpoint).get_clinical_data(),
            endpoint=endpoint,
            fit_softcut_function_in_gridanalysis=True,
            grid=GridConstructor(vaf_grid="fixed"),
            red_id_in_genomic_df=cv_splits["train"][i],
        )
        varsearcher_i.run_gridsearch_analysis_for_genes()
        varsearcher_i.compute_result_summary_dfs()
        varsearcher_i.dump_results_to_json(path_to_dump_fixedgrid_results, cv_idx=i)

    blue_print("COLLECTING CV RESULTS: FLEXIBLE GRID")
    for i in range(5):
        orange_print(f"Iteration {i} of variable selection...")
        varsearcher_i = VariableSearcher(
            variants_df=variants_df,
            clinical_df=DataBuilder(endpoint=endpoint).get_clinical_data(),
            endpoint=endpoint,
            fit_softcut_function_in_gridanalysis=True,
            red_id_in_genomic_df=cv_splits["train"][i],
        )
        varsearcher_i.run_gridsearch_analysis_for_genes()
        varsearcher_i.compute_result_summary_dfs()
        varsearcher_i.dump_results_to_json(path_to_dump_flexigrid_results, cv_idx=i)
