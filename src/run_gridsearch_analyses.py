"""Per-gene VAF grid searches in the discovery cohort.

Whole-cohort searches set the cutoffs for the TCGA models (NSCLC TTR, LUAD TTR,
and NSCLC DSS). The ESS-penalized search is the Norway side of Figure S12.
Cross-validation searches cover TTR, DSS, and OS.
"""

from utils.prelude import *
from utils.prepare_data.prepare_data import DataBuilder
from utils.prepare_data.variant_data import VariantData
from utils.variablesearch.variable_searcher import VariableSearcher
from utils.models.penalizers import EffectiveSampleSizePenalizer
from utils.gene_selection__discovery import run_gridsearch_analysis_on_subset
from utils.gene_selection__discovery_crossval import run_gridsearch_analysis_on_each_crossval_trainset

orange_print("Loading heavy genomic data...")
gatk_raw_df = VariantData().get_gatk()
gatk_raw_df = VariantData().add_red_sampleid_col(gatk_raw_df)
variants_filtered_df = VariantData().filter_variants(gatk_raw_df)
variants_filtered_df["IS_ONCOGENIC"] = variants_filtered_df["ONCOGENICITY_SCORE"] >= 3

clinical_df = DataBuilder().get_clinical_data()
red_id_norway = clinical_df.query("country=='norway'")["RED_ID"].to_list()
red_id_norway_luad = clinical_df.query("country=='norway' & histology=='LUAD'")["RED_ID"].to_list()

# ---------- WHOLE COHORT -----------------------------------------------------------------------------------------
blue_print("\n\n\n--- NORWAY, NSCLC (TTR) ---")
for dump_dir_path, vaf_grid in zip(
    [Paths().varselection_dump_flexigrid_norway_dir, Paths().varselection_dump_fixedgrid_norway_dir],
    ["adaptive", "fixed"],
):
    varsearcher = run_gridsearch_analysis_on_subset(
        variants_filtered_df,
        vaf_grid=vaf_grid,
        red_id_subset=red_id_norway,
        prelim_pval_required_for_cutoff_analysis=1,
    )
    varsearcher.dump_results_to_json(dump_dir_path)

blue_print("\n\n\n--- NORWAY, NSCLC (TTR, ESS-penalized) ---")
varsearcher = run_gridsearch_analysis_on_subset(
    variants_filtered_df,
    red_id_subset=red_id_norway,
    prelim_pval_required_for_cutoff_analysis=1,
    softcut_fit_params=VariableSearcher.SoftCutFitParams(
        effective_samplesize_regularizer=EffectiveSampleSizePenalizer(),
    ),
)
varsearcher.dump_results_to_json(Paths().varselection_dump_flexigrid_norway_ESSpen_dir)

blue_print("\n\n\n--- NORWAY, LUAD (TTR) ---")
for dump_dir_path, vaf_grid in zip(
    [Paths().varselection_dump_LUAD_norway_flexigrid_dir, Paths().varselection_dump_LUAD_norway_fixedgrid_dir],
    ["adaptive", "fixed"],
):
    varsearcher = run_gridsearch_analysis_on_subset(
        variants_filtered_df,
        vaf_grid=vaf_grid,
        red_id_subset=red_id_norway_luad,
        prelim_pval_required_for_cutoff_analysis=0.2,
    )
    varsearcher.dump_results_to_json(dump_dir_path)

blue_print("\n\n\n--- NORWAY, NSCLC (DSS) ---")
for dump_dir_path, vaf_grid in zip(
    [Paths().varselection_dump_norway_DSS_dir, Paths().varselection_dump_norway_DSS_fixedgrid_dir],
    ["adaptive", "fixed"],
):
    varsearcher = run_gridsearch_analysis_on_subset(
        variants_filtered_df,
        vaf_grid=vaf_grid,
        red_id_subset=red_id_norway,
        endpoint="DSS",
        prelim_pval_required_for_cutoff_analysis=1,
    )
    varsearcher.dump_results_to_json(dump_dir_path)

# ---------- CROSS-VALIDATION -----------------------------------------------------------------------------------------
blue_print("\n\n\n--- CROSS-VALIDATION: NORWAY (TTR) ---")
run_gridsearch_analysis_on_each_crossval_trainset(variants_filtered_df, subset="Norway")

blue_print("\n\n\n--- CROSS-VALIDATION: NORWAY, LUAD (TTR) ---")
run_gridsearch_analysis_on_each_crossval_trainset(variants_filtered_df, subset="Norway, LUAD")

blue_print("\n\n\n--- CROSS-VALIDATION: NORWAY (DSS) ---")
run_gridsearch_analysis_on_each_crossval_trainset(variants_filtered_df, subset="Norway, DSS")

blue_print("\n\n\n--- CROSS-VALIDATION: NORWAY (OS) ---")
run_gridsearch_analysis_on_each_crossval_trainset(variants_filtered_df, subset="Norway, OS")
