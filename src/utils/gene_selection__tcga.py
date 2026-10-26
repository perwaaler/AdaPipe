"""Per-gene VAF grid search on TCGA LUAD for Figure S12.

AdaPipe chooses a VAF cutoff for each gene in the discovery cohort, using time
to recurrence. To see whether those cutoffs are cohort-specific, the same
search is run independently in TCGA LUAD, with overall survival as the
endpoint. OS is used because it leaves more genes with evidence of an effect
in both cohorts than disease-specific survival does. Figure S12 then plots the
two cutoff estimates against each other for genes with p < 0.1 in both.
"""

import pandas as pd

from utils.display import *
from utils.models.penalizers import EffectiveSampleSizePenalizer
from utils.paths import Paths
from utils.prepare_data.prepare_data import DataBuilder
from utils.prepare_data.variant_data import VariantData
from utils.TCGA.prepare_data import PrepareTCGAClinData
from utils.TCGA.prepare_data import PrepareTCGAMutaData
from utils.variablesearch.grid_cutoff_selectors import MinPvalOverEligible
from utils.variablesearch.variable_searcher import VariableSearcher


def load_tcga_variants_filtered() -> pd.DataFrame:
    """Filtered TCGA LUAD mutation table."""
    mutas_tcga_df = PrepareTCGAMutaData().prepare_mutation_data()
    tcga_filtered_df = VariantData().filter_variants(mutas_tcga_df)
    tcga_filtered_df["IS_ONCOGENIC"] = tcga_filtered_df["ONCOGENIC"].str.contains("Oncogenic", na=False)
    return tcga_filtered_df


def run_gridsearch_analysis_on_TCGA(tcga_filtered_df: pd.DataFrame) -> VariableSearcher:
    """Fit one adaptive VAF cutoff per TSO500 gene on the full TCGA LUAD cohort.

    Overall survival is the endpoint. The effective-sample-size penalty is the
    same one used for the discovery-cohort softcut. A gene enters the cutoff
    search only when its preliminary p-value is below 0.5. Results are written
    to the TCGA OS grid-search folder.
    """
    genes_to_analyze = DataBuilder.get_tso500_gene_list()
    orange_print(f"Analyzing {len(genes_to_analyze)} genes (LUAD, endpoint=OS, grid=adaptive)")
    varsearcher = VariableSearcher(
        variants_df=tcga_filtered_df,
        genes_to_analyze=genes_to_analyze,
        red_id_used_to_fit_filters="no_restriction",
        genes_to_force=DataBuilder.get_geneselection_pool(),
        prelim_pval_cutoff=0.5,
        clinical_df=PrepareTCGAClinData(endpoint="OS", verbose=0).get_clinical_data(),
        endpoint="OS",
        grid_cutoff_selector=MinPvalOverEligible(min_n_mutas=10),
        fit_softcut_function_in_gridanalysis=True,
        softcut_fit_params=VariableSearcher.SoftCutFitParams(
            effective_samplesize_regularizer=EffectiveSampleSizePenalizer(),
        ),
    )
    varsearcher.prepare_analysis_dataframe()
    varsearcher.run_gridsearch_analysis_for_genes()
    varsearcher.compute_result_summary_dfs()
    varsearcher.dump_results_to_json(Paths().varselection_dump_TCGA_OS_dir)
    return varsearcher
