"""Re-estimate AdaPipe VAF cutoffs in TCGA LUAD for Figure S12.

The discovery cohort fits a cutoff per gene using time to recurrence. This
script repeats that search in TCGA LUAD, using overall survival, so the two
sets of cutoffs can be compared.
"""

from utils.display import *
from utils.gene_selection__tcga import load_tcga_variants_filtered
from utils.gene_selection__tcga import run_gridsearch_analysis_on_TCGA

orange_print("Loading TCGA LUAD mutation data...")
tcga_luad_filtered_df = load_tcga_variants_filtered()

blue_print("\n\n\n--- TCGA LUAD, whole cohort (adaptive, ESS-penalized softcut) ---")
run_gridsearch_analysis_on_TCGA(tcga_luad_filtered_df)
