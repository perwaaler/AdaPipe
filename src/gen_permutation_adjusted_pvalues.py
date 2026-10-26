"""Permutation-adjusted p-values for the discovery cohort.

Runs the ESS-penalized adaptive pipeline used in Tables S1 and S2, and the
fixed 5% cutoff used as the comparison.
"""

from utils.display import *
from utils.permutation_experiments import run_permutation__fluid_ess_penalized_adaptive
from utils.permutation_experiments import run_permutation__stiff_5pct
from utils.prepare_data.variant_data import VariantData

orange_print("Loading heavy genomic data...")
gatk_raw_df = VariantData().get_gatk()
gatk_raw_df = VariantData().add_red_sampleid_col(gatk_raw_df)
variants_filtered_df = VariantData().filter_variants(gatk_raw_df)
variants_filtered_df["IS_ONCOGENIC"] = variants_filtered_df["ONCOGENICITY_SCORE"] >= 3

run_permutation__fluid_ess_penalized_adaptive(variants_filtered_df)
run_permutation__stiff_5pct(variants_filtered_df)
