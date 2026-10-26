"""Summarize the discovery-cohort permutation p-values.

Writes Tables S1 and S2, compares the ESS-penalized run with the non-penalized
and fixed 5% runs, and counts genes lost under a fixed cutoff or a universal
oncogenicity filter.
"""

from ana.ana3__permutationAdjusted_pvals import ana1__genTable__genes_selected_using_permutation_adjusted_pvals
from ana.ana3__permutationAdjusted_pvals import ana2__analyze_permutation_pvalues
from ana.ana3__permutationAdjusted_pvals import ana3__impact_of_variant_filters_on_univariate_gene_selection
from utils.prepare_data.variant_data import VariantData

gatk0_raw_df = VariantData().get_gatk()
gatk0_raw_df = VariantData().add_red_sampleid_col(gatk0_raw_df)
variants_filtered_df = VariantData().filter_variants(gatk0_raw_df, min_vaf=0.0)
variants_filtered_df["IS_ONCOGENIC"] = variants_filtered_df["ONCOGENICITY_SCORE"] >= 3

ana1__genTable__genes_selected_using_permutation_adjusted_pvals(variants_filtered_df)
ana2__analyze_permutation_pvalues()
ana3__impact_of_variant_filters_on_univariate_gene_selection()
