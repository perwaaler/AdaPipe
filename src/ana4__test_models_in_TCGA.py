"""External validation of the discovery models in TCGA.

Runs the main TCGA models, the mutation-frequency comparison (Figure S3),
the VAF-cutoff concordance scatter (Figure S12), and the fixed-cutoff
table (Table S3).
"""

# %% ---------------------------------------------------------------------------------------------------
from utils.prelude import *
from utils.prepare_data.variant_data import VariantData
from utils.TCGA.prepare_data import PrepareTCGAMutaData
from ana.ana4__test_models_in_TCGA import ana1__main_analysis
from ana.ana4__test_models_in_TCGA import ana2__VAFcutoffs_TCGA_vs_TNMi
from ana.ana4__test_models_in_TCGA import ana3__compare_mutationrates_between_norway_and_TCGA
from ana.ana4__test_models_in_TCGA import ana4__produce_table_S5_stiff_VAFcutoffs_in_TCGA_LUAD

mutas_tcga_df = PrepareTCGAMutaData(histology="NSCLC").prepare_mutation_data()
tcga_filtered_df = VariantData().filter_variants(mutas_tcga_df, min_support=5)
tcga_filtered_df["IS_ONCOGENIC"] = tcga_filtered_df["ONCOGENIC"].str.contains("Oncogenic", na=False)

ana1__main_analysis(tcga_filtered_df)
ana4__produce_table_S5_stiff_VAFcutoffs_in_TCGA_LUAD(tcga_filtered_df)

orange_print("Loading TNM-I variants for supplementary TCGA comparisons...")
gatk0_raw_df = VariantData().get_gatk()
tnmi_filtered_df = VariantData().filter_variants(gatk0_raw_df, min_support=5)
tnmi_filtered_df["IS_ONCOGENIC"] = tnmi_filtered_df["ONCOGENICITY_SCORE"] >= 3

ana3__compare_mutationrates_between_norway_and_TCGA(tnmi_filtered_df, tcga_filtered_df)
ana2__VAFcutoffs_TCGA_vs_TNMi(tcga_filtered_df, tnmi_filtered_df)
