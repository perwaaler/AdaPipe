"""Build the TCGA gene pools used for external validation.

Keeps TSO500 genes with at least 5 oncogenic mutations at VAF >= 5%. Writes one
pool for LUAD and one for LUSC. NSCLC evaluation uses the LUAD pool.
"""

from utils.prelude import *
from utils.prepare_data.prepare_data import DataBuilder
from utils.prepare_data.variant_data import GeneLevelFeatures, VariantData
from utils.TCGA.prepare_data import PrepareTCGAMutaData


def load_TCGA_variants_data(histology: Literal["LUAD", "LUSC", "NSCLC"] = "NSCLC") -> pd.DataFrame:
    """Load TCGA mutations and apply the base variant filter."""
    mutas_tcga_df = PrepareTCGAMutaData(histology=histology).prepare_mutation_data()
    tcga_filtered_df = VariantData().filter_variants(mutas_tcga_df)
    tcga_filtered_df["IS_ONCOGENIC"] = tcga_filtered_df["ONCOGENIC"].str.contains("Oncogenic", na=False)
    return tcga_filtered_df


def get_mutacounts_for_tso500_genes(
    tcga_filtered_df,
    histology_tcga: Literal["LUSC", "LUAD", "NSCLC"] = "LUAD",
    min_vaf=0.05,
):
    """Computes the number of oncogenic mutations per TSO500 gene."""
    variant_info_tcga = GeneLevelFeatures.from_variants_using_max(
        variants_filtered_df=tcga_filtered_df,
        create_zerocols_for_genes_with_no_oncovariants=True,
    )
    if histology_tcga == "NSCLC":
        tcga_filtered_histsubset_df = tcga_filtered_df.copy()
    else:
        tcga_filtered_histsubset_df = tcga_filtered_df.query(f"histology=='{histology_tcga}'").copy()

    red_id_tcga_subset = list(tcga_filtered_histsubset_df["RED_ID"].unique())
    genomic_features_histsubset_df = variant_info_tcga.genomic_features_df.query(f"RED_ID in {red_id_tcga_subset}")
    max_oncovaf_tcga_histsubset_df = genomic_features_histsubset_df[[x for x in genomic_features_histsubset_df.columns if "VAF" in x and "ONCO" in x]]   # fmt: skip

    tso500_genes = DataBuilder.get_tso500_gene_list()
    mutacount_tcga = (max_oncovaf_tcga_histsubset_df >= min_vaf).sum().sort_values(ascending=False)
    mutacount_tcga.index = [x.replace("_MAX_ONCO_VAF", "") for x in mutacount_tcga.index]

    return mutacount_tcga.loc[[x for x in mutacount_tcga.index if x in tso500_genes]]


def get_oncobased_gene_pool(
    tcga_filtered_df,
    histology_tcga: Literal["LUSC", "LUAD", "NSCLC"] = "LUAD",
    dump_path=None,
    min_n_oncogenic_mutas=5,
    min_vaf=0.05,
) -> list[str]:
    """Finds TSO500 genes with enough oncogenic mutations. Writes the list when
    dump_path is set."""
    mutacount_tcga = get_mutacounts_for_tso500_genes(tcga_filtered_df, histology_tcga, min_vaf)
    gene_selection_pool = mutacount_tcga[mutacount_tcga >= min_n_oncogenic_mutas].index.to_list()
    green_print(f"{histology_tcga} gene selection pool: {len(gene_selection_pool)} genes")
    if dump_path is not None:
        Paths.dump_to_json(gene_selection_pool, dump_path, print_confirm=True)
    return gene_selection_pool


if __name__ == "__main__":
    tcga_filtered_df = load_TCGA_variants_data("NSCLC")
    get_oncobased_gene_pool(
        tcga_filtered_df,
        histology_tcga="LUAD",
        dump_path=Paths().genepool_tcgaLUAD_validation_oncoselection_json,
    )
    get_oncobased_gene_pool(
        tcga_filtered_df,
        histology_tcga="LUSC",
        dump_path=Paths().genepool_tcgaLUSC_validation_oncoselection_json,
    )
