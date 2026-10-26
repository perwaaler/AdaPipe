"""Save lists that get used often for filtering, such as IDs describing
histological subsets. Re-run when raw or filtered variant data change.
"""

from utils.prelude import *

from utils.prepare_data.prepare_data import DataBuilder
from utils.prepare_data.variant_data import VariantData


def genes_in_variants_df(variants_filtered_df) -> list[str]:
    """Unique gene symbols in a filtered variants dataframe."""
    return sorted(variants_filtered_df.dropna(subset="SYMBOL")["SYMBOL"].unique().tolist())


def build_genes_in_norway_json(variants_filtered_df) -> list[str]:
    """Dump sorted gene list for Norway variant data (see Paths.genes_that_appear_in_norway_json)."""
    genes = genes_in_variants_df(variants_filtered_df)
    Paths.dump_to_json(genes, Paths().tso500_genes_with_offpanel_hits_json, print_confirm=True)
    green_print(f"{len(genes)} genes → {Paths().tso500_genes_with_offpanel_hits_json}")


def sort_red_ids(red_ids) -> list[str]:
    """Unique RED_IDs as strings, in stable order."""
    return sorted({str(red_id) for red_id in red_ids if pd.notna(red_id)})


def build_red_id_histology_json(histology: str, path) -> list[str]:
    """Dump RED_IDs that correspond to a specified histology."""
    clinical_df = DataBuilder().get_clinical_data()
    red_ids = sort_red_ids(clinical_df.loc[clinical_df["histology"] == histology, "RED_ID"])
    Paths.dump_to_json(red_ids, path, print_confirm=True)
    green_print(f"{len(red_ids)} {histology} IDs → {path}")
    return red_ids


def build_red_id_genomic_json(variants_df) -> list[str]:
    """Dump RED_IDs of samples that have both clinical and mutation
    data."""
    clinical_df = DataBuilder().get_clinical_data()
    red_ids = sorted(set(sort_red_ids(variants_df["RED_ID"])) & set(sort_red_ids(clinical_df["RED_ID"])))
    Paths.dump_to_json(red_ids, Paths().red_id_genomic_data_json, print_confirm=True)
    green_print(f"{len(red_ids)} genomic IDs → {Paths().red_id_genomic_data_json}")
    return red_ids


# %% LOAD DATA
orange_print("Loading heavy genomic data...")
gatk0_raw_df = VariantData().get_gatk()
gatk0_raw_df = VariantData().add_red_sampleid_col(gatk0_raw_df)

variants_filtered_df = VariantData().filter_variants(
    gatk0_raw_df.query(f"RED_ID in {DataBuilder().get_red_id_norway_data()}"),
    min_support=5,
)

# %% BUILD ARTIFACTS
build_genes_in_norway_json(variants_filtered_df)
build_red_id_histology_json("LUAD", Paths().red_id_luad_json)
build_red_id_histology_json("LUSC", Paths().red_id_lusc_json)
build_red_id_genomic_json(gatk0_raw_df)
