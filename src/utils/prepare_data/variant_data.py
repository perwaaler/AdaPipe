"""Loading, filtering, and reshaping of somatic variant data."""

import re
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from utils.display import *
from utils.paths import Paths
from utils.prepare_data import genomic_metainfo


@dataclass
class GeneLevelFeatures:
    """Class that focuses on summarizing variant level data into gene-level
    features."""

    genomic_features_df: pd.DataFrame
    variant_id_df: pd.DataFrame

    @classmethod
    def from_variants_using_max(
        cls,
        variants_filtered_df,
        red_id_subset=None,
        create_zerocols_for_genes_with_no_oncovariants=False,
    ):
        """Wide max-VAF table: one row per patient, plus max oncogenic VAF
        from ``IS_ONCOGENIC``."""
        assert "IS_ONCOGENIC" in variants_filtered_df.columns, "variants_df must have IS_ONCOGENIC"
        variants_df = variants_filtered_df.copy()

        # RESTRICT TO SUBSET OF RED ID IF A LIST OF ID'S IS PROVIDED
        if red_id_subset is not None:
            red_id_subset = Paths.load_json(Paths().red_id_norway_path)
            variants_df = variants_df[variants_df["RED_ID"].isin(red_id_subset)]

        genes = list(set(variants_filtered_df["SYMBOL"]))
        subset_label = "ONCO"
        variants_df[subset_label] = variants_df["IS_ONCOGENIC"].fillna(False).astype(bool)

        # GET MAXIMUM VAF FOR EACH SUBSET
        idx = variants_df.groupby(["RED_ID", "SYMBOL", subset_label])["VAF_TUMOR"].idxmax()
        long_df = variants_df.loc[idx, ["RED_ID", "SYMBOL", subset_label, "VAF_TUMOR", "ID"]].copy()

        # Pivot to wide format with multi-index: 0=gene, 1=oncogenicity
        wide_maxvaf_df = long_df.pivot(index="RED_ID", columns=["SYMBOL", subset_label], values="VAF_TUMOR")
        wide_id_df = long_df.pivot(index="RED_ID", columns=["SYMBOL", subset_label], values="ID")

        # Keep only columns related to the requested genes
        wide_maxvaf_df = wide_maxvaf_df.loc[:, wide_maxvaf_df.columns.get_level_values(0).isin(genes)]
        wide_id_df = wide_id_df.loc[:, wide_id_df.columns.get_level_values(0).isin(genes)]
        # Fill Nan with 0
        wide_maxvaf_df = wide_maxvaf_df.fillna(0.0)

        def flatten_column_indexes(cols):
            return [f"{g}_MAX_{subset_label}_VAF" if onco else f"{g}_MAX_VAF" for (g, onco) in cols]

        # Convert from from multi-index columns to flat index
        wide_maxvaf_df.columns = flatten_column_indexes(wide_maxvaf_df.columns)
        wide_id_df.columns = flatten_column_indexes(wide_id_df.columns)
        subset_vaf_cols = [varname for varname in wide_maxvaf_df.columns if subset_label in varname]

        # Create maxVAF variable: Replace maxVAF in not-subset with max over subset and not-subset
        for i, max_subset_vaf_col in enumerate(subset_vaf_cols):
            # Name of not-subset column if it exists
            max_notsubset_vaf_col = max_subset_vaf_col.replace(f"{subset_label}_", "")
            if max_notsubset_vaf_col in wide_maxvaf_df.columns:
                # Get locations where VAF is larger in the subset than in not-subset
                mask = wide_maxvaf_df[max_subset_vaf_col] > wide_maxvaf_df[max_notsubset_vaf_col]
                # Replac
                wide_id_df.loc[mask, max_notsubset_vaf_col] = wide_id_df.loc[mask, max_subset_vaf_col]
                wide_maxvaf_df.loc[mask, max_notsubset_vaf_col] = wide_maxvaf_df.loc[mask, max_subset_vaf_col]
            else:
                # This means the "not-subset" is empty, so max VAF is max over subset
                wide_maxvaf_df.loc[:, max_notsubset_vaf_col] = wide_maxvaf_df.loc[:, max_subset_vaf_col]
                wide_id_df.loc[:, max_notsubset_vaf_col] = wide_id_df.loc[:, max_subset_vaf_col]

            if i > 50:
                # Defragment to avoid the fragmentation warning of pandas
                wide_id_df = wide_id_df.copy()
                wide_maxvaf_df = wide_maxvaf_df.copy()

        genomic_features_df = wide_maxvaf_df.reset_index()
        variant_id_df = wide_id_df.reset_index()

        # REPLACE '-' WITH '_' IN COLUMN NAMES
        genomic_features_df.columns = [c.replace("-", "_") for c in genomic_features_df.columns]
        variant_id_df.columns = [c.replace("-", "_") for c in variant_id_df.columns]

        if create_zerocols_for_genes_with_no_oncovariants:
            genomic_features_df = cls.insert_zerocolumns_for_genes_with_no_oncovariants(genomic_features_df)

        return cls(genomic_features_df, variant_id_df)

    @staticmethod
    def get_property_for_variants(
        variants_filtered_df,
        variant_id_df,
        genomic_features_df,
        variable_to_get="ONCOGENICITY",
        vaf_cutoff: float = None,
    ) -> pd.DataFrame:
        """For each ID in ``variant_id_df``, gets the corresponding value
        for the variable ``variable_to_get``. Setting ``vaf_cutoff`` will
        assign ``None`` to variants that fail this cutoff.

        Example of ``vaf_filter``: ``lambda df: df>=0.1``."""
        # HANDLE ID COLUMN
        red_id_col = None
        if "RED_ID" in variant_id_df.columns:
            assert "RED_ID" in genomic_features_df.columns
            red_id_col = variant_id_df["RED_ID"]
            variant_id_df = variant_id_df.drop(columns="RED_ID")
            genomic_features_df = genomic_features_df.drop(columns="RED_ID")

        variable_map = variants_filtered_df.set_index("ID")[variable_to_get]
        out_df = variant_id_df.map(lambda variant_id: variable_map.get(variant_id, None))
        if vaf_cutoff is not None:
            out_df = out_df.where(genomic_features_df >= vaf_cutoff, None)

        if red_id_col is not None:
            # RE-INSERT RED ID COLUMN
            out_df.insert(0, "RED_ID", red_id_col)

        return out_df

    @classmethod
    def insert_zerocolumns_for_genes_with_no_oncovariants(
        cls, genomic_features_df, genes: list[str] = None
    ) -> pd.DataFrame:
        """Ensures each gene gets a MAX ONCO VAF column by creating zero
        columns (the reason they do not get produced in the first place is
        because the gene has no oncogenic variants.)"""
        if genes is None:
            genes = [x.split("_MAX")[0] for x in genomic_features_df.columns if x != "RED_ID"]
        vaf_cols = [c for c in genomic_features_df.columns if any([g in c for g in genes])]
        genes_with_no_oncovaf_column = [
            x for x in genes if f"{x}_MAX_VAF" in vaf_cols and f"{x}_MAX_ONCO_VAF" not in vaf_cols
        ]
        fill_cols = [f"{x}_MAX_ONCO_VAF" for x in genes_with_no_oncovaf_column]
        filler_df = pd.DataFrame(0.0, index=genomic_features_df.index, columns=fill_cols)
        genomic_features_df = pd.concat([genomic_features_df, filler_df], axis=1)
        return genomic_features_df


@dataclass
class VariantData:
    """Class for loading, curating, and managing variant sequencing data."""

    def get_gatk(self, add_red_id=True):
        """Loads large dataset with Mutect2 analysis for resected tissue from
        ~500 patients."""
        df = pd.read_csv(Paths().gatk_data_path, delimiter="\t", low_memory=False)
        df = df.rename(columns={"VARIANT_CLASS_x": "VARIANT_CLASS"})
        df = df.drop(columns=["VARIANT_CLASS_y", "AMP_ASCO_CAP", "REGULATORY_ANNOTATION"], errors="ignore")

        if add_red_id:
            df = self.add_red_sampleid_col(df)
        return self._basic_cleanup_of_variants_df(df)

    def filter_variants(
        self,
        gatk0_raw_df,
        low_impact_consequences=genomic_metainfo.low_impact_or_non_coding_variant_types,
        remove_low_impact_variants=True,
        consequence_col: Literal["CONSEQUENCE", "ONCOKB_CONSEQUENCE"] = "CONSEQUENCE",
        consequence_sep: Literal["&", ";"] = "&",
        # ONCOGENICITY AND ACTIONABILITY
        max_actionability_tier: float = None,
        # VAF AND SR
        oncokb_oncogenic_only=False,
        min_onco_score=None,
        min_vaf=0.0,
        min_support=5,
        min_depth=None,
        add_dummy_rows=True,  # Creates dummy rows (SYMBOL='NONE') so every participant in the input is represented after filtering
        verbose=False,
        level1_variants_only=False,
        id_col: Literal["RED_ID", "SAMPLE_ID"] = "RED_ID",
    ):
        """Base variant filter: consequence, VAF, and supporting reads."""
        # GET ID OF ALL PARTICIPANTS BEFORE FILTRATION
        all_participant_ids = set(gatk0_raw_df[id_col])

        n_variants_raw = gatk0_raw_df.__len__()
        n_participants = all_participant_ids.__len__()
        # EXCLUDE VARIANTS WHERE VARIANT CONSEQUENCE IS UNKNOWN (FRAMESHIFT, STOP GAINED, ETC...)
        condition = gatk0_raw_df[consequence_col].notna() & (gatk0_raw_df[consequence_col] != "")
        df = gatk0_raw_df[condition].copy()
        green_print(f"Variants dropped due to not having consequence annotations: {(~condition).sum()} ({(~condition).sum() / n_variants_raw:.1%})") if verbose else None   # fmt: skip

        if remove_low_impact_variants:
            # KEEP VARIANT IF AT LEAST ONE CONSEQUENCE IS CLINICALLY RELEVANT
            indicator_atleast_one_important_effect = lambda S: any(
                [s not in low_impact_consequences for s in S.split(consequence_sep)]
            )
            condition = df[consequence_col].apply(indicator_atleast_one_important_effect)
            df = df[df[consequence_col].apply(indicator_atleast_one_important_effect)].copy()   # fmt: skip
            green_print(f"Variants dropped due to not having any impactfull consequences: {(~condition).sum()} ({(~condition).sum() / n_variants_raw:.1%})") if verbose else None   # fmt: skip

        # ACTIONABILITY AND ONCOGENICITY ACC. TO CLINGEN/CGC/VICC ONCOGENICITY GUIDELINES
        if max_actionability_tier is not None:  # KEEP ONLY ACTIONABLE VARIANTS
            df = df.query(f"ACTIONABILITY_TIER <= {max_actionability_tier}")  # fmt: skip
        if min_onco_score is not None:
            # KEEP ONLY VARIANTS THAT ARE ONCOGENIC
            condition = df["ONCOGENICITY_SCORE"] >= min_onco_score
            df = df[condition].copy()  # fmt: skip
            green_print(f"Variants dropped due to not meeting Oncoscore>= {min_onco_score}: {(~condition).sum()} ({(~condition).sum() / n_variants_raw:.1%})") if verbose else None   # fmt: skip
        if oncokb_oncogenic_only:
            condition = df["ONCOGENIC"].str.contains("Oncogenic")
            df = df[condition].copy()
            green_print(f"Variants dropped due to not being >='Likely Oncogenic' (OncoKB SOP): {(~condition).sum() / n_variants_raw:.1%}")   # fmt: skip
        # --- VAF FILTERING
        condition = df["VAF_TUMOR"] >= min_vaf
        df = df[condition].copy()
        green_print(f"Variants dropped due to not meeting VAF>= {min_vaf}: {(~condition).sum()} ({(~condition).sum() / n_variants_raw:.1%})") if verbose else None   # fmt: skip

        # EXCLUDE VARIANTS THAT ARE LIKELY GERMLINE
        condition = df["VAF_TUMOR"] <= 0.95
        df = df[condition].copy()
        green_print(f"Variants dropped due to not meeting VAF<=95%: {(~condition).sum()} ({(~condition).sum() / n_variants_raw:.6%})") if verbose else None   # fmt: skip

        # --- SUPPORTING READS & DEPTH FILTERING
        if min_depth is not None:
            condition = df["DP_TUMOR"] >= min_depth
            df = df[condition].copy()
        # CREATE SUPPORTING READS VARIABLE
        df["SUPPORTING_READS"] = (df["VAF_TUMOR"] * df["DP_TUMOR"]).astype(int)  # DP: Depth, Read Depth, Coverage, or Sequencing Depth   # fmt: skip
        # FILTER ROWS WITH LOW COVERAGE
        condition = df["SUPPORTING_READS"] >= min_support
        df = df[condition].copy()
        green_print(f"Variants dropped due to not meeting Coverage>={min_support}: {(~condition).sum()} ({(~condition).sum() / n_variants_raw:.1%})") if verbose else None   # fmt: skip

        teal_print(f"Variants included in analysis: {len(df)} ({len(df) / n_variants_raw:.1%}; {len(df) / n_participants:.0f} variants per patient)") if verbose else None   # fmt: skip

        if level1_variants_only:
            condition = df["IS_LEVEL_1"]
            df = df[condition].copy()

        if add_dummy_rows:
            # ADD DUMMY ROWS TO ENSURE THE ALL PATIENTS APPEAR IN THE FILTERED DF
            white_print("Adding dummy rows for participants with no variants after filtration (SYMBOL=='NONE')") if verbose else None   # fmt: skip
            df = self.add_dummy_rows_for_mutationless_samples(
                variants_df=df, id_full_dataset=all_participant_ids, id_col=id_col
            )

        return df

    def add_dummy_rows_for_mutationless_samples(
        self, variants_df, id_full_dataset: list[str], id_col: Literal["RED_ID", "SAMPLE_ID"] = "RED_ID"
    ):
        """Creates dummy rows (SYMBOL='NONE' and 'CONSEQUENCE'='nothing') to
        represent each person that does not appear in the dataset to ensure that
        number of samples in the dataset."""
        df_dummy = pd.DataFrame({id_col: list(set(id_full_dataset) - set(variants_df[id_col]))})   # fmt: skip
        df_dummy["SYMBOL"] = "NONE"
        df_dummy["CONSEQUENCE"] = "nothing"
        variants_df = pd.concat([variants_df, df_dummy])
        return variants_df

    def _basic_cleanup_of_variants_df(self, df):
        """Performs the most common cleanup steps (replacing Nan etc...)."""
        # STANDARDIZE MISSING VALUES IN MUTATION CONSEQUENCE COLUMNS
        df["ONCOKB_CONSEQUENCE"] = df["ONCOKB_CONSEQUENCE"].fillna("")
        df["CONSEQUENCE"] = df["CONSEQUENCE"].fillna("")

        # ADD SOME ANALYSIS-CONVENIENCE COLUMNS
        df["IS_LEVEL_1"] = df["LEVEL_1"].notna()
        df["EXON_NUMBER"] = df["CDS_CHANGE"].apply(VariantData.extract_exon_from_cds_change)

        # CALCULATE SUPPORTING READS FROM VAF AND COVERAGE
        df.loc[:, "SUPPORTING_READS"] = (df["VAF_TUMOR"] * df["DP_TUMOR"]).astype(int)  # DP: Depth, Read Depth, Coverage, or Sequencing Depth   # fmt: skip
        silent_print("Data prepared: filled Nan in CONSEQUENCE columns with '', created variable IS_LEVEL_1")  # fmt: skip
        return df

    def add_red_sampleid_col(self, df) -> pd.DataFrame:
        """Adds column RED_ID with RED-id values that match the id-values in
        SAMPLE_ID. Used to couple genomic clnical data."""
        if "RED_ID" in df.columns:
            return df

        # Get key dataframe which has both SAMPLE_ID and RED_ID
        dfkey = pd.read_csv(Paths().tmb_msi_key_filter_path, low_memory=False)

        # Remove duplicates (rows with suffixes like _v1 etc...)
        dfkey = dfkey[~self._identify_red_id_duplicates(dfkey)]
        # Standardize column names
        dfkey = dfkey.rename(columns={"Pair_ID_original": "SAMPLE_ID"})  # fmt: skip
        # 13 SAMPLE_ID are not found in key-dataframe
        # MERGE USING 'LEFT' (MAINTAIN SHAPE/ROWS OF LEFT DF)
        df = df.merge(dfkey[["SAMPLE_ID", "RED_ID"]], on="SAMPLE_ID", how="left")
        # CONFIRMED: NO ROWS IN VARIANT-CALLER DF LOST IN THE MERGE

        # REARRANGE COLUMNS so that RED_ID appears next to SAMPLE_ID
        df = df[["SAMPLE_ID", "RED_ID"] + [c for c in df.columns if c not in ["SAMPLE_ID", "RED_ID"]]]  # fmt: skip

        return df

    # ---------- VARIANT DATAFRAME UTILITIES (variants_filtered_df) -----------------
    @staticmethod
    def extract_exon_from_cds_change(s: str) -> int | None:
        """Extract exon number from ``CDS_CHANGE``. Returns None if non-coding/intron."""
        if isinstance(s, str):
            match = re.search(r"exon(\d+)", s)
            if match:
                return int(match.group(1))

    @staticmethod
    def _identify_red_id_duplicates(df: pd.Series) -> pd.Series:
        """Identifies rows that ends with the suffix like _v1."""
        return df["RED_ID"].str.contains(r"_v\d+$", regex=True)
