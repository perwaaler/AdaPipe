"""Load TCGA clinical and mutation tables for the reported analyses."""

import numpy as np
import pandas as pd
from utils.paths import Paths
from utils.TCGA import clinvar_metadata
from dataclasses import dataclass
from utils.display import *
from typing import Literal
from utils.models.models import CPHModel
from utils.prepare_data.prepare_data import DataBuilder


@dataclass
class PrepareTCGAClinData:
    """Loads and prepares clinical TCGA data. The time-to-event columns for the
    chosen endpoint determines which endpoint is used in subsequent analysis
    scripts."""

    drop_cols_with_all_missing: bool = True
    drop_cols_that_are_constants: bool = True
    drop_stage4: bool = True
    drop_rows_with_missing_timetoevent_data: bool = True
    endpoint: Literal["OS", "PFI", "DSS", "PFS"] = "DSS"
    histology: Literal["LUAD", "LUSC", "NSCLC"] = "LUAD"
    verbose: bool = 1
    infer_eventobserved_to_censored_if_timetoevent_exists: bool = True  # Some rows have time-to-event=nan but still have time-to-event data: decide how to handle    # fmt: skip

    def get_clinical_data(self):
        # --- LOAD DATA
        # CURATED (CDR PAPER) TCGA DATA
        curated_endpoints_df = pd.read_excel(Paths().tcga_CDRcurated_endpoints_csv, index_col=0, sheet_name="TCGA-CDR").rename(columns={"bcr_patient_barcode": "Patient ID"})   # fmt: skip
        self.log(f"Histology: {self.histology}", c="white")

        if self.histology == "LUAD":
            curated_endpoints_df = curated_endpoints_df.query("type=='LUAD'").copy()
            # TCGA DATA
            clinical_df0 = pd.read_csv(Paths().tcga_luad_clinical_data_tsv, sep="\t")
            clinical_df0["histology"] = "LUAD"

        elif self.histology == "LUSC":
            curated_endpoints_df = curated_endpoints_df.query("type=='LUSC'").copy()
            # TCGA DATA
            clinical_df0 = pd.read_csv(Paths().tcga_clinical_patient_csv).rename(columns=clinvar_metadata.map1_LUSC_clinicalPatient)   # fmt: skip
            clinical_df0["histology"] = "LUSC"

        elif self.histology == "NSCLC":
            curated_endpoints_df = curated_endpoints_df.query("type in ['LUAD', 'LUSC']").copy()   # fmt: skip
            # LUAD
            clinical_LUAD_df0 = pd.read_csv(Paths().tcga_luad_clinical_data_tsv, sep="\t")
            clinical_LUAD_df0["histology"] = "LUAD"
            # LUSC
            clinical_LUSC_df0 = pd.read_csv(Paths().tcga_clinical_patient_csv).rename(columns=clinvar_metadata.map1_LUSC_clinicalPatient)   # fmt: skip
            clinical_LUSC_df0["histology"] = "LUSC"
            # MERGE
            clinical_df0 = pd.concat([clinical_LUAD_df0, clinical_LUSC_df0], join="inner")

        self.log(f"Initial n {len(clinical_df0)}")
        # ATTACH CURATED ENDPOINTS
        curated_vars_to_extract = [
            "Patient ID",
            "OS",
            "OS.time",
            "PFI",
            "PFI.time",
            "DSS",
            "DSS.time",
        ]
        merged_df = clinical_df0.merge(curated_endpoints_df[curated_vars_to_extract], on="Patient ID")

        # DROP IRRELEVANT COLUMNS
        clinical_df = merged_df.drop(columns=clinvar_metadata.cols_to_drop, errors="ignore")

        # ---------- TIME TO EVENT ANALYSIS VARIABLES -----------------------------------------------------------------------------------------
        if self.endpoint == "OS":
            self.log("Using OS as endpoint...")
            clinical_df["event_observed"] = clinical_df["OS"]
            clinical_df["time_to_event"] = clinical_df["OS.time"]

        elif self.endpoint == "PFI":
            self.log("Using pre-curated Progression Free Survival...", c="magenta")
            # USE RELAPSE ENDPOINT THAT IS PRE-CURATED
            clinical_df["event_observed"] = clinical_df["PFI"]
            clinical_df["time_to_event"] = clinical_df["PFI.time"]

        elif self.endpoint == "DSS":
            self.log("Using pre-curated Disease Specific Survival...", c="magenta")
            # USE RELAPSE ENDPOINT THAT IS PRE-CURATED
            clinical_df["event_observed"] = clinical_df["DSS"]
            clinical_df["time_to_event"] = clinical_df["DSS.time"]

        elif self.endpoint == "PFS":
            self.log("Using pre-curated Progression Free Survival...", c="magenta")
            # Construct RFS from PFI and OS
            clinical_df["event_observed"] = clinical_df[["PFI", "OS"]].max(axis=1)
            clinical_df["time_to_event"] = clinical_df[["PFI.time", "OS.time"]].min(axis=1)

        if self.infer_eventobserved_to_censored_if_timetoevent_exists:
            clinical_df.loc[
                clinical_df["time_to_event"].notna() & clinical_df["event_observed"].isna(), "event_observed"
            ] = 0.0

        self.print_stats_for_timetoevent_data(clinical_df) if self.verbose > 0 else None

        # --- STANDARDIZE COLUMN NAMES AND VARIABLE VALUES
        clinical_df["RED_ID"] = clinical_df["Patient ID"]
        clinical_df["pstage_raw"] = clinical_df["Neoplasm Disease Stage American Joint Committee on Cancer Code"]   # fmt: skip
        clinical_df["pstage"] = clinical_df["pstage_raw"].map(clinvar_metadata.stage_map)
        clinical_df = clinical_df.rename(columns={"Diagnosis Age": "age"})

        clinical_df["pstage_linear"] = clinical_df["pstage_raw"].map(clinvar_metadata.stage_map_granular)

        # --- COLUMN FILTERING
        if self.drop_cols_with_all_missing:
            clinical_df = drop_vars_with_all_missing(clinical_df, self.verbose)

        if self.drop_cols_that_are_constants:
            clinical_df = drop_constant_vars(clinical_df, self.verbose)

        # --- BINARY INDICATOR COLUMNS FOR CONVENIENCE
        clinical_df["pstage2"] = clinical_df["pstage"] == "2"
        clinical_df["pstage3"] = clinical_df["pstage"] == "3"
        clinical_df["pstage4"] = clinical_df["pstage"] == "4"

        # --- ROW FILTERING
        # MISSING ENDPOINT-STATUS FILTERING
        if self.drop_rows_with_missing_timetoevent_data:
            self.log(f"Dropping {clinical_df[["time_to_event", "event_observed"]].isna().any(axis=1).sum()} rows with missing event data")   # fmt: skip
            clinical_df = clinical_df.dropna(subset=["time_to_event", "event_observed"])

        # STAGE FILTERING
        if self.drop_stage4:
            stages_to_discard = ["4"]
            clinical_df = clinical_df.query(f"pstage not in {stages_to_discard}").copy()

        self.log(f"Size of curated dataset: {len(clinical_df)}", c="blue")

        return clinical_df

    def print_stats_for_timetoevent_data(self, clinical_df):
        green_print(f"Value counts (Endpoint: {self.endpoint})")
        display(clinical_df["event_observed"].value_counts(dropna=False))
        green_print("Sample of time-to-event analysis rows with missing data")
        time_analysis_cols = ["time_to_event", "event_observed"]
        df = pd.concat(
            [
                clinical_df.query(f"time_to_event.isna() & event_observed.notna()")[time_analysis_cols].head(5),
                clinical_df.query(f"time_to_event.notna() & event_observed.isna()")[time_analysis_cols].head(5),
            ],
            keys=["time is nan, event exists", "time exists, event is nan"],
        )
        display(df)

    def log(self, txt: str, c="grey", verbose_lvl=1):
        if self.verbose >= verbose_lvl:
            color_print(txt, c=c)


@dataclass
class PrepareTCGAMutaData:
    """TCGA mutation table with OncoKB annotations. For consistency, maps 
    column names to match the discovery-cohort."""

    drop_cols_with_all_missing: bool = True
    drop_constants: bool = True
    keep_only_TSO500_genes: bool = True
    histology: Literal["LUAD", "LUSC", "NSCLC"] = "LUAD"
    verbose: bool = 1

    def prepare_mutation_data(self):
        """Load TCGA variants, attach OncoKB, and keep TSO500 genes."""
        assert self.histology in ["LUAD", "LUSC", "NSCLC"], f"histology {self.histology} not valid argument"

        # --- LOAD VARIANTS DATA (VARIANTS DATA AND ONCOKB ANNOTATIONS)
        if self.histology == "LUAD":
            tcga_mutas_df = pd.read_csv(Paths().tcga_luad_mutas_csv)
            tcga_OncoKB_df = pd.read_csv(Paths().tcga_luad_oncokb_out_txt, sep="\t", low_memory=False)
            tcga_OncoKB_df["histology"] = "LUAD"

        elif self.histology == "LUSC":
            tcga_mutas_df = pd.read_csv(Paths().tcga_lusc_mutas_csv)
            tcga_OncoKB_df = pd.read_csv(Paths().tcga_lusc_oncokb_out_txt, sep="\t", low_memory=False)
            tcga_OncoKB_df["histology"] = "LUSC"

        elif self.histology == "NSCLC":
            tcga_LUADmutas_df = pd.read_csv(Paths().tcga_luad_mutas_csv)
            tcga_LUSCmutas_df = pd.read_csv(Paths().tcga_lusc_mutas_csv)
            # LUAD and LUSC confirmed to have same columns
            tcga_mutas_df = pd.concat([tcga_LUADmutas_df, tcga_LUSCmutas_df])

            tcga_LUAD_OncoKB_df = pd.read_csv(Paths().tcga_luad_oncokb_out_txt, sep="\t", low_memory=False)
            tcga_LUSC_oncokb_df = pd.read_csv(Paths().tcga_lusc_oncokb_out_txt, sep="\t", low_memory=False)
            # LUAD and LUSC confirmed to have same columns
            tcga_OncoKB_df = pd.concat([tcga_LUAD_OncoKB_df, tcga_LUSC_oncokb_df], keys=["LUAD", "LUSC"])
            tcga_OncoKB_df.index = [x[0] for x in tcga_OncoKB_df.index]
            tcga_OncoKB_df = tcga_OncoKB_df.reset_index(names="histology")

        # --- ID COLUMNS (USE VARIABLE NAMES IN TNMI FOR COMPATIBILITY AND SIMPLICITY)
        tcga_mutas_df["RED_ID"] = tcga_mutas_df["Tumor_Sample_Barcode"].str[:12]
        tcga_mutas_df["SAMPLE_ID"] = tcga_mutas_df["RED_ID"]
        # Assign new variant identifiers
        tcga_mutas_df["ID"] = np.arange(len(tcga_mutas_df))

        # RENAME OTHER COLUMNS
        tcga_mutas_df = tcga_mutas_df.rename(columns={"Consequence": "CONSEQUENCE"})

        # MERGE MUTATION DATA WITH ONCOKB ANNOTATIONS
        shared_columns = list(tcga_OncoKB_df.columns.intersection(tcga_mutas_df.columns))
        tcga_mutas_df = tcga_mutas_df.merge(tcga_OncoKB_df, on=shared_columns)

        # --- DATA FORGE
        # VAF
        tcga_mutas_df["VAF_TUMOR"] = tcga_mutas_df["t_alt_count"] / tcga_mutas_df["t_depth"]
        # DEPTH
        tcga_mutas_df["DP_TUMOR"] = tcga_mutas_df["t_depth"]
        tcga_mutas_df["SUPPORTING_READS"] = tcga_mutas_df["t_alt_count"]

        if self.drop_cols_with_all_missing:
            tcga_mutas_df = drop_vars_with_all_missing(tcga_mutas_df, self.verbose)

        if self.drop_constants:
            tcga_mutas_df = drop_constant_vars(tcga_mutas_df, self.verbose)

        # ADD DUMMY ROWS FOR ID'S THAT HAVE NO TSO500 MUTATIONS
        tso500_genes = DataBuilder.get_tso500_gene_list()
        has_noTSO500_muta = tcga_mutas_df.groupby("RED_ID")["SYMBOL"].apply(lambda x: set(x).intersection(tso500_genes) == set())   # fmt: skip
        red_id_noTSO500_muta = has_noTSO500_muta[has_noTSO500_muta].index
        histology_noTSO500_muta = tcga_mutas_df.groupby("RED_ID")["histology"].first().loc[red_id_noTSO500_muta]
        tcga_mutas_df = add_dummy_rows(tcga_mutas_df, id_list=red_id_noTSO500_muta, histology_list=histology_noTSO500_muta.values)   # fmt: skip

        if self.keep_only_TSO500_genes:
            # DISCARD ALL ROWS THAT ARE NOT IN TSO500 OR MARKED AS 'NONE'
            genes_to_keep = DataBuilder.get_tso500_gene_list() + ["NONE"]
            tcga_mutas_df = tcga_mutas_df[tcga_mutas_df["SYMBOL"].isin(genes_to_keep)].copy()

        return tcga_mutas_df


def drop_vars_with_all_missing(df, verbose):
    all_nan_cols = df.columns[df.isna().all()]
    df = df[[x for x in df.columns if x not in all_nan_cols]].copy()
    orange_print(f"Dropping nan columns...\n {list(all_nan_cols)}") if verbose > 0 else None
    return df


def drop_constant_vars(df, verbose):
    constants = df.columns[df.nunique() == 1]
    df = df[[x for x in df.columns if x not in constants or x == "histology"]].copy()   # fmt: skip
    orange_print(f"Dropping constant columns...\n{list(constants)}") if verbose > 0 else None
    return df


def add_dummy_rows(variants_df, id_list: list[str], histology_list: list[str]):
    """Dummy rows (SYMBOL='NONE') so patients with no mutations stay in
    the table."""
    df_dummy = pd.DataFrame({"RED_ID": id_list, "histology": histology_list})
    df_dummy["SYMBOL"] = "NONE"
    df_dummy["CONSEQUENCE"] = "nothing"
    variants_df = pd.concat([variants_df, df_dummy])
    return variants_df
