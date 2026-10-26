"""Functions for loading, cleaning, and preparing datasets.

[cursor-agent-edit-marker: 2026-08-27 — buffer test]
"""

from typing import Literal

from utils.paths import Paths
from utils.prepare_data import relabel
from pathlib import Path

from utils.display import *

import pandas as pd


class DataBuilder:
    """Class for loading, curating, and managing data."""

    def __init__(
        self,
        use_2026_clindata=True,
        use_EE_endpoint_updates=True,
        count_only_death_from_initial_tumor_in_DSS=True,
        endpoint: Literal["TTR", "OS", "DSS", "DFS"] = "TTR",
    ):
        """Column dropsets specify which columns to drop when preparing
        datasets to de-clutter them.

        # ENDPOINTS
        * TTR: relapse
        * OS: death
        * DSS: death from lung cancer
        * DFS: death or relapse
        """
        self.cols_to_drop: Literal["almost_all", "many", "none"] = "almost_all"
        self.use_2026_clindata = use_2026_clindata
        self.use_EE_endpoint_updates = use_EE_endpoint_updates
        self.count_only_death_from_initial_tumor_in_DSS = count_only_death_from_initial_tumor_in_DSS
        self.time_to_event_vars = ["TTR", "OS", "DFS", "DSS"]
        self.endpoint = endpoint
        self.run_sanity_check = False

    # ---------- LOAD RAW DATASETS -----------------------------------------------------------------------------------------
    @classmethod
    def get_tso500_gene_list(cls):
        return [x.replace("*", "").replace("-", "_") for x in Paths.load_json(Paths().tso500_genes_json)]

    @classmethod
    def load_gene_selection_pool(cls, path) -> list[str]:
        """Load a gene-pool JSON. The raw TSO500 panel file uses HLA-A / * markers;
        those are rewritten to the varsearcher spelling (HLA_A)."""

        genes = Paths.load_json(path)
        if Path(path).resolve() == Path(Paths().tso500_genes_json).resolve():
            return cls.get_tso500_gene_list()
        return genes

    # ---------- LOAD CURATED ANALYSIS-READY DATASETS -----------------------------------------------------------------------------------------
    def get_clinical_data(self):
        """Loads and curates patient variables, most importantly those
        pertaining to survival, smoking, state of immune system, recurrence,
        age, gender, histology, pathological state, and performance status.

        Note: After 2026 update, the following columns have been updated or
        corrected, so extract these variables from
        ``FU26_TNMI_with_events.csv``:

        * ``date_last_journalcheck``
        * ``lungccancerdeath``
        * ``date_birth``
        * ``relapse_loco`
        * ``adjuv_treat``
        * ``FU_time``
        * ``surgerydate``
        * ``TTR``
        * ``OS``
        * ``DFS``
        * ``relapsed``
        * ``date_of_histological_relapse``
        * ``type_adjuv_treat___3``

        * For DSS: use ``date_last_journalcheck`` as event time for those who
          have not died.
        """

        # LOAD DATA
        readable_df = pd.read_csv(Paths().clinical_labels_path, low_memory=False)
        encoded_df = pd.read_csv(Paths().FU26_TNMI_with_events_csv, low_memory=False)

        # UNIFY NAMES OF OVERLAPPING VARIABLES
        readable_df = readable_df.rename(columns=relabel.var_name_map1)
        encoded_df = encoded_df.rename(columns=relabel.var_name_map2)

        # --- MERGE DATAFRAMES (ENCODED + NOT ENCODED)
        # GIVE PRIORITY TO COLUMNS IN ENCODED DF (MOST UPDATED) IF VARIABLE IS IN BOTH DATASETS
        cols_exclusive_to_readable_df = [x for x in readable_df.columns if x not in encoded_df.columns]
        clinical_df = encoded_df.merge(readable_df[["record_id"] + cols_exclusive_to_readable_df], on="record_id")

        # RENAME VARIABLES
        # Rename to shorter version of names used in encoded_df
        clinical_df = clinical_df.rename(columns=relabel.var_name_map1)
        # Some additional post-merge renaming for convenience
        clinical_df = clinical_df.rename(columns=relabel.var_name_map3_postmerge)
        # APPLY VALUE MAP 1: MAP VALUES TO MORE CONVENIENT LABELS
        clinical_df = clinical_df.replace(relabel.value_map1_postmerge)

        # CHANGE DATATYPE
        clinical_df["pstage"] = clinical_df["pstage"].astype(str)

        # --- SURGICAL MARGINS
        # Save original version of variable prior to value merge
        clinical_df["surgical_margins_of_tumor_orgl"] = clinical_df["surgical_margins_of_tumor"]
        # Collapse to binary Feature used in ML
        clinical_df["surgical_margins_of_tumor"] = clinical_df["surgical_margins_of_tumor"].replace(
            {
                "negative_probably": "negative",
                "positive_probably": "positive",
                "unknown": "negative",
            }
        )

        # --- ADJUVANT TREATMENT
        # Create single column with type of adjuvant treatment
        clinical_df["adjuvant_treatment"] = None
        for adjuvtreat_indicator in [x for x in clinical_df.columns if "adjuvant_treatment_" in x]:
            # Grab suffix of variable
            value = adjuvtreat_indicator.replace("adjuvant_treatment_", "")
            clinical_df.loc[clinical_df[adjuvtreat_indicator] == 1, "adjuvant_treatment"] = value

        # --- HISTOLOGY
        # Summary of all checked LUAD patterns (not exclusive; can be mixed e.g. "solid_lepidic")
        luad_cols = [c for c in clinical_df.columns if c.startswith("luad_")]
        clinical_df["luad_subtype"] = clinical_df[luad_cols].apply(lambda row: "_".join(row.index[row == "Checked"]).replace("luad_", ""), axis=1)   # fmt: skip
        clinical_df["luad_subtype"] = clinical_df["luad_subtype"].replace("", None)
        clinical_df[luad_cols] = clinical_df[luad_cols].eq("Checked")
        # Store original variable (5-category labels)
        clinical_df["histology_orgl"] = clinical_df["histology"].map({v: k for k, v in relabel.value_key["histology"].items()})   # fmt: skip
        # Merge histologies to LUAD, LUSC, and other for modeling
        clinical_df["histology"] = clinical_df["histology"].map(relabel.value_map2["histology_coarse"])
    
        # SPREAD THROUGH AIRWAYS
        # Re-classify into "confirmed" and "not confirmed"
        clinical_df["tumor_spread_through_airways"] = clinical_df["tumor_spread_through_airways"].map(
            relabel.value_map2["tumor_spread_through_airways"]
        )
        # INDICATOR COLUMNS (CONVENIENCE COLUMNSS)
        clinical_df["pstage1"] = clinical_df["pstage"] == "1"
        clinical_df["pstage2"] = clinical_df["pstage"] == "2"
        clinical_df["pstage3"] = clinical_df["pstage"] == "3"
        clinical_df["male"] = clinical_df["sex"] == "Male"
        clinical_df["female"] = clinical_df["sex"] == "Female"
        clinical_df["LUAD"] = clinical_df["histology"] == "LUAD"
        clinical_df["LUSC"] = clinical_df["histology"] == "LUSC"

        # Rename identifier column so it matches other dataframes
        clinical_df = clinical_df.rename(columns={"record_id": "RED_ID"})

        # ------ ENDPOINTS
        # CONVERT DATES TO DATE-TIME FORMAT
        for date_var in [
            "date_surgery",
            "date_birth",
            "date_death",
            "date_relapse",
            "date_last_followup",
            "date_last_journalcheck",
        ]:
            clinical_df[date_var] = pd.to_datetime(clinical_df[date_var], format="%d/%m/%Y")  # fmt: skip
        # --- EVENT OBSERVED VARIABLES
        clinical_df["lungcancerdeath_observed"] = clinical_df["lungcancerdeath_observed"].fillna(0.0)   # fmt: skip
        # Composite endpoints
        clinical_df["death_or_relapse_observed"] = (clinical_df["relapse_observed"] == 1) | (clinical_df["death_observed"] == 1)   # fmt: skip
        clinical_df["lungcancerdeath_or_relapse_observed"] = (clinical_df["relapse_observed"] == 1) | (clinical_df["lungcancerdeath_observed"] == 1)   # fmt: skip
        clinical_df["time__surgery_to_last_followup"] = (clinical_df["date_last_followup"] - clinical_df["date_surgery"]).dt.days   # fmt: skip
        clinical_df["time__surgery_to_last_journalcheck"] = (clinical_df["date_last_journalcheck"] - clinical_df["date_surgery"]).dt.days   # fmt: skip
        clinical_df["time__surgery_to_death"] = (clinical_df["date_death"] - clinical_df["date_surgery"]).dt.days   # fmt: skip

        if self.use_EE_endpoint_updates:
            # --- CORRECT DSS
            # Initiate time-to-event with time from surgery to last journal check
            clinical_df["DSS"] = clinical_df["time__surgery_to_last_journalcheck"] / 30.4
            # Time-to-event should be based on last journal check in case of any death
            subset = clinical_df["death_observed"] == 1
            clinical_df.loc[subset, "DSS"] = clinical_df.loc[subset, "time__surgery_to_death"] / 30.4
            # Define DSS event (not yet differentiating between new vs same tumors)
            clinical_df["DSS_event"] = clinical_df["lungcancerdeath_observed"].astype(float)
            if self.count_only_death_from_initial_tumor_in_DSS:
                # ADDITIONAL TNM-I CRITERIA (DEATH MUST BE FROM SAME TUMOR)
                # Patient IDs are kept in the data dir, not in the code
                endpoint_updates = Paths.load_json(Paths().endpoint_updates_from_EE_json)
                # Set DSS variables to missing if unknown if the LC tumor was same as the initial tumor
                subset = clinical_df["RED_ID"].isin(endpoint_updates["red_id_unknown_if_died_from_initial_tumor"])
                clinical_df.loc[subset, ["DSS_event", "DSS"]] = np.nan
                # Set DSS event to "0" for patients we know have died from new LC tumor and not the initial tumor
                subset = clinical_df["RED_ID"].isin(endpoint_updates["red_id_secondary_LCdeath"])
                clinical_df.loc[subset, "DSS_event"] = 0.0
                clinical_df.loc[subset, "DSS"] = clinical_df[subset]["time__surgery_to_last_journalcheck"] / 30.4
            # --- CORRECT TTR
            # Update rows where patient have died from LC treatment WITHOUT recorded relapse
            subset = (clinical_df["DSS_event"] == 1) & (clinical_df["TTR_event"] == 0)
            clinical_df.loc[subset, "TTR_event"] = 1.0
            clinical_df.loc[subset, "TTR"] = clinical_df.loc[subset, "DSS"]

        # CALCULATE TIME ELAPSED FROM SURGERY TO TIME LAST MONITORED
        clinical_df["days_monitored_since_surgery"] = (clinical_df["date_last_followup"] - clinical_df["date_surgery"]).dt.days.astype(float)   # fmt: skip
        clinical_df["age_at_surgery"] = (clinical_df["date_surgery"] - clinical_df["date_birth"]).dt.days / 365.25
        # CONVERT SCALE OF TIME-TO-EVENT VARS FROM MONTHS TO DAYS
        for endpoint in ["TTR", "OS", "DFS", "DSS"]:
            # Convert event cols to float
            clinical_df[f"{endpoint}_event"] = clinical_df[f"{endpoint}_event"].astype(float)
            if clinical_df[endpoint].max() < 1000:
                clinical_df[endpoint] = clinical_df[endpoint] * 30.44

        # ADD TIME-TO-EVENT AND EVENT-OBSERVED COLUMNS (USED IN CPH AND KAPLAN MEIER FITS)
        clinical_df = self.add_eventobserved_and_timetoevent_variables_for_cph_fit(clinical_df)

        # DEFINING SOME MISC VARIBLES
        # I have confirmed that filtered and loose exposures do not overlap. There are 62 cases where ever-smokers lack pack-years
        clinical_df["packyears_total"] = clinical_df[["packyears_filtered", "packyears_loose"]].sum(axis=1, min_count=1)   # fmt: skip
        clinical_df["smoking_status"] = clinical_df["prev_vs_current_smoker"].fillna("never")   # fmt: skip
        clinical_df["neutrophile_to_lymphocyte_ratio"] = clinical_df["neutrophile_count"] / clinical_df["lymphocyte_count"]   # fmt: skip
        clinical_df["age_over_65"] = clinical_df["age_at_surgery"] >= 65
        clinical_df["has_genomic_data"] = clinical_df["RED_ID"].isin(Paths.load_json(Paths().red_id_genomic_data_json))   # fmt: skip

        # ADD VARIABLE: PATIENT RELAPSED WITHIN 3 YEARS
        clinical_df["relapsed_within_3years"] = (clinical_df["TTR"] < 3 * 365).astype(float)
        # Consider cases where patient has not relapsed yet: if monitored < 3 years, undefined; if monitored > 3 years, then no 3 year relapse is confirmed
        # No relapse, but patient has been monitored for <3 years, so variable is indeterminate
        clinical_df.loc[(clinical_df["days_monitored_since_surgery"] < 3 * 365) & (clinical_df["relapse_observed"] == "No"), "relapsed_within_3years"] = np.nan   # fmt: skip
        # No relapse and patint has been monitored for >=3years, so it is a definitive 'No'
        clinical_df.loc[(clinical_df["days_monitored_since_surgery"] >= 3 * 365) & (clinical_df["relapse_observed"] == "No"), "relapsed_within_3years"] = 0   # fmt: skip

        # CREATE COLUMN INDICATING WHICH ROWS BELONG TO HOLDOUT SET (DENMARK SUBSET)
        clinical_df["is_in_holdout_set"] = clinical_df["surg_center"].isin(["odense", "copenhagen"])
        # CREATE COUNTRY VARIABLE
        clinical_df["country"] = clinical_df["is_in_holdout_set"].replace({False: "norway", True: "denmark"})

        # DECLUTTER: REMOVE VARIABLES THAT WONT BE FREQUENTLY USED IN MODELLING
        clinical_df = self._drop_irrelevant_columns(clinical_df)

        return clinical_df

    # ---------- CPH FIT UTILITIES -----------------------------------------------------------------------------------------
    def add_eventobserved_and_timetoevent_variables_for_cph_fit(self, model_df) -> pd.DataFrame:
        """Assumes model_df has columns ``TTR``, ``date_last_followup`` and
        ``date_surgery``.

        Adds columns ``event_observed`` (allows model to know which TTR values
        are censored) and ``time_to_event`` (replaces nans with
        time-to-last-followup)."""
        model_df["event_observed"] = model_df[relabel.endpoint_vars_map[self.endpoint]["event_observed"]]
        model_df["time_to_event"] = model_df[relabel.endpoint_vars_map[self.endpoint]["time_to_event"]]
        return model_df

    # ---------- DATA UTILITIES -----------------------------------------------------------------------------------------
    def _decode_surgical_center(self, surgical_center_col: pd.Series):
        """Maps from encoded form (0-6) to string labels as they appear in
        clinical data."""
        surge_center_map = {
            "Odense": 6.0,
            "Radium-/Rikshospitalet Oslo": 4.0,
            "Rigshospitalet Copenhagen": 5.0,
            "St.Olav Trondheim": 2.0,
            "UNN Tromso": 1.0,
        }
        # INVERT DICT SO IT MAPS INDEX TO STRING
        index_to_string_map = {v: k for k, v in surge_center_map.items()}
        return surgical_center_col.replace(index_to_string_map)

    def _create_adjuvant_treatment_column(self, clinical_df: pd.DataFrame) -> pd.Series:
        """Creates column 'adjuvant_treatment' based on column called 'adjuv' with
        values chemo, radiotherapy, both, and other"""

        # DEFINE FUNCTION THAT AGGRAGATES THE ADJUVANT-TREATMENT VARIABLES FOR EACH ROW
        def encode_adjuvant_treatment(row):
            if row["What type of adjuvant treatment? (choice=Platinum based chemotherapy)"] == "Checked":
                return "chemo"
            if row["What type of adjuvant treatment? (choice=Postoperative radiotherapy (PORT))"] == "Checked":
                return "radiotherapy"
            if row["What type of adjuvant treatment? (choice=Both (sequential or concomitant))"] == "Checked":
                return "both"
            if row["What type of adjuvant treatment? (choice=other)"] == "Checked":
                return "other"
            else:
                return "none"

        # CREATE SINGLE ADJUVANT TREATMENT COLUMN
        clinical_df["adjuvant_treatment"] = clinical_df[[c for c in clinical_df.columns if "adjuv" in c]].apply(encode_adjuvant_treatment, axis=1)   # fmt: skip

        return clinical_df["adjuvant_treatment"]

    def _drop_irrelevant_columns(self, df, index_of_column_dropset=0):
        """For decluttering the dataset by dropping columns that are unlikely
        to be used or are irrelevant for the current analysis."""
        if self.cols_to_drop == "almost_all":
            return df.drop(columns=relabel.dropset0, errors="ignore")
        elif index_of_column_dropset == "many":
            return df.drop(columns=relabel.dropset1, errors="ignore")
        elif index_of_column_dropset == 2:
            return df.drop(columns=relabel.dropset2, errors="ignore")

    def get_attributes_and_methods(self):
        return [x for x in dir(self) if "__" not in x]

    @staticmethod
    def _identify_red_id_duplicates(df: pd.Series) -> pd.Series:
        """Identifies rows that ends with the suffix like _v1."""
        return df["RED_ID"].str.contains(r"_v\d+$", regex=True)
    
    # ---------- CONVENIENCE FUNCTIONS: LOADING RED ID SUBSETS -----------------------------------------------------------------------------------------
    @classmethod
    def get_driver_genes_df(cls):
        return pd.read_csv(Paths().intogen_compendium_cancer_driver_genes_tsv, sep="\t")

    @classmethod
    def get_driver_genes(cls, cancer_subset: Literal["any", "NSCLC", "LUAD"] = "any"):
        df = pd.read_csv(Paths().intogen_compendium_cancer_driver_genes_tsv, sep="\t")
        if cancer_subset == "any":
            return list(df["SYMBOL"].unique())
        elif cancer_subset == "NSCLC":
            return list(df.query("CANCER_TYPE in ['LUAD', 'LUSC','NSCLC']")["SYMBOL"].unique())
        elif cancer_subset == "LUAD":
            return list(df.query("CANCER_TYPE in ['LUAD']")["SYMBOL"].unique())

    @classmethod
    def get_genes_in_variant_data(cls):
        return Paths.load_json(Paths().genes_that_appear_in_norway_json)
    
    @classmethod
    def get_red_id_genomic_data(cls):
        return cls().get_clinical_data().query(f"RED_ID in {Paths.load_json(Paths().red_id_genomic_data_json)}")["RED_ID"].to_list()   # fmt: skip

    @classmethod
    def get_red_id_luad_data(cls):
        return cls().get_clinical_data().query(f"RED_ID in {Paths.load_json(Paths().red_id_luad_json)}")["RED_ID"].to_list()   # fmt: skip

    @classmethod
    def get_red_id_lusc_data(cls):
        return cls().get_clinical_data().query(f"RED_ID in {Paths.load_json(Paths().red_id_lusc_json)}")["RED_ID"].to_list()   # fmt: skip

    @classmethod
    def get_red_id_high_stages(cls):
        """RED_IDs with pathological stage II or III."""
        return cls().get_clinical_data().query(f"RED_ID in {Paths.load_json(Paths().red_id_high_stages_json)}")["RED_ID"].to_list()   # fmt: skip

    @classmethod
    def get_red_id_norway_data(cls):
        return cls().get_clinical_data().query(f"RED_ID in {Paths.load_json(Paths().red_id_norway_path)}")["RED_ID"].to_list()   # fmt: skip

    @classmethod
    def get_geneselection_pool_path(cls, eval_histo: Literal["LUAD", "LUSC", "NSCLC"]):
        """JSON path for a TCGA validation gene pool."""
        if eval_histo == "LUAD":
            return Paths().genepool_tcgaLUAD_validation_oncoselection_json
        if eval_histo == "LUSC":
            return Paths().genepool_tcgaLUSC_validation_oncoselection_json
        if eval_histo == "NSCLC":
            return Paths().genepool_tcgaNSCLC_validation_oncoselection_json
        raise ValueError(f"No gene selection pool path for eval_histo={eval_histo!r}")

    @classmethod
    def get_geneselection_pool(cls, eval_histo: Literal["LUAD", "LUSC", "NSCLC"] = "LUAD") -> list[str]:
        """TSO500 genes with at least 5 oncogenic mutations in the TCGA histology."""
        return Paths.load_json(cls.get_geneselection_pool_path(eval_histo))

    @classmethod
    def get_config_for_main_model(cls) -> dict:
        """Returns a dict with info specifying the main model that gets
        evaluated in TCGA LUAD."""
        config_dict = Paths.load_json(Paths().hyperpar_for_TCGAval__fluid_NSCLCtrain_LUADval_json)
        config_dict["path_varsearcher"] = Paths().varselection_dump_flexigrid_norway_dir
        return config_dict