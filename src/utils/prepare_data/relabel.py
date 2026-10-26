# ---------- UNIFY NAMES OF OVERLAPPING VARIABLES IN PREPARATION FOR MERGE -----------------------------------------------------------------------------------------
# --- CLEANUP 2
# Maps names in master_FU1224_EEfilter_IP.csv to corresponding variables in encoded df
var_name_map1 = {
    # PARTICIPANT ID
    "RED_ID": "record_id",
    # DATES, TEMPPORAL VARIABLES, AND ENDPOINTS
    "Date of birth": "date_birth",  # Drop
    "Date of death": "date_death",  # Drop
    "Date of recurrence": "date_relapse",  # Drop
    "Date of surgery": "date_surgery",  # Drop
    "Date of histological verification of recurrence": "date_of_histological_relapse",  # Drop
    "date of last follow-up": "date_last_followup",  # Drop
    "Date of most recent journal assessment": "date_last_journalcheck",  # Drop
    "Dead within 30 days of first operation?": "postoperative_death",
    "Death from lung cancer": "lungcancerdeath_observed",  # Drop
    "Which type of recurrence?": "relapse_loco",  # Drop
    # MISC
    "Ever smoker?": "eversmoker",
    "Is the patient dead?": "death_observed",  # Drop
    "Name of surgical center": "surg_center",
    "PD-L1 expression in %": "pdl1expression",
    "PD-L1 expression assessed?": "pdl1_expression_assessed",
    "Which assay of PDL1 expression was used?": "pdl1assay",
    "Previous or current smoker?": "prev_vs_current_smoker",
    "Recurrence?": "relapse_observed",  # Drop
    "Sex": "sex",
    "Surgical procedure": "surgical_procedure",
    "WHO performance status": "who_ps",
    # --- PATHOLOGICAL
    # TNM STAGE
    "Pathological N stage": "pnstage",
    "Pathological T stage (8th edition)": "ptstage",
    "Pathological stage": "pstage",
    # Substages
    "pstage_granular": "pstage_granular",
    # Other
    "Surgical margins of primary tumor": "surgical_margins_of_tumor",
    # --- HISTOLOGY
    # MAIN TYPE
    "Histology": "histology",
    # LUAD SUBTYPES
    "Subtype adenocarcinoma (choice=acin\u00e6rt)": "luad_acinar",
    "Subtype adenocarcinoma (choice=lepidisk (invasive og minimal invasive))": "luad_lepidic",
    "Subtype adenocarcinoma (choice=micropapillary)": "luad_micropapillary",
    "Subtype adenocarcinoma (choice=mucinous)": "luad_mucinous",
    "Subtype adenocarcinoma (choice=not reported)": "luad_not_reported",
    "Subtype adenocarcinoma (choice=papillary)": "luad_papillary",
    "Subtype adenocarcinoma (choice=solid)": "luad_solid",
    "Subtype adenocarcinoma (choice=others (fetal, colloid, intestinal))": "luad_other",
    # LUSC SUBTYPES
    "Subtype squamous cell carcinoma": "lusc_subtype",
    # --- ADJUVANT TREATMENT
    "Did patient receive adjuvant treatment?": "adjuv_treat",  # Drop
    # TREATMENT TYPE
    "What type of adjuvant treatment? (choice=Platinum based chemotherapy)": "adjuvant_treatment_chemotherapy",
    "What type of adjuvant treatment? (choice=Postoperative radiotherapy (PORT))": "adjuvant_treatment_radiotherapy",
    "What type of adjuvant treatment? (choice=Both (sequential or concomitant))": "adjuvant_treatment_both",
    "What type of adjuvant treatment? (choice=other)": "adjuvant_treatment_other",
    # POST-RECURRENCE-TREATMENT VARIABLES
    "Following recurrence, which treatment has the patient received up to date of last follow-up? (choice=Radical surgery)": "postrelapse_treatment_surgery",
    "Following recurrence, which treatment has the patient received up to date of last follow-up? (choice=Stereotactic RT)": "postrelapse_treatment_stereotactic_rt",
    "Following recurrence, which treatment has the patient received up to date of last follow-up? (choice=Conventional radical RT)": "postrelapse_treatment_conventional_rt",
    "Following recurrence, which treatment has the patient received up to date of last follow-up? (choice=Chemotherapy)": "postrelapse_treatment_chemotherapy",
    "Following recurrence, which treatment has the patient received up to date of last follow-up? (choice=TKI (Small molecular inhibitors))": "postrelapse_treatment_tki",
    "Following recurrence, which treatment has the patient received up to date of last follow-up? (choice=Checkpoint inhibitors)": "postrelapse_treatment_checkpoint_inhibitors",
    "Following recurrence, which treatment has the patient received up to date of last follow-up? (choice=Palliative RT)": "postrelapse_treatment_palliative_rt",
    "Following recurrence, which treatment has the patient received up to date of last follow-up? (choice=Other)": "postrelapse_treatment_other",
    "Following recurrence, which treatment has the patient received up to date of last follow-up? (choice=No treatment)": "postrelapse_treatment_none",
    # POST RELAPSE METASTASIS SITE
    "Sites(s) of metastasis at time of recurrence (choice=Bone)": "postrelapse_met_site_bone",
    "Sites(s) of metastasis at time of recurrence (choice=Brain)": "postrelapse_met_site_brain",
    "Sites(s) of metastasis at time of recurrence (choice=Liver)": "postrelapse_met_site_liver",
    "Sites(s) of metastasis at time of recurrence (choice=Lung)": "postrelapse_met_site_lung",
    "Sites(s) of metastasis at time of recurrence (choice=Adrenal)": "postrelapse_met_site_adrenal",
    "Sites(s) of metastasis at time of recurrence (choice=Other)": "postrelapse_met_site_other",
    "Date of histological verification of recurrence": "date_of_histological_relapse",
}
var_name_map2 = {
    "luad_subtype___1": "luad_acinar",
    "luad_subtype___2": "luad_micropapillary",
    "luad_subtype___3": "luad_papillary",
    "luad_subtype___4": "luad_solid",
    "luad_subtype___5": "luad_lepidic",
    "luad_subtype___6": "luad_mucinous",
    "luad_subtype___7": "luad_other",
    "luad_subtype___8": "luad_not_reported",
    # ADJUVANT TREATMENT
    "type_adjuv_treat___1": "adjuvant_treatment_chemotherapy",
    "type_adjuv_treat___2": "adjuvant_treatment_radiotherapy",
    "type_adjuv_treat___3": "adjuvant_treatment_both",
    "type_adjuv_treat___4": "adjuvant_treatment_other",
    # ENDPOINTS
    "death_yesno": "death_observed",
    "relapsed": "relapse_observed",
    "lungccancerdeath": "lungcancerdeath_observed",
    # DATES
    "birthdate": "date_birth",
    "death_date": "date_death",
    "relapse_date": "date_relapse",
    "date_last_journalcheck": "date_last_journalcheck",
    "surgerydate": "date_surgery",
    "date_last_fu": "date_last_followup",
    "histological_dateofrecur": "date_of_histological_relapse",
    # OTHER
    "surg_proc": "surgical_procedure",
}
# ---------------------------------------------------------------------------------------------------
var_name_map3_postmerge = {
    "Number of years the patient has been smoking": "n_years_smoking",
    "Age at surgery": "age_at_surgery",
    "Loose tobacco or filtered cigarettes": "smoking_loose_or_filtered",
    "Number of loose packs per week?": "packs_per_week",
    "Cigarette pack years (loose packs)": "packyears_loose",
    "Cigarette pack years (filtered cigs)": "packyears_filtered",
    "Weight loss (weight loss of >10% from pre-illness weight during last 6 months)": "pre_illness_weightloss",
    "History of autoimmune disease?": "auto_immune_history",
    "Has the patient been given the diagnosis of COPD (chronic obstructive pulmonary disease)?": "copd",
    "Is 'tumor spread thorugh airspaces' (STAS) identified?": "tumor_spread_through_airways",
    "Vascular infiltration": "vascular_infiltration",
    "Tumor Size (in mm)": "tumor_size",
    "pstage": "pstage",
    "date of last follow-up": "date_last_followup",
    # --- BLOOD CELL VARIABLES
    "Total leukocyte count (preoperative blood count #)": "leukocyte_count",
    "Neutrophiles (preoperative blood count #)": "neutrophile_count",
    "Lymphocytes (preoperative blood count #)": "lymphocyte_count",
    "Monocytes (preoperative blood count#)": "monocyte_count",
    "Basofile (preoperative blood count#)": "basofile_count",
    "Eosinophile (preoperative blood count#)": "eosinophile_count",
    "Total leukocyte count (preoperative blood count #)": "leukocyte_count",  # General marker of immune response; high levels may indicate systemic inflammation, which can influence cancer progression.
    "Neutrophiles (preoperative blood count #)": "neutrophile_count",  # First-line defense against infections; high neutrophil counts (relative to lymphocytes) are associated with poor lung cancer prognosis.
    "Lymphocytes (preoperative blood count #)": "lymphocyte_count",  # Include T cells, B cells, and NK cells, which are crucial for anti-tumor immunity; high or low counts can indicate immune system activity.
    "Monocytes (preoperative blood count #)": "monocyte_count",  # Can differentiate into macrophages that either attack tumors or promote their growth (tumor-associated macrophages).
    "Basofile (preoperative blood count #)": "basofile_count",  # Rare immune cells that release histamine; may be linked to tumor-promoting inflammation.
    "Eosinophile (preoperative blood count #)": "eosinophile_count",
}

endpoint_vars_map = {
    "TTR": {"event_observed": "TTR_event", "time_to_event": "TTR"},
    "OS": {"event_observed": "OS_event", "time_to_event": "OS"},
    "DFS": {"event_observed": "DFS_event", "time_to_event": "DFS"},
    "DSS": {"event_observed": "DSS_event", "time_to_event": "DSS"},
}
# --- CLEANUP 3
# Maps to new values after renaming variables in cleanup 2
value_map1_postmerge = {
    "pnstage": {2: "pN0", 3: "pN1", 4: "pN2", 1: "pNx"},
    "ptstage": {1: "T1", 2: "T2", 3: "T3", 4: "T4"},
    "surg_center": {
        1: "tromso",
        2: "trondheim",
        4: "oslo",
        5: "copenhagen",
        6: "odense",
    },
    "vascular_infiltration": {
        "not reported": "not_reported",
        "no lymphovascular infiltration observed": "no",
        "lymphovascular infiltration observed": "yes",
    },
    "lusc_subtype": {
        "Keratinizing (including Keratin pearls and cytoplasmic keratinization)": "keratinizing",
        "Non-keratinizing": "non_keratinizing",
        "Not reported": "not_reported",
        "Basaloid": "basaloid",
    },
    "prev_vs_current_smoker": {1.0: "previous", 2.0: "current"},
    "surgical_procedure": {1: "Lobectomy", 2: "Pulmectomy", 3: "Wedge"},
    "surgical_margins_of_tumor": {
        "negative (clear)": "negative",
        "unsure, probably negative (clear)": "negative_probably",
        "positive": "positive",
        "unsure, probably positive": "positive_probably",
        "unknown": "unknown",
    },
    "sex": {1: "Male", 2: "Female"},
}
value_map2 = {
    "histology_coarse": {
        1: "LUAD",
        2: "LUSC",
        3: "other",
        4: "other",
        5: "other",
    },
    # # TNM STAGE
    # "pstage3_subgroup": {"pStage 1": 1.0, "pStage 2": 2.0, "pStage 3": 3.0},
    # "sex": {"Female": 2, "Male": 1},
    # "surgical_procedure": {"Lobectomy": 1, "Pulmectomy": 2, "Wedge": 3},
    # "surgical_procedure": {
    #     "Lobectomy": "lobectomy",
    #     "Wedge": "wedge",
    #     "Pulmectomy": "pulmectomy",
    # },
    "tumor_spread_through_airways": {
        "not reported": "not_confirmed",
        "no": "not_confirmed",
        "yes": "confirmed",
    },
    "Which type of recurrence?": {
        "local/locoregional recurrence": 1.0,
        "oligometastasis": 2.0,
        "systemic (metastasis)": 3.0,
    },
}

# ---------- DICTIONARIES FOR UNDERSTANDING ENCODING (NOT USED IN CODE) -----------------------------------------------------------------------------------------
WHO_map = {
    "0 (Fully active, without restriction)": "0",
    "1 (Restricted,but ambulatory,able to carry out light work)": "1",
    "2 (Ambulatory,but unable to carry out any work . Up and about more than 50% of waking hours)": "2",
    "3 (Capable of only limited selfcare, confined to bed or chair more than 50% of waking hours)": "3",
}
# MAPS VALUES IN DF1 (READABLE) TO VALUES IN DF2 (NUMERICALLY ENCODED)
value_key = {
    "eversmoker": {"No": 0, "Yes": 1},
    "histology": {
        "AD (adenocarcinoma)": 1,
        "ADSCC (Adenosquamous carcinoma)": 3,
        "LCC (large cell carcinoma)": 4,
        "Other (e.g. sarcomatoid, salivary gland like tumors, lymphoepithelial carcinoma, NUT-carcinoma, very poorly differentiated)": 5,
        "SCC (squamous cell carcinoma)": 2,
    },
    "prev_vs_current_smoker": {"Current smoker": 2.0, "Previous smoker": 1.0},
    "surgical_procedure": {"Lobectomy": 1, "Pulmectomy": 2, "Wedge": 3},
}

# COLUMNS TO DROP AFTER CLEANUP
dropset0 = [
    "Which specific OUS hospital has FU information about the patient",
    "Follow-up hospital",
    "Date of inclusion",
    "exclusion",
    # CIGARETTE VARIABLES
    "Specify, number of loose packs or loose cigs (rolled cigarettes/'rullings')",
    "Number of loose cigs per day (rolled cigarettes/'rullings')",
    "'Number of cigarettes per day?'",
    "Cigarette pack years (loose cigs)",
    "packyears_filtered",
    "packyears_loose",
    "Cigarette pack years  Whole numbers only (no decimals). See calculation based on type of cigarettes above",
    "Cigarette pack years  Whole numbers only (no decimals). See calculation based on type of cigarettes above.1",
    # OTHER
    "luad_not_reported",
    "Follow-up hospital",
    "Which specific OUS hospital has FU information about the patient",
    "auto_immune_history",
    "Which kind of autoimmune disease?",
    "In freetext, what is the name of the autoimmune disease?",
    "whichotheradjuvtreat",
    "Comments on adjuvant treatment?",
    "pathological Stage 2 subgroup",  # redundant: can be derived from pathological_substage
    "pathological Stage 3 subgroup",
    # HISTOLOGY VARIABLES
    "luad_subtype___1",
    "luad_subtype___2",
    "luad_subtype___3",
    "luad_subtype___4",
    "luad_subtype___5",
    "luad_subtype___6",
    "luad_subtype___7",
    "luad_subtype___8",
    # POST-RECURRENCE-TREATMENT VARIABLES
    "postrelapse_treatment_surgery",
    "postrelapse_treatment_stereotactic_rt",
    "postrelapse_treatment_conventional_rt",
    "postrelapse_treatment_chemotherapy",
    "postrelapse_treatment_tki",
    "postrelapse_treatment_checkpoint_inhibitors",
    "postrelapse_treatment_palliative_rt",
    "postrelapse_treatment_other",
    "postrelapse_treatment_none",
    "Comments on recurrence?",
    # POST RELAPSE METASTASIS SITE
    "postrelapse_met_site_bone",
    "postrelapse_met_site_brain",
    "postrelapse_met_site_liver",
    "postrelapse_met_site_lung",
    "postrelapse_met_site_adrenal",
    "postrelapse_met_site_other",
    # OTHER
    "date_of_histological_relapse",
    "Comments on recurrence treatment",
    "eversmoker",
    "prev_vs_current_smoker",
]
dropset1 = [
    "packs_per_week",
    "smoking_loose_or_filtered",
    "Specify, number of loose packs or loose cigs (rolled cigarettes/'rullings')",
    "packyears_loose",
    "Number of loose cigs per day (rolled cigarettes/'rullings')",
    "Number of cigarettes per day?",
    "Cigarette pack years (loose cigs)",
    "Cigarette pack years  Whole numbers only (no decimals). See calculation based on type of cigarettes above",
    "Cigarette pack years  Whole numbers only (no decimals). See calculation based on type of cigarettes above.1",
    "Which kind of autoimmune disease?",
    "In freetext, what is the name of the autoimmune disease?",
    "copd",
    "surgical_procedure",
    "Which assay of PDL1 expression was used?",
    "Comments on recurrence?",
]
dropset2 = [
    "Num_Positive_mm2_tumor",
    "AOO_IP",
    "total_mm2",
    "new_IP",
    "Num_Detections_tumor",
    "Area_um_tumor",
    "total_perc",
    "Num_Negative_stroma",
    "Num_Negative_tumor",
    "Num_Positive_tumor",
    "Num_Positive_stroma",
    "Num_Positive_mm2_stroma",
    "Positive_perc_stroma",
]
