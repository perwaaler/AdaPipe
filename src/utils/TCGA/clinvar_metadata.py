cols_to_drop = [
    "Tumor Site",  # All LUAD
    "Time between excision and freezing",  # All nan
    "Time between clamping and freezing",  # All nan
    "Stage Other",  # All nan
    "Somatic Status",  # All 'matched'
    "Specimen Freezing Means",  # All nan
    "Specimen Current Weight",  # All nan
    "Sample type id",  # All 1
    "Sample Type",  # All 'primary'
    "Number of Samples Per Patient",  # All '1'
    "Last Alive Less Initial Pathologic Diagnosis Date Calculated Day Value",  # All '0' or nan
    "Project code",  # Allmost all Nan
    "Primary Other Site of Disease Name",  # Nearly all nan
    "Lymph node location positive pathology name",  # All nan
    "Adjuvant Postoperative Pharmaceutical Therapy Administered Indicator",  # All nan
    "Other Patient ID",  # Not needed
    "Other Sample ID",  # Not needed
    "Specimen Collection Method",  # All nan
    "Oncotree Code",  # Only 'LUAD'
    "Oct embedded",  # Almost all nan
    "Number of lymphnodes positive by ihc",  # All nan
    "New Neoplasm Event Post Initial Therapy Indicator",  # ANear all nan
    "Mutation Type",  # Near all nan
    "First Pathologic Diagnosis Biospecimen Acquisition Method Type.1",  # All nan
    "First Pathologic Diagnosis Biospecimen Acquisition Other Method Type",  # All nan
    "First Pathologic Diagnosis Biospecimen Acquisition Method Type",  # All nan
    "Longest Dimension",  # Mostly irrelevant
    "Location lung parenchyma",  # Mostly nan
    "Primary Tumor Laterality",  # All nan
    "Karnofsky Performance Score",  # Mostly nan
    "Informed consent verified",  # Irrelevant
    "HIV Status",  # All nan
    "Prior infectious disease",  # All nan
    "Prior immunologic disease other",  # All nan
    "Prior immunologic disease type",  # All nan
    "Fev1 percent ref prebroncholiator",  # Irrelevant, and mostly nan
    "Fev1 percent ref postbroncholiator",  # Irrelevant, and mostly nan
    "Fev1 fvc ratio prebroncholiator",  # Irrelevant, and mostly nan
    "Fev1 fvc ratio postbroncholiator",  # Irrelevant, and mostly nan
    "Lymphomatous Extranodal Site Involvement Indicator",  # All nan
    "Disease code",  # Nearly all nan
    "days_to_tumor_progression",  # All nan
    "Days to Sample Procurement",  # All nan
    "days_to_patient_progression_free",  # All nan
    "Days to Sample Collection.",  # Mostly nan
    "Carbon monoxide diffusion dlco",  # Mostly nan
    "Cancer Type Detailed",  # All LUAD
    "Cancer Type",  # All NSCLC
    "ALK Translocation Variant",  # All nan
    "ALK Translocation Status",  # Mostly nan
    "ALK Analysis Type",  # Mostly nan
    "American Joint Committee on Cancer Lymph Node Stage Code",  # All nan
    "American Joint Committee on Cancer Lymph Node Stage Code.1",  # All nan
    "Neoplasm American Joint Committee on Cancer Clinical Distant Metastasis M Stage",  # All nan
    "Neoplasm Disease Stage American Joint Committee on Cancer Code.1",  # All nan
    "Study ID",  # All luad_tcga
]

# --- LUAD/LUSC COLMN UNIFICATION MAPS
# Maps names in TCGA_LUSC_Clinical_Patient.csv to the corresponding names in luad_tcga_clinical_data.tsv
map1_LUSC_clinicalPatient = {
    "bcr_patient_barcode": "Patient ID",
    "age_at_initial_pathologic_diagnosis": "Diagnosis Age",
    "eastern_cancer_oncology_group": "who_ps",
    "stage_event_pathologic_stage": "Neoplasm Disease Stage American Joint Committee on Cancer Code",
}

stage_map = {
    "Stage I": "1",
    "Stage IA": "1",
    "Stage IB": "1",
    "Stage II": "2",
    "Stage IIA": "2",
    "Stage IIB": "2",
    "Stage IIIA": "3",
    "Stage IIIB": "3",
    "Stage IV": "4",
}


stage_map_granular = {
    "Stage I": 1,
    "Stage IA": 2,
    "Stage IB": 3,
    "Stage II": 4,
    "Stage IIA": 5,
    "Stage IIB": 6,
    "Stage IIIA": 7,
    "Stage IIIB": 8,
}
