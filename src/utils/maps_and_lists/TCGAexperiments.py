"""Settings for the TCGA models reported in the paper.

E1 trains on NSCLC with TTR and is evaluated on LUAD, NSCLC, and LUSC.
E2 trains on LUAD with TTR and is evaluated on LUAD.
E3 trains on NSCLC with DSS and is evaluated on LUAD.
The 4–7% fixed VAF cutoffs are stored on the E1 LUAD evaluation.

fluidGS: AdaPipe
stiffGS: benchmark approach to VAF modeling
"""

from utils.paths import Paths
from utils.prepare_data.prepare_data import DataBuilder

EXPERIMENT_DICTS = {
    "E1: train on NSCLC": {
        # SHARED SETTINGS
        "crossval_path": Paths().cv_splits_norway_path,
        "path_to_varsearcher_fluidGS": Paths().varselection_dump_flexigrid_norway_dir,
        "path_to_varsearcher_stiffGS": Paths().varselection_dump_fixedgrid_norway_dir,
        "path_to_CVresults__fluidGS": Paths().crossval_varselection_dump_flexigrid_norway_dir,
        "path_to_CVresults__stiffGS": Paths().crossval_varselection_dump_fixedgrid_norway_dir,
        "endpoint": "TTR",
        "train_histology": "NSCLC",
        # --- SUBEXPERIMENTS
        "LUADeval": {
            "eval_histology": "LUAD",
            "hyperparam_dump_path__fluidGS": Paths().hyperpar_for_TCGAval__fluid_NSCLCtrain_LUADval_json,
            "hyperparam_dump_path__stiffGS": Paths().hyperpar_for_TCGAval__stiff_NSCLCtrain_LUADval_json,
            "hyperparam_dump_path__stiffGS_lookup": {
                0.04: Paths().hyperpar_for_TCGAval__stiff4pct_NSCLCtrain_LUADval_json,
                0.06: Paths().hyperpar_for_TCGAval__stiff6pct_NSCLCtrain_LUADval_json,
                0.07: Paths().hyperpar_for_TCGAval__stiff7pct_NSCLCtrain_LUADval_json,
            },
            "gene_selection_pool_path": DataBuilder.get_geneselection_pool_path("LUAD"),
        },
        "NSCLCeval": {
            "eval_histology": "NSCLC",
            "hyperparam_dump_path__fluidGS": Paths().hyperpar_for_TCGAval__fluid_NSCLCtrain_NSCLCval_json,
            "hyperparam_dump_path__stiffGS": Paths().hyperpar_for_TCGAval__stiff_NSCLCtrain_NSCLCval_json,
            "gene_selection_pool_path": DataBuilder.get_geneselection_pool_path("LUAD"),
        },
        "LUSCeval": {
            "eval_histology": "LUSC",
            "hyperparam_dump_path__fluidGS": Paths().hyperpar_for_TCGAval__fluid_NSCLCtrain_LUSCval_json,
            "hyperparam_dump_path__stiffGS": Paths().hyperpar_for_TCGAval__stiff_NSCLCtrain_LUSCval_json,
            "gene_selection_pool_path": DataBuilder.get_geneselection_pool_path("LUSC"),
        },
    },
    # ---------------------------------------------------------------------------------------------------
    "E2: train on LUAD": {
        # SHARED SETTINGS
        "crossval_path": Paths().cv_splits_norway_luad_path,
        "path_to_varsearcher_fluidGS": Paths().varselection_dump_LUAD_norway_flexigrid_dir,
        "path_to_varsearcher_stiffGS": Paths().varselection_dump_LUAD_norway_fixedgrid_dir,
        "path_to_CVresults__fluidGS": Paths().crossval_varselection_dump_flexigrid_luad_norway_dir,
        "path_to_CVresults__stiffGS": Paths().crossval_varselection_dump_fixedgrid_luad_norway_dir,
        "endpoint": "TTR",
        "train_histology": "LUAD",
        # --- SUBEXPERIMENTS
        "LUADeval": {
            "eval_histology": "LUAD",
            "hyperparam_dump_path__fluidGS": Paths().hyperpar_for_TCGAval__fluid_LUADtrain_LUADval_json,
            "hyperparam_dump_path__stiffGS": Paths().hyperpar_for_TCGAval__stiff_LUADtrain_LUADval_json,
            "gene_selection_pool_path": DataBuilder.get_geneselection_pool_path("LUAD"),
        },
    },
    # ---------------------------------------------------------------------------------------------------
    "E3: train on NSCLC and DSS": {
        # SHARED SETTINGS
        "crossval_path": Paths().cv_splits_norway_path,
        "path_to_varsearcher_fluidGS": Paths().varselection_dump_norway_DSS_dir,
        "path_to_varsearcher_stiffGS": Paths().varselection_dump_norway_DSS_fixedgrid_dir,
        "path_to_CVresults__fluidGS": Paths().crossval_varselection_dump_flexigrid_norway_DSS_dir,
        "path_to_CVresults__stiffGS": Paths().crossval_varselection_dump_fixedgrid_norway_DSS_dir,
        "endpoint": "DSS",
        "train_histology": "NSCLC",
        # --- SUBEXPERIMENTS
        "LUADeval": {
            "eval_histology": "LUAD",
            "hyperparam_dump_path__fluidGS": Paths().hyperpar_for_TCGAval__fluid_NSCLCtrain_LUADval_DSS_json,
            "hyperparam_dump_path__stiffGS": Paths().hyperpar_for_TCGAval__stiff_NSCLCtrain_LUADval_DSS_json,
            "gene_selection_pool_path": DataBuilder.get_geneselection_pool_path("LUAD"),
        },
    },
}
