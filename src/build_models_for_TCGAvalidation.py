"""Tune hyperparameters for the TCGA models in EXPERIMENT_DICTS.

Runs Bayesian optimization of n_selected and the L2 penalty for E1, E2, and E3,
including the 4–7% fixed VAF cutoffs. The last two runs repeat E1 and E2 with
gene selection nested inside cross-validation.
"""
# %% IMPORTS
from utils.prelude import *

from utils.display import *
from utils.prepare_data.variant_data import VariantData
from utils.build_models_for_TCGAvalidation import CrossValHyperParOptimizer
from utils.build_models_for_TCGAvalidation import BaysianOptimizerSettings
import os

# PYTHON SEED FOR REPRODUCABILITY
orange_print(f"PYTHON SEED: {os.environ.get("PYTHONHASHSEED")}")

# %% LOAD HEAVY VARIANTS DATA
orange_print("Loading heavy genomic data...")
gatk0_raw_df = VariantData().get_gatk()
gatk0_raw_df = VariantData().add_red_sampleid_col(gatk0_raw_df)

variants_filtered_df = VariantData().filter_variants(gatk0_raw_df, verbose=True)
variants_filtered_df["IS_ONCOGENIC"] = variants_filtered_df["ONCOGENICITY_SCORE"] >= 3

# %% ---------- LOAD DATAFACTORIES -----------------------------------------------------------------------------------------
# ---------- E1: TRAIN ON NSCLC (TTR) -----------------------------------------------------------------------------------------
e1_fluid_LUADeval_optimizer = CrossValHyperParOptimizer(experiment="E1: train on NSCLC", sub_experiment="LUADeval")   # fmt: skip
e1_fluid_LUADeval_optimizer.setup(variants_filtered_df)
e1_fluid_LUADeval_optimizer.run_bayesian_optimization(BaysianOptimizerSettings())
e1_fluid_LUADeval_optimizer.plot_histogram_of_hyperpar_training_history()
e1_fluid_LUADeval_optimizer.plot_roc_curve_for_best_hyperparams("LUAD", "TTR")
e1_fluid_LUADeval_optimizer.dump_results()

# STIFF PIPELINE (5% VAF CUTOFF)
e1_stiff_LUADeval_optimizer = CrossValHyperParOptimizer(experiment="E1: train on NSCLC", sub_experiment="LUADeval", stiff_cutoff=0.05)   # fmt: skip
e1_stiff_LUADeval_optimizer.setup(variants_filtered_df)
e1_stiff_LUADeval_optimizer.run_bayesian_optimization()
e1_stiff_LUADeval_optimizer.plot_histogram_of_hyperpar_training_history()
e1_stiff_LUADeval_optimizer.dump_results()

# STIFF PIPELINE (4% CUTOFF)
e1_stiff4pct_LUADeval_optimizer = CrossValHyperParOptimizer(experiment="E1: train on NSCLC", sub_experiment="LUADeval", stiff_cutoff=0.04)   # fmt: skip
e1_stiff4pct_LUADeval_optimizer.setup(variants_filtered_df)
e1_stiff4pct_LUADeval_optimizer.run_bayesian_optimization()
e1_stiff4pct_LUADeval_optimizer.plot_histogram_of_hyperpar_training_history()
e1_stiff4pct_LUADeval_optimizer.dump_results()

# STIFF PIPELINE (6% CUTOFF)
e1_stiff6pct_LUADeval_optimizer = CrossValHyperParOptimizer(experiment="E1: train on NSCLC", sub_experiment="LUADeval", stiff_cutoff=0.06)   # fmt: skip
e1_stiff6pct_LUADeval_optimizer.setup(variants_filtered_df)
e1_stiff6pct_LUADeval_optimizer.run_bayesian_optimization()
e1_stiff6pct_LUADeval_optimizer.plot_histogram_of_hyperpar_training_history()
e1_stiff6pct_LUADeval_optimizer.dump_results()

# STIFF PIPELINE (7% CUTOFF)
e1_stiff7pct_LUADeval_optimizer = CrossValHyperParOptimizer(experiment="E1: train on NSCLC", sub_experiment="LUADeval", stiff_cutoff=0.07)   # fmt: skip
e1_stiff7pct_LUADeval_optimizer.setup(variants_filtered_df)
e1_stiff7pct_LUADeval_optimizer.run_bayesian_optimization()
e1_stiff7pct_LUADeval_optimizer.plot_histogram_of_hyperpar_training_history()
e1_stiff7pct_LUADeval_optimizer.dump_results()

# FLUID PIPELINE (LUSC EVAL)
e1_fluid_LUSCeval_optimizer = CrossValHyperParOptimizer(experiment="E1: train on NSCLC", sub_experiment="LUSCeval")
e1_fluid_LUSCeval_optimizer.setup(variants_filtered_df)
e1_fluid_LUSCeval_optimizer.run_bayesian_optimization()
e1_fluid_LUSCeval_optimizer.plot_histogram_of_hyperpar_training_history()
e1_fluid_LUSCeval_optimizer.dump_results()

# ---------- E2: TRAIN ON LUAD & TTR -----------------------------------------------------------------------------------------
# --- TRAIN & EVAL ON LUAD
e2_fluid_LUADtrain_LUADeval_optimizer = CrossValHyperParOptimizer(
    experiment="E2: train on LUAD", sub_experiment="LUADeval"
)
e2_fluid_LUADtrain_LUADeval_optimizer.setup(variants_filtered_df)
e2_fluid_LUADtrain_LUADeval_optimizer.run_bayesian_optimization(BaysianOptimizerSettings())
e2_fluid_LUADtrain_LUADeval_optimizer.plot_histogram_of_hyperpar_training_history()
e2_fluid_LUADtrain_LUADeval_optimizer.dump_results()

# ---------- E3: TRAIN ON NSCLC & DSS -----------------------------------------------------------------------------------------
# --- EVAL ON LUAD
# FLUID PIPELINE
e3_fluid_DSS_LUADeval_optimizer = CrossValHyperParOptimizer(
    experiment="E3: train on NSCLC and DSS", sub_experiment="LUADeval"
)
e3_fluid_DSS_LUADeval_optimizer.setup(variants_filtered_df)
e3_fluid_DSS_LUADeval_optimizer.run_bayesian_optimization()
e3_fluid_DSS_LUADeval_optimizer.plot_histogram_of_hyperpar_training_history()
e3_fluid_DSS_LUADeval_optimizer.dump_results()

# ---------- NESTED GENE SELECTION (supplement: 64.6% vs 57.2%) --------------------------------
e1_fluid_LUADeval_nested_optimizer = CrossValHyperParOptimizer(experiment="E1: train on NSCLC", sub_experiment="LUADeval", freeze_selections=False)   # fmt: skip
e1_fluid_LUADeval_nested_optimizer.setup(variants_filtered_df)
e1_fluid_LUADeval_nested_optimizer.run_bayesian_optimization(BaysianOptimizerSettings())
e1_fluid_LUADeval_nested_optimizer.plot_histogram_of_hyperpar_training_history()

e2_fluid_LUADtrain_LUADeval_nested_optimizer = CrossValHyperParOptimizer(
    experiment="E2: train on LUAD", sub_experiment="LUADeval", freeze_selections=False
)
e2_fluid_LUADtrain_LUADeval_nested_optimizer.setup(variants_filtered_df)
e2_fluid_LUADtrain_LUADeval_nested_optimizer.run_bayesian_optimization(BaysianOptimizerSettings())
e2_fluid_LUADtrain_LUADeval_nested_optimizer.plot_histogram_of_hyperpar_training_history()
