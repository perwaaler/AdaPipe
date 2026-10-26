# %% IMPORTS
# anamaster: Re-run all Paper 1 discovery-cohort cross-validation analyses in one pass.
# Maps to Notion section "Cross validation analysis in the discovery cohort".
from utils.prelude import *

from utils.prepare_data.variant_data import VariantData
from utils.datafactory.datafactory_analysis import reset_defaults
from utils.datafactory.datafactory_analysis import build_paper1_discovery_fluid_datafactory
from utils.datafactory.datafactory_analysis import build_paper1_discovery_stiff_datafactories
from ana.ana2__discovery_cohort_crossvalidation import ana5__Endpoint_Comparison
from ana.ana2__discovery_cohort_crossvalidation import ana1b__CrossValComp_FluidVsStiff_trainOnNSCLC_testOnLUAD
from ana.ana2__discovery_cohort_crossvalidation import ana3__tune_ridge_L2_penalty
from ana.ana2__discovery_cohort_crossvalidation import ana1__FluidVsStiff_barplots
from ana.ana2__discovery_cohort_crossvalidation import ana1a__FluidVsStiff_barplotsV2__drivers_only
from ana.ana2__discovery_cohort_crossvalidation import ana4__AdaPipe_CrossVal_GeneSelection_heatmap
from ana.ana2__discovery_cohort_crossvalidation import ana2__Ranking_Features
from ana.ana2__discovery_cohort_crossvalidation import ana2a__OncofilterStrategies
from ana.ana2__discovery_cohort_crossvalidation import ana2b__Robustness_FluidEnsemble_vs_StiffCutoffs

# %% LOAD VARIANT DATA
print_header1("Loading heavy genomic data")
gatk0_raw_df = VariantData().get_gatk()
gatk0_raw_df = VariantData().add_red_sampleid_col(gatk0_raw_df)
variants_df = VariantData().filter_variants(gatk0_raw_df)
variants_df["IS_ONCOGENIC"] = variants_df["ONCOGENICITY_SCORE"] >= 3

# %% BUILD DATAFACTORIES (once — varsearcher load is the expensive step)
print_header1("Load datafactories with default settings")
stiff_datafactories = build_paper1_discovery_stiff_datafactories(variants_df)
datafactory_fluid = build_paper1_discovery_fluid_datafactory(variants_df)

# %% --- L2 PENALTY SENSITIVITY (Fig S6). Reported analyses use λ = 0.1.
print_header1("L2 penalty sensitivity (TTR)")
ana3__tune_ridge_L2_penalty(stiff_datafactories, datafactory_fluid)

# %% --- PIPELINE COMPARISON: ADAPIPE VS STIFF BENCHMARKS (Fig 2B/C, S4)
print_header1("Pipeline comparison: AdaPipe vs fixed-cutoff benchmarks")
reset_defaults(datafactory_fluid, stiff_datafactories)
ana1__FluidVsStiff_barplots(stiff_datafactories, datafactory_fluid)

# %% --- GENE SELECTION HEATMAP (Fig S6)
print_header1("Cross-val gene selection heatmap (k=20)")
reset_defaults(datafactory_fluid)
ana4__AdaPipe_CrossVal_GeneSelection_heatmap(datafactory_fluid)

# %% --- ENDPOINT TRAINING COMPARISON (Fig S7)
print_header1("Endpoint training comparison (TTR vs DSS vs OS)")
reset_defaults(datafactory_fluid, stiff_datafactories)
ana5__Endpoint_Comparison(variants_df, stiff_datafactories)

# %% --- HISTOLOGY: TRAIN NSCLC, EVAL ON LUAD / LUSC
print_header1("Histology (train on NSCLC, test on LUAD / LUSC)")
reset_defaults(datafactory_fluid, stiff_datafactories)
ana1b__CrossValComp_FluidVsStiff_trainOnNSCLC_testOnLUAD(stiff_datafactories, datafactory_fluid)

# %% --- ABLATION: FEATURIZATION FOR GENE SELECTION (Fig S8 A)
reset_defaults(datafactory_fluid)
ana2__Ranking_Features(datafactory_fluid)

# %% --- ROBUSTNESS: HARD+SOFT MODEL-FEATURE ENSEMBLE VS STIFF CUTOFFS (supplement)
reset_defaults(datafactory_fluid, stiff_datafactories)
ana2b__Robustness_FluidEnsemble_vs_StiffCutoffs(stiff_datafactories, datafactory_fluid)

# %% --- ABLATION: ONCOGENICITY FILTER STRATEGIES (Fig S8 B)
print_header1("Ablation: oncogenicity filter strategies")
reset_defaults(datafactory_fluid)
ana2a__OncofilterStrategies(datafactory_fluid)

# %% --- DRIVER-GENE POOL RESTRICTION (Fig S9; Notion subsection under gene selection)
print_header1("Driver-gene pool restriction CV")
reset_defaults(datafactory_fluid, stiff_datafactories)
ana1a__FluidVsStiff_barplotsV2__drivers_only(stiff_datafactories, datafactory_fluid)

green_print("Discovery-cohort CV master run complete.")
