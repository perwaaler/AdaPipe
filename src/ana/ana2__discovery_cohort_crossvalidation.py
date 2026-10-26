"""Discovery-cohort cross-validation analyses called by the Paper 1 orchestrator."""

import copy

import numpy as np

import utils.datafactory.design_matrixes as design_matrixes
from utils.datafactory.data_factory import DataFactory
from utils.datafactory.datafactory_analysis import CaptionConfig
from utils.datafactory.datafactory_analysis import DataFactoryUpdater
from utils.datafactory.datafactory_analysis import LABEL_BENCHMARK_ENSEMBLE
from utils.datafactory.datafactory_analysis import ListOfDataFactories
from utils.datafactory.datafactory_analysis import N_GENES_SELECTED
from utils.datafactory.datafactory_analysis import N_GENES_SELECTED_LINEPLOTS
from utils.datafactory.datafactory_analysis import PipelineComparisonAnalysis
from utils.datafactory.datafactory_analysis import build_paper1_discovery_fluid_datafactory
from utils.datafactory.gene_selectors import TopKBestGenes
from utils.models.models import CPHModel
from utils.paths import Paths
from utils.plotting import sns_colors
from utils.prelude import *
from utils.prepare_data.prepare_data import DataBuilder
from utils.split_data import CrossValidator
from utils.variablesearch.gridpoint_choosers import MinPvalOverEligible


def ana3__tune_ridge_L2_penalty(
    stiff_datafactories: ListOfDataFactories,
    datafactory_fluid: DataFactory,
):
    """Figure S6. Cross-validation at several L2 penalties. The paper uses λ = 0.1."""
    penalty_values = [0.05, 0.1, 0.25, 0.5, 1.0, 10]
    ensemble_range = [10, 20]
    for penalizer in penalty_values:
        white_print(f"\n\n\nPENALIZER: {penalizer}")
        pipelines = ListOfDataFactories({"AdaPipe": datafactory_fluid}) + stiff_datafactories
        analysis = PipelineComparisonAnalysis(
            pipelines=pipelines,
            n_genes_selected_list=np.arange(5, 50, 2),
            train_endpoint="TTR",
            penalizer=penalizer,
            verbose=0,
        )
        analysis.generate_results()
        analysis.plot_barplot_for_full_ensemble_models(ensemble_range)
        analysis.plot_auc_for_different_n_select()
        text = f"L2-penalty: {penalizer:g}; N selected range: {ensemble_range}"
        ax = analysis.axes_lineplots["AUC (2 yr)"]
        ax.text(0.0, -0.15, text, ha="left", va="top", transform=ax.transAxes, alpha=0.5)
        ax = analysis.axes_barplot_full_ensemble[1]
        ax.text(0.0, -0.15, text, ha="left", va="top", transform=ax.transAxes, alpha=0.5)
        Paths.dump_fig(analysis.figs_lineplots["AUC (2 yr)"], Paths().get_path_auc_as_fcn_of_n_genes_selected_L2penalty_optim(penalizer))   # fmt: skip
        Paths.dump_fig(analysis.fig_barplot_full_ensemble, Paths().get_path_barplot_ensemble_L2penalty_optim(penalizer))


def ana1__FluidVsStiff_barplots(stiff_datafactories: ListOfDataFactories, datafactory_fluid: DataFactory):
    """Main paper figure: AdaPipe vs fixed-cutoff benchmarks."""
    # *** PAPER 1 FIGURE 2 (A, B, C) ***
    endpoint = "TTR"
    # --- ANALYSIS
    pipelines = ListOfDataFactories({"AdaPipe": datafactory_fluid}) + stiff_datafactories
    self = PipelineComparisonAnalysis(
        pipelines=pipelines,
        n_genes_selected_list=N_GENES_SELECTED_LINEPLOTS,
        train_endpoint=endpoint,
        penalizer=0.1,
    )
    self.generate_results()
    self.plot_barplot(n_selected_range=[10, 20])
    self.plot_barplot_for_full_ensemble_models(ensemble_range=[10, 20])
    self.plot_auc_for_different_n_select()
    # *** PAPER 1 FIGURE (SUPPLEMENTARY) ***
    self.plot_roc_curve_for_specific_k_for_candidate(k=20)
    self.compare_candidate_vs_benchmark_stats()
    # SAVE FIGS
    Paths.dump_fig(self.figs_lineplots["AUC (2 yr)"], Paths().auc_as_fcn_of_n_genes_selected_pdf, dpi=300)
    Paths.dump_fig(self.fig_barplot_full_ensemble, Paths().barplot_crossval_2yrauc_averaged_over_n_genes_pdf, dpi=300)


def ana4__AdaPipe_CrossVal_GeneSelection_heatmap(datafactory_fluid: DataFactory):
    """Runs the main AdaPipe in CV and produces the gene-selection heatmap
    showing which genes are selected across folds."""
    ridge_penalty = 0.1
    n_genes = 20

    # CROSS VALIDATION (FLUID)
    dataclass_updater = DataFactoryUpdater(datafactory_fluid)
    dataclass_updater.update_gene_selector(n_genes=n_genes)
    dataclass_updater.update_gene_pool("TSO500")
    dataclass_updater.update_oncofilter_strategy(new_oncofilter_strategy="highest_neglogp")
    datafactory_fluid.verbose = 1

    crossval_fluid = CrossValidator(
        data_factory=datafactory_fluid.build_data_for_ith_fold,
        model_factory=lambda: CPHModel(penalizer=ridge_penalty),
        eval_endpoint="TTR",
        train_endpoint="TTR",
        verbose=1,
    )
    crossval_fluid.prepare_data()
    crossval_fluid.train_models_in_crossvalidation()
    crossval_fluid.run_roc_curve_analysis(n_years=2)
    datafactory_fluid.print_variable_consistency_in_heatmap(verbose=0)
    # *** PAPER 1 FIGURE (SUPPLEMENTARY X) ***
    Paths.dump_fig(datafactory_fluid.fig_geneselection_heatmap, Paths().gene_selection_heatmap_crossval_pdf, dpi=300)
    green_print(f"N selected genes: {datafactory_fluid.info.n_selected}")


def load_fluid_datafactories(variants_df):
    """Load adaptive-filter datafactories developed on DSS, TTR, and OS endpoints."""
    green_print("DSS-developed pipeline")
    datafactory_fluid_DSS = DataFactory(
        variants_filtered_df=variants_df,
        data_builder=DataBuilder(endpoint="DSS"),
        grid_cutoff_selector=MinPvalOverEligible(oncofilter_selection="highest_neglogp"),
        path_varsearcher_dumpdir=Paths().crossval_varselection_dump_flexigrid_norway_DSS_dir,
        gene_selector=TopKBestGenes(k=20, gene_rater="p_softcut", minimal_criteria=lambda row: row["p"] < 0.05),
        path_crossval_split=Paths().cv_splits_norway_path,
    )
    datafactory_fluid_DSS.load_varsearchers()

    green_print("TTR-developed pipeline")
    datafactory_fluid_TTR = DataFactory(
        variants_filtered_df=variants_df,
        grid_cutoff_selector=MinPvalOverEligible(oncofilter_selection="highest_neglogp"),
        path_varsearcher_dumpdir=Paths().crossval_varselection_dump_flexigrid_norway_dir,
        gene_selector=TopKBestGenes(k=20, gene_rater="p_softcut", minimal_criteria=lambda row: row["p"] < 0.05),
        path_crossval_split=Paths().cv_splits_norway_path,
    )
    datafactory_fluid_TTR.load_varsearchers()

    green_print("OS-developed pipeline")
    datafactory_fluid_OS = DataFactory(
        variants_filtered_df=variants_df,
        grid_cutoff_selector=MinPvalOverEligible(oncofilter_selection="highest_neglogp"),
        path_varsearcher_dumpdir=Paths().crossval_varselection_dump_flexigrid_norway_OS_dir,
        gene_selector=TopKBestGenes(k=20, gene_rater="p_softcut", minimal_criteria=lambda row: row["p"] < 0.05),
        path_crossval_split=Paths().cv_splits_norway_path,
    )
    datafactory_fluid_OS.load_varsearchers()

    return datafactory_fluid_DSS, datafactory_fluid_TTR, datafactory_fluid_OS


def ana5__Endpoint_Comparison(variants_df, stiff_datafactories: ListOfDataFactories):
    """Cross-validation comparison of DSS/OS prediction performance when varying
    the endpoint used for gene selection, filter tuning, and model fitting."""
    datafactory_fluid_DSS, datafactory_fluid_TTR, datafactory_fluid_OS = load_fluid_datafactories(variants_df)
    ensemble_range = [10, 20]

    # --- 1. DEVELOP & FIT ON DSS, EVAL ON DSS
    print_header1("1. DEVELOP & FIT ON DSS, EVAL ON DSS", vspace=5)
    analyzer_DSS_DSS = PipelineComparisonAnalysis(
        pipelines=ListOfDataFactories(
            {
                "Developed on TTR, trained on DSS": datafactory_fluid_TTR,
                "Developed & trained on DSS": datafactory_fluid_DSS,
            }
        ),
        candidate_label="Developed on TTR, trained on DSS",
        n_genes_selected_list=N_GENES_SELECTED,
        train_endpoint="DSS",
        eval_endpoint="DSS",
        penalizer=0.1,
        verbose=1,
    )
    analyzer_DSS_DSS.generate_results()
    analyzer_DSS_DSS.plot_barplot_for_ensemble_predictions(ensemble_range)
    analyzer_DSS_DSS.plot_barplot_for_full_ensemble_models(ensemble_range)
    analyzer_DSS_DSS.compare_candidate_vs_benchmark_stats(ensemble_range=ensemble_range)

    # --- 2. DEVELOP & FIT ON TTR, EVAL ON DSS
    print_header1("2. DEVELOP & FIT ON TTR, EVAL ON DSS", vspace=5)
    # *** PAPER 1 FIGURE (SUPPLEMENTARY 3A) ***
    analyzer_TTR_TTR = PipelineComparisonAnalysis(
        pipelines=ListOfDataFactories({"AdaPipe": datafactory_fluid_TTR}) + stiff_datafactories,
        n_genes_selected_list=N_GENES_SELECTED,
        train_endpoint="TTR",
        eval_endpoint="DSS",
        penalizer=0.1,
        verbose=1,
    )
    analyzer_TTR_TTR.generate_results()
    analyzer_TTR_TTR.plot_barplot_for_full_ensemble_models(ensemble_range)
    analyzer_TTR_TTR.compare_candidate_vs_benchmark_stats(ensemble_range=ensemble_range)
    Paths.dump_fig(analyzer_TTR_TTR.fig_barplot_full_ensemble, Paths().barplot_crossval_TrainDevTTR_evalDSS_pdf, dpi=300)   # fmt: skip

    # --- 3. DEVELOP ON TTR, FIT ON DSS, EVAL ON DSS (HYBRID)
    print_header1("3. DEVELOP ON TTR, FIT ON DSS, EVAL ON DSS (HYBRID)", vspace=5)
    analyzer_TTR_DSS = PipelineComparisonAnalysis(
        pipelines=ListOfDataFactories({"AdaPipe": datafactory_fluid_TTR}) + stiff_datafactories,
        n_genes_selected_list=N_GENES_SELECTED,
        train_endpoint="DSS",
        eval_endpoint="DSS",
        penalizer=0.1,
        verbose=1,
    )
    analyzer_TTR_DSS.generate_results()
    analyzer_TTR_DSS.plot_barplot_for_ensemble_predictions(ensemble_range)
    analyzer_TTR_DSS.plot_barplot_for_full_ensemble_models(ensemble_range)
    analyzer_TTR_DSS.compare_candidate_vs_benchmark_stats(ensemble_range=ensemble_range)

    # --- 4. DEVELOP & FIT ON TTR, EVAL ON OS
    print_header1("4. DEVELOP & FIT ON TTR, EVAL ON OS", vspace=5)
    # *** PAPER 1 FIGURE (SUPPLEMENTARY 3B) ***
    analyzer_TTR_TTR_OS = PipelineComparisonAnalysis(
        pipelines=ListOfDataFactories({"AdaPipe": datafactory_fluid_TTR}) + stiff_datafactories,
        n_genes_selected_list=N_GENES_SELECTED,
        train_endpoint="TTR",
        eval_endpoint="OS",
        penalizer=0.1,
        verbose=1,
    )
    analyzer_TTR_TTR_OS.generate_results()
    analyzer_TTR_TTR_OS.plot_barplot_for_full_ensemble_models(ensemble_range)
    analyzer_TTR_TTR_OS.compare_candidate_vs_benchmark_stats(ensemble_range=ensemble_range)
    Paths.dump_fig(analyzer_TTR_TTR_OS.fig_barplot_full_ensemble, Paths().barplot_crossval_TrainDevTTR_evalOS_pdf, dpi=300)   # fmt: skip

    # --- 5. DEVELOP & FIT ON OS, EVAL ON OS
    print_header1("5. DEVELOP & FIT ON OS, EVAL ON OS", vspace=5)
    analyzer_OS_OS = PipelineComparisonAnalysis(
        pipelines=ListOfDataFactories(
            {"Developed on TTR": datafactory_fluid_TTR, "Developed on OS": datafactory_fluid_OS}
        ),
        candidate_label="Developed on TTR",
        n_genes_selected_list=N_GENES_SELECTED,
        train_endpoint="OS",
        eval_endpoint="OS",
        penalizer=0.1,
        verbose=1,
    )
    analyzer_OS_OS.generate_results()
    analyzer_OS_OS.plot_barplot_for_full_ensemble_models(ensemble_range)
    analyzer_OS_OS.compare_candidate_vs_benchmark_stats(ensemble_range=ensemble_range)

    # ---------- SUMMARIZE RESULTS IN TABLE -----------------------------------------------------------------------------------------
    # *** PAPER 1 TABLE (SUPPLEMENTARY TABLE 1) ***
    ada_pipe = "AdaPipe"
    benchmark = LABEL_BENCHMARK_ENSEMBLE
    no_adaptive_tuning = "-"

    # Each row: analyzer, ensemble_summary_df index label(s), and endpoint metadata.
    # Gene-selection / filter-tuning endpoint follows the varsearcher dump used by the datafactory.
    table_row_specs = [
        # --- Evaluated on DSS
        dict(
            analyzer=analyzer_DSS_DSS,
            approach="Developed & trained on DSS",
            pipeline=ada_pipe,
            gene_sel="DSS",
            filter_tune="DSS",
            model_fit="DSS",
            eval_ep="DSS",
        ),
        dict(
            analyzer=analyzer_TTR_TTR,
            approach=ada_pipe,
            pipeline=ada_pipe,
            gene_sel="TTR",
            filter_tune="TTR",
            model_fit="TTR",
            eval_ep="DSS",
        ),
        dict(
            analyzer=analyzer_TTR_DSS,
            approach=ada_pipe,
            pipeline=ada_pipe,
            gene_sel="TTR",
            filter_tune="TTR",
            model_fit="DSS",
            eval_ep="DSS",
        ),
        dict(
            analyzer=analyzer_TTR_TTR,
            approach=benchmark,
            pipeline=benchmark,
            gene_sel="TTR",
            filter_tune=no_adaptive_tuning,
            model_fit="TTR",
            eval_ep="DSS",
        ),
        dict(
            analyzer=analyzer_TTR_DSS,
            approach=benchmark,
            pipeline=benchmark,
            gene_sel="TTR",
            filter_tune=no_adaptive_tuning,
            model_fit="DSS",
            eval_ep="DSS",
        ),
        # --- Evaluated on OS
        dict(
            analyzer=analyzer_TTR_TTR_OS,
            approach=ada_pipe,
            pipeline=ada_pipe,
            gene_sel="TTR",
            filter_tune="TTR",
            model_fit="TTR",
            eval_ep="OS",
        ),
        dict(
            analyzer=analyzer_OS_OS,
            approach="Developed on OS",
            pipeline=ada_pipe,
            gene_sel="OS",
            filter_tune="OS",
            model_fit="OS",
            eval_ep="OS",
        ),
        dict(
            analyzer=analyzer_TTR_TTR_OS,
            approach=benchmark,
            pipeline=benchmark,
            gene_sel="TTR",
            filter_tune=no_adaptive_tuning,
            model_fit="TTR",
            eval_ep="OS",
        ),
    ]

    results_records = []
    for spec in table_row_specs:
        analyzer = spec["analyzer"]
        analyzer.get_ensemble_summary_dfs(ensemble_range)
        stats = analyzer.ensemble_summary_df.loc[spec["approach"]].agg(["mean", "std"])
        results_records.append(
            {
                "Endpoint for gene selection and filter tuning": spec["gene_sel"],
                "Endpoint for filter tuning(a)": spec["filter_tune"],
                "Endpoint for model fitting": spec["model_fit"],
                "Evaluation endpoint": spec["eval_ep"],
                "Pipeline": spec["pipeline"],
                "AUC (%), mean (SD)": f"{stats['mean']:.1f} ({stats['std']:.1f})",
            }
        )

    results_df = pd.DataFrame(results_records)
    display(results_df)
    Paths.dump_to_csv(results_df, Paths().auc_DSSprediction_for_various_trainEndpoints_xlsx)


def ana1b__CrossValComp_FluidVsStiff_trainOnNSCLC_testOnLUAD(
    stiff_datafactories: ListOfDataFactories, datafactory_fluid: DataFactory
):
    """NSCLC-trained fluid vs stiff pipelines, evaluated on LUAD and LUSC test subsets."""
    # *** PAPER 1 FIGURE (SUPPLEMENTARY) ***
    n_genes_selected_list = np.arange(10, 30, 2)
    ensemble_range = [10, 20]
    caption_luad = CaptionConfig(alpha=0.8, extratxt="\n\nPipelines trained on NSCLC, tested on LUAD")
    caption_lusc = CaptionConfig(alpha=0.8, extratxt="\n\nPipelines trained on NSCLC, tested on LUSC")

    # ---------- LUAD -----------------------------------------------------------------------------------------
    pipelines = ListOfDataFactories({"AdaPipe": datafactory_fluid}) + stiff_datafactories
    self_luad = PipelineComparisonAnalysis(
        pipelines=pipelines,
        eval_histology="LUAD",
        n_genes_selected_list=n_genes_selected_list,
        penalizer=0.1,
    )
    self_luad.generate_results()
    self_luad.plot_auc_for_different_n_select()
    ax, fig = self_luad.axes_lineplots["AUC (2 yr)"], self_luad.figs_lineplots["AUC (2 yr)"]
    ax.text(0.0, -0.15, "Pipelines trained on NSCLC, tested on LUAD", ha="left", va="top", transform=ax.transAxes)
    Paths.dump_fig(fig, Paths().auc_as_fcn_of_n_genes_selected_trainNSCLC_testLUAD_pdf)
    self_luad.plot_barplot(n_selected_range=ensemble_range, captionconfig=caption_luad)
    self_luad.plot_barplot_for_full_ensemble_models(ensemble_range=ensemble_range)
    ax, fig = self_luad.axes_barplot[1], self_luad.fig_barplot
    ax.axhline(y=1, linestyle="--", c="k", alpha=0.3)
    self_luad.compare_candidate_vs_benchmark_stats(ensemble_range=ensemble_range)
    Paths.dump_fig(fig, Paths().barplot_crossval_2yrauc_averaged_over_n_genes_trainNSCLC_testLUAD_pdf)

    # ---------- LUSC -----------------------------------------------------------------------------------------
    pipelines = ListOfDataFactories({"AdaPipe": datafactory_fluid}) + stiff_datafactories
    self_lusc = PipelineComparisonAnalysis(
        pipelines=pipelines,
        eval_histology="LUSC",
        n_genes_selected_list=n_genes_selected_list,
        penalizer=0.1,
    )
    self_lusc.generate_results()
    self_lusc.plot_auc_for_different_n_select()
    ax, fig = self_lusc.axes_lineplots["AUC (2 yr)"], self_lusc.figs_lineplots["AUC (2 yr)"]
    ax.set_ylim([0.5, 0.8])
    ax.text(0.0, -0.15, "Pipelines trained on NSCLC, tested on LUSC", ha="left", va="top", transform=ax.transAxes)
    Paths.dump_fig(fig, Paths().auc_as_fcn_of_n_genes_selected_trainNSCLC_testLUSC_pdf)
    self_lusc.plot_barplot(n_selected_range=ensemble_range, captionconfig=caption_lusc)
    self_lusc.plot_barplot_for_full_ensemble_models(ensemble_range=ensemble_range)
    self_lusc.compare_candidate_vs_benchmark_stats(ensemble_range=ensemble_range)
    ax, fig = self_lusc.axes_barplot[1], self_lusc.fig_barplot
    ax.axhline(y=1, linestyle="--", c="k", alpha=0.3)
    Paths.dump_fig(fig, Paths().barplot_crossval_2yrauc_averaged_over_n_genes_trainNSCLC_testLUSC_pdf)


def ana2__Ranking_Features(datafactory_fluid: DataFactory):
    """Compares which p-value/gene-level featurization method is best for gene
    selection, measured in terms of downstream performance of multivariate
    model."""
    print_header1("Ablation: gene-selection featurization (soft vs hard vs raw VAF)", 5, 1)

    # *** PAPER 1 FIGURE (SUPPLEMENTARY) ***
    # USING P-VALUE OF BINARY MUTATION TO SELECT GENES
    datafactory_fluid_HardCut = copy.deepcopy(datafactory_fluid)
    datafactory_fluid_HardCut.gene_selector.gene_rater = "p"
    # USING P-VALUE OF RAW VAF TO SELECT GENES
    datafactory_fluid_RawVAF = copy.deepcopy(datafactory_fluid)
    datafactory_fluid_RawVAF.gene_selector.gene_rater = "p_vaf"

    # COMBINE INTO LIST OF DATAFACTORIES
    fluid_datafactories = ListOfDataFactories(
        {
            "soft thresholding (logistic)": datafactory_fluid,
            "hard thresholding": datafactory_fluid_HardCut,
            "no transform": datafactory_fluid_RawVAF,
        }
    )
    analyzer = PipelineComparisonAnalysis(
        pipelines=fluid_datafactories,
        candidate_label="soft thresholding (logistic)",
        n_genes_selected_list=N_GENES_SELECTED,
        penalizer=0.1,
    )
    # GENERATE RESULTS
    analyzer.generate_results()
    # --- PLOT
    analyzer.colors_benchmarks = ["teal", "lime"]
    analyzer.plot_barplot_for_ensemble_predictions(
        ensemble_range=[10, 20], plot_mean_of_benchmarks=False, tick_rotation=0
    )
    analyzer.compare_candidate_vs_benchmark_stats(ensemble_range=[10, 20])
    analyzer.fig_barplot_ensemble.suptitle("Cross Validation Comparison of versions of AdaPipe", fontsize=16, y=1.01)   # fmt: skip

    Paths.dump_fig(analyzer.fig_barplot_ensemble, Paths().barplot_crossval_featurization_used_in_GeneSelect_pdf)


def ana2b__Robustness_FluidEnsemble_vs_StiffCutoffs(
    stiff_datafactories: ListOfDataFactories,
    datafactory_fluid: DataFactory,
):
    """Supplement robustness analysis where I ensemble two versions of AdaPipe:
    one that uses dichotmoized features and one that uses the features from the
    logistic transform that were used for gene selection."""
    print_header1("Robustness: AdaPipe hard+soft ensemble vs all stiff cutoffs", vspace1=5)

    datafactory_soft = copy.deepcopy(datafactory_fluid)
    datafactory_soft.feature_type = "SOFTCUT"

    pipelines = (
        ListOfDataFactories({"AdaPipe": datafactory_fluid})
        + ListOfDataFactories({"AdaPipe (SOFTCUT features)": datafactory_soft})
        + stiff_datafactories
    )

    analyzer = PipelineComparisonAnalysis(
        pipelines=pipelines,
        candidate_label="AdaPipe",
        pipeline_ensemble_groups={
            "AdaPipe ensemble": ["AdaPipe", "AdaPipe (SOFTCUT features)"],
            "Stiff ensemble": list(stiff_datafactories.labels),
        },
        n_genes_selected_list=N_GENES_SELECTED_LINEPLOTS,
        train_endpoint="TTR",
        penalizer=0.1,
    )
    analyzer.colors_benchmarks = [sns_colors[c] for c in ["yellow", "orange", "red", "dark_red"]]
    analyzer.generate_results()
    analyzer.plot_barplot_for_full_ensemble_models(ensemble_range=[10, 20])
    analyzer.compare_candidate_vs_benchmark_stats(ensemble_range=[10, 20])
    analyzer.plot_auc_for_different_n_select()


def ana2a__OncofilterStrategies(datafactory_fluid: DataFactory):
    """Compares different strategies for incorporating oncoscore into filter
    tuning and gene selection."""
    # *** PAPER 1 FIGURE (SUPPLEMENTARY) ***
    # ADAPTIVE PIPELINE (CHOOSE ONCOGENICITY FILTER PER GENE)
    datafactory_fluid.update_oncofilter_strategy("highest_neglogp")
    # NO ONCOFILTER: IGNORE ONCOGENICITY IN FILTERS
    datafactory_fluid_NoOnco = copy.deepcopy(datafactory_fluid)
    datafactory_fluid_NoOnco.update_oncofilter_strategy("pick_off")
    # USE ONCOFILTER: CONSIDER ONLY VHIGH-ONCOSCORE VARIANTS
    datafactory_fluid_Onco = copy.deepcopy(datafactory_fluid)
    datafactory_fluid_Onco.update_oncofilter_strategy("pick_on")

    # COMBINE INTO LIST OF DATAFACTORIES
    pipelines = ListOfDataFactories(
        {
            "Adaptive oncogenicity filtering": datafactory_fluid,
            "No oncogenicity filtering": datafactory_fluid_NoOnco,
            "Only oncogenic variants": datafactory_fluid_Onco,
        }
    )
    analyzer = PipelineComparisonAnalysis(
        pipelines=pipelines,
        candidate_label="Adaptive oncogenicity filtering",
        penalizer=0.1,
    )
    # --- GENERATE RESULTS
    analyzer.generate_results()
    # --- PLOT
    analyzer.colors_benchmarks = ["teal", "lime"]
    analyzer.plot_barplot_for_ensemble_predictions(
        ensemble_range=[10, 20], plot_mean_of_benchmarks=False, tick_rotation=0
    )
    analyzer.compare_candidate_vs_benchmark_stats(ensemble_range=[10, 20])
    analyzer.fig_barplot_ensemble.suptitle("Cross Validation Comparison of Oncogenicity Filtering Strategies", fontsize=16, y=1.01)   # fmt: skip

    Paths.dump_fig(analyzer.fig_barplot_ensemble, Paths().barplot_crossval_comparing_oncofilter_strategies_pdf)


def ana1a__FluidVsStiff_barplotsV2__drivers_only(
    stiff_datafactories: ListOfDataFactories, datafactory_fluid: DataFactory
):
    """Compares AdaPipe against stiff benchmarks, but restricts selection to
    genes that are known drivers in cancer. Compares against unrestricted TSO500
    selection."""
    # --- RESTRICT TO KNOWN DRIVER GENES
    driver_genes = DataBuilder.get_driver_genes()
    gene_selection_pool = driver_genes

    datafactory_fluid.gene_selector.gene_selection_pool = gene_selection_pool
    for datafactory in stiff_datafactories.datafactories:
        datafactory.gene_selector.gene_selection_pool = gene_selection_pool
    green_print(f"Restricting gene-selection set to {len(set(driver_genes) & set(datafactory_fluid._variants_filtered_df["SYMBOL"]))} known driver genes (in cancer)")   # fmt: skip

    endpoint = "TTR"
    # --- ANALYSIS
    self = PipelineComparisonAnalysis(
        pipelines=ListOfDataFactories({"AdaPipe": datafactory_fluid}) + stiff_datafactories,
        n_genes_selected_list=N_GENES_SELECTED_LINEPLOTS,
        train_endpoint=endpoint,
        candidate_label="AdaPipe",
        penalizer=0.1,
    )
    self.generate_results()
    # *** PAPER 1 FIGURE (SUPPLEMENTARY) ***
    self.plot_auc_for_different_n_select()
    self.plot_roc_curve_for_specific_k_for_candidate(k=20)
    self.plot_barplot_for_full_ensemble_models(ensemble_range=[10, 20])
    self.compare_candidate_vs_benchmark_stats(ensemble_range=[10, 20])
    # SAVE FIGS
    Paths.dump_fig(
        self.fig_barplot_full_ensemble, Paths().barplot_crossval_2yrauc_averaged_over_n_genes_drivers_only_pdf, dpi=300
    )
