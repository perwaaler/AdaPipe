"""TCGA external validation reported in the paper.

ana1 fits the main models. ana2 is the VAF-cutoff concordance scatter
(Figure S12). ana3 is the mutation-frequency comparison (Figure S3).
ana4 is the fixed VAF-cutoff table (Table S3).
"""

from utils.prelude import *
from utils.analysis_and_calculations import pearson_correlation
from utils.plotting import sns_colors
from utils.prepare_data.prepare_data import DataBuilder
from utils.TCGA.prepare_data import PrepareTCGAClinData
from utils.TCGA.tcga_mutamodel_analyser import MainTCGA_Analysis
from utils.variablesearch.variable_searcher import VariableSearcher
import statsmodels.formula.api as smf


def ana1__main_analysis(tcga_filtered_df):
    """How well do discovery-cohort models (AdaPipe vs stiff) predict survival in
    TCGA LUAD, LUSC, and NSCLC?"""
    # --- SETUP
    shared_args = {"tcga_filtered_df": tcga_filtered_df}
    experiments: Dict[str, MainTCGA_Analysis] = {}

    # --- TRAIN: NSCLC | EVAL: LUAD
    # PIPELINE: FluidCutGS
    experiments["NSCLCevalLUAD"] = MainTCGA_Analysis(experiment="E1: train on NSCLC", sub_experiment="LUADeval", pipeline_type="FluidCutGS",  publication_mode=True, **shared_args)   # fmt: skip
    experiments["NSCLCevalLUAD"].load_varsearchers_and_hyperparams()
    experiments["NSCLCevalLUAD"].model_preparation()
    experiments["NSCLCevalLUAD"].get_performance_evaluation_metrics("DSS")
    experiments["NSCLCevalLUAD"].plot_heatmap_for_pStage_vs_mutaRisk_crosstable()
    # PLOT THAT SHOWS AUC AS GENES ARE ADDED ON AT A TIME
    experiments["NSCLCevalLUAD"].incremental_gene_auc()
    PERCENTILE_CUTOFFS = experiments["NSCLCevalLUAD"].percentile_cutoffs
    plt.show()
    # *** PAPER 1 FIGURE ***
    # *** PAPER 1 SUPPLEMENTARY FIGURE ***
    Paths.dump_fig(experiments["NSCLCevalLUAD"].fig_incremental_auc, Paths().incremental_gene_auc_TCGAval__NSCLCtrain_LUADeval_pdf, dpi=300)   # fmt: skip
    Paths.dump_fig(experiments["NSCLCevalLUAD"].roc_output.fig, Paths().roc_TCGAval__NSCLCtrain_LUADeval_DSS_pdf, dpi=300)   # fmt: skip
    Paths.dump_fig(experiments["NSCLCevalLUAD"].analyzer.fig_coefplot, Paths().coef_TCGAval__NSCLCtrain_LUADeval_pdf, dpi=300)   # fmt: skip
    Paths.dump_fig(experiments["NSCLCevalLUAD"].km_analyzer.fig, Paths().km_TCGAval__NSCLCtrain_LUADeval_DSS_pdf, dpi=300, dual_dump=True)   # fmt: skip
    Paths.dump_fig(experiments["NSCLCevalLUAD"].heatmap_fig, Paths().heatmap_TCGAval__NSCLCtrain_LUADeval_DSS_pdf, dpi=300, dual_dump=True)   # fmt: skip

    # PIPELINE: StiffCutGS (5% VAF CUTOFF)
    experiments["NSCLCevalLUAD_Stiff"] = MainTCGA_Analysis(experiment="E1: train on NSCLC", sub_experiment="LUADeval", pipeline_type="StiffCutGS", **shared_args)   # fmt: skip
    experiments["NSCLCevalLUAD_Stiff"].load_varsearchers_and_hyperparams()
    experiments["NSCLCevalLUAD_Stiff"].model_preparation()
    experiments["NSCLCevalLUAD_Stiff"].get_performance_evaluation_metrics()
    plt.show()

    # --- AUC COMPARISON USING BOOTSTRAPPING
    results_df = experiments["NSCLCevalLUAD"].compare_auc_using_bootstrap(
        other_scores=experiments["NSCLCevalLUAD_Stiff"],
    )

    # --- TRAIN: NSCLC | EVAL: LUSC
    experiments["NSCLCevalLUSC"] = MainTCGA_Analysis(experiment="E1: train on NSCLC", sub_experiment="LUSCeval", pipeline_type="FluidCutGS", publication_mode=True,  **shared_args)   # fmt: skip
    experiments["NSCLCevalLUSC"].load_varsearchers_and_hyperparams()
    experiments["NSCLCevalLUSC"].model_preparation()
    # Re-use percentile cutoffs from main model
    experiments["NSCLCevalLUSC"].percentile_cutoffs = PERCENTILE_CUTOFFS
    experiments["NSCLCevalLUSC"].get_performance_evaluation_metrics()
    experiments["NSCLCevalLUSC"].plot_heatmap_for_pStage_vs_mutaRisk_crosstable("PFI")
    plt.show()
    # *** PAPER 1 FIGURE SXXX (TCGA LUSC KM CURVE)***
    Paths.dump_fig(experiments["NSCLCevalLUSC"].roc_output.fig, Paths().roc_TCGAval__NSCLCtrain_LUSCeval_DSS_pdf, dpi=300)   # fmt: skip
    Paths.dump_fig(experiments["NSCLCevalLUSC"].km_analyzer.fig, Paths().km_TCGAval__NSCLCtrain_LUSCeval_DSS_pdf, dpi=300)   # fmt: skip
    Paths.dump_fig(experiments["NSCLCevalLUSC"].heatmap_fig, Paths().heatmap_TCGAval__NSCLCtrain_LUSCeval_DSS_pdf, dpi=300)   # fmt: skip

    # --- TRAIN: NSCLC | EVAL: NSCLC
    experiments["NSCLCevalNSCLC"] = MainTCGA_Analysis(experiment="E1: train on NSCLC", sub_experiment="NSCLCeval", pipeline_type="FluidCutGS", **shared_args)   # fmt: skip
    experiments["NSCLCevalNSCLC"].load_varsearchers_and_hyperparams()
    experiments["NSCLCevalNSCLC"].model_preparation()
    experiments["NSCLCevalNSCLC"].get_performance_evaluation_metrics()
    # *** PAPER 1 FIGURE 3D ***
    plt.show()
    Paths.dump_fig(experiments["NSCLCevalNSCLC"].roc_output.fig, Paths().roc_TCGAval__NSCLCtrain_NSCLCeval_DSS_pdf, dpi=300)   # fmt: skip

    # --- TRAIN: NSCLC DSS | EVAL: LUAD
    experiments["NSCLCevalLUAD_trainDSS"] = MainTCGA_Analysis(experiment="E3: train on NSCLC and DSS", sub_experiment="LUADeval", pipeline_type="FluidCutGS", publication_mode=True, **shared_args)   # fmt: skip
    experiments["NSCLCevalLUAD_trainDSS"].load_varsearchers_and_hyperparams()
    experiments["NSCLCevalLUAD_trainDSS"].model_preparation()
    experiments["NSCLCevalLUAD_trainDSS"].get_performance_evaluation_metrics()
    plt.show()

    # --- TRAIN: LUAD | EVAL: LUAD
    experiments["LUADevalLUAD"] = MainTCGA_Analysis(experiment="E2: train on LUAD", sub_experiment="LUADeval", pipeline_type="FluidCutGS", publication_mode=True, **shared_args)   # fmt: skip
    experiments["LUADevalLUAD"].load_varsearchers_and_hyperparams()
    experiments["LUADevalLUAD"].model_preparation()
    experiments["LUADevalLUAD"].percentile_cutoffs = experiments["NSCLCevalLUAD"].percentile_cutoffs
    experiments["LUADevalLUAD"].get_performance_evaluation_metrics()
    experiments["LUADevalLUAD"].plot_heatmap_for_pStage_vs_mutaRisk_crosstable()
    plt.show()
    # *** PAPER 1 FIGURE (SUPPLEMENTARY) ***
    Paths.dump_fig(experiments["LUADevalLUAD"].roc_output.fig, Paths().roc_TCGAval__LUADtrain_LUADeval_DSS_pdf, dpi=300)   # fmt: skip
    Paths.dump_fig(experiments["LUADevalLUAD"].km_analyzer.fig, Paths().km_TCGAval__LUADtrain_LUADeval_DSS_pdf, dpi=300)   # fmt: skip
    Paths.dump_fig(experiments["LUADevalLUAD"].heatmap_fig, Paths().heatmap_TCGAval__LUADtrain_LUADeval_DSS_pdf, dpi=300)   # fmt: skip

    # ---------- ROC CURVES FOR OTHER ENDPOINTS: PFI AND OS -----------------------------------------------------------------------------------------
    # OS
    experiments["NSCLCevalLUAD"].get_performance_evaluation_metrics(endpoint="OS")
    Paths.dump_fig(experiments["NSCLCevalLUAD"].roc_output.fig, Paths().roc_TCGAval__NSCLCtrain_LUADeval_OS_pdf, dpi=300)   # fmt: skip
    # PFI
    experiments["NSCLCevalLUAD"].get_performance_evaluation_metrics(endpoint="PFI")
    Paths.dump_fig(experiments["NSCLCevalLUAD"].roc_output.fig, Paths().roc_TCGAval__NSCLCtrain_LUADeval_PFI_pdf, dpi=300)   # fmt: skip


def ana2__VAFcutoffs_TCGA_vs_TNMi(tcga_filtered_df, tnmi_filtered_df):
    """Do per-gene adaptive VAF cutoffs agree between TNM-I and TCGA LUAD?"""
    endpoint: Literal["DSS", "OS"] = "OS"
    dump_dir_path_TCGA = Paths().varselection_dump_TCGA_DSS_dir if endpoint == "DSS" else Paths().varselection_dump_TCGA_OS_dir   # fmt: skip
    dump_dir_path_NORW = Paths().varselection_dump_flexigrid_norway_ESSpen_dir

    # --- LOAD VARSEARCHERS
    # NORWAY
    varsearcher_norw = VariableSearcher(
        variants_df=tnmi_filtered_df,
        clinical_df=DataBuilder().get_clinical_data(),
    )
    varsearcher_norw.prepare_analysis_dataframe()
    varsearcher_norw.load_results_from_json(dump_dir_path_NORW, compute_result_dfs=True)
    # TCGA
    varsearcher_tcga = VariableSearcher(
        variants_df=tcga_filtered_df,
        clinical_df=PrepareTCGAClinData(endpoint=endpoint).get_clinical_data(),
        endpoint=endpoint,
        red_id_used_to_fit_filters="no_restriction",
        fit_softcut_function_in_gridanalysis=True,
        prelim_pval_cutoff=0.3,
    )
    varsearcher_tcga.prepare_analysis_dataframe()
    varsearcher_tcga.load_results_from_json(dump_dir_path_TCGA, compute_result_dfs=True)

    genepool = [x for x in DataBuilder.get_tso500_gene_list() if x in tcga_filtered_df["SYMBOL"].unique()]

    oncofilter_setting: Literal["off", "on", "adaptive"] = "adaptive"
    if oncofilter_setting == "adaptive":
        df_TCGA = varsearcher_tcga.summary_best_models_df.loc[genepool]
        df_NORW = varsearcher_norw.summary_best_models_df.loc[genepool]
    elif oncofilter_setting == "off":
        df_TCGA = varsearcher_tcga.summary_df__off.loc[genepool]
        df_NORW = varsearcher_norw.summary_df__off.loc[genepool]
    elif oncofilter_setting == "on":
        df_TCGA = varsearcher_tcga.summary_df__on.loc[genepool]
        df_NORW = varsearcher_norw.summary_df__on.loc[genepool]
    else:
        raise ValueError("ERROR")

    cols = ["p_softcut", "vaf_cutoff", "scale_softcut", "loc_softcut", "coef", "neglogp", "p"]
    df_merged = df_TCGA[cols].merge(df_NORW[cols], left_index=True, right_index=True, suffixes=("_TCGA", "_NORW")).dropna()   # fmt: skip
    p_cutoff = 0.1
    p_var: Literal["p", "p_softcut"] = "p_softcut"
    subset: Literal["significant_in_both", "significant_in_TCGA", "both_p_signif"] = "significant_in_both"
    if subset == "significant_in_both":
        comparable_genes = df_merged.query(f"{p_var}_NORW<{p_cutoff} & {p_var}_TCGA<{p_cutoff}").index
    elif subset == "both_p_signif":
        comparable_genes = df_merged.query(f"p_NORW<{p_cutoff} & p_softcut_NORW<{p_cutoff} & p_TCGA<{p_cutoff} &  p_softcut_TCGA<{p_cutoff}").index   # fmt: skip
    else:
        comparable_genes = df_merged.query(f"{p_var}_TCGA<{p_cutoff}").index
    green_print(f"N genes with p_softcut<{p_cutoff} in both datasets: {len(comparable_genes)}")
    df_merged[sorted(df_merged.columns)].loc[comparable_genes]

    # --- COMPUTE CORRELATIONS (WITH P-VALUES)
    f = lambda df, var: pearson_correlation(df[f"{var}_TCGA"], df[f"{var}_NORW"])
    vaf_cutoff_corr = f(df_merged.loc[comparable_genes], "vaf_cutoff")
    coef_corr = f(df_merged.loc[comparable_genes], "coef")
    neglogp_corr = f(df_merged.loc[comparable_genes], "neglogp")

    # FIT REGRESSION LINE FOR PLOTTING
    results = smf.ols("vaf_cutoff_TCGA ~ vaf_cutoff_NORW", data=df_merged.loc[comparable_genes]).fit()
    xx = np.linspace(0, 0.5, 100)
    yy = results.params["Intercept"] + np.linspace(0, 0.5, 100) * results.params["vaf_cutoff_NORW"]

    # *** PAPER 1 FIGURE (SUPPLEMENT) ***
    # --- PLOT SCATTER PLOT FOR VAF CUTOFF ESTIMATES
    vaf_cutoffs_NORW = df_merged.loc[comparable_genes, "vaf_cutoff_NORW"]
    vaf_cutoffs_TCGA = df_merged.loc[comparable_genes, "vaf_cutoff_TCGA"]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(vaf_cutoffs_NORW, vaf_cutoffs_TCGA, edgecolor="k", c="k", s=50, alpha=0.7)
    ax.plot(xx, yy, linestyle="--", c="r", label=rf"regression line ($\rho={vaf_cutoff_corr.value:.3f}$)")
    ax.plot([0, 0.5], [0, 0.5], "--", c="k", label="1-1 line")
    ax.set_xlabel("VAF cutoff (discovery cohort)")
    ax.set_ylabel("VAF cutoff (TCGA LUAD)")
    ax.set_title("Correlation of VAF cutoff estimates between cohorts")
    ax.legend()

    green_print(f"\n{'='*10} ANALYSIS OF GENES WITH P<0.1 IN BOTH DATASETS (n={len(comparable_genes)}) {"=="*10}")
    green_print(f"Correlation of VAF cutoffs: {vaf_cutoff_corr.value:.1%} ({vaf_cutoff_corr.p:.1e})")
    green_print(f"Correlation of coefficients: {coef_corr.value:.1%} ({coef_corr.p:.1e})")
    green_print(f"Correlation of -log10(p): {neglogp_corr.value:.1%} ({neglogp_corr.p:.1e})")

    Paths.dump_fig(fig, Paths().scatter_vaf_cutoff_concordance_norway_vs_TCGA_pdf, dpi=300)


def ana3__compare_mutationrates_between_norway_and_TCGA(tnmi_filtered_df, tcga_filtered_df):
    """How do per-gene mutation frequencies compare between TNM-I and TCGA LUAD?"""
    tso500_genes = DataBuilder.get_tso500_gene_list()  # fmt: skip
    tnmi_country: Literal["Norway", "Denmark", "all"] = "all"
    tnmi_histology: Literal["LUAD", "LUSC", "NSCLC"] = "LUAD"
    histology_map = {
        "LUAD": DataBuilder.get_red_id_luad_data(),
        "LUSC": DataBuilder.get_red_id_lusc_data(),
        "NSCLC": DataBuilder.get_red_id_genomic_data(),
    }

    red_id_specification = list(set(DataBuilder.get_red_id_norway_data() + histology_map[tnmi_histology]))  # fmt: skip
    norway_filtered_df = tnmi_filtered_df[tnmi_filtered_df["RED_ID"].isin(red_id_specification)]  # fmt: skip
    tcga_luad_filtered_df = tcga_filtered_df.query("histology in ['LUAD']").copy()

    # ---------- UTILITY FUNCTIONS -----------------------------------------------------------------------------------------
    restrict_to_tso500 = lambda df: df.loc[[x for x in df.index if x in tso500_genes]]
    is_oncogenic = lambda df: df["ONCOGENIC"].apply(lambda x: "Oncogenic" in x if pd.notna(x) else False)  # fmt: skip
    high_oncoscore = lambda df: df["ONCOGENICITY_SCORE"] >= 3
    exceeds_vaf_cutoff = lambda df: df["VAF_TUMOR"] >= 0.05
    get_onco_rates = lambda df: df[is_oncogenic(df) & exceeds_vaf_cutoff(df)].groupby("SYMBOL")["RED_ID"].nunique().sort_values(ascending=True) / df["RED_ID"].nunique()   # fmt: skip
    get_onco_counts = (
        lambda df: df[is_oncogenic(df) & exceeds_vaf_cutoff(df)]
        .groupby("SYMBOL")["RED_ID"]
        .nunique()
        .sort_values(ascending=True)
    )
    get_high_oncorate = lambda df: df[high_oncoscore(df) & exceeds_vaf_cutoff(df)].groupby("SYMBOL")["RED_ID"].nunique().sort_values(ascending=True) / df["RED_ID"].nunique()  # fmt: skip
    get_rates = lambda df: df[exceeds_vaf_cutoff(df)].groupby("SYMBOL")["RED_ID"].nunique().sort_values(ascending=True) / df["RED_ID"].nunique()   # fmt: skip

    # ---------- WITH ONCOKB ONCOGENICITY FILTERING -----------------------------------------------------------------------------------------
    # --- CALCULATE RATES
    tcga_oncorates = get_onco_rates(tcga_luad_filtered_df)
    tcga_oncocounts = get_onco_counts(tcga_luad_filtered_df)
    norway_oncorates = get_onco_rates(norway_filtered_df)
    norway_oncocounts = get_onco_counts(norway_filtered_df)
    # RESTRICT TO TSO500
    norway_oncorates = restrict_to_tso500(norway_oncorates)
    tcga_oncorates = restrict_to_tso500(tcga_oncorates)

    rate_diff = (norway_oncorates - tcga_oncorates).dropna().mean()
    rate_proportions = tcga_oncorates / norway_oncorates
    rate_proportions_IQR = rate_proportions.agg([lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]).to_list()
    magenta_print(f"N genes with onco-mutas in TCGA LUAD: {(tcga_oncocounts>0).sum()}")
    magenta_print(f"Mean difference in mutation rate: {rate_diff:.1%} (STD={rate_diff.std():.4f})")
    magenta_print(f"Median mutation rate ratio (TCGA/NORWAY): {(rate_proportions).median():.3f} (IQR={", ".join([f"{x:.3f}" for x in rate_proportions_IQR])})")   # fmt: skip
    blue_print(f"MeanMutaRateNorw / MeanMutaRateTCGA: {norway_oncorates.mean() / tcga_oncorates.mean():.1f}")

    col_names = ["tcga_rate", "norw_rate", "tcga_count", "norw_count"]
    rates_wide_df = pd.concat([tcga_oncorates, norway_oncorates, tcga_oncocounts, norway_oncocounts], axis=1, keys=col_names).dropna()  # fmt: skip
    rates_wide_df["mean_rate"] = rates_wide_df.sum(axis=1)
    rates_wide_df["prop_TCGAoverTNMI"] = rates_wide_df.apply(lambda r: r["tcga_rate"] / r["norw_rate"], axis=1)
    rates_wide_df = rates_wide_df.sort_values("tcga_rate", ascending=False).reset_index()  # fmt: skip
    rates_wide_df = rates_wide_df.head(50)
    melted_df = rates_wide_df.melt(value_name="mutation rate", var_name="Cohort", value_vars=["tcga_rate", "norw_rate"], id_vars=["SYMBOL"])  # fmt: skip

    barplot_settings = {"palette": [sns_colors["orange"], sns_colors["blue"]], "edgecolor": "k"}

    # *** PAPER 1 FIGURE (SUPPLEMENTARY) ***
    fig, axes = plt.subplots(2, 1, figsize=(20, 10))

    ax = axes[0]
    melted_df["Cohort"] = melted_df["Cohort"].replace({"tcga_rate": "TCGA", "norw_rate": "TNM-I"})
    sns.barplot(melted_df, x="SYMBOL", hue="Cohort", y="mutation rate", ax=ax, **barplot_settings)
    ax.tick_params(rotation=90)
    ax.legend()
    ax.set_xlabel("")
    ax.set_ylabel("mutation frequency")
    ax.set_title("Oncogenic variants (OncoKB SOP)")

    # ---------- WITHOUT ONCOGENICITY FILTERING -----------------------------------------------------------------------------------------
    # GET MUTATION RATES
    tcga_rates = get_rates(tcga_luad_filtered_df)
    # NORWAY LUAD
    norway_rates = get_rates(norway_filtered_df)

    norway_rates = restrict_to_tso500(norway_rates)
    tcga_rates = restrict_to_tso500(tcga_rates)

    rate_diff = (norway_rates - tcga_rates).dropna().mean()
    rate_proportions = tcga_rates / norway_rates
    blue_print(f"Mean difference in mutation rate: {rate_diff:.1%}")
    blue_print(f"Mean mutation rate ratio (TCGA/NORWAY): {(tcga_rates / norway_rates).median():.3f}")
    blue_print(f"MeanMutaRateNorw / MeanMutaRateTCGA: {norway_rates.mean() / tcga_rates.mean():.1f}")
    rate_proportions_IQR = rate_proportions.agg([lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]).to_list()
    magenta_print(f"Mean difference in mutation rate: {rate_diff:.1%} (STD={rate_diff.std():.4f})")
    magenta_print(f"Median mutation rate ratio (TCGA/NORWAY): {(rate_proportions).median():.3f} (IQR={", ".join([f"{x:.3f}" for x in rate_proportions_IQR])})")   # fmt: skip

    rates_wide_df = pd.concat([tcga_rates, norway_rates], axis=1, keys=["TCGA", "TNM-I"]).dropna()  # fmt: skip
    rates_wide_df["mean_rate"] = rates_wide_df.sum(axis=1)
    rates_wide_df = rates_wide_df.sort_values("TCGA", ascending=False).reset_index()
    rates_wide_df = rates_wide_df.head(50)
    melted_df = rates_wide_df.melt(value_name="mutation rate", var_name="Cohort", value_vars=["TCGA", "TNM-I"], id_vars=["SYMBOL"])  # fmt: skip

    ax = axes[1]
    sns.barplot(melted_df, x="SYMBOL", hue="Cohort", y="mutation rate", ax=ax, **barplot_settings)   # fmt: skip
    ax.tick_params(rotation=90)
    ax.set_title("All variants")
    ax.legend()
    ax.set_xlabel("")
    ax.set_ylabel("mutation frequency")

    fig.subplots_adjust(hspace=0.4)
    fig.suptitle(
        f"Mutation frequency in TCGA LUAD vs TNM-I ({tnmi_country}) {tnmi_histology} sorted by TCGA mutation frequency",
        fontsize=20,
        y=0.95,
    )

    Paths.dump_fig(fig, Paths().mutation_rates_TNMILUAD_vs_TCGALUAD_pdf)


def ana4__produce_table_S5_stiff_VAFcutoffs_in_TCGA_LUAD(
    tcga_filtered_df,
) -> tuple[pd.DataFrame, dict[float, MainTCGA_Analysis]]:
    """Evaluates TNM-I models that use VAF cutoffs ranging from 4% to 7% in TCGA
    LUAD (Table S3)."""
    vaf_cutoffs = [0.04, 0.05, 0.06, 0.07]
    rows = []
    experiments: dict[float, MainTCGA_Analysis] = {}

    def _evaluate_stiff_pipeline_at_vaf_cutoff(
        *,
        stiff_cutoff: float,
        tcga_filtered_df,
        endpoint: Literal["DSS", "OS", "PFI"] = "DSS",
        n_years: int = 2,
        show_plots: bool = True,
    ) -> MainTCGA_Analysis:
        shared_args = {"tcga_filtered_df": tcga_filtered_df}
        kwargs = {
            "experiment": "E1: train on NSCLC",
            "sub_experiment": "LUADeval",
            "pipeline_type": "StiffCutGS",
            **shared_args,
        }
        if stiff_cutoff != 0.05:
            kwargs["stiff_cutoff"] = stiff_cutoff
        experiment = MainTCGA_Analysis(**kwargs)
        experiment.load_varsearchers_and_hyperparams()
        experiment.model_preparation()
        experiment.get_performance_evaluation_metrics(endpoint=endpoint, n_years=n_years)
        if show_plots:
            plt.show()
        return experiment

    for cutoff in vaf_cutoffs:
        experiment = _evaluate_stiff_pipeline_at_vaf_cutoff(
            stiff_cutoff=cutoff,
            tcga_filtered_df=tcga_filtered_df,
        )
        experiments[cutoff] = experiment
        roc = experiment.roc_output
        rows.append(
            {
                "VAF cutoff": f"{int(round(cutoff * 100))}%",
                "AUC (DSS), (%)": round(roc.auc * 100, 1),
                "Cox p-value": round(roc.pval, 3),
                "Cox stage-adjusted p-value": round(roc.pval_adjusted, 3),
            }
        )

    results_df = pd.DataFrame(rows)
    Paths.dump_to_csv(results_df, Paths().table_S5_stiff_vaf_cutoffs_TCGA_LUAD_DSS_csv)
    display(results_df)
