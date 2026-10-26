"""Paths for the Paper 1 data, saved searches, and figure dumps."""

from typing import Union
from pathlib import Path
from utils.display import *
from matplotlib.figure import Figure
import os
import pandas as pd
import json
import yaml


class Paths:
    """File locations used by the discovery and TCGA analyses."""

    root = Path(__file__).parents[2]

    def __init__(self):
        data_dir = self.root / "data"
        output_dir = self.root / "output"
        results_dir = self.root / "results"
        master_data_dir = data_dir / "master-data"
        figdir = results_dir / "datafusion/figures"
        miscdir = results_dir / "datafusion/misc"
        clinvar_ana_dir = results_dir / "clinvar-analysis-tables"
        norway = output_dir / "varselect/varsearch-results/norway"
        tcga_search = output_dir / "varselect/varsearch-results/TCGA"
        hyper_dir = output_dir / "hyperparams"
        perm_dir = output_dir / "varselect/permutation-adjusted-pvals"
        auc_dir = figdir / "auc-vs-ngenes"
        bar_dir = figdir / "crossval-barplots"
        l2_dir = figdir / "sensitivity-l2-penalty"
        tcga_figdir = figdir / "tcga-validation"

        # --- TNM-I
        self.clinical_labels_path = master_data_dir / "clinical_data_label.csv"
        self.FU26_TNMI_with_events_csv = master_data_dir / "FU26_TNMI_with_events.csv"
        self.endpoint_updates_from_EE_json = master_data_dir / "endpoint_updates_from_EE.json"
        self.gatk_data_path = master_data_dir / "GATK_1224.txt"
        self.tmb_msi_key_filter_path = master_data_dir / "TMB_MSI_key_renamed.csv"
        self.tso500_genes_json = data_dir / "gene-panels/tso500_genes.json"
        self.intogen_compendium_cancer_driver_genes_tsv = data_dir / "gene-sets/2024-06-18_IntOGen-Drivers/Compendium_Cancer_Genes.tsv"

        # --- TCGA
        self.tcga_CDRcurated_endpoints_csv = data_dir / "tcga/TCGA-CDR-SupplementalTableS1.xlsx"
        self.tcga_luad_mutas_csv = data_dir / "tcga/LUAD/TCGA_LUAD_Mutations.csv"
        self.tcga_luad_oncokb_out_txt = data_dir / "tcga/LUAD/TCGA_LUAD_OncoKB_out.txt"
        self.tcga_luad_clinical_data_tsv = data_dir / "tcga/LUAD/luad_tcga_clinical_data.tsv"
        self.tcga_lusc_mutas_csv = data_dir / "tcga/LUSC/TCGA_LUSC_Mutations.csv"
        self.tcga_lusc_oncokb_out_txt = data_dir / "tcga/LUSC/TCGA_LUSC_OncoKB_out.txt"
        self.tcga_clinical_patient_csv = data_dir / "tcga/LUSC/TCGA_LUSC_Clinical_Patient.csv"

        # --- GENE POOLS AND ID LISTS
        gene_sets = output_dir / "gene-sets"
        self.genepool_tcgaLUAD_validation_oncoselection_json = gene_sets / "genepool_tcgaLUAD_validation_allonco_min5.json"
        self.genepool_tcgaNSCLC_validation_oncoselection_json = gene_sets / "genepool_tcgaNSCLC_validation_allonco.json"
        self.genepool_tcgaLUSC_validation_oncoselection_json = gene_sets / "genepool_tcgaLUSC_validation_allonco.json"
        self.genes_that_appear_in_norway_json = gene_sets / "genes_in_norway.json"

        splits = output_dir / "data-splits"
        self.cv_splits_norway_path = splits / "cv_splits_norway.json"
        self.cv_splits_norway_luad_path = splits / "cv_splits_norway_luad.json"
        self.red_id_norway_path = splits / "red_id_norway.json"
        self.red_id_genomic_data_json = splits / "red_id_genomic_data.json"
        self.red_id_luad_json = splits / "red_id_luad.json"
        self.red_id_lusc_json = splits / "red_id_lusc.json"
        self.red_id_high_stages_json = splits / "red_id_high_stages.json"

        # --- NORWAY VAF SEARCHES (whole cohort)
        self.varselection_dump_flexigrid_norway_dir = norway / "whole/flexigrid"
        self.varselection_dump_fixedgrid_norway_dir = norway / "whole/fixedgrid"
        self.varselection_dump_flexigrid_norway_ESSpen_dir = norway / "whole/ESSpenalized/flexigrid"
        self.varselection_dump_norway_DSS_dir = norway / "whole/DSS"
        self.varselection_dump_norway_DSS_fixedgrid_dir = norway / "whole/DSS/fixedgrid"
        self.varselection_dump_LUAD_norway_flexigrid_dir = norway / "whole/luad/flexigrid"
        self.varselection_dump_LUAD_norway_fixedgrid_dir = norway / "whole/luad/fixedgrid"
        self.bootstrapped_vaf_cutoffs_json = output_dir / "varselect/VAF-cutoff-analysis/bootstrapped_vaf_cutoffs.json"

        # --- NORWAY VAF SEARCHES (cross-validation)
        self.crossval_varselection_dump_flexigrid_norway_dir = norway / "crossval/flexigrid"
        self.crossval_varselection_dump_fixedgrid_norway_dir = norway / "crossval/fixedgrid"
        self.crossval_varselection_dump_flexigrid_norway_DSS_dir = norway / "crossval/DSS/flexigrid"
        self.crossval_varselection_dump_fixedgrid_norway_DSS_dir = norway / "crossval/DSS/fixedgrid"
        self.crossval_varselection_dump_flexigrid_norway_OS_dir = norway / "crossval/OS/flexigrid"
        self.crossval_varselection_dump_fixedgrid_norway_OS_dir = norway / "crossval/OS/fixedgrid"
        self.crossval_varselection_dump_flexigrid_luad_norway_dir = norway / "crossval/luad/flexigrid"
        self.crossval_varselection_dump_fixedgrid_luad_norway_dir = norway / "crossval/luad/fixedgrid"

        # --- TCGA VAF SEARCHES
        self.varselection_dump_TCGA_OS_dir = tcga_search / "whole/OS"
        self.varselection_dump_TCGA_DSS_dir = tcga_search / "whole/DSS"

        # --- HYPERPARAMETERS FOR THE TCGA MODELS
        self.hyperpar_for_TCGAval__fluid_NSCLCtrain_LUADval_json = hyper_dir / "hyperpar_for_TCGAval__fluid_NSCLCtrain_LUADval.json"
        self.hyperpar_for_TCGAval__stiff_NSCLCtrain_LUADval_json = hyper_dir / "hyperpar_for_TCGAval__stiff_NSCLCtrain_LUADval.json"
        self.hyperpar_for_TCGAval__stiff4pct_NSCLCtrain_LUADval_json = hyper_dir / "hyperpar_for_TCGAval__stiff4pct_NSCLCtrain_LUADval.json"
        self.hyperpar_for_TCGAval__stiff6pct_NSCLCtrain_LUADval_json = hyper_dir / "hyperpar_for_TCGAval__stiff6pct_NSCLCtrain_LUADval.json"
        self.hyperpar_for_TCGAval__stiff7pct_NSCLCtrain_LUADval_json = hyper_dir / "hyperpar_for_TCGAval__stiff7pct_NSCLCtrain_LUADval.json"
        self.hyperpar_for_TCGAval__fluid_NSCLCtrain_NSCLCval_json = hyper_dir / "hyperpar_for_TCGAval__fluid_NSCLCtrain_NSCLCval.json"
        self.hyperpar_for_TCGAval__stiff_NSCLCtrain_NSCLCval_json = hyper_dir / "hyperpar_for_TCGAval__stiff_NSCLCtrain_NSCLCval.json"
        self.hyperpar_for_TCGAval__fluid_NSCLCtrain_LUSCval_json = hyper_dir / "hyperpar_for_TCGAval__fluid_NSCLCtrain_LUSCval.json"
        self.hyperpar_for_TCGAval__stiff_NSCLCtrain_LUSCval_json = hyper_dir / "hyperpar_for_TCGAval__stiff_NSCLCtrain_LUSCval.json"
        self.hyperpar_for_TCGAval__fluid_LUADtrain_LUADval_json = hyper_dir / "hyperpar_for_TCGAval__fluid_LUADtrain_LUADval.json"
        self.hyperpar_for_TCGAval__stiff_LUADtrain_LUADval_json = hyper_dir / "hyperpar_for_TCGAval__stiff_LUADtrain_LUADval.json"
        self.hyperpar_for_TCGAval__fluid_NSCLCtrain_LUADval_DSS_json = hyper_dir / "hyperpar_for_TCGAval__fluid_NSCLCtrain_LUADval_DSS.json"
        self.hyperpar_for_TCGAval__stiff_NSCLCtrain_LUADval_DSS_json = hyper_dir / "hyperpar_for_TCGAval__stiff_NSCLCtrain_LUADval_DSS.json"

        # --- PERMUTATION P-VALUES
        self.permutated_pvals_norway_json = perm_dir / "permutated_pvals_norway.json"
        self.permuted_pvals_norway_csv = perm_dir / "permutation_adjusted_pvals_norway.csv"
        self.permutation_adjusted_pvals_soft_onco_norway_csv = perm_dir / "permutation_adjusted_pvals_soft_onco_norway.csv"
        self.permutated_pvals_fixedcut_norway_json = perm_dir / "permutated_pvals_fixedcut_norway.json"
        self.permutated_pvals_fixedcut_norway_csv = perm_dir / "permutation_adjusted_pvals_fixedcut_norway.csv"
        self.permutated_pvals_esspenalized_adaptive_norway_json = perm_dir / "experiment__esspenalized_adaptive_permutated_pvals_norway.json"
        self.permutated_pvals_esspenalized_adaptive_norway_csv = perm_dir / "experiment__esspenalized_adaptive_permutation_adjusted_pvals_norway.csv"

        # --- TABLES
        self.cohort_characteristics_table_xlsx = clinvar_ana_dir / "cohort_characteristics_table.xlsx"
        self.auc_DSSprediction_for_various_trainEndpoints_xlsx = miscdir / "auc_DSS_for_various_pipelines.xlsx"
        self.results_from_univariate_tso500_gene_analyses_csv = miscdir / "results_from_univariate_tso500_gene_analyses.csv"
        self.results_from_univariate_nontso500_gene_analyses_csv = miscdir / "results_from_univariate_nontso500_gene_analyses.csv"
        self.table_S5_stiff_vaf_cutoffs_TCGA_LUAD_DSS_csv = miscdir / "table_S5_stiff_vaf_cutoffs_TCGA_LUAD_DSS.csv"

        # --- DISCOVERY FIGURES
        self.comut_plot_pdf = figdir / "comut_plot.pdf"
        self.mutation_rates_TNMILUAD_vs_TCGALUAD_pdf = figdir / "mutation_rates_TNMILUAD_vs_TCGALUAD.pdf"
        self.gene_selection_heatmap_crossval_pdf = figdir / "gene_selection_heatmap_crossval.pdf"
        self.sigmoid_curves_showign_VAF_relapse_relationship_for_flowdiagram_pdf = figdir / "sigmoid_curves_showign_VAF_relapse_relationship_for_flowdiagram.pdf"
        self.hazard_ratio_curves_for_flowdiagram_pdf = figdir / "hazard_ratio_curves_for_flowdiagram.pdf"
        self.auc_as_fcn_of_n_genes_selected_pdf = auc_dir / "auc_as_fcn_of_n_genes_selected.pdf"
        self.auc_as_fcn_of_n_genes_selected_trainNSCLC_testLUAD_pdf = auc_dir / "auc_as_fcn_of_n_genes_selected_trainNSCLC_testLUAD.pdf"
        self.auc_as_fcn_of_n_genes_selected_trainNSCLC_testLUSC_pdf = auc_dir / "auc_as_fcn_of_n_genes_selected_trainNSCLC_testLUSC.pdf"
        self.barplot_crossval_2yrauc_averaged_over_n_genes_pdf = bar_dir / "barplot_crossval_2yrauc_averaged_over_n_genes.pdf"
        self.barplot_crossval_2yrauc_averaged_over_n_genes_drivers_only_pdf = bar_dir / "barplot_crossval_2yrauc_averaged_over_n_genes_drivers_only.pdf"
        self.barplot_crossval_2yrauc_averaged_over_n_genes_trainNSCLC_testLUAD_pdf = bar_dir / "barplot_crossval_2yrauc_averaged_over_n_genes_trainNSCLC_testLUAD.pdf"
        self.barplot_crossval_2yrauc_averaged_over_n_genes_trainNSCLC_testLUSC_pdf = bar_dir / "barplot_crossval_2yrauc_averaged_over_n_genes_trainNSCLC_testLUSC.pdf"
        self.barplot_crossval_TrainDevTTR_evalDSS_pdf = bar_dir / "barplot_crossval_TrainDevTTR_evalDSS.pdf"
        self.barplot_crossval_TrainDevTTR_evalOS_pdf = bar_dir / "barplot_crossval_TrainDevTTR_evalOS.pdf"
        self.barplot_crossval_featurization_used_in_GeneSelect_pdf = bar_dir / "barplot_crossval_featurization_used_in_GeneSelect.pdf"
        self.barplot_crossval_comparing_oncofilter_strategies_pdf = bar_dir / "barplot_crossval_comparing_oncofilter_strategies.pdf"
        self.get_path_barplot_ensemble_L2penalty_optim = lambda penalty: l2_dir / f"barplot_ensemble_L2penalty_optim_{float_as_str(penalty)}.pdf"
        self.get_path_auc_as_fcn_of_n_genes_selected_L2penalty_optim = lambda penalty: l2_dir / f"auc_as_fcn_of_n_genes_selected_L2penalty_optim_{float_as_str(penalty)}.pdf"

        # --- TCGA FIGURES
        self.roc_TCGAval__NSCLCtrain_LUADeval_PFI_pdf = tcga_figdir / "roc_TCGAval__NSCLCtrain_LUADeval_PFI.pdf"
        self.roc_TCGAval__NSCLCtrain_LUADeval_OS_pdf = tcga_figdir / "roc_TCGAval__NSCLCtrain_LUADeval_OS.pdf"
        self.roc_TCGAval__NSCLCtrain_LUADeval_DSS_pdf = tcga_figdir / "roc_TCGAval__NSCLCtrain_LUADeval_DSS.pdf"
        self.roc_TCGAval__NSCLCtrain_NSCLCeval_DSS_pdf = tcga_figdir / "roc_TCGAval__NSCLCtrain_NSCLCeval_DSS.pdf"
        self.roc_TCGAval__NSCLCtrain_LUSCeval_DSS_pdf = tcga_figdir / "roc_TCGAval__NSCLCtrain_LUSCeval_DSS.pdf"
        self.roc_TCGAval__LUADtrain_LUADeval_DSS_pdf = tcga_figdir / "roc_TCGAval__LUADtrain_LUADeval_DSS.pdf"
        self.coef_TCGAval__NSCLCtrain_LUADeval_pdf = tcga_figdir / "coef_TCGAval__NSCLCtrain_LUADeval.pdf"
        self.km_TCGAval__NSCLCtrain_LUADeval_DSS_pdf = tcga_figdir / "km_TCGAval__NSCLCtrain_LUADeval_DSS.pdf"
        self.km_TCGAval__NSCLCtrain_LUSCeval_DSS_pdf = tcga_figdir / "km_TCGAval__NSCLCtrain_LUSCeval_DSS.pdf"
        self.km_TCGAval__LUADtrain_LUADeval_DSS_pdf = tcga_figdir / "km_TCGAval__LUADtrain_LUADeval_DSS.pdf"
        self.get_tcga_kaplanmeier_fig_dump_path = lambda pipeline_tag, endpoint: tcga_figdir / f"km_TCGAval__{pipeline_tag}__{endpoint}.pdf"
        self.heatmap_TCGAval__NSCLCtrain_LUADeval_DSS_pdf = tcga_figdir / "heatmap_TCGAval__NSCLCtrain_LUADeval_DSS.pdf"
        self.heatmap_TCGAval__NSCLCtrain_LUSCeval_DSS_pdf = tcga_figdir / "heatmap_TCGAval__NSCLCtrain_LUSCeval_DSS.pdf"
        self.heatmap_TCGAval__LUADtrain_LUADeval_DSS_pdf = tcga_figdir / "heatmap_TCGAval__LUADtrain_LUADeval_DSS.pdf"
        self.incremental_gene_auc_TCGAval__NSCLCtrain_LUADeval_pdf = tcga_figdir / "incremental_gene_auc_TCGAval__NSCLCtrain_LUADeval.pdf"
        self.scatter_vaf_cutoff_concordance_norway_vs_TCGA_pdf = tcga_figdir / "scatter_vaf_cutoff_concordance_norway_vs_TCGA.pdf"

    @classmethod
    def get_full_path(cls, path):
        return cls.root / path

    @classmethod
    def create_dir(cls, p):
        path_full = cls.get_full_path(p)
        dirpath = os.path.dirname(path_full)
        if dirpath != "":
            os.makedirs(dirpath, exist_ok=True)

    @classmethod
    def get_full_path_and_create_dir(cls, path_relative):
        """Full path, creating the parent directory when it is missing."""
        path_full = cls.get_full_path(path_relative)
        cls.create_dir(path_full)
        return path_full

    @classmethod
    def dump_to_json(cls, dump_dict: dict, path: str, print_confirm=False):
        """Write a JSON file, converting numpy values to lists."""
        full_path = cls.get_full_path_and_create_dir(path)
        dump_dict = convert_ndarrays_to_list(dump_dict)
        with open(full_path, mode="w", encoding="utf-8") as json_file:
            json.dump(dump_dict, json_file, sort_keys=True, indent=4, cls=NpEncoder)
            silent_print(f"dumped to {full_path}") if print_confirm else None

    @classmethod
    def dump_to_csv(cls, data: Union[dict, pd.DataFrame], path) -> None:
        """Write a dataframe, or a dict converted to one."""
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        full_path = cls.get_full_path_and_create_dir(path)
        if os.path.isdir(full_path):
            full_path = full_path / "temp.csv"
        data.to_csv(full_path)
        silent_print(f"df dumped to {full_path}")

    @classmethod
    def dump_fig(cls, fig: Figure, path: str | Path, dual_dump=False, dpi=None, verbose=True):
        """Save a figure. ``dual_dump`` also writes a PNG next to it."""
        path = Path(path)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        path_print(verbose, f"Dumping figure to {path}")
        if dual_dump:
            path_print(verbose, f"Dumping figure to {path.with_suffix('.png')}")
            fig.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight", transparent=True)

    @classmethod
    def load_yaml(cls, path: str, print_confirm=False) -> dict:
        full_path = cls.get_full_path(path)
        with open(full_path, mode="r", encoding="utf-8") as jfl:
            yaml_loaded = yaml.safe_load(jfl)
            silent_print(f"Loading {full_path}") if print_confirm else None
        return yaml_loaded

    @classmethod
    def load_json(cls, path: str, print_confirm=False) -> dict:
        full_path = cls.get_full_path(path)
        with open(full_path, mode="r", encoding="utf-8") as jfl:
            loaded = json.load(jfl)
            silent_print(f"Loading {full_path}") if print_confirm else None
        return loaded


def path_print(verbose, *args, sep=" "):
    if verbose:
        silent_print(*args, sep=sep)


def convert_ndarrays_to_list(item) -> dict:
    """Replace numpy arrays in a dict so it can be written as JSON."""
    if isinstance(item, dict):
        for key, value in item.items():
            if isinstance(value, np.ndarray):
                item[key] = list(value)
        return item
    return item


def float_as_str(penalty):
    """``0.1`` becomes ``0p1``, for filenames."""
    return f"{penalty:g}".replace(".", "p")


class NpEncoder(json.JSONEncoder):
    """Encode numpy scalars and arrays as JSON numbers and lists."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


PATHS = Paths()
