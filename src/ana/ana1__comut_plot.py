import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import palettable
from typing import Literal
from typing import List
from typing import Tuple
from comut import comut
from dataclasses import dataclass
from typing import ClassVar


@dataclass
class Maps:
    """Container for all maps used in plotting and labeling."""

    @dataclass
    class Spelling:
        """Mapping raw data strings to print-friendly names (lowercase)."""

        consequence: ClassVar[dict] = {
            "multihit": "Multi hit",
            "frameshift_insertion": "Frameshift insertion",
            "frameshift_deletion": "Frameshift deletion",
            "inframe_deletion": "Inframe deletion",
            "splice_site": "Splice site",
            "missense": "Missense",
            "nonsense": "Nonsense",
        }
        consequence_tmb: ClassVar[dict] = {f"_{k}": f"* {v}" for k, v in consequence.items()}
        pstage: ClassVar[dict] = {
            "1": "Stage I",
            "2": "Stage II",
            "3": "Stage III",
        }
        histology: ClassVar[dict] = {
            "LUAD": "LUAD",
            "LUSC": "LUSC",
            "other": "Other",
        }
        smoking: ClassVar[dict] = {
            "never": "Never",
            "previous": "Previous",
            "current": "Current",
        }

    @dataclass
    class Color:
        """Color mappings keyed by the print-friendly names."""

        consequence: ClassVar[dict] = {
            "Multi hit": "#7806c9",
            "Nonsense": "#06c97b",
            "Missense": "#0667c9",
            "Frameshift insertion": "#c98106",
            "Frameshift deletion": "#a41f24",
            "Splice site": "#915000",
            "Inframe deletion": "#ef63eb",
        }
        consequence_tmb: ClassVar[dict] = {f"* {k}": v for k, v in consequence.items()}
        pstage: ClassVar[dict] = {
            "Stage I": "#c400ad",
            "Stage II": "#2ea001",
            "Stage III": "#bdb000",
        }
        histology: ClassVar[dict] = {
            "LUAD": "#990505",
            "LUSC": "#dd6700",
            "Other": "#0ef7ff",
        }
        smoking: ClassVar[dict] = {
            "Never": "#d9d9d9",
            "Previous": "#737373",
            "Current": "#000000",
        }


def add_mutatype_columns(variant_df) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Retursn dataframe with column 'variant_type' and multihit dataframe
    indicating number of mutations per gene."""
    types_sorted_by_priority = [
        "multihit",
        "frameshift_insertion",
        "frameshift_deletion",
        "nonsense",
        "splice_site",
        "inframe_deletion",
        "missense",
    ]
    consequence = variant_df["CONSEQUENCE"]
    variant_class = variant_df["VARIANT_CLASS"]

    # --- CREATE VARIANT TYPE COLUMN FOR COMUT PLOT
    variant_df["missense"] = consequence.str.contains("missense")
    variant_df["splice_site"] = consequence.str.contains("splice_acceptor_variant") | consequence.str.contains("splice_donor_variant")   # fmt: skip
    variant_df["frameshift_deletion"] = consequence.str.contains("frameshift_variant") & variant_class.str.contains("deletion")   # fmt: skip
    variant_df["frameshift_insertion"] = consequence.str.contains("frameshift_variant") & variant_class.str.contains("insertion")   # fmt: skip
    variant_df["nonsense"] = consequence.str.contains("stop_gained")
    variant_df["inframe_deletion"] = consequence.str.contains("inframe_deletion")

    # --- CREATE MULTIHIT DATAFRAME
    multihit_df = variant_df.groupby(["RED_ID", "SYMBOL"]).size().reset_index(name="hits")
    multihit_df["multihit"] = multihit_df["hits"] > 1
    variant_df = variant_df.merge(multihit_df[["RED_ID", "SYMBOL", "multihit"]], on=["RED_ID", "SYMBOL"], how="left")

    def get_highest_priority_type(row):
        variant_type = "other"
        for variant_type in types_sorted_by_priority:
            if row[variant_type] == True:
                return variant_type
        return variant_type

    variant_df["variant_type"] = variant_df[types_sorted_by_priority].apply(get_highest_priority_type, axis=1)

    return variant_df, multihit_df


def prepare_consequence_column(mutation_df) -> pd.Series:
    """Create consequence column to be used for plotting."""
    consequence_mapping = {
        "Missense": ["missense_variant"],
        "Truncating": ["frameshift_variant", "stop_gained", "stop_lost"],
        "In-Frame": ["inframe_insertion"],
        "Splice site": ["splice_acceptor_variant", "splice_donor_variant"],
        "Multi-Hit": [
            "frameshift_variant&splice_region_variant",
            "missense_variant&splice_region_variant",
            "splice_region_variant&stop_gained",
        ],
    }

    def find_consequence_group(consequence):
        for consequence_group, consequence_items in consequence_mapping.items():
            if consequence in consequence_items:
                return consequence_group
        return "nothing"

    return mutation_df["CONSEQUENCE"].apply(find_consequence_group)


def sort_samples(mutation_df, priority_genes):
    """Sort samples first by histology, then by presence of prioritized genes."""

    # Map samples to histology and mutated genes
    sample_to_histology_map = mutation_df.groupby("sample")["histology"].first().to_dict()
    mutated_genes_per_sample = mutation_df.groupby("sample")["category"].agg(set).to_dict()

    # Define histology priority (higher value means higher priority in rankings)
    histology_order = {"LUAD": 3, "LUSC": 2, "other": 1}

    def sort_key(sample):
        histo_rank = histology_order[sample_to_histology_map[sample]]
        genes_mutated_in_sample = mutated_genes_per_sample.get(sample, set())

        # Build tuples used to order sample after grouping by histology
        # - lexinographic sorting: in case of a tie, looks at next value in the
        #   tuple to determine order
        gene_flags = tuple(int(gene in genes_mutated_in_sample) for gene in priority_genes)

        return (histo_rank,) + gene_flags

    # Sort samples
    sorted_samples = sorted(mutation_df["sample"].unique(), key=sort_key, reverse=True)

    return sorted_samples


def count_mutations_per_sample(mutation_df) -> pd.DataFrame:
    mutation_count_per_sample = mutation_df.query("category!='NONE'").groupby("sample")["category"].nunique()   # fmt: skip
    return mutation_count_per_sample.reset_index().rename(columns={"category": "n mutations"})   # fmt: skip


def add_clinical_categories(comut_plotter: comut.CoMut, variant_df, clinical_df):
    # --- ADD SMOKING ROW
    smoking_df = clinical_df.merge(variant_df[["RED_ID", "sample"]], on="RED_ID", how="left")
    smoking_df = smoking_df.query("sample in @comut_plotter.samples").copy()
    smoking_df["value"] = smoking_df["smoking_status"].replace(Maps.Spelling.smoking)
    smoking_df = smoking_df[["sample", "value"]].drop_duplicates(subset="sample").copy()
    smoking_df["category"] = "Smoking"
    comut_plotter.add_categorical_data(
        smoking_df,
        name="Smoking",
        mapping=Maps.Color.smoking,
    )
    # --- ADD PSTAGE ROW
    pstage_df = clinical_df.merge(variant_df[["RED_ID", "sample"]], on="RED_ID", how="left")
    pstage_df = pstage_df.query("sample in @comut_plotter.samples").copy()
    pstage_df["value"] = pstage_df["pstage"].replace(Maps.Spelling.pstage)
    pstage_df = pstage_df[["sample", "value"]].drop_duplicates(subset="sample").copy()
    pstage_df["category"] = "pStage"
    comut_plotter.add_categorical_data(
        pstage_df,
        name="pStage",
        mapping=Maps.Color.pstage,
        value_order=["Stage I", "Stage II", "Stage III"],
    )
    # --- ADD HISTOLOGY ROW
    histology_df = clinical_df.merge(variant_df[["RED_ID", "sample"]], on="RED_ID", how="left")
    histology_df = histology_df.query("sample in @comut_plotter.samples").copy()
    histology_df["value"] = histology_df["histology"].replace(Maps.Spelling.histology)
    histology_df = histology_df[["sample", "value"]].drop_duplicates(subset="sample").copy()
    histology_df["category"] = "histology"
    comut_plotter.add_categorical_data(
        histology_df,
        name="Histology",
        mapping=Maps.Color.histology,
    )


def add_barplots(comut_plotter: comut.CoMut, mutation_df, genes_sorted):
    # ---------- BARPLOTS -----------------------------------------------------------------------------------------
    # --- BARPLOT ON THE LEFT (PER-GENE MUTATION RATE)

    mutation_hist_df = mutation_df.groupby(["category", "value"])["sample"].nunique().unstack(fill_value=0)
    mutation_hist_df = mutation_hist_df.loc[genes_sorted, :].reset_index()
    comut_plotter.add_side_bar_data(
        mutation_hist_df,
        paired_name="Consequence",  # aligns with your main matrix
        name="Mutated samples",
        position="left",
        mapping=Maps.Color.consequence,
        stacked=True,  # <- this enables stacked bars
        xlabel="Mutated\nsamples [%]",
    )

    # --- ADD BARPLOT ON TOP: TMB PER SAMPLE
    purp_7 = palettable.cartocolors.sequential.Purp_7.mpl_colormap
    df = mutation_df.copy()
    df["variant_type"] = mutation_df["variant_type"].apply(lambda x: f"_{x}")
    df["variant_type"] = df["variant_type"].replace(Maps.Spelling.consequence_tmb)
    tmb_per_mutatype_df = df.groupby(["sample", "variant_type"]).size().unstack(fill_value=0)
    tmb_per_mutatype_df = tmb_per_mutatype_df.loc[comut_plotter.samples, :].reset_index()
    comut_plotter.add_bar_data(
        tmb_per_mutatype_df.rename(columns={"variant_type": "value"}),
        name="",  # Empty string since we dont want a legend for this barplot
        mapping=Maps.Color.consequence_tmb,
        stacked=True,
        ylabel="TMB",
    )

    return mutation_hist_df, tmb_per_mutatype_df


def add_legend(
    comut_plotter: comut.CoMut,
    xpos=0.0,
    ypos=-0.15,
    x_separation=0.15,
):
    # Remove labels for TMB plot since these are the same as in left barplot
    # comut_plotter.add_unified_legend(ncol=1, bbox_to_anchor=(1, 1), ignored_values=ignored_labels)
    axes = comut_plotter.axes
    fig = comut_plotter.figure
    # --- CONSEQUENCE
    # Handl ignored labels
    ignored_labels = list(Maps.Color.consequence_tmb.keys()) + ["Absent"]
    handles1, labels1 = axes["Consequence"].get_legend_handles_labels()
    filtered = [(h, l) for h, l in zip(handles1, labels1) if l not in ignored_labels]
    if filtered:
        handles1, labels1 = zip(*filtered)
    leg1 = fig.legend(
        handles1,
        labels1,
        title="Consequence",
        loc="upper left",
        bbox_to_anchor=(xpos, ypos),  # horizontal positioning
        ncol=1,
        frameon=False,
    )
    # 2. Histology legend
    handles2, labels2 = axes["Histology"].get_legend_handles_labels()
    handles2, labels2 = zip(*[(h, l) for h, l in zip(handles2, labels2) if l != "Absent"])
    leg2 = fig.legend(
        handles2,
        labels2,
        title="Histology",
        loc="upper left",
        bbox_to_anchor=(xpos + x_separation, ypos),
        ncol=1,
        frameon=False,
    )
    # 3. pStage legend
    handles3, labels3 = axes["pStage"].get_legend_handles_labels()
    handles3, labels3 = zip(*[(h, l) for h, l in zip(handles3, labels3) if l != "Absent"])
    handles3, labels3 = list(handles3), list(labels3)
    stage_order = ["Stage I", "Stage II", "Stage III"]
    order = sorted(range(len(labels3)), key=lambda i: stage_order.index(labels3[i]))
    handles3 = [handles3[i] for i in order]
    labels3 = [labels3[i] for i in order]
    leg3 = fig.legend(
        handles3,
        labels3,
        title="Pathological stage",
        loc="upper left",
        bbox_to_anchor=(xpos + 2 * x_separation, ypos),
        ncol=1,
        frameon=False,
    )
    # 4. Smoking legend
    handles4, labels4 = axes["Smoking"].get_legend_handles_labels()
    handles4, labels4 = zip(*[(h, l) for h, l in zip(handles4, labels4) if l != "Absent"])
    handles4, labels4 = list(handles4), list(labels4)
    smoking_order = ["Never", "Previous", "Current"]
    order = sorted(range(len(labels4)), key=lambda i: smoking_order.index(labels4[i]))
    handles4 = [handles4[i] for i in order]
    labels4 = [labels4[i] for i in order]
    leg4 = fig.legend(
        handles4,
        labels4,
        title="Smoking",
        loc="upper left",
        bbox_to_anchor=(xpos + 3 * x_separation, ypos),
        ncol=1,
        frameon=False,
    )
