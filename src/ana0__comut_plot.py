# %% ---------- SETUP -----------------------------------------------------------------------------------------
import warnings
from utils.misc import *
from utils.display import *
from utils.prepare_data.variant_data import VariantData
from utils.prepare_data.prepare_data import DataBuilder
from utils.paths import Paths

# Disable deprecation warning that appears when I use groupby(...).apply(...)
warnings.filterwarnings("ignore", message=".*DataFrameGroupBy.apply operated on the grouping columns.*", category=DeprecationWarning)  # fmt: skip

pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 100)

# %% ---------- LOAD RAW GENOMIC DATA -----------------------------------------------------------------------------------------
# PREPARE DRAGEN DATAFRAME
raw_variant_df = VariantData().get_gatk()
clinical_df = DataBuilder().get_clinical_data()

raw_variant_df["EXON_NUMBER"] = raw_variant_df["CDS_CHANGE"].apply(VariantData.extract_exon_from_cds_change)
raw_variant_df["SUPPORTING_READS"] = (raw_variant_df["VAF_TUMOR"] * raw_variant_df["DP_TUMOR"]).astype(int)  # fmt: skip

# %% ---------- FILTER VARIANTS -----------------------------------------------------------------------------------------
dumpfigure = False
variants_df = VariantData().filter_variants(
    gatk0_raw_df=raw_variant_df,
    consequence_col="CONSEQUENCE",
    min_onco_score=3,
    min_vaf=0.05,
    min_support=5,
    oncokb_oncogenic_only=True,
)
variants_df = VariantData().add_red_sampleid_col(variants_df)
variants_df["pipeline"] = "Mutect2"

# ADD CLINICAL DATA
variants_df = variants_df.merge(
    clinical_df[["RED_ID", "histology", "pstage"]], on="RED_ID", how="left"
)
# RESTRICT TO NORWAY COHORT
red_id_to_plot = Paths.load_json(Paths().red_id_norway_path)
variants_df = variants_df.query(f"RED_ID.isin({red_id_to_plot})").copy()

# ---------- COMMUT PLOT -----------------------------------------------------------------------------------------

from comut import comut
import palettable

# from utils.comut_plot_utils import prepare_consequence_column
from ana.ana1__comut_plot import sort_samples
from ana.ana1__comut_plot import count_mutations_per_sample
from ana.ana1__comut_plot import add_barplots
from ana.ana1__comut_plot import add_mutatype_columns
from ana.ana1__comut_plot import add_clinical_categories
from ana.ana1__comut_plot import Maps

vivid_10 = palettable.cartocolors.qualitative.Vivid_10.mpl_colors
red_id_to_plot = red_id_to_plot[:]
n_genes_to_show = 20

# Rename samples to format "Patient i"
new_sample_names = {sample_id: f"{i+1}" for i, sample_id in enumerate(variants_df["RED_ID"].unique())}   # fmt: skip
# Find 20 most common gene mutations
variants_df["sample"] = variants_df["RED_ID"].replace(new_sample_names)
# GET LIST OF GENES (ORDERED BY MUTA COUNT) TO PLOT IN COMUT PLOT
df = variants_df.query("SYMBOL!='NONE'")
n_mutas_per_gene = df.groupby(["SYMBOL"])["RED_ID"].nunique()
genes_sorted = list(n_mutas_per_gene.sort_values().index[-n_genes_to_show:])

# variant_df1["CONSEQUENCE_GROUP"] = prepare_consequence_column(variant_df1)
variants_df, multihit_df = add_mutatype_columns(variants_df)

# RENAME TO MEET EXPECTATIONS OF COMUT
mutation_df = variants_df.copy()
mutation_df["value"] = mutation_df["variant_type"]
mutation_df["category"] = mutation_df["SYMBOL"]

comut_plotter = comut.CoMut()


# Apply ordering to the comut plot
comut_plotter.samples = sort_samples(mutation_df, priority_genes=genes_sorted[::-1])

# ---------- CLINICAL VARIABLES -----------------------------------------------------------------------------------------
add_clinical_categories(comut_plotter, variants_df, clinical_df)

# ---------- CREATE MAIN MUTATION MATRIX (X: SAMPLE, Y: GENE, VALUE: MUTATED T/F) -----------------------------------------------------------------------------------------
mutation_df["value"] = mutation_df["value"].replace(Maps.Spelling.consequence)
comut_plotter.add_categorical_data(
    mutation_df[["value", "category", "sample"]],
    name="Consequence",
    category_order=genes_sorted,
    # mapping=Maps.Color.consequence,
    mapping={**Maps.Color.consequence, "Absent": {"facecolor": "#EEEEEE"}},
    tick_style="italic",
)

# ---------- BARPLOTS ---------------------------------------------q--------------------------------------------
mutation_hist_df, _ = add_barplots(comut_plotter, mutation_df, genes_sorted)

# ---------- PLOT -----------------------------------------------------------------------------------------
x_padding = 0.1  # the x distance between patches in comut
y_padding = 0.1  # the y distance between patches in comut
tri_padding = 0.05  # the distance between triangles in comut
comut_plotter.plot_comut(
    x_padding=x_padding,
    y_padding=y_padding,
    tri_padding=tri_padding,
    hspace=0.03,
    figsize=(20, 10),
    widths=[0.5, 5],
)

# --- LEGEND
from ana.ana1__comut_plot import add_legend
add_legend(comut_plotter, xpos=0.25, ypos=0.1)

# Remove patient ticks
s = comut_plotter.axes["Consequence"].set_xticklabels([])
# PLOT ONE DOT FOR EACH PATIENT
ax = comut_plotter.axes["Consequence"]
x_ticks = ax.get_xticks()
ax.scatter(x_ticks, [ax.get_ylim()[0]] * len(x_ticks), color="black", s=1)
comut_plotter.axes["Consequence"].set_xlabel("Participant")
comut_plotter.axes["Consequence"].set_facecolor("#F5F3F3")

# --- ADD PERCENTAGE LABELS NEXT TO BARS
ax_left = comut_plotter.axes["Mutated samples"]
# x positions for labels: sum of the stacked bars per gene
patch_height = list(ax_left.patches)[0].get_height()
variant_types = mutation_df["value"].unique()
for idx, row in mutation_hist_df.iterrows():
    total_width = row[variant_types].sum()  # sum all Consequence columns
    y = idx  # y-position corresponds to row index
    ax_left.text(
        total_width,
        y + patch_height / 2,
        f"{total_width / mutation_df["sample"].nunique():.1%}",
        va="center",
        ha="left",
        fontsize=8,
    )

# --- DRAW A HORIZONTAL LINE SHOWING THE RANGE OF THE BARS
max_val = mutation_hist_df[[c for c in mutation_hist_df.columns if c != "category"]].sum(axis=1).max()
ax_left.hlines(y=0, xmin=0, xmax=max_val, color="black", linewidth=1)   # fmt: skip
# Add tick labels at 0 and max
ax_left.set_xticks([0, max_val])
ax_left.set_xticklabels([0, int(max_val)])
ax_left.set_xlim(0, max_val)
ax_left.set_xlabel("No. samples")

if dumpfigure:
    Paths.dump_fig(fig=comut_plotter.figure, path=Paths().comut_plot_pdf)