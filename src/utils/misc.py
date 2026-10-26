import pandas as pd
from utils.display import *
from matplotlib.lines import Line2D


def replace_hyphen(gene: str) -> str:
    return gene.replace("-", "_")


def get_rows_where_nyear_relapse_is_defined(modelfit_df, n_years=2) -> pd.Series:
    """Returns series of True where the target is defined."""
    return (modelfit_df["event_observed"] == 1) | (modelfit_df["time_to_event"] >= n_years * 365)


# ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤
# ¤¤ DATAFRAME AGGREGATION AND DISPLAY FUNCTIONS ¤¤
# ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤


def add_text_to_legend(text, ax):
    extra_legend_text = Line2D([0], [0], linestyle="none", color="none", label=text)
    _ = ax.legend(handles=ax.get_legend_handles_labels()[0] + [extra_legend_text])
    handles, labels = ax.get_legend_handles_labels()
    handles.append(extra_legend_text)
    ax.legend(handles=handles)
    return handles
