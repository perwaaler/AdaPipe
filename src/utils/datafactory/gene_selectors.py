"""Logic that determines which genes enter the Cox model in each
cross-validation fold.

Top-k by softcut p-value is the discovery default. A fixed list is used when the
genes chosen in Norway are frozen.
"""

from dataclasses import dataclass
from typing import Callable
from typing import List
from typing import Union

from utils.variablesearch.variable_searcher import VariableSearcher
import pandas as pd


# ---------- GENE SELECTORS -----------------------------------------------------------------------------------------
@dataclass
class GeneSelectorBase:
    """Picks genes from one fold's cutoff summary."""

    def select_genes(self, varsearcher_i: VariableSearcher) -> List[str]:
        """Gene symbols for this fold."""
        raise NotImplementedError

    def get_subset_of_gene_stats_df_with_vaf_cutoff_estimate(self, varsearcher_i: VariableSearcher) -> pd.DataFrame:
        """Cutoff summary, dropping genes that have no cutoff."""
        return varsearcher_i.summary_best_models_df.dropna(subset=["vaf_cutoff"]).copy()

    def sort_rows_with_score_column(self, summary_df) -> pd.DataFrame:
        """Sort so the best score is first. Lower is better when score moves with the p-value."""
        assert "score" in summary_df.columns, "column 'score' not found in summary_df; create score colum first!"   # fmt: skip
        positive_correlation_with_pvalue = summary_df[["p", "score"]].corr().iloc[0, 1] > 0
        ascending = True if positive_correlation_with_pvalue else False
        return summary_df.sort_values("score", ascending=ascending)


@dataclass
class ManualGeneSelection(GeneSelectorBase):
    """A fixed gene list. A list of lists is one list per fold.

    Genes with no cutoff in that fold are dropped.
    """

    selection: Union[List[str], List[List[str]]]

    def __post_init__(self):
        self.list_of_genes_is_static = True
        if isinstance(self.selection[0], list):
            self.list_of_genes_is_static = False

    def select_genes(self, varsearcher_i: VariableSearcher, cv_idx=None) -> List[str]:
        # Check if there is a separate list for each CV-fold or just a single static list
        if self.list_of_genes_is_static:
            selected_genes = self.selection
        else:
            assert cv_idx is not None
            selected_genes = self.selection[cv_idx]
        summary_df = self.get_subset_of_gene_stats_df_with_vaf_cutoff_estimate(varsearcher_i)
        gene_with_cutoff = summary_df.index
        # Find genes that have cutoff
        selected_genes = list(set(gene_with_cutoff).intersection(selected_genes))   # fmt: skip
        assert len(selected_genes) > 0, f"No genes selected!"
        return selected_genes


@dataclass
class TopKBestGenes(GeneSelectorBase):
    """Keep the k best genes.

    ``gene_rater`` is a summary column (``p_softcut`` in the paper) or a function of one row. ``minimal_criteria`` drops genes before that cut. ``gene_selection_pool`` limits the candidates (TSO500 for the heatmap).
    """

    k: int = 20
    gene_rater: str | Callable[[pd.Series], float] = "p"
    minimal_criteria: Callable[[pd.Series], float] | None = None
    gene_selection_pool: list[str] = None

    def __post_init__(self):
        gene_rater_val = self.gene_rater
        if isinstance(gene_rater_val, str):
            self.gene_rater = lambda row: row[gene_rater_val]

    def select_genes(self, varsearcher_i: VariableSearcher, cv_idx=None) -> List[str]:
        summary_df = self.get_subset_of_gene_stats_df_with_vaf_cutoff_estimate(varsearcher_i)
        # Use the gene rating function to rankorder genes and select the top k best
        summary_df["score"] = summary_df.apply(self.gene_rater, axis=1)
        summary_df_sorted = self.sort_rows_with_score_column(summary_df)
        if self.minimal_criteria is not None:
            candidates_sorted_summary_df = summary_df_sorted[self.minimal_criteria(summary_df_sorted)]
            assert len(candidates_sorted_summary_df) > 0, f"No genes meet minimal criteria!\n\nSorted info (first 10 rows)\n{summary_df_sorted.head(10)}"   # fmt: skip
        else:
            candidates_sorted_summary_df = summary_df_sorted

        gene_candidates = candidates_sorted_summary_df.index
        if self.gene_selection_pool:
            gene_candidates = [x for x in gene_candidates if x in self.gene_selection_pool]
        # Grab top k best genes
        selected_genes = gene_candidates[: self.k]

        return selected_genes
        selected_genes = forced_genes + remaining_genes[:n_remaining_slots]

        return selected_genes
