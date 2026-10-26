"""Design matrix for the Paper 1 mutation Cox models: one column per selected gene."""

from typing import List

import pandas as pd


class BaseDesignMatrix:
    def __init__(self):
        self.model_variables: List[str] = []

    def prepare_modelfit_df(
        self,
        genomic_modelfit_df: pd.DataFrame | None = None,
        fit_new_data_transforms: bool = False,
        red_id_list: list[str] | None = None,
        clinical_df=None,
    ) -> pd.DataFrame:
        raise NotImplementedError

    def inferr_model_variables(self, modelfit_df) -> List[str]:
        """Columns other than the patient id and the Cox outcome."""
        return [c for c in modelfit_df.columns if c not in ["RED_ID", "time_to_event", "event_observed"]]


class Genomic(BaseDesignMatrix):
    """Selected genes as they are, with no clinical variables added."""

    def prepare_modelfit_df(
        self,
        genomic_modelfit_df,
        fit_new_data_transforms=False,
        red_id_list: list[str] = None,
        clinical_df=None,
    ) -> pd.DataFrame:
        """Return the gene columns unchanged."""
        self.model_variables = self.inferr_model_variables(genomic_modelfit_df)
        return genomic_modelfit_df
