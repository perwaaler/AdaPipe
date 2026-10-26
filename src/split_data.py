"""Write the discovery-cohort cross-validation splits.

One 5-fold split for Norwegian NSCLC and one for Norwegian LUAD.
Folds are stratified on 3-year relapse and availability of genomic data.
Also writes the Norway RED_ID list.
"""

import hashlib
from typing import Literal

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from utils.display import *
from utils.paths import Paths
from utils.prepare_data.prepare_data import DataBuilder


def cohort_red_ids(clinical_df: pd.DataFrame, subset: Literal["Norway", "Norway, LUAD"]):
    """RED_IDs and output path for one discovery split."""
    red_id_subset = Paths.load_yaml(Paths().red_id_norway_path)
    dump_path = Paths().cv_splits_norway_path
    if subset == "Norway, LUAD":
        red_id_luad = Paths.load_yaml(Paths().red_id_luad_json)
        red_id_subset = list(set(red_id_subset).intersection(red_id_luad))
        dump_path = Paths().cv_splits_norway_luad_path
    red_id_subset = clinical_df.query(f"RED_ID.isin({red_id_subset})")["RED_ID"]
    return red_id_subset, dump_path


def gen_splits(clinical_df, red_id_subset, dump_path, subset, n_splits=5, random_state=1):
    """Stratified 5-fold split. Balances 3-year relapse and genomic-data availability."""
    cv_split_indices = {"train": [], "test": [], "split_hash": []}
    red_id_genomic = Paths.load_json(Paths().red_id_genomic_data_json)
    devset_df = clinical_df.query(f"RED_ID.isin({list(red_id_subset)})").reset_index(drop=True).copy()

    devset_df["relapse_3yr_defined"] = devset_df["event_observed"] | ((devset_df["event_observed"] == False) & (devset_df["time_to_event"] > 365 * 3))   # fmt: skip
    devset_df["relapse_3yr"] = devset_df[["time_to_event", "relapse_3yr_defined"]].apply(
        lambda row: (row["time_to_event"] < 365 * 3) if row["relapse_3yr_defined"] else None, axis=1
    )
    devset_df["has_genomic_data"] = devset_df["RED_ID"].isin(red_id_genomic)

    stratif_variables = ["relapse_3yr", "has_genomic_data"]
    stratif_variable = ""
    for i, var in enumerate(stratif_variables):
        if i == 0:
            stratif_variable += f"{var}_" + devset_df[var].astype(str)
        else:
            stratif_variable += f"__{var}_" + devset_df[var].astype(str)
    devset_df["stratif_var"] = stratif_variable

    cv_splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_idx, test_idx in cv_splitter.split(devset_df, devset_df["stratif_var"]):
        cv_split_indices["train"].append(devset_df["RED_ID"].iloc[train_idx].values)
        cv_split_indices["test"].append(devset_df["RED_ID"].iloc[test_idx].values)
        cv_split_indices["split_hash"].append(get_split_identifier(devset_df["RED_ID"].iloc[test_idx].values))

    for i, (train_id, test_id) in enumerate(zip(cv_split_indices["train"], cv_split_indices["test"])):
        train_id = list(train_id)
        test_id = list(test_id)
        blue_print(f"\n--- Fold {i} ---")
        train_df = devset_df.query(f"RED_ID in {train_id} and RED_ID in {red_id_genomic}")
        time_to_3yrelapse = train_df.query("relapse_3yr_defined==True")["time_to_event"]
        red_print("TRAIN")
        red_print("N cases where 3yr relapse is defined: ", time_to_3yrelapse.__len__(), f"({(time_to_3yrelapse<3*365).sum()}+, {(time_to_3yrelapse>3*365).sum()}-)")   # fmt: skip
        red_print("N relapses observed: ", train_df["event_observed"].sum())   # fmt: skip
        red_print("N with genomic data: ", train_df["has_genomic_data"].sum())   # fmt: skip
        train_df = devset_df.query(f"RED_ID in {test_id} and RED_ID in {red_id_genomic}")
        time_to_3yrelapse = train_df.query("relapse_3yr_defined==True")["time_to_event"]
        pink_print("TEST")
        pink_print("N cases where 3yr relapse is defined: ", time_to_3yrelapse.__len__(), f"({(time_to_3yrelapse<3*365).sum()}+, {(time_to_3yrelapse>3*365).sum()}-)")   # fmt: skip
        pink_print("N relapses observed: ", train_df["event_observed"].sum())   # fmt: skip
        pink_print("N with genomic data: ", train_df["has_genomic_data"].sum())   # fmt: skip
        assert set(train_id).intersection(test_id) == set(), "Test-train splits have overlapping elements!"

    white_print(f"\nMODEL DEVELOPENT SPLITS CREATED\nSubset: {subset}\nStats for last split:")
    red_print("Red: train")
    pink_print("Pink: test")
    red_print(f"n train: {len(red_id_subset.iloc[train_idx].values)}")
    pink_print(f"n test: {len(red_id_subset.iloc[test_idx].values)}")
    red_print(f"hash: {get_split_identifier(red_id_subset.iloc[test_idx].values)}")

    Paths.dump_to_json(dump_dict=cv_split_indices, path=dump_path)
    white_print("\n------ END -----")


def get_split_identifier(id_list):
    """Short hash of a fold's RED_IDs."""
    joined = ",".join(sorted(id_list))
    return hashlib.sha256(joined.encode()).hexdigest()[:10]


def dump_norway_red_ids(clinical_df):
    """Write the Norway RED_ID list."""
    red_id_norway = clinical_df.query("~is_in_holdout_set")["RED_ID"]
    Paths.dump_to_json(list(red_id_norway), path=Paths().red_id_norway_path)


if __name__ == "__main__":
    clinical_df = DataBuilder().get_clinical_data()
    for subset in ("Norway", "Norway, LUAD"):
        red_id_subset, dump_path = cohort_red_ids(clinical_df, subset)
        gen_splits(clinical_df, red_id_subset, dump_path, subset)
    dump_norway_red_ids(clinical_df)
