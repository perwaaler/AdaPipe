"""Discovery-cohort summary for Table 1."""

from collections import namedtuple

import pandas as pd
from scipy.stats import chi2_contingency
from scipy.stats import mannwhitneyu

from utils.display import *
from utils.paths import Paths
from utils.prepare_data.prepare_data import DataBuilder


def ana1__produce_cohort_characteristics_table():
    """Cohort characteristics table for the discovery cohort.

    LUAD pattern rows are checkbox counts (a patient can appear in more than
    one row). Percentages use LUAD n as the denominator and need not sum to 100.
    """
    # *** PAPER 1 TABLE ***
    group_micropapillary_with_other = True
    include_denmark = False

    clinical_df = DataBuilder().get_clinical_data().dropna(subset=["time_to_event"])
    clinical_df = clinical_df.query(f"RED_ID in {DataBuilder.get_red_id_genomic_data()}").copy()

    # RELABEL VALUES TO NICER NAMES
    clinical_df["pnstage"] = clinical_df["pnstage"].replace(
        {"pN0": "N0", "pN1": "N1 or N2", "pN2": "N1 or N2", "pNx": "Nx"}
    )
    clinical_df["pstage"] = clinical_df["pstage"].replace({"1": "Stage I", "2": "Stage II", "3": "Stage IIIA"})   # fmt: skip
    clinical_df["ecog_group"] = clinical_df["who_ps"].astype("Int64").map(
        lambda x: "≥2" if pd.notna(x) and x >= 2 else (str(round(x)) if pd.notna(x) else pd.NA)
    )
    clinical_df["ecog_group"] = pd.Categorical(clinical_df["ecog_group"], categories=["0", "1", "≥2"], ordered=True)
    clinical_df["smoking_status"] = clinical_df["smoking_status"].replace({"previous": "Former/current", "current": "Former/current", "never": "Never"})   # fmt: skip
    clinical_df["pstage"] = pd.Categorical(clinical_df["pstage"], categories=["Stage I", "Stage II", "Stage IIIA"], ordered=True)   # fmt: skip
    clinical_df["pnstage"] = pd.Categorical(clinical_df["pnstage"], categories=["N0", "N1 or N2", "Nx"], ordered=True)
    clinical_df["ptstage"] = pd.Categorical(clinical_df["ptstage"], categories=sorted(clinical_df["ptstage"].dropna().unique()), ordered=True)   # fmt: skip
    clinical_df["adjuvant_treatment"] = clinical_df["adjuvant_treatment"].fillna("None").str.capitalize()
    clinical_df["adjuvant_treatment"] = pd.Categorical(clinical_df["adjuvant_treatment"], categories=["None", "Chemotherapy", "Radiotherapy", "Both"], ordered=True)   # fmt: skip


    Result = namedtuple("Result", ["value_counts", "txt_df"])
    non_categorical_vars = ["age_at_surgery"]

    def get_data(df, var):
        if var in non_categorical_vars:
            txt_df = df[["age_at_surgery"]].agg(lambda x: f"{round(x.median())} ({round(x.min())}, {round(x.max())})")   # fmt: skip
            summary_metrics = None
        else:
            summary_metrics = df[var].value_counts(sort=False).dropna()
            counts = df[var].value_counts(sort=False).dropna()
            pcts = df[var].value_counts(sort=False, normalize=True).dropna()
            txt_df = counts.astype(str) + " " + pcts.apply(lambda x: f"({round(x * 100)})")   # fmt: skip

        txt_df = txt_df.to_frame(name="value")
        return Result(summary_metrics, txt_df)

    luad_pattern_cols = {
        "Acinar": ["luad_acinar"],
        "Lepidic": ["luad_lepidic"],
        "Papillary": ["luad_papillary"],
        "Solid": ["luad_solid"],
        "Other": ["luad_mucinous", "luad_other"],
    }
    if group_micropapillary_with_other:
        luad_pattern_cols["Other"] = luad_pattern_cols["Other"] + ["luad_micropapillary"]
    else:
        other_cols = luad_pattern_cols.pop("Other")
        luad_pattern_cols["Micropapillary"] = ["luad_micropapillary"]
        luad_pattern_cols["Other"] = other_cols

    def get_luad_pattern_data(df):
        luad = df.query("histology=='LUAD'")
        n = len(luad)
        counts = pd.Series(
            {label: luad[cols].eq(True).any(axis=1).sum() for label, cols in luad_pattern_cols.items()}
        )
        if n == 0:
            pcts = pd.Series(0.0, index=counts.index)
        else:
            pcts = counts / n
        txt_df = counts.astype(str) + " " + pcts.map(lambda p: f"({round(p * 100)})")
        return Result(counts, txt_df.to_frame(name="value"))

    formatted_dataframes = []
    counts_dataframes = []
    sizes = []
    if include_denmark:
        subsets = {
            "TNM-I Norway": clinical_df.query(f"country=='norway'"),
            "TNM-I Denmark": clinical_df.query(f"country=='denmark'"),
            "With relapse": clinical_df.query("event_observed==1"),
            "Without relapse": clinical_df.query("event_observed==0"),
            "Total": clinical_df,
        }
    else:
        subsets = {
            "TNM-I Norway": clinical_df.query(f"country=='norway'"),
            "With relapse": clinical_df.query("country=='norway' & event_observed==1"),
            "Without relapse": clinical_df.query("country=='norway' & event_observed==0"),
        }
    for subset_name in subsets:
        df = subsets[subset_name]
        pstage_results = get_data(df, "pstage")
        nstage_results = get_data(df, "pnstage")
        tstage_results = get_data(df, "ptstage")
        adjuv_chemo = get_data(df, "adjuvant_treatment")
        age_results = get_data(df, "age_at_surgery")
        sex_results = get_data(df, "sex")
        histology_results = get_data(df, "histology")
        luad_subtype_results = get_luad_pattern_data(df)
        smoking_results = get_data(df, "smoking_status")
        who_ps_results = get_data(df, "ecog_group")
        surgical_procedure_results = get_data(df.dropna(subset="surgical_procedure"), "surgical_procedure")   # fmt: skip
        inputs = {
            "pStage": pstage_results,
            "pN status": nstage_results,
            "pT status": tstage_results,
            "Age median, (range)": age_results,
            "Sex": sex_results,
            "Histology findings": histology_results,
            "LUAD subtype": luad_subtype_results,
            "Smoking": smoking_results,
            "ECOG": who_ps_results,
            "Surgery type": surgical_procedure_results,
            "Adjuvant treatment": adjuv_chemo, 
        }
        pretty_strings_df = pd.concat([x.txt_df for x in inputs.values()], keys=inputs.keys())
        valuecounts_df = pd.concat(
            [x.value_counts for x in inputs.values() if x.value_counts is not None],
            keys=[k for k, x in inputs.items() if x.value_counts is not None],
        )
        counts_dataframes.append(valuecounts_df)
        formatted_dataframes.append(pretty_strings_df)
        sizes.append(len(df))

    def _format_pvalue(p: float | None) -> str:
        if p is None:
            return ""
        if p < 0.001:
            return "<0.001"
        return f"{p:.3f}"

    def _chi2_pvalue(block: pd.DataFrame) -> float | None:
        block = block.fillna(0)
        block = block.loc[block.sum(axis=1) > 0, block.sum(axis=0) > 0]
        if block.shape[0] < 2 or block.shape[1] < 2:
            return None
        try:
            return chi2_contingency(block.T.values).pvalue
        except ValueError:
            return None

    # -- STATISTICAL TESTS
    # CATEGORICAL COUNTS (CHI SQUARED TEST)
    subset_labels = subsets.keys()
    counts_df = pd.concat(counts_dataframes, keys=subset_labels, axis=1).fillna(0)
    variable_pvalues: dict[str, float | None] = {}
    for key in set(counts_df.index.get_level_values(0)):
        if key in ["LUAD subtype", "Adjuvant treatment"]:
            variable_pvalues[key] = None
            orange_print("Skipping chi-squared for LUAD subtype: pattern rows overlap (checkbox counts)")
            continue
        pvalue = _chi2_pvalue(counts_df.loc[key])
        variable_pvalues[key] = pvalue
        if pvalue is None:
            orange_print(f"Skipping chi-squared for {key}: fewer than 2 non-empty rows/columns after dropping zeros")
            display(counts_df.loc[key].fillna(0))
            continue
        blue_print(f"Chi squared p-value for {key}: {pvalue:.7f}")
    # LOCATION OF CONTINUOUS DISTRIBUTIONS (MANN-WHITNEY U TEST)
    if include_denmark:
        age_x = clinical_df.query("country=='norway'")["age_at_surgery"]
        age_y = clinical_df.query("country=='denmark'")["age_at_surgery"]
        age_test_label = "Norway vs Denmark"
    else:
        age_x = subsets["With relapse"]["age_at_surgery"]
        age_y = subsets["Without relapse"]["age_at_surgery"]
        age_test_label = "with vs without relapse"
    age_pvalue = mannwhitneyu(age_x, age_y, alternative="two-sided").pvalue
    variable_pvalues["Age median, (range)"] = age_pvalue
    green_print(f"P-value Mann-Whitney U test for age ({age_test_label}): {age_pvalue:.3f}")

    headers = [f"{s}, n={n} (%)" for s, n in zip(subset_labels, sizes)]
    final_table = pd.concat(formatted_dataframes, keys=headers, axis=1)
    final_table.columns = [x[0] for x in final_table.columns]

    pvalue_col = pd.Series("", index=final_table.index, dtype=object)
    for variable in final_table.index.get_level_values(0).unique():
        first_row = final_table.index[final_table.index.get_level_values(0) == variable][0]
        pvalue_col.loc[first_row] = _format_pvalue(variable_pvalues.get(variable))
    final_table["p-value"] = pvalue_col

    silent_print(f"Dumping cohort characteristics table to {Paths().cohort_characteristics_table_xlsx}")
    final_table.to_csv(Paths().cohort_characteristics_table_xlsx)

    display(final_table)


def ana1a__cohort_stats_for_paper1():
    """Event counts and follow-up time for the discovery cohort."""
    red_id_paper1_cohort = list(set(DataBuilder.get_red_id_norway_data()).intersection(DataBuilder.get_red_id_genomic_data()))   # fmt: skip
    clinical_df = DataBuilder().get_clinical_data().query(f"RED_ID in {red_id_paper1_cohort}")

    get_median_w_IQR = lambda s: s.agg({"median": "median", "IQR lower": lambda x: x.quantile(0.25), "IQR upper": lambda x: x.quantile(0.75)})   # fmt: skip
    get_stats = lambda s, dropna=False: pd.concat([s.value_counts(dropna=dropna), s.value_counts(normalize=True, dropna=dropna)], axis=1).apply(
        lambda r: f"{r["count"]} ({r["proportion"]:.1%})", axis=1
    )
    display(get_stats(clinical_df["OS_event"]))
    display(get_stats(clinical_df["TTR_event"]))
    display(get_stats(clinical_df["DSS_event"]))
    clinical_df.query("TTR_event==1")["TTR"].median() / 30.44
    clinical_df.query("DSS_event==1")["DSS"].median() / 30.44
    clinical_df["age_at_surgery"].median()
    clinical_df["age_at_surgery"].agg(
        {"lower 25%": lambda x: x.quantile(0.25), "upper 75%": lambda x: x.quantile(0.75)}
    )
    # VARIABLE COUNTS
    get_stats(clinical_df["sex"])
    get_stats(clinical_df["smoking_status"] != "never")
    get_stats(clinical_df["histology"] == "LUAD")
    get_stats(clinical_df["histology"] == "LUSC")
    get_stats(clinical_df["histology"] == "other")
    get_stats(clinical_df["pstage"])
    get_stats(clinical_df["adjuvant_treatment"])
    get_stats(clinical_df["adjuv_treat"])
    get_stats(clinical_df["surg_center"])

    white_print("Median follow up time")
    display((get_median_w_IQR(clinical_df["time__surgery_to_last_followup"]) / 30.44).round(1))

    # N EVENTS IN 2 YEARS
    get_stats(clinical_df.query("TTR_event==1")["TTR"] < 2 * 365).rename_axis("Relapses that were within two years")

    pd.concat([clinical_df["OS_event"].value_counts(), clinical_df["OS_event"].value_counts(normalize=True)], axis=1)

    red_id_paper1_cohort = list(set(DataBuilder.get_red_id_norway_data()).intersection(DataBuilder.get_red_id_genomic_data()))
    clinical_df[clinical_df["RED_ID"].isin(red_id_paper1_cohort)]["TTR_event"].sum()
    red_id_paper1_cohort = list(set(DataBuilder.get_red_id_norway_data()).intersection(DataBuilder.get_red_id_genomic_data()))
    clinical_df[clinical_df["RED_ID"].isin(red_id_paper1_cohort)]["DSS_event"].sum()


def plot_endpoint_info():
    """Event counts and median time-to-event in the Norwegian cohort."""
    red_id_genomic_genomic_data = DataBuilder.get_red_id_genomic_data()
    clinical_df = DataBuilder().get_clinical_data().query(f"RED_ID in {red_id_genomic_genomic_data}")
    for endpoint in ["TTR", "DFS", "DSS", "OS"]:
        n_events, p_events = clinical_df.query(f"country=='norway'")[f"{endpoint}_event"].agg(["sum", "mean"]).values
        median_timetoevent = clinical_df.query(f"country=='norway' and {endpoint}_event==1")["DSS"].median() / 365
        white_print(endpoint)
        green_print(f"N events={n_events} ({p_events:.1%}); Median time-to-event={median_timetoevent:.1f} years")

    n_deaths = clinical_df.query("country=='norway' and OS_event==1").__len__()
    n_nonLC_deaths = clinical_df.query("country=='norway' and OS_event==1 and DSS_event==0")["OS_event"].sum()
    orange_print(f"N deaths NOT caused by lung cancer: {n_nonLC_deaths} ({n_nonLC_deaths / n_deaths:.1%})")
