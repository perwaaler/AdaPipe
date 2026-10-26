"""Synthetic variants + survival for pipeline demos."""

# %% ---------- SETTINGS -----------------------------------------------------------------------------------------
from utils.prelude import *
from utils.models.models import CPHModel
from utils.prepare_data.variant_data import GeneLevelFeatures, VariantData

n_patients = 1000
rng = np.random.default_rng(2)

GENES = [
    "BRCA1",
    "KMT2D",
    "TET2",
    "KEAP1",
    "KAT6A",
    "FRS2",
    "ANKRD26",
    "FANCD2",
    "MYD88",
    "STK11",
]

# Mean variant count per (patient, gene); used in survival step (gene mutated = max VAF >= cutoff).
POISSON_MEAN = {
    "BRCA1": 4.4,
    "KMT2D": 15.3,
    "TET2": 5.6,
    "KEAP1": 1.2,
    "KAT6A": 5.1,
    "FRS2": 1.1,
    "ANKRD26": 4.8,
    "FANCD2": 2.1,
    "MYD88": 0.8,
    "STK11": 1.1,
}

P_ONCOGENIC = 1 / 3
VAF_MEAN = 0.05
VAF_MIN, VAF_MAX = 0.01, 0.97  # clip tail; keeps VAF below germline proxy used in the real pipeline
DP_TUMOR_MEAN = 800
# Depth sampling: Poisson(mean) is very tight (CV ≈ 1/sqrt(mean)). Alternatives:
#   "shifted_gamma" — base + Gamma noise; wider, avoids depths far below base (default)
#   "gamma"         — Gamma(shape, mean/shape); simple overdispersion
#   "lognormal"     — skewed high; many values below mean unless you clip
DP_TUMOR_SAMPLER = "lognormal"
DP_TUMOR_BASE = 420  # used by shifted_gamma; typical minimum-ish depth
DP_TUMOR_GAMMA_SHAPE = 2.0


def sample_dp_tumor(size: int, rng: np.random.Generator) -> np.ndarray:
    """Positive integer depths with mean ≈ DP_TUMOR_MEAN."""
    if DP_TUMOR_SAMPLER == "poisson":
        return rng.poisson(DP_TUMOR_MEAN, size=size)
    if DP_TUMOR_SAMPLER == "gamma":
        scale = DP_TUMOR_MEAN / DP_TUMOR_GAMMA_SHAPE
        return rng.gamma(DP_TUMOR_GAMMA_SHAPE, scale, size=size).astype(int)
    if DP_TUMOR_SAMPLER == "shifted_gamma":
        scale = (DP_TUMOR_MEAN - DP_TUMOR_BASE) / DP_TUMOR_GAMMA_SHAPE
        return (DP_TUMOR_BASE + rng.gamma(DP_TUMOR_GAMMA_SHAPE, scale, size=size)).astype(int)
    if DP_TUMOR_SAMPLER == "lognormal":
        # log-normal with mean DP_TUMOR_MEAN (sigma controls spread; larger → wider + more lows)
        sigma = 0.35
        mu = np.log(DP_TUMOR_MEAN) - 0.5 * sigma**2
        return rng.lognormal(mu, sigma, size=size).astype(int)
    raise ValueError(f"Unknown DP_TUMOR_SAMPLER={DP_TUMOR_SAMPLER!r}")


# Empirical CONSEQUENCE mix (counts from discovery cohort driver-gene variants).
CONSEQUENCE = [
    "frameshift_variant",
    "missense_variant",
    "inframe_deletion",
    "coding_sequence_variant&intron_variant&splice_donor_5th_base_variant&splice_donor_variant",
    "coding_sequence_variant&intron_variant&splice_acceptor_variant",
    "protein_altering_variant",
    "inframe_insertion",
    "protein_altering_variant&stop_gained",
    "frameshift_variant&stop_gained",
    "splice_acceptor_variant",
    "inframe_insertion&stop_retained_variant",
    "intron_variant&splice_donor_region_variant",
    "intron_variant&splice_donor_5th_base_variant",
    "coding_sequence_variant&intron_variant&splice_acceptor_variant&splice_donor_5th_base_variant&splice_donor_variant",
    "intron_variant&splice_donor_5th_base_variant&splice_donor_variant",
]
CONSEQUENCE_COUNT = [26, 18, 13, 11, 6, 5, 5, 3, 3, 3, 3, 1, 1, 1, 1]
CONSEQUENCE_P = np.array(CONSEQUENCE_COUNT, dtype=float) / sum(CONSEQUENCE_COUNT)

# Per-gene filters from discovery whole-cohort AdaPipe fit (used for MUTAS features).
FILTER_FIT = pd.DataFrame(
    {
        "vaf_cutoff": [0.078, 0.188, 0.072, 0.183, 0.06, 0.036, 0.039, 0.032, 0.108, 0.15],
        "coef": [0.296, 0.319, 0.097, 0.513, 0.319, -0.16, -0.195, -0.077, 0.356, 0.379],
        "best_oncofiltersetting": ["on", "off", "off", "on", "off", "off", "off", "on", "off", "off"],
    },
    index=GENES,
)

# Survival (TTR-like; times in days)
TARGET_EVENT_RATE = 0.37
CENSOR_LO, CENSOR_HI = 900, 2900
# Discovery multivar coefs are not a generative baseline hazard; center eta so lambda0 sets overall rate.
CENTER_ETA = True


def simulate_survival_from_eta(
    eta: np.ndarray,
    rng: np.random.Generator,
    lambda0: float,
    unit_exp: np.ndarray | None = None,
    censor_times: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Exponential PH with independent uniform censoring.

    If ``unit_exp`` (Exp(1)) and ``censor_times`` are passed, event times are a
    deterministic function of ``lambda0`` (used by the calibrator).
    """
    if unit_exp is None:
        unit_exp = rng.exponential(scale=1.0, size=len(eta))
    if censor_times is None:
        censor_times = rng.uniform(CENSOR_LO, CENSOR_HI, len(eta))
    T = unit_exp / (lambda0 * np.exp(eta))
    event_observed = (T < censor_times).astype(int)
    time_to_event = np.minimum(T, censor_times)
    return time_to_event, event_observed


def calibrate_lambda0(
    eta: np.ndarray,
    rng: np.random.Generator,
    target_rate: float,
    tol: float = 0.005,
    max_iter: int = 80,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Bisect baseline hazard on one fixed Exp(1)/censoring draw."""
    unit_exp = rng.exponential(scale=1.0, size=len(eta))
    censor_times = rng.uniform(CENSOR_LO, CENSOR_HI, len(eta))

    def rate_at(lambda0: float) -> float:
        _, event_observed = simulate_survival_from_eta(eta, rng, lambda0, unit_exp=unit_exp, censor_times=censor_times)
        return float(event_observed.mean())

    lo, hi = 1e-9, 1.0
    while rate_at(hi) < target_rate:
        hi *= 2.0
        if hi > 1e3:
            break
    for _ in range(max_iter):
        mid = np.sqrt(lo * hi)
        rate = rate_at(mid)
        if abs(rate - target_rate) < tol:
            lo = hi = mid
            break
        if rate > target_rate:
            hi = mid
        else:
            lo = mid
    lambda0 = np.sqrt(lo * hi)
    time_to_event, event_observed = simulate_survival_from_eta(
        eta, rng, lambda0, unit_exp=unit_exp, censor_times=censor_times
    )
    return lambda0, time_to_event, event_observed


def mutation_features_from_filter_fit(
    genomic_features_df: pd.DataFrame, filter_fit: pd.DataFrame, red_ids: list[str]
) -> pd.DataFrame:
    """Binary MUTAS columns using AdaPipe cutoffs (same rule as VariableSearcher MUTAS)."""
    vaf_wide = genomic_features_df.set_index("RED_ID").reindex(red_ids, fill_value=0.0)
    mutated = {}
    for gene in filter_fit.index:
        onco = filter_fit.loc[gene, "best_oncofiltersetting"]
        cutoff = filter_fit.loc[gene, "vaf_cutoff"]
        vaf_col = f"{gene}_MAX_ONCO_VAF" if onco == "on" else f"{gene}_MAX_VAF"
        if vaf_col in vaf_wide.columns:
            vaf_series = vaf_wide[vaf_col]
        else:
            vaf_series = pd.Series(0.0, index=vaf_wide.index)
        mutated[f"{gene}_MUTATED"] = (vaf_series >= cutoff).astype(float)
    out = pd.DataFrame(mutated, index=vaf_wide.index).reset_index()
    return out


# ---------- PATIENT IDS -----------------------------------------------------------------------------------------
red_ids = [f"SIM_{i:04d}" for i in range(n_patients)]

# ---------- DRAW VARIANT COUNTS AND ROWS -----------------------------------------------------------------------
rows = []
variant_counter = 0

for red_id in red_ids:
    for gene in GENES:
        n_variants = rng.poisson(POISSON_MEAN[gene])
        if n_variants == 0:
            continue
        vafs = np.clip(rng.exponential(scale=VAF_MEAN, size=n_variants), a_min=VAF_MIN, a_max=VAF_MAX)
        dp_tumor = sample_dp_tumor(n_variants, rng)
        supporting_reads = (vafs * dp_tumor).astype(int)
        is_onco = rng.random(n_variants) < P_ONCOGENIC
        consequences = rng.choice(CONSEQUENCE, size=n_variants, p=CONSEQUENCE_P)
        for vaf, dp, alt_reads, onco, consequence in zip(vafs, dp_tumor, supporting_reads, is_onco, consequences):
            rows.append(
                {
                    "RED_ID": red_id,
                    "SYMBOL": gene,
                    "VAF_TUMOR": float(vaf),
                    "DP_TUMOR": int(dp),
                    "SUPPORTING_READS": int(alt_reads),
                    "CONSEQUENCE": consequence,
                    "ID": f"simvar_{variant_counter:07d}",
                    "IS_ONCOGENIC": bool(onco),
                }
            )
            variant_counter += 1

variants_sim_df = pd.DataFrame(rows)

# %% ---------- SUMMARY --------------------------------------------------------------------------------------------
print_header1("Simulated variant data")
display(variants_sim_df)

orange_print(f"n_patients={n_patients}, variant rows={len(variants_sim_df):,}, unique genes={variants_sim_df['SYMBOL'].nunique()}")  # fmt: skip
if len(variants_sim_df):
    orange_print(f"IS_ONCOGENIC rate: {variants_sim_df['IS_ONCOGENIC'].mean():.1%} (target {P_ONCOGENIC:.1%})")

print_header1("Mean variants per patient×gene (simulated vs target Poisson mean):")
sim_means = variants_sim_df.groupby("SYMBOL").size() / n_patients
summary = pd.DataFrame({"target_mean": POISSON_MEAN, "sim_mean": sim_means}).sort_index()
summary["sim_mean"] = summary["sim_mean"].fillna(0)
display(summary)

print_header1("Patients with ≥1 variant per gene (fraction of cohort):")
patients_with_gene = variants_sim_df.groupby("SYMBOL")["RED_ID"].nunique() / n_patients
display(patients_with_gene.sort_values(ascending=False).to_frame("frac_patients"))

white_print(f"VAF_TUMOR overall mean: {variants_sim_df['VAF_TUMOR'].mean():.3f} (target {VAF_MEAN})")
white_print("VAF_TUMOR by gene (mean / median / max):")
vaf_summary = variants_sim_df.groupby("SYMBOL")["VAF_TUMOR"].agg(["mean", "median", "max", "count"])
display(vaf_summary)

white_print(f"DP_TUMOR ({DP_TUMOR_SAMPLER}): mean={variants_sim_df['DP_TUMOR'].mean():.1f} (target {DP_TUMOR_MEAN}), std={variants_sim_df['DP_TUMOR'].std():.1f}, p10={variants_sim_df['DP_TUMOR'].quantile(0.1):.0f}, p90={variants_sim_df['DP_TUMOR'].quantile(0.9):.0f}")  # fmt: skip
white_print(f"SUPPORTING_READS mean: {variants_sim_df['SUPPORTING_READS'].mean():.1f}")

fig, axs = plt.subplots(1, 2, figsize=(10, 3))
sns.histplot(variants_sim_df["DP_TUMOR"], ax=axs[0])
sns.histplot(variants_sim_df["SUPPORTING_READS"], ax=axs[1])

# %% ---------- FILTER VARIANTS (same rules as discovery pipeline) ------------------------------------------------
filtered_variants_sim_df = VariantData().filter_variants(variants_sim_df)
orange_print(f"After filter_variants: {len(filtered_variants_sim_df):,} rows ({len(filtered_variants_sim_df)/len(variants_sim_df):.1%} of raw)")  # fmt: skip

# %% ---------- GENE-LEVEL VAF + MUTAS FEATURES --------------------------------------------------------------------
genelvlfeatures = GeneLevelFeatures.from_variants_using_max(filtered_variants_sim_df)
mutation_features_df = mutation_features_from_filter_fit(genelvlfeatures.genomic_features_df, FILTER_FIT, red_ids)

print_header1("Filter fit settings (discovery AdaPipe):")
display(FILTER_FIT)

print_header1("Mutation feature prevalence (fraction of cohort with gene_MUTATED == 1):")
display(mutation_features_df.filter(like="_MUTATED").mean().sort_values(ascending=False).to_frame("prevalence"))

print_header1("Mutation matrix")
display(mutation_features_df)

print_header1("Mutations ber gene")
display(mutation_features_df.set_index("RED_ID").sum().to_frame("mutated samples").join(mutation_features_df.set_index("RED_ID").mean().to_frame("proportion of samples")))   # fmt: skip


# %% ---------- SIMULATE SURVIVAL DATA -------------------------------------------------------------------------------------------
# Cox PH: eta = sum coef_g * {g}_MUTATED (coef from FILTER_FIT / discovery multivar fit).


print_header1("SIMULATE SURVIVAL DATA")
mutated_predictors = [f"{gene}_MUTATED" for gene in GENES]
coef_series = pd.Series({f"{g}_MUTATED": FILTER_FIT.loc[g, "coef"] for g in GENES})
eta = mutation_features_df[mutated_predictors].to_numpy() @ coef_series.to_numpy()
if CENTER_ETA:
    eta = eta - np.mean(eta)

lambda0, time_to_event, event_observed = calibrate_lambda0(eta, rng, TARGET_EVENT_RATE)

clinical_sim_df = mutation_features_df.copy()
clinical_sim_df["time_to_event"] = time_to_event
clinical_sim_df["event_observed"] = event_observed

orange_print(f"Survival: events={clinical_sim_df['event_observed'].mean():.1%} (target {TARGET_EVENT_RATE:.0%}), lambda0={lambda0:.5g}")  # fmt: skip
orange_print(f"TTE | event: mean={clinical_sim_df.query('event_observed==1')['time_to_event'].mean():.0f} d   TTE | censored: mean={clinical_sim_df.query('event_observed==0')['time_to_event'].mean():.0f} d   median TTE={clinical_sim_df['time_to_event'].median():.0f} d")  # fmt: skip
display(clinical_sim_df[mutated_predictors].mean().sort_values(ascending=False).to_frame("mutated_rate"))
display(clinical_sim_df[["time_to_event", "event_observed"]].describe())

from lifelines import KaplanMeierFitter

high_risk = eta >= np.median(eta)
fig, ax = plt.subplots()
for mask, label in [(~high_risk, "low risk"), (high_risk, "high risk")]:
    KaplanMeierFitter().fit(time_to_event[mask], event_observed[mask], label=label).plot(ax=ax)

# %% ---------- SANITY CHECK: REFIT COX ON MUTATED GENES -----------------------------------------------------------
cph = CPHModel(predictors=mutated_predictors, penalizer=0.1)
cph.fit(clinical_sim_df)
white_print("Refit Cox PH on simulated survival (truth coef in FILTER_FIT['coef'])")
results_df = cph.get_light_summary()

# --- PLOT ESTIMATED COEFFICIENTS AGAINST TRUE VALUES
results_df.index = [x.replace("_MUTATED", "") for x in results_df.index]
results_df.index.name = "gene"
results_df = results_df.join(FILTER_FIT["coef"], rsuffix="_true").reset_index()

long_df = pd.melt(results_df, id_vars="gene", value_vars=["coef", "coef_true"], var_name="coefficient")
long_df["coefficient"] = long_df["coefficient"].replace({"coef_true": "true", "coef": "estimated"})
fig, ax = plt.subplots()
sns.barplot(long_df, x="gene", y="value", hue="coefficient", edgecolor="k")
ax.axhline(0, ls="-", color="k", alpha=0.2)
ax.tick_params(rotation=90)
ax.set_title("Estimated vs true coefficient values")

# %% ---------- UNIVARIATE ANALYSIS -----------------------------------------------------------------------------------------
from utils.variablesearch.variable_searcher import VariableSearcher

# %% ---------- NAME -----------------------------------------------------------------------------------------
varsearcher_sim = VariableSearcher(
    variants_df=filtered_variants_sim_df,
    red_id_used_to_fit_filters="no_restriction",
    clinical_df=clinical_sim_df[["RED_ID", "time_to_event", "event_observed"]].copy(),
    prelim_pval_cutoff=1,
)
varsearcher_sim.run_gridsearch_analysis_for_genes()
varsearcher_sim.prelim_results
varsearcher_sim.compute_result_summary_dfs()

# %% --- PLOT FIT RESULTS VERSUS TRUE VALUES
df = varsearcher_sim.summary_best_models_df.sort_values("p")[["p", "vaf_cutoff", "coef", "p_softcut", "neglogp"]]
df = df.join(FILTER_FIT, rsuffix="_true")
df.index.name = "gene"
df = df.reset_index()
long_df = pd.melt(df, id_vars="gene", value_vars=["vaf_cutoff", "vaf_cutoff_true"], var_name="cutoff")
long_df["cutoff"] = long_df["cutoff"].replace({"vaf_cutoff": "estimated", "vaf_cutoff_true": "true"})

fig, axs = plt.subplots(1, 2, figsize=(10, 3))
ax = axs[0]
sns.barplot(long_df, x="gene", y="value", hue="cutoff", edgecolor="k", ax=ax)
ax.axhline(0, ls="-", color="k", alpha=0.2)
ax.tick_params(rotation=90)
ax.set_title("Estimated vs true coefficient values")

ax = axs[1]
ax.scatter(FILTER_FIT.set_index(df["gene"])["coef"], df["neglogp"], s=100, edgecolor="k")
ax.set_xlabel("Coefficient magnitude")
ax.set_ylabel("-log10(p)")
ax.axhline(-np.log10(0.05), ls="--", c="orange", label="p=0.05")
ax.axhline(-np.log10(0.01), ls="--", c="red", label="p=0.01")
ax.set_title("Fit significance versus true coefficient value")
ax.legend()

# %% ---------- NAME -----------------------------------------------------------------------------------------
from utils.variablesearch.grid_cutoff_selectors import ManuallySelectedCutoff
from utils.variablesearch.grid_constructor import GridConstructor

varsearcher_fixed_sim = VariableSearcher(
    variants_df=filtered_variants_sim_df,
    red_id_used_to_fit_filters="no_restriction",
    clinical_df=clinical_sim_df[["RED_ID", "time_to_event", "event_observed"]].copy(),
    prelim_pval_cutoff=1,
    grid=GridConstructor("fixed"),
    grid_cutoff_selector=ManuallySelectedCutoff(0.05, oncofilter_selection="pick_off"),
)
varsearcher_fixed_sim.run_gridsearch_analysis_for_genes()
varsearcher_fixed_sim.prelim_results
varsearcher_fixed_sim.compute_result_summary_dfs()
varsearcher_fixed_sim.summary_best_models_df.sort_values("p")[["p", "vaf_cutoff", "coef", "p_softcut", "neglogp"]]


# %% --- PLOT ESTIMATED COEFFICIENTS AGAINST TRUE VALUES
df = varsearcher_fixed_sim.summary_best_models_df.sort_values("p")[["p", "vaf_cutoff", "coef", "p_softcut", "neglogp"]]
df = df.join(FILTER_FIT, rsuffix="_true")
df.index.name = "gene"
df = df.reset_index()

fig, ax = plt.subplots()
ax.scatter(FILTER_FIT.set_index(df["gene"])["coef"], df["neglogp"], s=100, edgecolor="k")
ax.set_xlabel("Coefficient magnitude")
ax.set_ylabel("-log10(p)")
ax.axhline(-np.log10(0.05), ls="--", c="orange", label="p=0.05")
ax.axhline(-np.log10(0.01), ls="--", c="red", label="p=0.01")
ax.set_title("Fit significance versus true coefficient value")
ax.legend()
