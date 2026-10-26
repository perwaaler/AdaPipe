This repository contains code for "Gene-Level Optimization of Variant Allele
Frequency Thresholds Improves Performance of Prognostic Models in Resected
NSCLC", which introduces AdaPipe: a pipeline for estimating optimal variant
filters at the gene level. Specifically, AdaPipe evaluates combinations of
oncogenicity filters and VAF cutoffs, and selects the filter combination that
maximizes the association between mutation status and relapse (or other survival
endpoints). To assess and rank gene significance, a two-parameter logistic
function is fitted to a gene's VAF values. The logistic model represents a
continuous analogue to a threshold model, with the location parameter of the
sigmoid function corresponding to the threshold parameter of the dichotomization
approach, and the scale parameter controlling the smoothness of the transition
(behaves like a threshold model for low scale values).

## Setup

- **Python:** 3.12.
- **Working directory:** run scripts from `src/` (`cd src`).
- **Install:** `pip install -r requirements.txt` from the repo root (includes
  **torch**, **lifelines**, **scikit-optimize**, and **scikit-survival** for
  AdaPipe / Cox / survival-tree code). Use a [PyTorch install
  command](https://pytorch.org/get-started/locally/) if the default CPU wheel
  does not match your platform.
- Paths to inputs and dumps are defined in ``src/utils/paths.py`` (typically
  ``data/``, ``output/``, and ``results/`` at the repo root).

The main code for running AdaPipe analysis per gene is in ``VariableSearcher``
in ``src/utils/variablesearch/variable_searcher.py``. To use this code on new
data, you must provide two dataframes: ``variants_df`` and ``clinical_df``:

- ``clinical_df``: one row per patient. Required: ``RED_ID`` (patient
  identifier), ``time_to_event``, ``event_observed`` (0 or 1).
- ``variants_df``: one row per variant. Required: ``RED_ID``, ``SYMBOL`` (gene
  identifier), ``VAF_TUMOR``, ``ID`` (variant identifier), ``IS_ONCOGENIC``
  (boolean indicating whether the variant is oncogenic).

Run ``src/simulate_example_data.py`` for a demo with simulated data showing what
the input data can look like and how to use ``VariableSearcher``.

## Running the analyses of the paper

To run the scripts that generate the results of the paper (the "ana" scripts),
start by running ``src/build_convenience_objects.py``, followed by
``src/split_data.py`` (one split for the whole cohort and one for the LUAD
subset) to generate the cross-validation data splits. See the section below for data dependencies. Then, the following
scripts build the objects studied in the analyses:

- ``src/run_gridsearch_analyses.py``: fits gene-level variant filters in the
  discovery cohort for different subsets (LUAD or NSCLC) and endpoints (TTR, OS,
  and DSS). The results used in the cross-validation analysis are dumped in
  ``output/varselect/varsearch-results/norway/crossval/``, with results for
  different folds stored in folders ``fold0/``, ``fold1/``, ... ``fold4/``.
  Run before ``src/ana2__discovery_cohort_crossvalidation.py``.
- ``src/gen_permutation_adjusted_pvalues.py``: runs permutation testing to
  adjust for bias in p-values as a result of endpoint-based feature
  construction. Results are dumped in
  ``output/varselect/permutation-adjusted-pvals`` and analyzed in
  ``src/ana3__permutationAdjusted_pvals.py``.
- ``src/get_gene_selectionpool_for_TCGAanalysis.py`` identifies genes that have
  sufficiently high mutation rates in TCGA LUAD and LUSC respectively to be
  included in the sets of selectable genes. Sets of candidate genes are saved in
  ``output/gene-sets/``.
- ``src/build_models_for_TCGAvalidation.py`` runs hyperparameter optimization
  for different modeling strategies (restrictions to histology, choice of
  clinical endpoint for model development, and variant filtering strategy) in
  the discovery cohort. Depends on lists in ``output/gene-sets/``. Saves
  hyperparameters in ``output/hyperparams``. Fitted models are analyzed in
  ``src/ana4__test_models_in_TCGA.py``.

Paper analysis scripts:

- ``src/ana0__comut_plot.py``: produces the comut plot for the discovery cohort
- ``src/ana1__data_overview.py``: code describing cohort characteristics of the
  discovery cohort
- ``src/ana2__discovery_cohort_crossvalidation.py``: discovery cohort cross-validation
  analysis (first run: ``run_gridsearch_analyses.py``)
- ``src/ana3__permutationAdjusted_pvals.py``: analysis of permutation-adjusted p-values
  (first run: ``gen_permutation_adjusted_pvalues.py``; ``run_gridsearch_analyses.py`` for
  Table S1)
- ``src/ana4__test_models_in_TCGA.py``: external validation analysis in TCGA
  LUAD and LUSC (first run: ``get_gene_selectionpool_for_TCGAanalysis.py``,
  ``build_models_for_TCGAvalidation.py``, ``run_gridsearch_analyses_TCGA.py``)

## Data dependencies

File locations are centralized in ``src/utils/paths.py``. Paths are relative to
the repo root.

### Layout

| Directory | Role |
|-----------|------|
| ``data/`` | Curated inputs (you provide these; not produced by the paper pipeline) |
| ``output/`` | Intermediate artifacts (splits, grid searches, hyperparameters) |
| ``results/`` | Tables and figure exports written by the ``ana*`` scripts |

### Inputs in ``data/``

**TNM-I discovery cohort**

| File | Purpose |
|------|---------|
| ``data/master-data/GATK_1224.txt`` | Mutect2 variant calls |
| ``data/master-data/FU26_TNMI_with_events.csv`` | Clinical data and survival endpoints |
| ``data/master-data/clinical_data_label.csv`` | Readable labels merged into the clinical table |
| ``data/master-data/TMB_MSI_key_renamed.csv`` | Maps GATK ``SAMPLE_ID`` to clinical ``RED_ID`` |

**Gene lists**

| File | Purpose |
|------|---------|
| ``data/gene-panels/tso500_genes.json`` | TSO500 panel gene list |
| ``data/gene-sets/2024-06-18_IntOGen-Drivers/Compendium_Cancer_Genes.tsv`` | IntOGen compendium cancer driver genes |

**TCGA (external validation)**

| File | Purpose |
|------|---------|
| ``data/tcga/TCGA-CDR-SupplementalTableS1.xlsx`` | Curated TCGA survival endpoints |
| ``data/tcga/LUAD/TCGA_LUAD_Mutations.csv`` | LUAD somatic mutations |
| ``data/tcga/LUAD/TCGA_LUAD_OncoKB_out.txt`` | OncoKB annotations (LUAD) |
| ``data/tcga/LUAD/luad_tcga_clinical_data.tsv`` | LUAD clinical variables |
| ``data/tcga/LUSC/TCGA_LUSC_Mutations.csv`` | LUSC somatic mutations |
| ``data/tcga/LUSC/TCGA_LUSC_OncoKB_out.txt`` | OncoKB annotations (LUSC) |
| ``data/tcga/LUSC/TCGA_LUSC_Clinical_Patient.csv`` | LUSC clinical variables |

### Generated under ``output/``

Precursor scripts write artifacts under ``output/data-splits/`` (sample lists and CV folds),
``output/gene-sets/``, ``output/varselect/`` (grid-search and permutation dumps), and
``output/hyperparams/``. Exact filenames and subfolders are defined on ``Paths`` in
``paths.py``.

### Outputs in ``results/``

The ``ana*`` scripts write figures and tables under ``results/datafusion/``; export paths
are likewise on ``Paths``.

## Data availability

The **TNM-I** variant and clinical files listed under ``data/master-data/`` are **not**
included in this repository (patient privacy and cohort data-use agreements). Access may
be granted to qualified researchers **upon reasonable request**, subject to approval by
the data controllers and any required collaboration or material-transfer agreements.
Contact the corresponding author of the paper.

**TCGA** data are public via the
[NCI Genomic Data Commons](https://portal.gdc.cancer.gov/). This repo expects curated
exports under ``data/tcga/`` (file names in the table above); you must obtain and place
them locally.

## License

Source code is released under the **MIT License** (see ``LICENSE``). The MIT license
does not grant rights to the restricted cohort data described above.
