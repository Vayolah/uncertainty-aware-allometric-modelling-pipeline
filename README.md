# Supplementary Material: Uncertainty-Aware Allometric Modelling Pipeline

This supplementary package accompanies the manuscript:

**An Uncertainty-Aware Pipeline for Forest Biomass Allometric Model Development: Demonstration in Restored Mangrove Forests**

It contains the Python implementation and harmonised demonstration dataset used to reproduce the analytical workflow and principal results reported in the manuscript.

## Contents

- `uncertainty_aware_allometric_pipeline.py`  
  Complete implementation of the baseline OLS fit and the six-step analytical pipeline described in the manuscript.

- `gazi_bay_rhizophora_mucronata_allometry_data.csv`  
  Harmonised destructive-harvest dataset for restored *Rhizophora mucronata* stands aged 5, 6, and 12 years in Gazi Bay, Kenya, drawn from previously published studies.

- `requirements.txt`  
  Python package dependencies required to run the pipeline.

- `README.md`  
  Documentation for the supplementary code and data package.

- `LICENSE`  
  MIT License for the Python source code.

- `DATA_USE_NOTICE.txt`  
  Terms of use for the harmonised dataset.

- `CITATION.cff`  
  Citation metadata for this repository.

## Dataset

The harmonised dataset contains 73 trees distributed across three restored stand-age cohorts:

- 5-year cohort: n = 11
- 6-year cohort: n = 12
- 12-year cohort: n = 50

The pipeline uses the following harmonised variables:

| Harmonised column | Description |
|---|---|
| `cohort` | Restored stand age in years (5, 6, or 12) |
| `dbh` | Stem diameter measurement used in the source dataset, expressed in centimetres |
| `height` | Total tree height in metres |
| `agb` | Total above-ground biomass dry weight in kilograms |

## Data harmonisation

The three demonstration datasets were originally reported in different source studies (Kairo, 2001; Tamooh et al., 2009; Kairo et al., 2008). Tree-level raw data for all three cohorts were provided directly by the original authors.

For analysis, the relevant variables were harmonised to the common schema shown above. The numerical observations used in model fitting were preserved. No trees were added, removed, filtered, or altered during the harmonisation or modelling stages.

This is consistent with the data-integrity principle described in the manuscript.

## Candidate model forms

Four candidate log-linear model forms are evaluated:

- M1: ln(AGB) = beta0 + beta1 * ln(DBH)
- M2: ln(AGB) = beta0 + beta1 * ln(DBH) + beta2 * ln(H)
- M3a: ln(AGB) = beta0 + beta1 * ln(DBH^2 * H)
- M3b: ln(AGB) = beta0 + beta1 * ln(DBH * H)

## Analytical workflow

For each cohort, every candidate model form is processed independently through a baseline fit followed by six statistical steps:

- Baseline: Ordinary Least Squares (OLS) fit of the candidate model form
- Step 1: Breusch-Pagan heteroscedasticity test
- Step 2: Conditional estimator selection (OLS retained, or Weighted Least Squares (WLS) applied with weights w_i = 1 / DBH_i^2)
- Step 3: Non-parametric bootstrap confidence intervals (B = 5,000)
- Step 4: Leave-One-Out Cross-Validation (LOOCV)
- Step 5: Duan smearing correction for log back-transformation
- Step 6: Prediction and confidence interval estimation

For WLS-fitted models, training R2 is computed from unweighted log-scale residuals, so that it is directly comparable with LOOCV-R2 and with OLS-fitted models.

After all candidate forms have been processed, M2 is screened for eligibility (excluded if the height coefficient beta2 has p >= 0.05). Eligible candidate equations are then ranked using five equally weighted criteria:

- LOOCV-R2: higher is better
- Overfitting gap (Delta R2 = training R2 - LOOCV-R2): lower is better
- AIC: lower is better
- Biomass-scale RMSE (kg): lower is better
- Bootstrap confidence-interval width for beta1: lower is better

The pipeline returns a ranked list of eligible candidate equations. Final equation retention is informed by the ranked statistical outputs together with analyst review for biological plausibility.

## Requirements

Python 3.12 or later is required (the pipeline was verified with Python 3.13).

Install the required packages with:

```bash
pip install -r requirements.txt
```

The package dependencies are listed in `requirements.txt`.

## Usage

1. Place `uncertainty_aware_allometric_pipeline.py` and `gazi_bay_rhizophora_mucronata_allometry_data.csv` in the same directory.
2. Install the dependencies.
3. Run:

```bash
python uncertainty_aware_allometric_pipeline.py
```

4. The script processes the 5-, 6-, and 12-year cohorts and writes outputs to an `outputs/` directory created alongside the script.

## Outputs

The pipeline produces:

- console output reporting the baseline fit and each analytical step for every cohort and candidate model;
- one nine-panel diagnostic figure for the top-ranked eligible equation in each cohort;
- `allometric_pipeline_results.xlsx`, containing:
  - `Summary`: model diagnostics and performance statistics for all candidate models and cohorts, including bootstrap confidence intervals, LOOCV statistics, the overfitting gap and Duan smearing factors;
  - `Predictions`: fitted values and influence diagnostics for the top-ranked eligible equation in each cohort;
  - `Prediction Intervals`: 95% confidence and prediction intervals at the mean DBH and mean height of each cohort;
- composite-ranking results, reported in the console output.

## Verification and reproducibility

The supplied script and dataset were run end-to-end to verify that they reproduce the principal values reported in the manuscript, including the retained model form, beta1, LOOCV-R2, the overfitting gap, biomass-scale RMSE, and Duan smearing factor for all three cohorts.

The bootstrap procedure uses a fixed random seed (`42`) to support reproducibility.

## Software dependencies

See `requirements.txt`.

## Citation

If this code or dataset is reused, please cite the associated manuscript and this Zenodo record (concept DOI: https://doi.org/10.5281/zenodo.22789431).

## Licence and data use

The Python code in this repository is released under the MIT License (see `LICENSE`).

The harmonised dataset is not released under an open licence. The original source datasets were used with permission from the data owners, and the harmonised file is shared solely to support reproducibility of the associated study. Users should consult the original publications and data owners regarding any further reuse or redistribution of the underlying data (see `DATA_USE_NOTICE.txt`).
