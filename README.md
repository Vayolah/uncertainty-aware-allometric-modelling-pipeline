# Supplementary Material: Uncertainty-Aware Allometric Modelling Pipeline

This supplementary package accompanies the manuscript:

**A Python-Based Pipeline for Uncertainty-Aware Allometric Model Development: A Case Study of Restored *Rhizophora mucronata* Stands in Gazi Bay, Kenya**

It contains the Python implementation and harmonised dataset used to reproduce the analytical workflow and principal results reported in the manuscript.

## Contents

- `uncertainty_aware_allometric_pipeline.py`  
  Complete implementation of the seven-step analytical pipeline described in Section 2.1.2 of the manuscript.

- `gazi_bay_rhizophora_mucronata_allometry_data.csv`  
 Previously published destructive harvest harmonised dataset for restored *Rhizophora mucronata* stands aged 5, 6, and 12 years in Gazi Bay, Kenya.

- `requirements.txt`  
  Python package dependencies required to run the pipeline.

- `README.md`  
  Documentation for the supplementary code and data package.

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

The three demonstration datasets were originally reported in different source studies (Kairo, 2001; Tamooh et al., 2009; Kairo et al., 2008).

For analysis, the relevant variables were harmonised to the common schema shown above. The numerical observations used in model fitting were preserved. No trees were added, removed, filtered, or altered during the harmonisation or modelling stages.

This is consistent with the data-integrity principle adopted for this work and is described in Section 2.2.2 of the manuscript.

## Candidate model forms

Four candidate log-linear model forms are evaluated:

- M1: ln(AGB) = β₀ + β₁ ln(DBH)
- M2: ln(AGB) = β₀ + β₁ ln(DBH) + β₂ ln(H)
- M3a: ln(AGB) = β₀ + β₁ ln(DBH²·H)
- M3b: ln(AGB) = β₀ + β₁ ln(DBH·H)

## Analytical workflow

For each cohort and candidate model form, the script implements the following seven-step workflow:

1. Ordinary Least Squares (OLS) baseline fitting
2. Breusch–Pagan heteroscedasticity testing
3. Conditional estimator selection: OLS retained or Weighted Least Squares (WLS) applied
4. Non-parametric bootstrap confidence intervals (B = 5,000)
5. Leave-One-Out Cross-Validation (LOOCV)
6. Duan smearing correction for log back-transformation
7. Prediction and confidence interval estimation

Eligible candidate equations are subsequently ranked using five equally weighted criteria:

- LOOCV-R² — higher is better
- ΔR² — lower is better
- AIC — lower is better
- biomass-scale RMSE — lower is better
- bootstrap confidence-interval width for β₁ — lower is better

The pipeline returns a ranked list of eligible candidate equations. Final equation retention is informed by the ranked statistical outputs together with analyst review for biological plausibility.

## Requirements

Python 3 is required.

Install the required packages with:

```bash
pip install -r requirements.txt
```

The package dependencies are listed in `requirements.txt`.

## Usage

1. Place `uncertainty_aware_allometric_pipeline.py`, `gazi_bay_rhizophora_mucronata_allometry_data.csv`, and `requirements.txt` in the same directory.
2. Install the dependencies.
3. Run:

```bash
python uncertainty_aware_allometric_pipeline.py
```

4. The script processes the 5-, 6-, and 12-year cohorts and writes outputs to an `outputs/` directory created alongside the script.

## Outputs

The pipeline produces:

- console output reporting each analytical step for every cohort and candidate model;
- one diagnostic figure for the top-ranked eligible equation in each cohort;
- `allometric_pipeline_results.xlsx`, containing:
  - `Summary` — model diagnostics and performance statistics for all candidate models and cohorts;
  - `Predictions` — fitted values and influence diagnostics for the top-ranked eligible equation in each cohort;
  - `Prediction Intervals` — 95% confidence and prediction intervals at the mean predictor value;
- bootstrap confidence intervals;
- LOOCV statistics;
- Duan smearing factors;
- influence diagnostics;
- composite-ranking results.

## Verification and reproducibility

The supplied script and dataset were run end-to-end to verify that they reproduce the principal values reported in the manuscript, including the retained model form, β₁, LOOCV-R², ΔR², biomass-scale RMSE, and Duan smearing factor for all three cohorts.

The bootstrap procedure uses a fixed random seed (`42`) to support reproducibility.

## Software dependencies

See `requirements.txt`.

## Citation

If this code or dataset is reused, please cite the associated manuscript and the archived repository record available through Zenodo.
Zenodo DOI: 10.5281/zenodo.22789432

## Licence

The Python code in this repository is released under the MIT License.

The accompanying harmonised dataset is made available under the Creative Commons Attribution 4.0 International (CC BY 4.0) licence.
