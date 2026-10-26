"""
===============================================================================
UNCERTAINTY-AWARE ALLOMETRIC MODELLING PIPELINE
Gazi Bay Rhizophora mucronata - Age-Specific Biomass Equations

===============================================================================

METHODOLOGY REFERENCE: Integrated Analytical Pipeline
  Baseline : Classical OLS fit of the candidate model form
  Step 1   : Breusch-Pagan heteroscedasticity test
  Step 2   : Weighted Least Squares (conditional on Step 1)
  Step 3   : Bootstrap resampling (B = 5,000)
  Step 4   : Leave-One-Out Cross-Validation (LOOCV)
  Step 5   : Duan smearing factor (back-transformation bias correction)
  Step 6   : Prediction intervals (95% CI and 95% PI)
  The baseline fit and Steps 1-6 are repeated independently for each
  candidate model form within each cohort, followed by composite ranking.

DATA INTEGRITY NOTE:
  The harmonised dataset preserves the numerical observations from the
  original source datasets. No observations are excluded or altered
  during model fitting. Influence diagnostics are reported for interpretation
  only and do not trigger automatic observation removal.

USAGE:
  1. Place your harmonised CSV in the same folder as this script.
  2. Set DATA_FILE below to your CSV's filename.
  3. Run: python uncertainty_aware_allometric_pipeline.py
  4. Results saved to the outputs/ folder.

REQUIRED COLUMNS IN CSV:
  - cohort    : integer stand/tree age in years (5, 6, 12)
  - dbh       : stem diameter measurement used in the source dataset (cm)
  - height    : total tree height (m)
  - agb       : above-ground biomass (kg dry weight)

DEPENDENCIES:
  pip install -r requirements.txt   (Python 3.12 or later)
===============================================================================
"""

# --- Imports ---
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from statsmodels.regression.linear_model import OLS, WLS
from statsmodels.tools import add_constant
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
import statsmodels.api as sm
from itertools import combinations

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION - edit these to match your file
# =============================================================================
DATA_FILE      = "gazi_bay_rhizophora_mucronata_allometry_data.csv"   # your CSV filename
OUTPUT_DIR     = "outputs"                    # results folder
B_BOOTSTRAP    = 5000                         # bootstrap iterations
ALPHA          = 0.05                         # significance level
CI_LEVEL       = 0.95                         # confidence/prediction interval
RANDOM_SEED    = 42                           # reproducibility

# Cohorts to process - set to None to process all, or list e.g. [5, 12]
COHORTS        = None   # e.g. [5] to run only 5-year cohort

# =============================================================================
# COLOUR PALETTE (consistent across all figures)
# =============================================================================
COLOURS = {
    'ols'        : '#1E3A5F',   # dark navy
    'wls'        : '#C0392B',   # deep red
    'ci'         : '#2980B9',   # blue
    'pi'         : '#85C1E9',   # light blue
    'scatter'    : '#2ECC71',   # green
    'residual'   : '#E67E22',   # orange
    'bootstrap'  : '#8E44AD',   # purple
    'grid'       : '#ECF0F1',
}

# =============================================================================
# HELPER UTILITIES
# =============================================================================

def make_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def sep(char='=', n=72):
    return char * n

def section(title):
    print(f"\n{sep()}")
    print(f"  {title}")
    print(sep())

def subsection(title):
    print(f"\n  {'-'*60}")
    print(f"  {title}")
    print(f"  {'-'*60}")

def fmt(val, decimals=4):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:.{decimals}f}"

# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

def load_data(filepath):
    """Load and validate the harmonised dataset."""
    section("DATA LOADING AND VALIDATION")

    if not os.path.exists(filepath):
        print(f"\n  ERROR: File not found: {filepath}")
        print(f"  Please place your CSV file in: {os.path.abspath('.')}")
        sys.exit(1)

    df = pd.read_csv(filepath)
    print(f"\n  File loaded: {filepath}")
    print(f"  Total records: {len(df)}")
    print(f"  Columns found: {list(df.columns)}")

    # Normalise column names
    df.columns = df.columns.str.lower().str.strip()

    # Check required columns
    required = ['cohort', 'dbh', 'agb']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"\n  ERROR: Missing required columns: {missing}")
        print(f"  Required columns: {required}")
        sys.exit(1)

    has_height = 'height' in df.columns

    # Type coercion
    for col in ['dbh', 'agb'] + (['height'] if has_height else []):
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['cohort'] = pd.to_numeric(df['cohort'], errors='coerce').astype(int)

    # Report missing values
    print(f"\n  Missing values per column:")
    print(df[['cohort','dbh','agb'] + (['height'] if has_height else [])
             ].isnull().sum().to_string(header=False))

    # Drop rows with missing key variables
    before = len(df)
    df = df.dropna(subset=['dbh', 'agb'])
    after = len(df)
    if before != after:
        print(f"\n  WARNING: Dropped {before-after} rows with missing DBH or AGB")

    # Remove non-positive values (log transformation requires > 0)
    for col in ['dbh', 'agb'] + (['height'] if has_height else []):
        n_nonpos = (df[col] <= 0).sum()
        if n_nonpos > 0:
            print(f"  WARNING: {n_nonpos} non-positive values in {col} - removed")
            df = df[df[col] > 0]

    # Cohort summary
    print(f"\n  Cohort summary:")
    for cohort, grp in df.groupby('cohort'):
        print(f"    {cohort}-year cohort: n = {len(grp)}")

    print(f"\n  OK: Data validation complete. {len(df)} records ready.")
    return df, has_height

# =============================================================================
# DESCRIPTIVE STATISTICS
# =============================================================================

def descriptive_statistics(df, cohort, has_height):
    """Compute and display descriptive statistics for one cohort."""
    subsection(f"DESCRIPTIVE STATISTICS - {cohort}-YEAR COHORT")

    cols = ['dbh', 'agb'] + (['height'] if has_height else [])
    stats_rows = []

    for col in cols:
        v = df[col].dropna()
        stats_rows.append({
            'Variable'  : col.upper(),
            'n'         : len(v),
            'Mean'      : v.mean(),
            'SD'        : v.std(),
            'Min'       : v.min(),
            'Median'    : v.median(),
            'Max'       : v.max(),
            'CV (%)'    : (v.std() / v.mean()) * 100
        })

    stats_df = pd.DataFrame(stats_rows)
    print(stats_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Log-transformed
    print(f"\n  Log-transformed variables:")
    log_rows = []
    for col in cols:
        v = np.log(df[col].dropna())
        log_rows.append({
            'Variable'  : f"ln({col.upper()})",
            'Mean'      : v.mean(),
            'SD'        : v.std(),
            'Min'       : v.min(),
            'Max'       : v.max(),
            'Skewness'  : stats.skew(v),
            'Kurtosis'  : stats.kurtosis(v)
        })
    log_df = pd.DataFrame(log_rows)
    print(log_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    return stats_df

# =============================================================================
# MODEL FORMS
# =============================================================================

def build_model_matrix(df, model_form, has_height):
    """
    Build X matrix and y vector for a given model form.

    Model forms:
      'dbh'        : ln(AGB) = a + b*ln(DBH)
      'dbh_h'      : ln(AGB) = a + b1*ln(DBH) + b2*ln(H)
      'dbh2h'      : ln(AGB) = a + b*ln(DBH^2*H)
      'dbhh'       : ln(AGB) = a + b*ln(DBH*H)
    """
    y = np.log(df['agb'].values)

    if model_form == 'dbh':
        X = add_constant(np.log(df['dbh'].values))
        labels = ['Intercept (beta0)', 'ln(DBH) (beta1)']

    elif model_form == 'dbh_h':
        if not has_height:
            return None, None, None
        X = add_constant(np.column_stack([
            np.log(df['dbh'].values),
            np.log(df['height'].values)
        ]))
        labels = ['Intercept (beta0)', 'ln(DBH) (beta1)', 'ln(H) (beta2)']

    elif model_form == 'dbh2h':
        if not has_height:
            return None, None, None
        composite = np.log(df['dbh'].values**2 * df['height'].values)
        X = add_constant(composite)
        labels = ['Intercept (beta0)', 'ln(DBH^2*H) (beta1)']

    elif model_form == 'dbhh':
        if not has_height:
            return None, None, None
        composite = np.log(df['dbh'].values * df['height'].values)
        X = add_constant(composite)
        labels = ['Intercept (beta0)', 'ln(DBH*H) (beta1)']

    else:
        raise ValueError(f"Unknown model form: {model_form}")

    return X, y, labels

MODEL_NAMES = {
    'dbh'   : 'Model 1 - ln(AGB) = a + b*ln(DBH)',
    'dbh_h' : 'Model 2 - ln(AGB) = a + b1*ln(DBH) + b2*ln(H)',
    'dbh2h' : 'Model 3a - ln(AGB) = a + b*ln(DBH^2*H)',
    'dbhh'  : 'Model 3b - ln(AGB) = a + b*ln(DBH*H)',
}

# =============================================================================
# BASELINE: CLASSICAL OLS
# =============================================================================

def run_ols(X, y, labels):
    """Fit OLS model and return result object with diagnostics."""
    model  = OLS(y, X).fit()
    y_hat  = model.fittedvalues
    resid  = model.resid
    n, p   = len(y), X.shape[1]

    # AIC (manual - consistent with log-likelihood definition)
    aic = model.aic

    # RMSE on log scale
    rmse_log = np.sqrt(np.mean(resid**2))

    return {
        'model'     : model,
        'params'    : model.params,
        'labels'    : labels,
        'y'         : y,
        'y_hat'     : y_hat,
        'resid'     : resid,
        'r2'        : model.rsquared,
        'r2_adj'    : model.rsquared_adj,
        'aic'       : aic,
        'rmse_log'  : rmse_log,
        'n'         : n,
        'p'         : p,
        'X'         : X,
    }

# =============================================================================
# STEP 1: BREUSCH-PAGAN TEST
# =============================================================================

def breusch_pagan_test(ols_result):
    """
    Formal heteroscedasticity test.
    Returns (bp_stat, p_value, heteroscedastic: bool)
    """
    resid = ols_result['resid']
    X     = ols_result['X']

    # het_breuschpagan returns (lm_stat, lm_p, fstat, f_p)
    lm_stat, lm_p, fstat, f_p = het_breuschpagan(resid, X)

    heteroscedastic = lm_p < ALPHA

    print(f"\n  Breusch-Pagan Test:")
    print(f"    LM statistic : {lm_stat:.4f}")
    print(f"    p-value      : {lm_p:.4f}")
    print(f"    Decision     : {'WARNING: HETEROSCEDASTICITY DETECTED -> WLS will be applied'
                                if heteroscedastic
                                else 'OK: Homoscedasticity assumption not violated -> OLS retained'}")

    return lm_stat, lm_p, heteroscedastic

# =============================================================================
# STEP 2: WEIGHTED LEAST SQUARES (conditional)
# =============================================================================

def unweighted_r2(y, y_hat, p):
    """
    Coefficient of determination from unweighted log-scale residuals.
    Used for WLS fits so that training R2 is directly comparable with
    LOOCV-R2 (computed unweighted) and with OLS-fitted models.
    p = number of model parameters, including the intercept.
    """
    y, y_hat = np.asarray(y), np.asarray(y_hat)
    n = len(y)
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - p)
    return r2, r2_adj


def run_wls(X, y, ols_result, labels, dbh_vals):
    """
    WLS with inverse-square DBH weights: w_i = 1 / DBH_i^2
    Addresses residual heteroscedasticity associated with tree size
    using inverse-square DBH weights (Parresol, 1999).
    Only called when BP test is significant.
    """
    weights = 1.0 / (dbh_vals ** 2)
    weights = np.clip(weights, 1e-10, None)  # numerical stability

    model  = WLS(y, X, weights=weights).fit()
    y_hat  = model.fittedvalues
    resid  = model.resid
    n, p   = len(y), X.shape[1]

    rmse_log = np.sqrt(np.mean(resid**2))

    # Training R2 from unweighted log-scale residuals, so that it is on the
    # same scale as LOOCV-R2 and as R2 from OLS-fitted candidate models.
    r2, r2_adj = unweighted_r2(y, y_hat, p)

    return {
        'model'     : model,
        'params'    : model.params,
        'labels'    : labels,
        'y'         : y,
        'y_hat'     : y_hat,
        'resid'     : resid,
        'r2'        : r2,
        'r2_adj'    : r2_adj,
        'aic'       : model.aic,
        'rmse_log'  : rmse_log,
        'weights'   : weights,
        'n'         : n,
        'p'         : p,
        'X'         : X,
    }

# =============================================================================
# STEP 3: BOOTSTRAP (B = 5,000)
# =============================================================================

def bootstrap_model(X, y, B=B_BOOTSTRAP, use_wls=False,
                    dbh_vals=None, seed=RANDOM_SEED):
    """
    Non-parametric bootstrap resampling.
    Returns 95% percentile CIs for all parameters and R2.

    B = 5,000 iterations - sufficient for stable percentile CIs
    (Efron & Tibshirani, 1993; Davison & Hinkley, 1997).
    WLS weights: w_i = 1/DBH_i^2 (Parresol, 1999).
    """
    rng    = np.random.default_rng(seed)
    n      = len(y)
    params = np.zeros((B, X.shape[1]))
    r2s    = np.zeros(B)

    for b in range(B):
        idx = rng.integers(0, n, size=n)
        Xb, yb = X[idx], y[idx]

        try:
            if use_wls and dbh_vals is not None:
                w  = 1.0 / (dbh_vals[idx] ** 2)
                w  = np.clip(w, 1e-10, None)
                m  = WLS(yb, Xb, weights=w).fit()
            else:
                m  = OLS(yb, Xb).fit()
            params[b] = m.params
            r2s[b]    = unweighted_r2(yb, m.fittedvalues, Xb.shape[1])[0]
        except Exception:
            params[b] = np.nan
            r2s[b]    = np.nan

    # Remove failed iterations
    valid  = ~np.isnan(params).any(axis=1)
    params = params[valid]
    r2s    = r2s[valid]
    n_valid = valid.sum()

    lo, hi = (1 - CI_LEVEL) / 2 * 100, (1 + CI_LEVEL) / 2 * 100

    ci_params = np.percentile(params, [lo, hi], axis=0)
    ci_r2     = np.percentile(r2s,   [lo, hi])

    return {
        'params_dist' : params,
        'r2_dist'     : r2s,
        'ci_params'   : ci_params,   # shape (2, p): [lower, upper]
        'ci_r2'       : ci_r2,
        'n_valid'     : n_valid,
        'B'           : B,
    }

def print_bootstrap_results(boot, labels):
    print(f"\n  Bootstrap Results (B = {boot['B']:,}, "
          f"valid = {boot['n_valid']:,}):")
    print(f"  {'Parameter':<25} {'Point Est':>10} {'95% CI Lower':>14} "
          f"{'95% CI Upper':>14}")
    print(f"  {'-'*65}")
    # Note: point estimates printed separately from calling context
    for i, lbl in enumerate(labels):
        lo = boot['ci_params'][0, i]
        hi = boot['ci_params'][1, i]
        print(f"  {lbl:<25} {'[see OLS/WLS]':>10} {lo:>14.4f} {hi:>14.4f}")

    print(f"\n  R2 Bootstrap 95% CI: [{boot['ci_r2'][0]:.4f}, "
          f"{boot['ci_r2'][1]:.4f}]")

# =============================================================================
# STEP 4: LEAVE-ONE-OUT CROSS-VALIDATION
# =============================================================================

def loocv(X, y, use_wls=False, dbh_vals=None):
    """
    LOOCV: fit model n times, each time predicting the left-out observation.

    Returns LOOCV-R2, LOOCV-RMSE, and the overfitting gap dR2.
    WLS weights: w_i = 1/DBH_i^2 (Parresol, 1999).
    """
    n          = len(y)
    y_pred_cv  = np.zeros(n)

    for i in range(n):
        mask   = np.ones(n, dtype=bool)
        mask[i] = False
        Xtrain, ytrain = X[mask], y[mask]
        xtest          = X[i:i+1]

        try:
            if use_wls and dbh_vals is not None:
                w = 1.0 / (dbh_vals[mask] ** 2)
                w = np.clip(w, 1e-10, None)
                m = WLS(ytrain, Xtrain, weights=w).fit()
            else:
                m = OLS(ytrain, Xtrain).fit()
            y_pred_cv[i] = m.predict(xtest)[0]
        except Exception:
            y_pred_cv[i] = np.nan

    # Remove failed predictions
    valid       = ~np.isnan(y_pred_cv)
    y_cv        = y_pred_cv[valid]
    y_true      = y[valid]

    ss_res_cv   = np.sum((y_true - y_cv) ** 2)
    ss_tot      = np.sum((y_true - y_true.mean()) ** 2)
    loocv_r2    = 1 - ss_res_cv / ss_tot
    loocv_rmse  = np.sqrt(np.mean((y_true - y_cv) ** 2))

    return loocv_r2, loocv_rmse, y_pred_cv

# =============================================================================
# STEP 5: DUAN SMEARING FACTOR
# =============================================================================

def duan_smearing_factor(resid):
    """
    Non-parametric smearing factor (Duan, 1983).
    SF = mean(exp(residuals))

    Applied independently to OLS and WLS residuals.
    Corrects systematic underestimation bias from log back-transformation.
    """
    sf = np.mean(np.exp(resid))
    return sf

def back_transform(y_hat_log, sf):
    """Back-transform log-scale predictions to original scale (kg)."""
    return np.exp(y_hat_log) * sf

def compute_rmse_kg(y_kg_true, y_kg_pred):
    return np.sqrt(np.mean((y_kg_true - y_kg_pred) ** 2))

# =============================================================================
# STEP 6: PREDICTION INTERVALS
# =============================================================================

def prediction_intervals(fit_result, X_new, sf, ci_level=CI_LEVEL):
    """
    Compute 95% CI (mean prediction) and 95% PI (individual tree).

    CI: y0_hat +/- t * s * sqrt(x0' (X'X)^-1 x0)
    PI: y0_hat +/- t * s * sqrt(1 + x0' (X'X)^-1 x0)

    Both intervals are back-transformed using the Duan SF. The interval
    calculation uses the common unweighted X'X approximation for both OLS and
    WLS fits.
    """
    model  = fit_result['model']
    n, p   = fit_result['n'], fit_result['p']
    X      = fit_result['X']
    resid  = fit_result['resid']
    sigma  = np.sqrt(np.sum(resid**2) / (n - p))
    t_crit = stats.t.ppf((1 + ci_level) / 2, df=n - p)

    # XtX inverse
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(X.T @ X)

    y_hat_new  = model.predict(X_new)
    leverage   = np.array([x @ XtX_inv @ x for x in X_new])

    se_mean    = sigma * np.sqrt(leverage)
    se_pred    = sigma * np.sqrt(1 + leverage)

    ci_lower_log = y_hat_new - t_crit * se_mean
    ci_upper_log = y_hat_new + t_crit * se_mean
    pi_lower_log = y_hat_new - t_crit * se_pred
    pi_upper_log = y_hat_new + t_crit * se_pred

    return {
        'y_hat_log'    : y_hat_new,
        'y_hat_kg'     : np.exp(y_hat_new) * sf,
        'ci_lower_kg'  : np.exp(ci_lower_log) * sf,
        'ci_upper_kg'  : np.exp(ci_upper_log) * sf,
        'pi_lower_kg'  : np.exp(pi_lower_log) * sf,
        'pi_upper_kg'  : np.exp(pi_upper_log) * sf,
    }

# =============================================================================
# INFLUENCE DIAGNOSTICS 
# =============================================================================

def influence_diagnostics(fit_result):
    """
    Cook's Distance, leverage, standardised residuals.
    Flagged observations are REPORTED ONLY - not removed.
    Data integrity principle: all decisions made prior to pipeline.
    """
    model  = fit_result['model']
    n, p   = fit_result['n'], fit_result['p']
    X      = fit_result['X']
    resid  = fit_result['resid']
    sigma  = np.sqrt(np.sum(resid**2) / (n - p))

    # Standardised residuals
    std_resid = resid / sigma

    # Leverage (hat matrix diagonal)
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(X.T @ X)
    leverage = np.array([x @ XtX_inv @ x for x in X])

    # Cook's Distance
    cooks_d = (std_resid**2 * leverage) / (p * (1 - leverage + 1e-10))

    # Thresholds
    thresh_std    = 2.0
    thresh_lev    = 2 * p / n
    thresh_cooks  = 4 / n

    flagged = {
        'std_resid'  : np.where(np.abs(std_resid) > thresh_std)[0],
        'leverage'   : np.where(leverage > thresh_lev)[0],
        'cooks'      : np.where(cooks_d > thresh_cooks)[0],
    }

    print(f"\n  Influence Diagnostics (thresholds: |e*|>{thresh_std}, "
          f"h>{thresh_lev:.3f}, D>{thresh_cooks:.3f}):")

    for name, idx in flagged.items():
        if len(idx) > 0:
            print(f"    {name:>12}: observations {idx+1} flagged "
                  f"(tree index {list(idx+1)})")
        else:
            print(f"    {name:>12}: no observations flagged")

    print(f"  NOTE: Flagged observations reported for interpretation in "
          f"Results section only.\n        No data modifications applied "
          f"(data integrity principle - see accompanying documentation).")

    return {
        'std_resid' : std_resid,
        'leverage'  : leverage,
        'cooks_d'   : cooks_d,
        'flagged'   : flagged,
    }

# =============================================================================
# FULL PIPELINE FOR ONE MODEL FORM
# =============================================================================

def run_pipeline(df, model_form, cohort, has_height):
    """
    Execute the baseline OLS fit and the six-step pipeline for one model
    form on one cohort.
    Returns a results dictionary.
    """
    print(f"\n{'='*72}")
    print(f"  {MODEL_NAMES.get(model_form, model_form)}")
    print(f"{'='*72}")

    # Build model matrix
    X, y, labels = build_model_matrix(df, model_form, has_height)
    if X is None:
        print(f"  WARNING: Skipped - height data not available for this model form.")
        return None

    n = len(y)
    if n < 5:
        print(f"  WARNING: Skipped - insufficient observations (n={n})")
        return None

    y_kg = np.exp(y)   # original scale AGB

    # --- BASELINE: OLS ---
    print(f"\n  BASELINE: Classical OLS")
    ols = run_ols(X, y, labels)

    print(f"  {'Parameter':<25} {'Estimate':>12} {'Std Error':>12} "
          f"{'t-stat':>10} {'p-value':>10}")
    print(f"  {'-'*70}")
    for i, lbl in enumerate(labels):
        p_val = ols['model'].pvalues[i]
        t_val = ols['model'].tvalues[i]
        se    = ols['model'].bse[i]
        star  = '***' if p_val<0.001 else '**' if p_val<0.01 else \
                '*' if p_val<0.05 else ''
        print(f"  {lbl:<25} {ols['params'][i]:>12.4f} {se:>12.4f} "
              f"{t_val:>10.3f} {p_val:>10.4f} {star}")

    print(f"\n  R2={ols['r2']:.4f}  Adj.R2={ols['r2_adj']:.4f}  "
          f"AIC={ols['aic']:.3f}  RMSE(log)={ols['rmse_log']:.4f}")

    # Extract raw DBH values for WLS weights (Parresol, 1999)
    dbh_vals = df['dbh'].values

    # --- STEP 1: BREUSCH-PAGAN ---
    print(f"\n  STEP 1: Breusch-Pagan Heteroscedasticity Test")
    bp_stat, bp_p, heteroscedastic = breusch_pagan_test(ols)

    # --- STEP 2: WLS (conditional) ---
    if heteroscedastic:
        print(f"\n  STEP 2: Weighted Least Squares (BP p={bp_p:.4f} < {ALPHA})")
        wls = run_wls(X, y, ols, labels, dbh_vals)
        final = wls
        estimator = 'WLS'

        print(f"  {'Parameter':<25} {'Estimate':>12} {'Std Error':>12}")
        print(f"  {'-'*52}")
        for i, lbl in enumerate(labels):
            se = wls['model'].bse[i]
            print(f"  {lbl:<25} {wls['params'][i]:>12.4f} {se:>12.4f}")

        print(f"\n  WLS: R2={wls['r2']:.4f}  Adj.R2={wls['r2_adj']:.4f}  "
              f"AIC={wls['aic']:.3f}  RMSE(log)={wls['rmse_log']:.4f}")
    else:
        print(f"\n  STEP 2: WLS not required (BP p={bp_p:.4f} >= {ALPHA})")
        wls = None
        final = ols
        estimator = 'OLS'

    # --- STEP 3: BOOTSTRAP ---
    print(f"\n  STEP 3: Bootstrap (B = {B_BOOTSTRAP:,})")
    boot = bootstrap_model(
        X, y, B=B_BOOTSTRAP,
        use_wls=heteroscedastic,
        dbh_vals=dbh_vals
    )
    print_bootstrap_results(boot, labels)

    # Print point estimates alongside bootstrap CIs
    print(f"\n  Point estimates vs bootstrap CIs:")
    print(f"  {'Parameter':<25} {'Point Est':>12} {'95% CI':>22}")
    print(f"  {'-'*62}")
    for i, lbl in enumerate(labels):
        lo = boot['ci_params'][0, i]
        hi = boot['ci_params'][1, i]
        print(f"  {lbl:<25} {final['params'][i]:>12.4f} "
              f"  [{lo:.4f}, {hi:.4f}]")

    r2_pt = final['r2']
    print(f"  {'R2':<25} {r2_pt:>12.4f}   "
          f"[{boot['ci_r2'][0]:.4f}, {boot['ci_r2'][1]:.4f}]")

    # --- STEP 4: LOOCV ---
    print(f"\n  STEP 4: Leave-One-Out Cross-Validation")
    loocv_r2, loocv_rmse, y_cv = loocv(
        X, y,
        use_wls=heteroscedastic,
        dbh_vals=dbh_vals
    )
    delta_r2 = final['r2'] - loocv_r2
    print(f"  Training R2    : {final['r2']:.4f}")
    print(f"  LOOCV-R2       : {loocv_r2:.4f}")
    print(f"  Overfit gap dR2: {delta_r2:.4f} "
          f"({'WARNING: substantial' if delta_r2>0.15 else 'OK: acceptable'})")
    print(f"  LOOCV-RMSE     : {loocv_rmse:.4f} (log scale)")

    # --- STEP 5: DUAN SMEARING FACTOR ---
    print(f"\n  STEP 5: Duan Smearing Factor")
    sf = duan_smearing_factor(final['resid'])
    y_kg_pred = back_transform(final['y_hat'], sf)
    rmse_kg   = compute_rmse_kg(y_kg, y_kg_pred)

    print(f"  Smearing Factor (SF) : {sf:.6f}")
    print(f"  RMSE (kg dry weight) : {rmse_kg:.4f} kg")

    # --- STEP 6: PREDICTION INTERVALS ---
    print(f"\n  STEP 6: Prediction Intervals")

    # At dataset means (mean DBH, plus mean height for height-based forms)
    dbh_mean  = df['dbh'].mean()
    dbh_range = np.linspace(df['dbh'].min(), df['dbh'].max(), 100)

    # Build X_new at the dataset means
    def make_x_new(dbh_val, h_val=None):
        if model_form == 'dbh':
            return np.array([[1, np.log(dbh_val)]])
        elif model_form == 'dbh_h' and h_val is not None:
            return np.array([[1, np.log(dbh_val), np.log(h_val)]])
        elif model_form == 'dbh2h' and h_val is not None:
            return np.array([[1, np.log(dbh_val**2 * h_val)]])
        elif model_form == 'dbhh' and h_val is not None:
            return np.array([[1, np.log(dbh_val * h_val)]])
        return None

    h_mean = df['height'].mean() if has_height else None
    x_mean = make_x_new(dbh_mean, h_mean)

    if x_mean is not None:
        pi_mean = prediction_intervals(final, x_mean, sf)
        if model_form == 'dbh':
            print(f"  At mean DBH = {dbh_mean:.2f} cm:")
        else:
            print(f"  At mean DBH = {dbh_mean:.2f} cm and mean H = {h_mean:.2f} m:")
        print(f"    Point prediction : {pi_mean['y_hat_kg'][0]:.3f} kg")
        print(f"    95% CI           : [{pi_mean['ci_lower_kg'][0]:.3f}, "
              f"{pi_mean['ci_upper_kg'][0]:.3f}] kg")
        print(f"    95% PI           : [{pi_mean['pi_lower_kg'][0]:.3f}, "
              f"{pi_mean['pi_upper_kg'][0]:.3f}] kg")

    # Build prediction curve across DBH range
    if model_form == 'dbh':
        X_range = np.column_stack([np.ones(100), np.log(dbh_range)])
    elif model_form == 'dbh_h' and has_height:
        X_range = np.column_stack([np.ones(100), np.log(dbh_range),
                                   np.full(100, np.log(h_mean))])
    elif model_form == 'dbh2h' and has_height:
        X_range = np.column_stack([np.ones(100),
                                   np.log(dbh_range**2 * h_mean)])
    elif model_form == 'dbhh' and has_height:
        X_range = np.column_stack([np.ones(100),
                                   np.log(dbh_range * h_mean)])
    else:
        X_range = None

    pi_curve = prediction_intervals(final, X_range, sf) if X_range is not None else None

    # --- INFLUENCE DIAGNOSTICS ---
    influence = influence_diagnostics(final)

    # --- RETURN ALL RESULTS ---
    return {
        'model_form'        : model_form,
        'cohort'            : cohort,
        'estimator'         : estimator,
        'n'                 : n,
        'labels'            : labels,
        'ols'               : ols,
        'wls'               : wls,
        'final'             : final,
        'bp_stat'           : bp_stat,
        'bp_p'              : bp_p,
        'heteroscedastic'   : heteroscedastic,
        'bootstrap'         : boot,
        'loocv_r2'          : loocv_r2,
        'loocv_rmse'        : loocv_rmse,
        'loocv_predictions' : y_cv,
        'delta_r2'          : delta_r2,
        'sf'                : sf,
        'rmse_kg'           : rmse_kg,
        'y_kg'              : y_kg,
        'y_kg_pred'         : y_kg_pred,
        'pi_at_mean'        : pi_mean if x_mean is not None else None,
        'dbh_mean'          : dbh_mean,
        'h_mean'            : h_mean if model_form != 'dbh' else None,
        'dbh_range'         : dbh_range,
        'pi_curve'          : pi_curve,
        'influence'         : influence,
        'X'                 : X,
        'y'                 : y,
        'dbh_vals'          : dbh_vals,
        'beta2_pval'        : (final['model'].pvalues[2]
                               if (model_form == 'dbh_h'
                               and len(final['model'].pvalues) > 2)
                               else None),
        # CI width of beta1 - narrower = more precise = better for MRV
        'ci_width'          : float(boot['ci_params'][1, 1]
                               - boot['ci_params'][0, 1]),
    }

# =============================================================================
# MODEL SELECTION SUMMARY
# =============================================================================

def rank_eligible_models(results_list):
    """
    Rank eligible candidate equations across five equally weighted criteria:
    LOOCV-R2 (higher), dR2 (lower), AIC (lower), RMSE(kg) (lower),
    and bootstrap CI width (lower).

    The function returns the full ranked table and the top-ranked eligible
    candidate. The pipeline does not designate a final retained equation;
    biological plausibility review remains an analyst-driven step.
    """
    valid = [r for r in results_list if r is not None]
    if not valid:
        return None

    # Exclude M2 (dbh_h) if beta2 (height coefficient) not significant
    # Retaining a non-significant predictor inflates complexity without
    # improving predictive value (Sileshi, 2014; Dutca et al., 2020).
    eligible = [
        r for r in valid
        if not (r['model_form'] == 'dbh_h'
                and r.get('beta2_pval') is not None
                and r['beta2_pval'] >= 0.05)
    ]
    if not eligible:
        eligible = valid
        print('  NOTE: All models have non-significant beta2; using full set.')
    else:
        for r in valid:
            if r not in eligible:
                pv = r.get('beta2_pval', None)
                print(f'  NOTE: M2 excluded - beta2 p={pv:.4f} >= 0.05')

    df_sel = pd.DataFrame([{
        'model_form'  : r['model_form'],
        'estimator'   : r['estimator'],
        'n'           : r['n'],
        'r2'          : r['final']['r2'],
        'r2_adj'      : r['final']['r2_adj'],
        'aic'         : r['final']['aic'],
        'loocv_r2'    : r['loocv_r2'],
        'loocv_rmse'  : r['loocv_rmse'],
        'delta_r2'    : r['delta_r2'],
        'rmse_kg'     : r['rmse_kg'],
        'sf'          : r['sf'],
        'bp_p'        : r['bp_p'],
        'ci_width'    : r['ci_width'],
    } for r in eligible])

    # Composite ranking across five equally weighted criteria.
    # Each eligible model receives a rank for each criterion; the lowest
    # summed composite score indicates the highest statistical support.
    # CI width rewards MRV-relevant parameter precision:
    #   LOOCV-R2  higher is better (ascending=False)
    #   dR2       lower is better  (ascending=True)  - overfitting gap
    #   AIC       lower is better  (ascending=True)
    #   RMSE(kg)  lower is better  (ascending=True)  - original-scale prediction error
    #   CI width  lower is better  (ascending=True)  - parameter precision (MRV alignment)
    df_sel['loocv_rank']    = df_sel['loocv_r2'].rank(ascending=False)
    df_sel['delta_r2_rank'] = df_sel['delta_r2'].rank(ascending=True)
    df_sel['aic_rank']      = df_sel['aic'].rank(ascending=True)
    df_sel['rmse_kg_rank']  = df_sel['rmse_kg'].rank(ascending=True)
    df_sel['ci_width_rank'] = df_sel['ci_width'].rank(ascending=True)
    df_sel['composite']     = (df_sel['loocv_rank']
                               + df_sel['delta_r2_rank']
                               + df_sel['aic_rank']
                               + df_sel['rmse_kg_rank']
                               + df_sel['ci_width_rank'])
    df_sel = df_sel.sort_values('composite')

    # Return the full ranked table plus the top-ranked eligible candidate.
    # Final equation retention remains subject to analyst review for biological
    # plausibility, consistent with the manuscript workflow.
    top_ranked_result = eligible[df_sel.index[0]]
    return df_sel, top_ranked_result

def print_model_comparison(df_sel, cohort):
    subsection(f"MODEL COMPARISON TABLE - {cohort}-YEAR COHORT")
    # Performance metrics
    cols = ['model_form','estimator','n','r2','r2_adj','aic',
            'loocv_r2','delta_r2','rmse_kg','sf','bp_p']
    print(df_sel[cols].to_string(index=False,
          float_format=lambda x: f"{x:.4f}"))
    # Composite ranking breakdown
    print(f"\n  Composite ranking (1 = best per criterion):")
    rank_cols = ['model_form','loocv_rank','delta_r2_rank',
                 'aic_rank','rmse_kg_rank','ci_width_rank','composite']
    available = [c for c in rank_cols if c in df_sel.columns]
    print(df_sel[available].to_string(index=False,
          float_format=lambda x: f"{x:.0f}"))
    print(f"\n  Eligible models ranked from highest to lowest statistical support")
    print(f"  using five equally weighted criteria.")
    print(f"  Review equation form, beta1 and its bootstrap 95% CI for biological")
    print(f"  plausibility before retaining a final equation.")
    print(f"  Rank 1 is the top-ranked eligible equation; it is not an automatic")
    print(f"  final recommendation.")
# =============================================================================
# VISUALISATION
# =============================================================================

def plot_cohort(cohort_results, top_ranked_result, cohort):
    """
    9-panel diagnostic figure for the top-ranked eligible equation.
    """
    r   = top_ranked_result
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        f"Allometric Model Diagnostics - {cohort}-Year Cohort\n"
        f"{MODEL_NAMES.get(r['model_form'], r['model_form'])} "
        f"[{r['estimator']}] | n={r['n']}",
        fontsize=14, fontweight='bold', y=0.98
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    y      = r['y']
    y_hat  = r['final']['y_hat']
    resid  = r['final']['resid']
    y_kg   = r['y_kg']
    y_pred = r['y_kg_pred']

    # Panel 1 - Log-log scatter with regression line
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(r['X'][:, 1], y, color=COLOURS['scatter'],
                edgecolors='white', zorder=3, s=60, label='Observed')
    x_line = np.linspace(r['X'][:, 1].min(), r['X'][:, 1].max(), 100)
    if r['model_form'] == 'dbh':
        y_line = r['final']['params'][0] + r['final']['params'][1] * x_line
    else:
        y_line = r['final']['model'].predict(
            np.column_stack([np.ones(100), x_line,
                             np.full(100, r['X'][:, -1].mean())])
            if r['X'].shape[1] == 3 else
            np.column_stack([np.ones(100), x_line])
        )
    ax1.plot(x_line, y_line, color=COLOURS['ols'], lw=2, label=r['estimator'])
    ax1.set_xlabel('ln(DBH)', fontsize=9)
    ax1.set_ylabel('ln(AGB)', fontsize=9)
    ax1.set_title('Log-log Regression', fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(True, color=COLOURS['grid'], alpha=0.8)

    # Panel 2 - Observed vs Predicted (kg scale)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(y_kg, y_pred, color=COLOURS['ci'],
                edgecolors='white', zorder=3, s=60)
    lim = [min(y_kg.min(), y_pred.min()) * 0.9,
           max(y_kg.max(), y_pred.max()) * 1.1]
    ax2.plot(lim, lim, 'k--', lw=1, alpha=0.7, label='1:1 line')
    ax2.set_xlabel('Observed AGB (kg)', fontsize=9)
    ax2.set_ylabel('Predicted AGB (kg)', fontsize=9)
    ax2.set_title('Observed vs Predicted (kg)', fontsize=10)
    ax2.set_xlim(lim); ax2.set_ylim(lim)
    ax2.legend(fontsize=8)
    ax2.grid(True, color=COLOURS['grid'], alpha=0.8)

    # Panel 3 - Prediction intervals across DBH range
    ax3 = fig.add_subplot(gs[0, 2])
    if r['pi_curve'] is not None:
        dbh_r = r['dbh_range']
        pc    = r['pi_curve']
        ax3.fill_between(dbh_r, pc['pi_lower_kg'], pc['pi_upper_kg'],
                         alpha=0.2, color=COLOURS['pi'], label='95% PI')
        ax3.fill_between(dbh_r, pc['ci_lower_kg'], pc['ci_upper_kg'],
                         alpha=0.4, color=COLOURS['ci'], label='95% CI')
        ax3.plot(dbh_r, pc['y_hat_kg'],
                 color=COLOURS['ols'], lw=2, label='Predicted')
        # Use raw DBH for all model forms
        dbh_obs = np.exp(r['X'][:, 1]) if r['model_form'] == 'dbh' else r['dbh_vals']
        ax3.scatter(dbh_obs, y_kg, color=COLOURS['scatter'],
                    edgecolors='white', zorder=3, s=60, label='Observed')
    ax3.set_xlabel('DBH (cm)', fontsize=9)
    ax3.set_ylabel('AGB (kg)', fontsize=9)
    ax3.set_title('Prediction & Confidence Intervals', fontsize=10)
    ax3.legend(fontsize=7)
    ax3.grid(True, color=COLOURS['grid'], alpha=0.8)

    # Panel 4 - Residuals vs Fitted
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.scatter(y_hat, resid, color=COLOURS['residual'],
                edgecolors='white', zorder=3, s=60)
    ax4.axhline(0, color='black', lw=1, linestyle='--')
    ax4.set_xlabel('Fitted values (log scale)', fontsize=9)
    ax4.set_ylabel('Residuals', fontsize=9)
    ax4.set_title('Residuals vs Fitted', fontsize=10)
    ax4.grid(True, color=COLOURS['grid'], alpha=0.8)

    # Panel 5 - Q-Q plot
    ax5 = fig.add_subplot(gs[1, 1])
    (osm, osr), (slope, intercept, _) = stats.probplot(resid, dist='norm')
    ax5.scatter(osm, osr, color=COLOURS['residual'],
                edgecolors='white', zorder=3, s=60)
    line_x = np.array([osm[0], osm[-1]])
    ax5.plot(line_x, slope * line_x + intercept,
             color=COLOURS['ols'], lw=2)
    sw_stat, sw_p = stats.shapiro(resid)
    ax5.set_xlabel('Theoretical Quantiles', fontsize=9)
    ax5.set_ylabel('Sample Quantiles', fontsize=9)
    ax5.set_title(f'Normal Q-Q Plot\n(Shapiro-Wilk: W={sw_stat:.3f}, p={sw_p:.3f})',
                  fontsize=10)
    ax5.grid(True, color=COLOURS['grid'], alpha=0.8)

    # Panel 6 - Residual histogram
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.hist(resid, bins=min(10, r['n']//2 + 1),
             color=COLOURS['residual'], edgecolor='white', alpha=0.8)
    x_norm = np.linspace(resid.min(), resid.max(), 100)
    ax6_twin = ax6.twinx()
    ax6_twin.plot(x_norm, stats.norm.pdf(x_norm, resid.mean(), resid.std()),
                  color=COLOURS['ols'], lw=2)
    ax6_twin.set_yticks([])
    ax6.set_xlabel('Residuals', fontsize=9)
    ax6.set_ylabel('Frequency', fontsize=9)
    ax6.set_title('Residual Distribution', fontsize=10)
    ax6.grid(True, color=COLOURS['grid'], alpha=0.8)

    # Panel 7 - Bootstrap distribution of beta1
    ax7 = fig.add_subplot(gs[2, 0])
    boot_b1 = r['bootstrap']['params_dist'][:, 1]
    ax7.hist(boot_b1, bins=50, color=COLOURS['bootstrap'],
             edgecolor='white', alpha=0.8)
    ci_lo = r['bootstrap']['ci_params'][0, 1]
    ci_hi = r['bootstrap']['ci_params'][1, 1]
    ax7.axvline(r['final']['params'][1], color=COLOURS['ols'],
                lw=2, linestyle='-', label=f'Point={r["final"]["params"][1]:.3f}')
    ax7.axvline(ci_lo, color='red', lw=1.5, linestyle='--',
                label=f'95% CI [{ci_lo:.3f}, {ci_hi:.3f}]')
    ax7.axvline(ci_hi, color='red', lw=1.5, linestyle='--')
    b1_desc = ('DBH coefficient' if r['model_form'] == 'dbh_h'
               else 'allometric exponent')
    ax7.set_xlabel(rf'$\beta_1$ ({b1_desc})', fontsize=9)
    ax7.set_ylabel('Frequency', fontsize=9)
    ax7.set_title(f'Bootstrap Distribution of $\\beta_1$\n(B={B_BOOTSTRAP:,})', fontsize=10)
    ax7.legend(fontsize=7)
    ax7.grid(True, color=COLOURS['grid'], alpha=0.8)

    # Panel 8 - LOOCV: predicted vs observed
    ax8 = fig.add_subplot(gs[2, 1])
    valid_mask = ~np.isnan(r['loocv_predictions'])
    y_cv_valid = r['loocv_predictions'][valid_mask]
    y_true_cv  = r['y'][valid_mask]
    ax8.scatter(y_true_cv, y_cv_valid, color=COLOURS['bootstrap'],
                edgecolors='white', zorder=3, s=60)
    lim2 = [min(y_true_cv.min(), y_cv_valid.min()) * 0.95,
            max(y_true_cv.max(), y_cv_valid.max()) * 1.05]
    ax8.plot(lim2, lim2, 'k--', lw=1, alpha=0.7)
    ax8.set_xlabel('Observed ln(AGB)', fontsize=9)
    ax8.set_ylabel('LOOCV Predicted ln(AGB)', fontsize=9)
    ax8.set_title(f'LOOCV Validation\n($R^2$={r["loocv_r2"]:.4f}, '
                  f'$\\Delta R^2$={r["delta_r2"]:.4f})', fontsize=10)
    ax8.grid(True, color=COLOURS['grid'], alpha=0.8)

    # Panel 9 - Cook's Distance
    ax9 = fig.add_subplot(gs[2, 2])
    cooks = r['influence']['cooks_d']
    ax9.bar(range(1, r['n']+1), cooks,
            color=COLOURS['residual'], edgecolor='white', alpha=0.8)
    thresh_c = 4 / r['n']
    ax9.axhline(thresh_c, color='red', lw=1.5, linestyle='--',
                label=f'Threshold = 4/n = {thresh_c:.3f}')
    ax9.set_xlabel("Tree index", fontsize=9)
    ax9.set_ylabel("Cook's Distance", fontsize=9)
    ax9.set_title("Cook's Distance\n(influence diagnostics)", fontsize=10)
    ax9.legend(fontsize=8)
    ax9.grid(True, color=COLOURS['grid'], alpha=0.8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fname = f"{OUTPUT_DIR}/{cohort}yr_{r['model_form']}_diagnostics.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Figure saved: {fname}")

# =============================================================================
# RESULTS EXPORT TO EXCEL
# =============================================================================

def export_results(all_results, df_original):
    """Export comprehensive results to Excel workbook."""
    fname = f"{OUTPUT_DIR}/allometric_pipeline_results.xlsx"

    with pd.ExcelWriter(fname, engine='openpyxl') as writer:

        # Sheet 1 - Summary table across all cohorts and models
        summary_rows = []
        for cohort, results_list in all_results.items():
            for r in results_list:
                if r is None:
                    continue
                row = {
                    'Cohort'            : cohort,
                    'Model Form'        : MODEL_NAMES.get(r['model_form'], ''),
                    'Estimator'         : r['estimator'],
                    'n'                 : r['n'],
                    'R2'                : round(r['final']['r2'], 4),
                    'Adj R2'            : round(r['final']['r2_adj'], 4),
                    'AIC'               : round(r['final']['aic'], 3),
                    'RMSE (log)'        : round(r['final']['rmse_log'], 4),
                    'LOOCV R2'          : round(r['loocv_r2'], 4),
                    'LOOCV RMSE (log)'  : round(r['loocv_rmse'], 4),
                    'Delta R2 (overfit)'     : round(r['delta_r2'], 4),
                    'RMSE (kg)'         : round(r['rmse_kg'], 4),
                    'Smearing Factor'   : round(r['sf'], 6),
                    'BP p-value'        : round(r['bp_p'], 4),
                    'Heteroscedastic'   : r['heteroscedastic'],
                }
                # Parameters
                for i, lbl in enumerate(r['labels']):
                    row[f'Param: {lbl}'] = round(r['final']['params'][i], 4)
                    row[f'Bootstrap CI lower: {lbl}'] = \
                        round(r['bootstrap']['ci_params'][0, i], 4)
                    row[f'Bootstrap CI upper: {lbl}'] = \
                        round(r['bootstrap']['ci_params'][1, i], 4)
                row['Bootstrap R2 CI lower'] = round(r['bootstrap']['ci_r2'][0], 4)
                row['Bootstrap R2 CI upper'] = round(r['bootstrap']['ci_r2'][1], 4)
                summary_rows.append(row)

        pd.DataFrame(summary_rows).to_excel(
            writer, sheet_name='Summary', index=False)

        # Sheet 2 - Raw data with predictions from top-ranked eligible model
        pred_rows = []
        for cohort, results_list in all_results.items():
            # Use the top-ranked eligible equation (Rank 1 per composite ranking).
            valid_r = [r for r in results_list if r is not None]
            if not valid_r:
                continue
            _, best_r = rank_eligible_models(valid_r)
            if best_r is None:
                continue
            cohort_df = df_original[df_original['cohort'] == cohort].copy()
            cohort_df = cohort_df.reset_index(drop=True)
            cohort_df['ln_dbh']     = np.log(cohort_df['dbh'])
            cohort_df['ln_agb']     = np.log(cohort_df['agb'])
            cohort_df['ln_agb_pred'] = best_r['final']['y_hat']
            cohort_df['agb_pred_kg'] = best_r['y_kg_pred']
            cohort_df['residual_log'] = best_r['final']['resid']
            cohort_df['std_resid']   = best_r['influence']['std_resid']
            cohort_df['leverage']    = best_r['influence']['leverage']
            cohort_df['cooks_d']     = best_r['influence']['cooks_d']
            pred_rows.append(cohort_df)

        if pred_rows:
            pd.concat(pred_rows).to_excel(
                writer, sheet_name='Predictions', index=False)

        # Sheet 3 - Prediction intervals at dataset means
        #   (mean DBH, and mean height for height-based model forms)
        pi_rows = []
        for cohort, results_list in all_results.items():
            for r in results_list:
                if r is None or r['pi_at_mean'] is None:
                    continue
                pi = r['pi_at_mean']
                pi_rows.append({
                    'Cohort'        : cohort,
                    'Model Form'    : r['model_form'],
                    'Mean DBH (cm)' : round(r['dbh_mean'], 2),
                    'Mean H (m)'    : (round(r['h_mean'], 2)
                                       if r.get('h_mean') is not None
                                       else None),
                    'Predicted AGB (kg)' : round(pi['y_hat_kg'][0], 3),
                    '95% CI lower (kg)' : round(pi['ci_lower_kg'][0], 3),
                    '95% CI upper (kg)' : round(pi['ci_upper_kg'][0], 3),
                    '95% PI lower (kg)' : round(pi['pi_lower_kg'][0], 3),
                    '95% PI upper (kg)' : round(pi['pi_upper_kg'][0], 3),
                })
        pd.DataFrame(pi_rows).to_excel(
            writer, sheet_name='Prediction Intervals', index=False)

    print(f"\n  OK: Results exported to: {fname}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    np.random.seed(RANDOM_SEED)
    make_output_dir()

    print(sep())
    print("  UNCERTAINTY-AWARE ALLOMETRIC MODELLING PIPELINE")
    print("  Six-Step Integrated Analytical Pipeline")
    print(sep())

    # Load data
    df, has_height = load_data(DATA_FILE)

    # Determine cohorts to process
    available_cohorts = sorted(df['cohort'].unique().tolist())
    cohorts_to_run    = COHORTS if COHORTS else available_cohorts
    print(f"\n  Cohorts to process: {cohorts_to_run}")
    print(f"  Height data available: {has_height}")

    # Determine model forms to test
    if has_height:
        model_forms = ['dbh', 'dbh_h', 'dbh2h', 'dbhh']
    else:
        model_forms = ['dbh']
        print("  WARNING: Height not available - only Model 1 (DBH only) will be tested.")

    all_results = {}

    # Process each cohort
    for cohort in cohorts_to_run:
        section(f"COHORT: {cohort}-YEAR-OLD STAND")

        cohort_df = df[df['cohort'] == cohort].copy().reset_index(drop=True)
        n = len(cohort_df)

        if n < 5:
            print(f"  WARNING: Only {n} observations - cohort skipped.")
            continue

        print(f"\n  n = {n} trees")

        # Descriptive statistics
        descriptive_statistics(cohort_df, cohort, has_height)

        # Run pipeline for each model form
        results_list = []
        for mf in model_forms:
            result = run_pipeline(cohort_df, mf, cohort, has_height)
            results_list.append(result)

        all_results[cohort] = results_list

        # Model selection
        valid_results = [r for r in results_list if r is not None]
        if valid_results:
            df_sel, top_ranked = rank_eligible_models(valid_results)
            print_model_comparison(df_sel, cohort)

            # Diagnostic figure for top-ranked eligible equation
            plot_cohort(results_list, top_ranked, cohort)

    # Cross-cohort comparison
    if len(all_results) > 1:
        section("CROSS-COHORT COMPARISON - TOP-RANKED ELIGIBLE EQUATIONS")
        comp_rows = []
        for cohort, rlist in all_results.items():
            # Use the top-ranked eligible equation per cohort.
            valid_r = [x for x in rlist if x is not None]
            if not valid_r:
                continue
            _, r = rank_eligible_models(valid_r)
            if r is None:
                continue
            comp_rows.append({
                    'Cohort'          : f"{cohort}-year",
                    'Model'           : r['model_form'],
                    'n'               : r['n'],
                    'beta0'              : round(r['final']['params'][0], 4),
                    'beta1'              : round(r['final']['params'][1], 4),
                    'R2'              : round(r['final']['r2'], 4),
                    'LOOCV-R2'        : round(r['loocv_r2'], 4),
                    'Delta R2'        : round(r['delta_r2'], 4),
                    'RMSE (kg)'       : round(r['rmse_kg'], 4),
                    'SF'              : round(r['sf'], 6),
                    'Estimator'       : r['estimator'],
                    'BP p-value'      : round(r['bp_p'], 4),
                })
        if comp_rows:
            cdf = pd.DataFrame(comp_rows)
            print(cdf.to_string(index=False,
                  float_format=lambda x: f"{x:.4f}"))

    # Export to Excel
    export_results(all_results, df)

    section("PIPELINE COMPLETE")
    print(f"  All results saved to: {os.path.abspath(OUTPUT_DIR)}/")
    print(f"  Files generated:")
    for f in os.listdir(OUTPUT_DIR):
        print(f"    - {f}")
    print(f"\n{sep()}\n")

# =============================================================================

if __name__ == "__main__":
    main()
