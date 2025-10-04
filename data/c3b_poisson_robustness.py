"""
c3_poisson_robustness.py
=========================
Poisson regression robustness check for C3 event study.

Key differences from C3v2 OLS:
- Outcome: attempts_2b (COUNT) instead of attempt_rate
- Model: Poisson GLM with log(opportunities_2b) offset
- No pitcher FE (GLM limitation), only year FE via dummies
- Cluster-robust SE by pitcher

Interpretation:
- Poisson coefficients = log(rate ratio)
- exp(coef) = multiplicative effect on attempt rate
- Compare to OLS: if similar, validates rate model

Treatment context:
- 2023: 15s/20s pitch timer + 2 pickoff limit + larger bases
- 2024: 18s (runners on) + same limits
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Poisson
from statsmodels.genmod.cov_struct import Exchangeable
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FILE = Path("analysis/c_running_game/c_panel_with_baseline.csv")
OUTPUT_DIR = Path("analysis/c3_poisson")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_YEAR = 2022
BASE_TERCILE = 'T2'
EXCLUDE_2020 = True

# ============================================================================
# LOAD DATA
# ============================================================================

print("=" * 70)
print("C3 POISSON ROBUSTNESS CHECK")
print("=" * 70)

print("\n" + "-" * 70)
print("LOADING DATA")
print("-" * 70)

try:
    df = pd.read_csv(INPUT_FILE)
    print(f"\nLoaded: {len(df):,} pitcher-season observations")
except Exception as e:
    print(f"\nERROR loading data: {e}")
    sys.exit(1)

# Filter to baseline sample
df_baseline = df[df['in_baseline_2022'] == 1].copy()
print(f"Filtered to baseline sample: {len(df_baseline):,} observations")

# Exclude 2020
if EXCLUDE_2020:
    n_before = len(df_baseline)
    df_baseline = df_baseline[df_baseline['season'] != 2020].copy()
    n_dropped = n_before - len(df_baseline)
    print(f"Excluded 2020 (COVID): dropped {n_dropped} observations")

print(f"Final sample: {len(df_baseline):,} observations")
print(f"  Pitchers: {df_baseline['pitcher_id'].nunique():,}")
print(f"  Years: {sorted(df_baseline['season'].unique())}")

# ============================================================================
# PREPARE DATA
# ============================================================================

print("\n" + "-" * 70)
print("PREPARING POISSON MODEL DATA")
print("-" * 70)

# Create year dummies
years = sorted(df_baseline['season'].unique())
years_non_base = [y for y in years if y != BASE_YEAR]

for year in years_non_base:
    df_baseline[f'year_{year}'] = (df_baseline['season'] == year).astype(int)

print(f"\nYear dummies created (base={BASE_YEAR}):")
print(f"  {years_non_base}")

# Create year × tercile interactions
interaction_cols = []
for year in years_non_base:
    for tercile in ['T1', 'T3']:
        int_col = f'y{year}_x_{tercile}'
        df_baseline[int_col] = (
            (df_baseline['season'] == year) & 
            (df_baseline['baseline_group'] == tercile)
        ).astype(int)
        interaction_cols.append(int_col)

print(f"Year×Tercile interactions: {len(interaction_cols)}")

# Create tercile dummies (needed since no entity FE)
df_baseline['tercile_T1'] = (df_baseline['baseline_group'] == 'T1').astype(int)
df_baseline['tercile_T3'] = (df_baseline['baseline_group'] == 'T3').astype(int)

# Check for missing values
df_baseline = df_baseline.dropna(subset=['attempts_2b', 'opportunities_2b'])
print(f"\nAfter dropping NA: {len(df_baseline):,} observations")

# Create log offset
df_baseline['log_opp'] = np.log(df_baseline['opportunities_2b'])

print(f"\nPoisson outcome: attempts_2b (count)")
print(f"  Mean: {df_baseline['attempts_2b'].mean():.2f}")
print(f"  Range: [{df_baseline['attempts_2b'].min():.0f}, {df_baseline['attempts_2b'].max():.0f}]")

print(f"\nOffset: log(opportunities_2b)")
print(f"  Mean opportunities: {df_baseline['opportunities_2b'].mean():.1f}")
print(f"  Range: [{df_baseline['opportunities_2b'].min():.0f}, {df_baseline['opportunities_2b'].max():.0f}]")

# ============================================================================
# POISSON MODEL
# ============================================================================

print("\n" + "-" * 70)
print("FITTING POISSON GLM")
print("-" * 70)

# Prepare regressors
year_dummy_cols = [f'year_{y}' for y in years_non_base]
tercile_cols = ['tercile_T1', 'tercile_T3']
all_regressors = tercile_cols + year_dummy_cols + interaction_cols

X = df_baseline[all_regressors]
X = sm.add_constant(X, has_constant='add')

y = df_baseline['attempts_2b']
offset = df_baseline['log_opp']

print(f"\nModel specification:")
print(f"  Outcome: attempts_2b (count)")
print(f"  Offset: log(opportunities_2b)")
print(f"  Regressors: {len(all_regressors)} (terciles + years + interactions)")
print(f"  Entity FE: None (GLM limitation)")
print(f"  Cluster SE: By pitcher ({df_baseline['pitcher_id'].nunique()} clusters)")

# Fit Poisson GLM
print(f"\nFitting Poisson GLM...")
poisson_model = sm.GLM(
    endog=y,
    exog=X,
    offset=offset,
    family=Poisson()
)

try:
    poisson_result = poisson_model.fit(
        cov_type='cluster',
        cov_kwds={'groups': df_baseline['pitcher_id'].values}
    )
    print(f"✓ Model converged")
    print(f"  Log-likelihood: {poisson_result.llf:.1f}")
    print(f"  AIC: {poisson_result.aic:.1f}")
    print(f"  Observations: {poisson_result.nobs:.0f}")
except Exception as e:
    print(f"✗ Model failed: {e}")
    sys.exit(1)

# ============================================================================
# LOAD C3V2 OLS RESULTS FOR COMPARISON
# ============================================================================

print("\n" + "-" * 70)
print("LOADING C3V2 OLS RESULTS FOR COMPARISON")
print("-" * 70)

try:
    df_ols = pd.read_csv("analysis/c3_event_study/c3_v2_coefficients.csv")
    print(f"✓ Loaded C3v2 OLS coefficients: {len(df_ols)} rows")
except Exception as e:
    print(f"✗ Could not load C3v2 results: {e}")
    print(f"  Will save Poisson results without comparison")
    df_ols = None

# ============================================================================
# EXTRACT AND COMPARE COEFFICIENTS
# ============================================================================

print("\n" + "-" * 70)
print("EXTRACTING COEFFICIENTS")
print("-" * 70)

params = poisson_result.params
se = poisson_result.bse
pvals = poisson_result.pvalues

# Build coefficient comparison table
coef_data = []

for year in years:
    # T2 (reference): main year effect only
    if year == BASE_YEAR:
        # Base year = 0 by construction
        coef_data.append({
            'year': year,
            'tercile': 'T2',
            'poisson_coef': 0.0,
            'poisson_se': 0.0,
            'poisson_pval': np.nan,
            'poisson_rate_ratio': 1.0
        })
    else:
        year_col = f'year_{year}'
        if year_col in params.index:
            coef = params[year_col]
            se_val = se[year_col]
            pval = pvals[year_col]
        else:
            coef = 0.0
            se_val = 0.0
            pval = np.nan
        
        coef_data.append({
            'year': year,
            'tercile': 'T2',
            'poisson_coef': coef,
            'poisson_se': se_val,
            'poisson_pval': pval,
            'poisson_rate_ratio': np.exp(coef)
        })
    
    # T1 and T3: year effect + interaction
    for tercile in ['T1', 'T3']:
        if year == BASE_YEAR:
            # Base year: only tercile main effect
            tercile_col = f'tercile_{tercile}'
            if tercile_col in params.index:
                coef = params[tercile_col]
                se_val = se[tercile_col]
                pval = pvals[tercile_col]
            else:
                coef = 0.0
                se_val = 0.0
                pval = np.nan
            
            coef_data.append({
                'year': year,
                'tercile': tercile,
                'poisson_coef': coef,
                'poisson_se': se_val,
                'poisson_pval': pval,
                'poisson_rate_ratio': np.exp(coef)
            })
        else:
            # Non-base year: tercile + year + interaction
            tercile_col = f'tercile_{tercile}'
            year_col = f'year_{year}'
            int_col = f'y{year}_x_{tercile}'
            
            # Get components
            tercile_coef = params.get(tercile_col, 0.0)
            year_coef = params.get(year_col, 0.0)
            int_coef = params.get(int_col, 0.0) if int_col in params.index else 0.0
            int_se = se.get(int_col, 0.0) if int_col in se.index else 0.0
            int_pval = pvals.get(int_col, np.nan) if int_col in pvals.index else np.nan
            
            # Total effect
            total_coef = tercile_coef + year_coef + int_coef
            
            coef_data.append({
                'year': year,
                'tercile': tercile,
                'poisson_coef': total_coef,
                'poisson_se': int_se,  # SE of interaction (conservative)
                'poisson_pval': int_pval,
                'poisson_rate_ratio': np.exp(total_coef)
            })

df_poisson = pd.DataFrame(coef_data)

# Add confidence intervals
df_poisson['poisson_ci_lower'] = df_poisson['poisson_coef'] - 1.96 * df_poisson['poisson_se']
df_poisson['poisson_ci_upper'] = df_poisson['poisson_coef'] + 1.96 * df_poisson['poisson_se']

print(f"\nExtracted {len(df_poisson)} Poisson coefficients")

# ============================================================================
# MERGE WITH OLS RESULTS
# ============================================================================

if df_ols is not None:
    print("\n" + "-" * 70)
    print("COMPARING POISSON VS OLS")
    print("-" * 70)
    
    # Merge
    df_compare = df_poisson.merge(
        df_ols[['year', 'tercile', 'coef_ols', 'se_ols', 'coef_wls', 'se_wls']],
        on=['year', 'tercile'],
        how='left'
    )
    
    # Compute differences
    df_compare['diff_poisson_ols'] = df_compare['poisson_coef'] - df_compare['coef_ols']
    df_compare['diff_poisson_wls'] = df_compare['poisson_coef'] - df_compare['coef_wls']
    
    # Print key comparisons
    print("\nKey treatment effects (T2, 2023-2024):")
    print("-" * 70)
    
    for year in [2023, 2024]:
        row = df_compare[(df_compare['year'] == year) & (df_compare['tercile'] == 'T2')]
        if len(row) > 0:
            row = row.iloc[0]
            print(f"\n{year} (T2):")
            print(f"  Poisson coef:  {row['poisson_coef']:+.5f} (rate ratio: {row['poisson_rate_ratio']:.3f})")
            print(f"  OLS coef:      {row['coef_ols']:+.5f}")
            print(f"  WLS coef:      {row['coef_wls']:+.5f}")
            print(f"  Poisson - OLS: {row['diff_poisson_ols']:+.5f}")
            print(f"  Poisson - WLS: {row['diff_poisson_wls']:+.5f}")
    
    # Summary stats
    print(f"\n" + "-" * 70)
    print(f"Overall comparison (all year×tercile combinations):")
    print(f"-" * 70)
    print(f"  Mean |Poisson - OLS|: {np.abs(df_compare['diff_poisson_ols']).mean():.5f}")
    print(f"  Max |Poisson - OLS|:  {np.abs(df_compare['diff_poisson_ols']).max():.5f}")
    print(f"  Correlation (Poisson, OLS): {df_compare[['poisson_coef', 'coef_ols']].corr().iloc[0,1]:.4f}")
    
else:
    df_compare = df_poisson

# ============================================================================
# SAVE OUTPUTS
# ============================================================================

print("\n" + "-" * 70)
print("SAVING OUTPUTS")
print("-" * 70)

# 1. Coefficient comparison
output_coef = OUTPUT_DIR / "c3_poisson_comparison.csv"
df_compare.to_csv(output_coef, index=False)
print(f"\nSaved: {output_coef}")

# 2. Full Poisson summary
output_summary = OUTPUT_DIR / "c3_poisson_summary.txt"
with open(output_summary, 'w', encoding='utf-8') as f:
    f.write("C3 POISSON ROBUSTNESS CHECK\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Outcome: attempts_2b (count)\n")
    f.write(f"Offset: log(opportunities_2b)\n")
    f.write(f"Sample: Pitchers with ≥50 pitches in 2022\n")
    if EXCLUDE_2020:
        f.write(f"Robustness: 2020 (COVID) excluded\n")
    f.write(f"Base year: {BASE_YEAR}\n")
    f.write(f"Base tercile: {BASE_TERCILE}\n")
    f.write(f"Cluster SE: By pitcher ({df_baseline['pitcher_id'].nunique()} clusters)\n\n")
    
    f.write("-" * 70 + "\n")
    f.write("POISSON GLM RESULTS\n")
    f.write("-" * 70 + "\n")
    f.write(str(poisson_result.summary()))
    f.write("\n\n")
    
    f.write("-" * 70 + "\n")
    f.write("INTERPRETATION\n")
    f.write("-" * 70 + "\n")
    f.write("Poisson coefficients = log(rate ratio)\n")
    f.write("exp(coefficient) = multiplicative effect on attempt rate\n")
    f.write("Example: coef=0.05 → rate ratio=1.051 → +5.1% increase\n")

print(f"Saved: {output_summary}")

# 3. Poisson-specific parameters
poisson_params = pd.DataFrame({
    'parameter': params.index,
    'coefficient': params.values,
    'std_error': se.values,
    'p_value': pvals.values,
    'rate_ratio': np.exp(params.values),
    'rr_ci_lower': np.exp(params.values - 1.96 * se.values),
    'rr_ci_upper': np.exp(params.values + 1.96 * se.values)
})

output_params = OUTPUT_DIR / "c3_poisson_parameters.csv"
poisson_params.to_csv(output_params, index=False)
print(f"Saved: {output_params}")

# ============================================================================
# PLOT COMPARISON
# ============================================================================

if df_ols is not None:
    print("\n" + "-" * 70)
    print("CREATING COMPARISON PLOTS")
    print("-" * 70)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel 1: Poisson vs OLS coefficients
    for tercile in ['T1', 'T2', 'T3']:
        df_plot = df_compare[df_compare['tercile'] == tercile].sort_values('year')
        
        ax1.plot(df_plot['year'], df_plot['poisson_coef'], 
                marker='o', linewidth=2, label=f'{tercile} (Poisson)', linestyle='-')
        ax1.plot(df_plot['year'], df_plot['coef_ols'], 
                marker='s', linewidth=1.5, label=f'{tercile} (OLS)', 
                linestyle='--', alpha=0.7)
    
    ax1.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax1.axvline(BASE_YEAR, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
    ax1.axvline(2023, color='green', linestyle=':', linewidth=1.5, alpha=0.5)
    ax1.set_xlabel('Year', fontsize=11)
    ax1.set_ylabel('Coefficient', fontsize=11)
    ax1.set_title('Poisson vs OLS Coefficients', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8, loc='best', ncol=2)
    ax1.grid(alpha=0.3)
    
    # Panel 2: Differences (Poisson - OLS)
    for tercile in ['T1', 'T2', 'T3']:
        df_plot = df_compare[df_compare['tercile'] == tercile].sort_values('year')
        ax2.plot(df_plot['year'], df_plot['diff_poisson_ols'], 
                marker='o', linewidth=2, label=tercile)
    
    ax2.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax2.axvline(BASE_YEAR, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
    ax2.axvline(2023, color='green', linestyle=':', linewidth=1.5, alpha=0.5)
    ax2.set_xlabel('Year', fontsize=11)
    ax2.set_ylabel('Difference (Poisson - OLS)', fontsize=11)
    ax2.set_title('Model Differences', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, loc='best')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    output_plot = OUTPUT_DIR / "c3_poisson_comparison_plot.png"
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_plot}")
    plt.close()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("C3 POISSON ROBUSTNESS COMPLETE")
print("=" * 70)

print(f"\nModel: Poisson GLM with log(opportunities) offset")
print(f"Observations: {poisson_result.nobs:.0f}")
print(f"Clusters: {df_baseline['pitcher_id'].nunique()}")
print(f"Log-likelihood: {poisson_result.llf:.1f}")

if df_ols is not None:
    print(f"\nComparison to C3v2 OLS:")
    print(f"  Mean absolute difference: {np.abs(df_compare['diff_poisson_ols']).mean():.5f}")
    print(f"  Correlation: {df_compare[['poisson_coef', 'coef_ols']].corr().iloc[0,1]:.4f}")
    
    # Check if results are "similar"
    mean_abs_diff = np.abs(df_compare['diff_poisson_ols']).mean()
    if mean_abs_diff < 0.001:
        print(f"\n✓ ROBUSTNESS CHECK PASSED")
        print(f"  Poisson and OLS yield very similar coefficients")
        print(f"  This validates the linear rate model in C3v2")
    elif mean_abs_diff < 0.003:
        print(f"\n✓ ROBUSTNESS CHECK: MOSTLY CONSISTENT")
        print(f"  Poisson and OLS show similar patterns")
        print(f"  Minor differences are expected due to model structure")
    else:
        print(f"\n⚠ ROBUSTNESS CHECK: INVESTIGATE DIFFERENCES")
        print(f"  Poisson and OLS show notable differences")
        print(f"  Review coefficient comparison table")

print(f"\nOutputs:")
print(f"  1. c3_poisson_comparison.csv (Poisson vs OLS)")
print(f"  2. c3_poisson_parameters.csv (all Poisson params)")
print(f"  3. c3_poisson_summary.txt (full GLM output)")
if df_ols is not None:
    print(f"  4. c3_poisson_comparison_plot.png")

print("\n" + "=" * 70)
print("Next: Run placebo test (c3_placebo_2021.py)")
print("=" * 70)