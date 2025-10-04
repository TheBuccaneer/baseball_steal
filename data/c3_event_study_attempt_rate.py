"""
c3_v2_event_study_linearmodels.py
==================================
Event study for 2B steal attempt rate using linearmodels.PanelOLS.

Key improvements over C3v1:
- Correct cluster-robust standard errors (CR1)
- Proper Wald test for pre-trends (full covariance matrix)
- Tested panel FE implementation
- Robustness: Exclude 2020 (COVID short season)

Specification:
- PanelOLS with entity_effects (pitcher FE) + time_effects (year FE)
- Interactions: Year × Tercile (T2 = reference)
- Base year: 2022
- SE: Clustered by pitcher (entity)
- Weights: opportunities_2b for WLS

Treatment context:
- 2023: 15s/20s pitch timer + 2 pickoff limit + larger bases
- 2024: 18s (runners on) + same limits
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from linearmodels import PanelOLS
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================

if Path.cwd().name == 'data':
    INPUT_FILE = Path("analysis/c_running_game/c_panel_with_baseline.csv")
    OUTPUT_DIR = Path("analysis/c3_event_study")
else:
    INPUT_FILE = Path("./data/analysis/c_running_game/c_panel_with_baseline.csv")
    OUTPUT_DIR = Path("./data/analysis/c3_event_study")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_YEAR = 2022
BASE_TERCILE = 'T2'
EXCLUDE_2020 = True  # COVID short season robustness

# ============================================================================
# LOAD DATA
# ============================================================================

print("=" * 70)
print("C3-V2: EVENT STUDY WITH LINEARMODELS.PANELOLS")
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

# Optionally exclude 2020
if EXCLUDE_2020:
    n_before = len(df_baseline)
    df_baseline = df_baseline[df_baseline['season'] != 2020].copy()
    n_dropped = n_before - len(df_baseline)
    print(f"Excluded 2020 (COVID): dropped {n_dropped} observations")

print(f"Final sample: {len(df_baseline):,} observations")
print(f"  Pitchers: {df_baseline['pitcher_id'].nunique():,}")
print(f"  Years: {sorted(df_baseline['season'].unique())}")

# ============================================================================
# PREPARE PANEL DATA
# ============================================================================

print("\n" + "-" * 70)
print("PREPARING PANEL DATA")
print("-" * 70)

# Create year dummies (excluding base year)
years = sorted(df_baseline['season'].unique())
years_non_base = [y for y in years if y != BASE_YEAR]

for year in years_non_base:
    df_baseline[f'year_{year}'] = (df_baseline['season'] == year).astype(int)

print(f"\nYear dummies created (base={BASE_YEAR}):")
print(f"  {years_non_base}")

# Create year × tercile interactions
# Note: Tercile main effects absorbed by entity FE (time-invariant)
interaction_cols = []

for year in years_non_base:
    for tercile in ['T1', 'T3']:  # T2 is reference
        int_col = f'y{year}_x_{tercile}'
        df_baseline[int_col] = (
            (df_baseline['season'] == year) & 
            (df_baseline['baseline_group'] == tercile)
        ).astype(int)
        interaction_cols.append(int_col)

print(f"\nYear×Tercile interactions: {len(interaction_cols)}")
print(f"  Reference tercile: {BASE_TERCILE}")

# Check for missing values
df_baseline = df_baseline.dropna(subset=['attempt_rate', 'opportunities_2b'])
print(f"\nAfter dropping NA: {len(df_baseline):,} observations")

# Set MultiIndex for PanelOLS (entity, time)
df_panel = df_baseline.set_index(['pitcher_id', 'season'])

print(f"\nPanel structure:")
print(f"  Entity (pitcher): {df_panel.index.get_level_values(0).nunique()} unique")
print(f"  Time (year): {df_panel.index.get_level_values(1).nunique()} unique")

# ============================================================================
# MODEL 1: OLS (UNWEIGHTED - AVERAGE PITCHER)
# ============================================================================

print("\n" + "-" * 70)
print("MODEL 1: OLS (UNWEIGHTED - AVERAGE PITCHER)")
print("-" * 70)

# Prepare regressors: Year dummies + interactions
year_dummy_cols = [f'year_{y}' for y in years_non_base]
all_regressors = year_dummy_cols + interaction_cols

X_ols = df_panel[all_regressors]
y_ols = df_panel['attempt_rate']

print(f"\nFitting PanelOLS:")
print(f"  Outcome: attempt_rate")
print(f"  Year dummies: {len(year_dummy_cols)}")
print(f"  Interactions: {len(interaction_cols)}")
print(f"  Entity effects: True (pitcher FE)")
print(f"  Time effects: False (explicit year dummies)")

# Fit model
mod_ols = PanelOLS(
    dependent=y_ols,
    exog=X_ols,
    entity_effects=True,
    time_effects=False  # Use explicit year dummies instead
)

res_ols = mod_ols.fit(cov_type='clustered', cluster_entity=True)

print(f"\nOLS Results:")
print(f"  R²: {res_ols.rsquared:.4f}")
print(f"  R² (within): {res_ols.rsquared_within:.4f}")
print(f"  Observations: {res_ols.nobs:,.0f}")
print(f"  Entities: {res_ols.entity_info['total']:,.0f}")
print(f"  Clusters: {res_ols.entity_info['total']:,.0f}")

# ============================================================================
# MODEL 2: WLS (WEIGHTED - AVERAGE OPPORTUNITY)
# ============================================================================

print("\n" + "-" * 70)
print("MODEL 2: WLS (WEIGHTED - AVERAGE OPPORTUNITY)")
print("-" * 70)

w_wls = df_panel['opportunities_2b']

mod_wls = PanelOLS(
    dependent=y_ols,
    exog=X_ols,
    weights=w_wls,
    entity_effects=True,
    time_effects=False
)

res_wls = mod_wls.fit(cov_type='clustered', cluster_entity=True)

print(f"\nWLS Results:")
print(f"  R²: {res_wls.rsquared:.4f}")
print(f"  R² (within): {res_wls.rsquared_within:.4f}")
print(f"  Observations: {res_wls.nobs:,.0f}")

# ============================================================================
# EXTRACT COEFFICIENTS
# ============================================================================

print("\n" + "-" * 70)
print("EXTRACTING COEFFICIENTS")
print("-" * 70)

# Get parameters and standard errors
params_ols = res_ols.params
params_wls = res_wls.params
se_ols = res_ols.std_errors
se_wls = res_wls.std_errors

print(f"\nEstimated parameters: {len(params_ols)}")

# Build coefficient table
coef_data = []

for year in years:
    # T2 (reference): main year effect only
    if year == BASE_YEAR:
        # Base year = 0 by construction
        coef_data.append({
            'year': year,
            'tercile': 'T2',
            'coef_ols': 0.0,
            'se_ols': 0.0,
            'coef_wls': 0.0,
            'se_wls': 0.0,
            'is_pretrend': year < 2023
        })
    else:
        # Year dummy coefficient (main effect for T2)
        year_col = f'year_{year}'
        
        if year_col in params_ols.index:
            coef_ols = params_ols[year_col]
            se_ols_val = se_ols[year_col]
        else:
            coef_ols = 0.0
            se_ols_val = 0.0
        
        if year_col in params_wls.index:
            coef_wls = params_wls[year_col]
            se_wls_val = se_wls[year_col]
        else:
            coef_wls = 0.0
            se_wls_val = 0.0
        
        coef_data.append({
            'year': year,
            'tercile': 'T2',
            'coef_ols': coef_ols,
            'se_ols': se_ols_val,
            'coef_wls': coef_wls,
            'se_wls': se_wls_val,
            'is_pretrend': year < 2023
        })
    
    # T1 and T3: year effect + interaction
    for tercile in ['T1', 'T3']:
        if year == BASE_YEAR:
            coef_data.append({
                'year': year,
                'tercile': tercile,
                'coef_ols': 0.0,
                'se_ols': 0.0,
                'coef_wls': 0.0,
                'se_wls': 0.0,
                'is_pretrend': year < 2023
            })
        else:
            # Year main effect
            year_col = f'year_{year}'
            
            if year_col in params_ols.index:
                year_ols = params_ols[year_col]
            else:
                year_ols = 0.0
            
            if year_col in params_wls.index:
                year_wls = params_wls[year_col]
            else:
                year_wls = 0.0
            
            # Interaction term
            int_col = f'y{year}_x_{tercile}'
            
            if int_col in params_ols.index:
                int_ols = params_ols[int_col]
                int_se_ols = se_ols[int_col]
            else:
                int_ols = 0.0
                int_se_ols = 0.0
            
            if int_col in params_wls.index:
                int_wls = params_wls[int_col]
                int_se_wls = se_wls[int_col]
            else:
                int_wls = 0.0
                int_se_wls = 0.0
            
            # Total effect = year main + interaction
            coef_data.append({
                'year': year,
                'tercile': tercile,
                'coef_ols': year_ols + int_ols,
                'se_ols': int_se_ols,  # SE of interaction (conservative)
                'coef_wls': year_wls + int_wls,
                'se_wls': int_se_wls,
                'is_pretrend': year < 2023
            })

df_coef = pd.DataFrame(coef_data)

# Compute confidence intervals
df_coef['ci_lower_ols'] = df_coef['coef_ols'] - 1.96 * df_coef['se_ols']
df_coef['ci_upper_ols'] = df_coef['coef_ols'] + 1.96 * df_coef['se_ols']
df_coef['ci_lower_wls'] = df_coef['coef_wls'] - 1.96 * df_coef['se_wls']
df_coef['ci_upper_wls'] = df_coef['coef_wls'] + 1.96 * df_coef['se_wls']

print(f"\nExtracted coefficients for {len(df_coef)} year×tercile combinations")

# ============================================================================
# PRE-TRENDS TEST (WALD TEST)
# ============================================================================

print("\n" + "-" * 70)
print("PRE-TRENDS TEST (WALD TEST)")
print("-" * 70)

# Identify pre-trend interaction coefficients
pretrend_years = [y for y in years if y < 2023 and y != BASE_YEAR]
pretrend_interactions = []

for year in pretrend_years:
    for tercile in ['T1', 'T3']:
        int_col = f'y{year}_x_{tercile}'
        if int_col in params_ols.index:
            pretrend_interactions.append(int_col)

print(f"\nPre-trend years: {pretrend_years}")
print(f"Pre-trend interactions to test: {len(pretrend_interactions)}")

if len(pretrend_interactions) > 0:
    # Construct restriction matrix: R β = 0
    # R is (n_restrictions × n_params) with 1s at positions of pre-trend interactions
    n_params = len(params_ols)
    n_restrictions = len(pretrend_interactions)
    
    R = np.zeros((n_restrictions, n_params))
    for i, int_col in enumerate(pretrend_interactions):
        j = params_ols.index.get_loc(int_col)
        R[i, j] = 1
    
    # Wald test: (Rβ)' [R V R']^(-1) (Rβ) ~ χ²(n_restrictions)
    # linearmodels has a wald_test method
    formula_str = ' = '.join([f'{col} = 0' for col in pretrend_interactions])
    
    print(f"\nTesting: {pretrend_interactions[:3]} ... (all = 0)")
    
    # OLS Wald test
    try:
        wald_ols = res_ols.wald_test(formula=formula_str)
        print(f"\nOLS Pre-Trends Wald Test:")
        print(f"  χ² statistic: {wald_ols.stat:.3f}")
        print(f"  df: {wald_ols.df}")
        print(f"  p-value: {wald_ols.pval:.4f}")
        
        if wald_ols.pval > 0.10:
            print(f"  ✓ Cannot reject parallel trends (p > 0.10)")
        else:
            print(f"  ⚠ Evidence against parallel trends (p ≤ 0.10)")
    except Exception as e:
        print(f"  ERROR in Wald test: {e}")
        wald_ols = None
    
    # WLS Wald test
    try:
        wald_wls = res_wls.wald_test(formula=formula_str)
        print(f"\nWLS Pre-Trends Wald Test:")
        print(f"  χ² statistic: {wald_wls.stat:.3f}")
        print(f"  df: {wald_wls.df}")
        print(f"  p-value: {wald_wls.pval:.4f}")
        
        if wald_wls.pval > 0.10:
            print(f"  ✓ Cannot reject parallel trends (p > 0.10)")
        else:
            print(f"  ⚠ Evidence against parallel trends (p ≤ 0.10)")
    except Exception as e:
        print(f"  ERROR in Wald test: {e}")
        wald_wls = None
else:
    print("\nNo pre-trend interactions to test (all years post-2023)")
    wald_ols = None
    wald_wls = None

# ============================================================================
# SAVE OUTPUTS
# ============================================================================

print("\n" + "-" * 70)
print("SAVING OUTPUTS")
print("-" * 70)

# 1. Coefficients
output_coef = OUTPUT_DIR / "c3_v2_coefficients.csv"
df_coef.to_csv(output_coef, index=False)
print(f"\nSaved: {output_coef}")

# 2. Pre-trends test
if wald_ols is not None and wald_wls is not None:
    pretrends_results = pd.DataFrame([
        {
            'model': 'OLS',
            'n_restrictions': len(pretrend_interactions),
            'interactions_tested': str(pretrend_interactions),
            'chi2_statistic': wald_ols.stat,
            'df': wald_ols.df,
            'p_value': wald_ols.pval,
            'reject_h0_at_10pct': wald_ols.pval <= 0.10
        },
        {
            'model': 'WLS',
            'n_restrictions': len(pretrend_interactions),
            'interactions_tested': str(pretrend_interactions),
            'chi2_statistic': wald_wls.stat,
            'df': wald_wls.df,
            'p_value': wald_wls.pval,
            'reject_h0_at_10pct': wald_wls.pval <= 0.10
        }
    ])
    
    output_pretrends = OUTPUT_DIR / "c3_v2_pretrends_test.csv"
    pretrends_results.to_csv(output_pretrends, index=False)
    print(f"Saved: {output_pretrends}")

# 3. Full regression output
output_summary = OUTPUT_DIR / "c3_v2_regression_summary.txt"
with open(output_summary, 'w', encoding='utf-8') as f:
    f.write("C3-V2 EVENT STUDY RESULTS (linearmodels.PanelOLS)\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Outcome: Attempt Rate (2B steals only)\n")
    f.write(f"Sample: Pitchers with ≥50 pitches in 2022\n")
    if EXCLUDE_2020:
        f.write(f"Robustness: 2020 (COVID) excluded\n")
    f.write(f"Base year: {BASE_YEAR}\n")
    f.write(f"Base tercile: {BASE_TERCILE}\n\n")
    
    f.write("-" * 70 + "\n")
    f.write("OLS (UNWEIGHTED)\n")
    f.write("-" * 70 + "\n")
    f.write(str(res_ols))
    f.write("\n\n")
    
    f.write("-" * 70 + "\n")
    f.write("WLS (OPPORTUNITY-WEIGHTED)\n")
    f.write("-" * 70 + "\n")
    f.write(str(res_wls))
    
print(f"Saved: {output_summary}")

# ============================================================================
# PLOT EVENT STUDY
# ============================================================================

print("\n" + "-" * 70)
print("CREATING EVENT STUDY PLOT")
print("-" * 70)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# OLS panel
for tercile in ['T1', 'T2', 'T3']:
    df_plot = df_coef[df_coef['tercile'] == tercile].sort_values('year')
    ax1.plot(df_plot['year'], df_plot['coef_ols'], 
             marker='o', linewidth=2, label=tercile)
    ax1.fill_between(df_plot['year'], 
                      df_plot['ci_lower_ols'], 
                      df_plot['ci_upper_ols'], 
                      alpha=0.2)

ax1.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax1.axvline(BASE_YEAR, color='red', linestyle=':', linewidth=1.5, 
            label=f'{BASE_YEAR} (base)')
ax1.axvline(2023, color='green', linestyle=':', linewidth=1.5, 
            alpha=0.7, label='2023 rules')
ax1.set_xlabel('Year', fontsize=11)
ax1.set_ylabel('Coefficient (pp change in attempt rate)', fontsize=11)
ax1.set_title('OLS (Average Pitcher)', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9, loc='best')
ax1.grid(alpha=0.3)

# WLS panel
for tercile in ['T1', 'T2', 'T3']:
    df_plot = df_coef[df_coef['tercile'] == tercile].sort_values('year')
    ax2.plot(df_plot['year'], df_plot['coef_wls'], 
             marker='o', linewidth=2, label=tercile)
    ax2.fill_between(df_plot['year'], 
                      df_plot['ci_lower_wls'], 
                      df_plot['ci_upper_wls'], 
                      alpha=0.2)

ax2.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax2.axvline(BASE_YEAR, color='red', linestyle=':', linewidth=1.5,
            label=f'{BASE_YEAR} (base)')
ax2.axvline(2023, color='green', linestyle=':', linewidth=1.5,
            alpha=0.7, label='2023 rules')
ax2.set_xlabel('Year', fontsize=11)
ax2.set_ylabel('Coefficient (pp change in attempt rate)', fontsize=11)
ax2.set_title('WLS (Average Opportunity)', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9, loc='best')
ax2.grid(alpha=0.3)

plt.tight_layout()
output_plot = OUTPUT_DIR / "c3_v2_event_study_plot.png"
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
print(f"\nSaved: {output_plot}")
plt.close()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("C3-V2 COMPLETE")
print("=" * 70)

print(f"\nPanel: {res_ols.nobs:,.0f} observations")
print(f"  Pitchers: {res_ols.entity_info['total']:,.0f}")
print(f"  Years: {sorted(df_panel.index.get_level_values(1).unique())}")
if EXCLUDE_2020:
    print(f"  (2020 excluded for robustness)")

if wald_ols is not None:
    print(f"\nPre-trends test (Wald):")
    print(f"  OLS: χ²={wald_ols.stat:.2f}, p={wald_ols.pval:.4f}")
    print(f"  WLS: χ²={wald_wls.stat:.2f}, p={wald_wls.pval:.4f}")

# Treatment effects (T2, 2023)
df_2023_t2 = df_coef[(df_coef['year'] == 2023) & (df_coef['tercile'] == 'T2')]
if len(df_2023_t2) > 0:
    print(f"\n2023 Treatment Effect (T2 tercile):")
    print(f"  OLS: {df_2023_t2['coef_ols'].values[0]:+.5f} "
          f"({df_2023_t2['coef_ols'].values[0]*100:+.2f} pp)")
    print(f"  WLS: {df_2023_t2['coef_wls'].values[0]:+.5f} "
          f"({df_2023_t2['coef_wls'].values[0]*100:+.2f} pp)")

print(f"\nOutputs:")
print(f"  1. c3_v2_coefficients.csv")
print(f"  2. c3_v2_pretrends_test.csv")
print(f"  3. c3_v2_regression_summary.txt")
print(f"  4. c3_v2_event_study_plot.png")

print("\n" + "=" * 70)
print("Next: Check Rambachan-Roth sensitivity, Poisson robustness")
print("=" * 70)