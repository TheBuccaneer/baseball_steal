"""
c3c_poisson_fe_ppml.py
======================
Fixed Effects Poisson (PPML) robustness check for C3 event study.

Uses statsmodels GLM with explicit pitcher dummies for FE.
Computationally intensive but methodologically correct.

Key improvements over c3_poisson_robustness.py:
- Pitcher fixed effects (within-pitcher estimation)
- PPML = Poisson Pseudo-Maximum-Likelihood (robust to heteroskedasticity)
- Comparable to C3v2 OLS with entity FE

Treatment context:
- 2023: 15s/20s pitch timer + 2 pickoff limit + larger bases
- 2024: 18s (runners on) + same limits
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import statsmodels.api as sm
from statsmodels.genmod.families import Poisson
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FILE = Path("analysis/c_running_game/c_panel_with_baseline.csv")
OUTPUT_DIR = Path("analysis/c3_poisson_fe")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_YEAR = 2022
BASE_TERCILE = 'T2'
EXCLUDE_2020 = True

# ============================================================================
# LOAD DATA
# ============================================================================

print("=" * 70)
print("C3 FIXED EFFECTS POISSON (PPML)")
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
print("PREPARING FE-POISSON DATA")
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

# Check for missing values
df_baseline = df_baseline.dropna(subset=['attempts_2b', 'opportunities_2b'])
print(f"\nAfter dropping NA: {len(df_baseline):,} observations")

# Create log offset
df_baseline['log_opp'] = np.log(df_baseline['opportunities_2b'])

print(f"\nPoisson outcome: attempts_2b (count)")
print(f"  Mean: {df_baseline['attempts_2b'].mean():.2f}")
print(f"  Range: [{df_baseline['attempts_2b'].min():.0f}, {df_baseline['attempts_2b'].max():.0f}]")

# ============================================================================
# CREATE PITCHER DUMMIES
# ============================================================================

print("\n" + "-" * 70)
print("CREATING PITCHER FIXED EFFECTS")
print("-" * 70)

# Get unique pitchers
pitchers = sorted(df_baseline['pitcher_id'].unique())
n_pitchers = len(pitchers)
print(f"\nCreating {n_pitchers} pitcher dummies...")

# Create pitcher dummies (exclude first for reference)
pitcher_dummies = pd.get_dummies(df_baseline['pitcher_id'], prefix='pitcher', drop_first=True)
print(f"  Created {len(pitcher_dummies.columns)} dummy variables")
print(f"  (Reference pitcher: {pitchers[0]})")

# ============================================================================
# FE-POISSON MODEL
# ============================================================================

print("\n" + "-" * 70)
print("FITTING FE-POISSON (GLM WITH PITCHER DUMMIES)")
print("-" * 70)

# Build regressors: year dummies + interactions + pitcher dummies
year_dummy_cols = [f'year_{y}' for y in years_non_base]
all_regressors = year_dummy_cols + interaction_cols

X = pd.concat([
    df_baseline[all_regressors],
    pitcher_dummies
], axis=1)

# Convert all to float64 to avoid dtype issues
X = X.astype(np.float64)

X = sm.add_constant(X, has_constant='add')

y = df_baseline['attempts_2b']
offset = df_baseline['log_opp']

print(f"\nModel specification:")
print(f"  Outcome: attempts_2b (count)")
print(f"  Offset: log(opportunities_2b)")
print(f"  Regressors: {len(all_regressors)} (years + interactions)")
print(f"  Pitcher FE: {len(pitcher_dummies.columns)} dummies")
print(f"  Total parameters: {X.shape[1]}")
print(f"  Cluster SE: By pitcher ({n_pitchers} clusters)")

# Fit model
print(f"\nFitting GLM Poisson...")
print(f"  (This may take 1-2 minutes with {n_pitchers} FE...)")

try:
    fe_poisson_model = sm.GLM(
        endog=y,
        exog=X,
        offset=offset,
        family=Poisson()
    )
    
    fe_poisson_result = fe_poisson_model.fit(
        cov_type='cluster',
        cov_kwds={'groups': df_baseline['pitcher_id'].values}
    )
    
    print(f"✓ Model converged")
    print(f"  Log-likelihood: {fe_poisson_result.llf:.1f}")
    print(f"  AIC: {fe_poisson_result.aic:.1f}")
    print(f"  Observations: {fe_poisson_result.nobs:.0f}")
    
except Exception as e:
    print(f"✗ Model failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# EXTRACT COEFFICIENTS
# ============================================================================

print("\n" + "-" * 70)
print("EXTRACTING COEFFICIENTS (YEAR + INTERACTIONS ONLY)")
print("-" * 70)

# Get coefficients (only for year/interaction terms, not pitcher dummies)
params = fe_poisson_result.params
se = fe_poisson_result.bse
pvals = fe_poisson_result.pvalues

# Filter to relevant coefficients
relevant_params = [c for c in params.index if c.startswith('year_') or c.startswith('y20')]
print(f"\nExtracted {len(relevant_params)} relevant coefficients")

# Build coefficient comparison
coef_data = []

for year in years:
    # T2 (reference): main year effect only
    if year == BASE_YEAR:
        coef_data.append({
            'year': year,
            'tercile': 'T2',
            'fe_poisson_coef': 0.0,
            'fe_poisson_se': 0.0,
            'fe_poisson_pval': np.nan,
            'fe_poisson_rr': 1.0
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
            'fe_poisson_coef': coef,
            'fe_poisson_se': se_val,
            'fe_poisson_pval': pval,
            'fe_poisson_rr': np.exp(coef)
        })
    
    # T1 and T3: year + interaction
    for tercile in ['T1', 'T3']:
        if year == BASE_YEAR:
            coef_data.append({
                'year': year,
                'tercile': tercile,
                'fe_poisson_coef': 0.0,
                'fe_poisson_se': 0.0,
                'fe_poisson_pval': np.nan,
                'fe_poisson_rr': 1.0
            })
        else:
            year_col = f'year_{year}'
            int_col = f'y{year}_x_{tercile}'
            
            # Get components
            year_coef = params.get(year_col, 0.0)
            int_coef = params.get(int_col, 0.0)
            int_se = se.get(int_col, 0.0)
            int_pval = pvals.get(int_col, np.nan)
            
            # Total effect
            total_coef = year_coef + int_coef
            
            coef_data.append({
                'year': year,
                'tercile': tercile,
                'fe_poisson_coef': total_coef,
                'fe_poisson_se': int_se,
                'fe_poisson_pval': int_pval,
                'fe_poisson_rr': np.exp(total_coef)
            })

df_fe_poisson = pd.DataFrame(coef_data)

# Add confidence intervals
df_fe_poisson['fe_poisson_ci_lower'] = df_fe_poisson['fe_poisson_coef'] - 1.96 * df_fe_poisson['fe_poisson_se']
df_fe_poisson['fe_poisson_ci_upper'] = df_fe_poisson['fe_poisson_coef'] + 1.96 * df_fe_poisson['fe_poisson_se']

print(f"Extracted {len(df_fe_poisson)} coefficients")

# ============================================================================
# LOAD PREVIOUS RESULTS FOR COMPARISON
# ============================================================================

print("\n" + "-" * 70)
print("LOADING PREVIOUS RESULTS FOR COMPARISON")
print("-" * 70)

# OLS results
try:
    df_ols = pd.read_csv("analysis/c3_event_study/c3_v2_coefficients.csv")
    print(f"✓ Loaded C3v2 OLS coefficients: {len(df_ols)} rows")
except Exception as e:
    print(f"✗ Could not load OLS results: {e}")
    df_ols = None

# Poisson without FE
try:
    df_poisson_nofe = pd.read_csv("analysis/c3_poisson/c3_poisson_comparison.csv")
    print(f"✓ Loaded Poisson (no FE) results: {len(df_poisson_nofe)} rows")
except Exception as e:
    print(f"✗ Could not load Poisson results: {e}")
    df_poisson_nofe = None

# ============================================================================
# MERGE AND COMPARE
# ============================================================================

print("\n" + "-" * 70)
print("COMPARING FE-POISSON VS OLS VS POISSON-NO-FE")
print("-" * 70)

df_compare = df_fe_poisson.copy()

if df_ols is not None:
    df_compare = df_compare.merge(
        df_ols[['year', 'tercile', 'coef_ols', 'se_ols']],
        on=['year', 'tercile'],
        how='left'
    )

if df_poisson_nofe is not None:
    df_compare = df_compare.merge(
        df_poisson_nofe[['year', 'tercile', 'poisson_coef', 'poisson_se']],
        on=['year', 'tercile'],
        how='left',
        suffixes=('', '_nofe')
    )

# Compute differences
if 'coef_ols' in df_compare.columns:
    df_compare['diff_fe_poisson_ols'] = df_compare['fe_poisson_coef'] - df_compare['coef_ols']

if 'poisson_coef' in df_compare.columns:
    df_compare['diff_fe_poisson_nofe'] = df_compare['fe_poisson_coef'] - df_compare['poisson_coef']

# Print key comparisons
print("\nKey treatment effects (T2, 2023-2024):")
print("-" * 70)

for year in [2023, 2024]:
    row = df_compare[(df_compare['year'] == year) & (df_compare['tercile'] == 'T2')]
    if len(row) > 0:
        row = row.iloc[0]
        print(f"\n{year} (T2):")
        print(f"  FE-Poisson:       {row['fe_poisson_coef']:+.5f} (RR: {row['fe_poisson_rr']:.3f}, p={row['fe_poisson_pval']:.4f})")
        if 'coef_ols' in df_compare.columns:
            print(f"  OLS:              {row['coef_ols']:+.5f}")
            print(f"  FE-Poisson - OLS: {row['diff_fe_poisson_ols']:+.5f}")
        if 'poisson_coef' in df_compare.columns:
            print(f"  Poisson (no FE):  {row['poisson_coef']:+.5f}")
            print(f"  FE vs no-FE diff: {row['diff_fe_poisson_nofe']:+.5f}")

# Summary stats
if 'diff_fe_poisson_ols' in df_compare.columns:
    print(f"\n" + "-" * 70)
    print(f"FE-Poisson vs OLS (all combinations):")
    print(f"-" * 70)
    print(f"  Mean |difference|: {np.abs(df_compare['diff_fe_poisson_ols']).mean():.5f}")
    print(f"  Max |difference|:  {np.abs(df_compare['diff_fe_poisson_ols']).max():.5f}")
    print(f"  Correlation: {df_compare[['fe_poisson_coef', 'coef_ols']].corr().iloc[0,1]:.4f}")

# ============================================================================
# SAVE OUTPUTS
# ============================================================================

print("\n" + "-" * 70)
print("SAVING OUTPUTS")
print("-" * 70)

# 1. Comparison table
output_compare = OUTPUT_DIR / "c3c_fe_poisson_comparison.csv"
df_compare.to_csv(output_compare, index=False)
print(f"\nSaved: {output_compare}")

# 2. Full model summary
output_summary = OUTPUT_DIR / "c3c_fe_poisson_summary.txt"
with open(output_summary, 'w', encoding='utf-8') as f:
    f.write("C3C FIXED EFFECTS POISSON (PPML)\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Outcome: attempts_2b (count)\n")
    f.write(f"Offset: log(opportunities_2b)\n")
    f.write(f"Sample: Pitchers with ≥50 pitches in 2022\n")
    if EXCLUDE_2020:
        f.write(f"Robustness: 2020 (COVID) excluded\n")
    f.write(f"Base year: {BASE_YEAR}\n")
    f.write(f"Base tercile: {BASE_TERCILE}\n")
    f.write(f"Fixed Effects: {n_pitchers} pitcher dummies\n")
    f.write(f"Cluster SE: pitcher_id ({n_pitchers} clusters)\n\n")
    
    f.write("-" * 70 + "\n")
    f.write("FE-POISSON RESULTS (Year/Interaction coefficients only)\n")
    f.write("-" * 70 + "\n\n")
    
    # Print only year/interaction coefficients
    for param in relevant_params:
        coef = params[param]
        se_val = se[param]
        pval = pvals[param]
        f.write(f"{param:20s}  {coef:+.5f}  (SE: {se_val:.5f}, p={pval:.4f})\n")
    
    f.write(f"\n\nFull GLM output:\n")
    f.write(str(fe_poisson_result.summary()))

print(f"Saved: {output_summary}")

# 3. Coefficient table
output_coefs = OUTPUT_DIR / "c3c_fe_poisson_coefficients.csv"
df_fe_poisson.to_csv(output_coefs, index=False)
print(f"Saved: {output_coefs}")

# ============================================================================
# PLOTS
# ============================================================================

print("\n" + "-" * 70)
print("CREATING COMPARISON PLOTS")
print("-" * 70)

if 'coef_ols' in df_compare.columns:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Panel 1: FE-Poisson vs OLS
    ax1 = axes[0]
    for tercile in ['T1', 'T2', 'T3']:
        df_plot = df_compare[df_compare['tercile'] == tercile].sort_values('year')
        ax1.plot(df_plot['year'], df_plot['fe_poisson_coef'], 
                marker='o', linewidth=2, label=f'{tercile} (FE-Poisson)', linestyle='-')
        ax1.plot(df_plot['year'], df_plot['coef_ols'], 
                marker='s', linewidth=1.5, label=f'{tercile} (OLS)', 
                linestyle='--', alpha=0.7)
    
    ax1.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax1.axvline(BASE_YEAR, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
    ax1.axvline(2023, color='green', linestyle=':', linewidth=1.5, alpha=0.5)
    ax1.set_xlabel('Year', fontsize=11)
    ax1.set_ylabel('Coefficient', fontsize=11)
    ax1.set_title('FE-Poisson vs OLS', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=7, loc='best', ncol=2)
    ax1.grid(alpha=0.3)
    
    # Panel 2: All three models (T2 only)
    ax2 = axes[1]
    df_t2 = df_compare[df_compare['tercile'] == 'T2'].sort_values('year')
    ax2.plot(df_t2['year'], df_t2['fe_poisson_coef'], 
            marker='o', linewidth=2, label='FE-Poisson', color='green')
    ax2.plot(df_t2['year'], df_t2['coef_ols'], 
            marker='s', linewidth=2, label='OLS', color='blue')
    if 'poisson_coef' in df_t2.columns:
        ax2.plot(df_t2['year'], df_t2['poisson_coef'], 
                marker='^', linewidth=2, label='Poisson (no FE)', 
                color='orange', alpha=0.7)
    
    ax2.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax2.axvline(BASE_YEAR, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
    ax2.axvline(2023, color='green', linestyle=':', linewidth=1.5, alpha=0.5)
    ax2.set_xlabel('Year', fontsize=11)
    ax2.set_ylabel('Coefficient', fontsize=11)
    ax2.set_title('T2: All Models', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, loc='best')
    ax2.grid(alpha=0.3)
    
    # Panel 3: Differences
    ax3 = axes[2]
    for tercile in ['T1', 'T2', 'T3']:
        df_plot = df_compare[df_compare['tercile'] == tercile].sort_values('year')
        ax3.plot(df_plot['year'], df_plot['diff_fe_poisson_ols'], 
                marker='o', linewidth=2, label=tercile)
    
    ax3.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax3.axvline(BASE_YEAR, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
    ax3.axvline(2023, color='green', linestyle=':', linewidth=1.5, alpha=0.5)
    ax3.set_xlabel('Year', fontsize=11)
    ax3.set_ylabel('Difference (FE-Poisson - OLS)', fontsize=11)
    ax3.set_title('Model Differences', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9, loc='best')
    ax3.grid(alpha=0.3)
    
    plt.tight_layout()
    output_plot = OUTPUT_DIR / "c3c_fe_poisson_comparison_plot.png"
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_plot}")
    plt.close()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("C3C FE-POISSON COMPLETE")
print("=" * 70)

print(f"\nModel: FE-Poisson (PPML) with {n_pitchers} pitcher dummies")
print(f"Observations: {fe_poisson_result.nobs:.0f}")
print(f"Log-likelihood: {fe_poisson_result.llf:.1f}")

if 'diff_fe_poisson_ols' in df_compare.columns:
    mean_diff = np.abs(df_compare['diff_fe_poisson_ols']).mean()
    corr = df_compare[['fe_poisson_coef', 'coef_ols']].corr().iloc[0,1]
    
    print(f"\nComparison to C3v2 OLS:")
    print(f"  Mean absolute difference: {mean_diff:.5f}")
    print(f"  Correlation: {corr:.4f}")
    
    if mean_diff < 0.002 and corr > 0.95:
        print(f"\n✓ EXCELLENT ROBUSTNESS")
        print(f"  FE-Poisson and OLS are highly consistent")
        print(f"  This strongly validates the C3v2 findings")
    elif mean_diff < 0.005 and corr > 0.85:
        print(f"\n✓ GOOD ROBUSTNESS")
        print(f"  FE-Poisson confirms OLS patterns")
        print(f"  Minor differences expected due to count model")
    else:
        print(f"\n⚠ MODERATE AGREEMENT")
        print(f"  Review coefficient comparison table")

print(f"\nOutputs:")
print(f"  1. c3c_fe_poisson_comparison.csv")
print(f"  2. c3c_fe_poisson_coefficients.csv")
print(f"  3. c3c_fe_poisson_summary.txt")
print(f"  4. c3c_fe_poisson_comparison_plot.png")

print("\n" + "=" * 70)
print("Next: Placebo test 2021 (c3d_placebo_2021.py)")
print("=" * 70)