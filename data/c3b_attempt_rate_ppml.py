"""
c3_attempt_rate_ppml.py
=======================
Attempt rate analysis via PPML (for Rate Ratio compatibility with C5)

Purpose:
- Direct estimation of attempt rate effect via PPML
- Provides Rate Ratios on same scale as C5 (total steal rate)
- Enables mechanistic decomposition: C3 (attempts) × C4 (success) ≈ C5 (total)

Model: FE-PPML with log(Opportunities) offset
- Endog: attempts_2b (attempt counts)
- Exposure: opportunities_2b (pitches with runner on 1B only)
- Fixed effects: Pitcher + Year
- Standard errors: Cluster-robust by pitcher

Comparison to original C3:
- Original: FE-OLS/WLS on attempt_rate (linear coefficients, pp)
- This version: PPML on attempt counts (log-coefficients, RR)
- Both are valid, but PPML provides RR for decomposition

Context:
- 2023: 15s BE / 20s RO pitch timer + pickoff limit + larger bases
- 2024: 18s RO
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import statsmodels.api as sm
from statsmodels.genmod.families import Poisson
from scipy import stats
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FILE = Path("analysis/c_running_game/c_panel_with_baseline.csv")
OUTPUT_DIR = Path("analysis/c3_ppml")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_YEAR = 2022
EXCLUDE_2020 = True

# ============================================================================
# LOAD DATA
# ============================================================================

print("=" * 70)
print("C3 ATTEMPT RATE ANALYSIS (PPML)")
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

if EXCLUDE_2020:
    n_before = len(df_baseline)
    df_baseline = df_baseline[df_baseline['season'] != 2020].copy()
    n_dropped = n_before - len(df_baseline)
    print(f"Excluded 2020 (COVID): dropped {n_dropped} observations")

# Handle column names robustly
att_col = next(c for c in ["attempts_2b", "att_2b", "n_attempts_2b"] if c in df_baseline.columns)
opp_col = next(c for c in ["opportunities_2b", "opps_2b", "n_opps_2b"] if c in df_baseline.columns)

# Ensure opportunities > 0 (needed for offset)
df_baseline = df_baseline[df_baseline[opp_col] > 0].copy()
print(f"\nFiltered to opportunities > 0: {len(df_baseline):,} observations")
print(f"  Pitchers: {df_baseline['pitcher_id'].nunique()}")
print(f"  Years: {sorted(df_baseline['season'].unique())}")

group_col = "baseline_group" if "baseline_group" in df_baseline.columns else "baseline_tercile_2022"

# ============================================================================
# DESCRIPTIVE STATISTICS
# ============================================================================

print("\n" + "-" * 70)
print("DESCRIPTIVE STATISTICS")
print("-" * 70)

desc = df_baseline.groupby('season').agg({
    'pitcher_id': 'count',
    att_col: ['sum', 'mean'],
    opp_col: ['sum', 'mean']
}).round(3)

desc.columns = ['n_obs', 'total_attempts', 'mean_att_per_pitcher', 'total_opps', 'mean_opps_per_pitcher']
desc['att_per_opp'] = (desc['total_attempts'] / desc['total_opps']).round(4)

print("\n" + desc.to_string())

# Baseline rate (2022)
baseline_rate = desc.loc[BASE_YEAR, 'att_per_opp']
print(f"\nBaseline attempt rate (2022): {baseline_rate:.4f}")

# ============================================================================
# PREPARE PANEL DATA
# ============================================================================

print("\n" + "-" * 70)
print("PREPARING PANEL DATA")
print("-" * 70)

years = sorted(df_baseline['season'].unique())
years_non_base = [y for y in years if y != BASE_YEAR]

# Create year dummies
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
            (df_baseline[group_col] == tercile)
        ).astype(int)
        interaction_cols.append(int_col)

print(f"Year×Tercile interactions: {len(interaction_cols)}")

# Drop NA
df_baseline = df_baseline.dropna(subset=[att_col, opp_col])
print(f"\nAfter dropping NA: {len(df_baseline):,} observations")

# ============================================================================
# MODEL: FE-PPML WITH OFFSET
# ============================================================================

print("\n" + "=" * 70)
print("MODEL: FE-PPML (POISSON WITH OFFSET)")
print("=" * 70)

# Create pitcher dummies
pitchers = sorted(df_baseline['pitcher_id'].unique())
n_pitchers = len(pitchers)
print(f"\nCreating {n_pitchers} pitcher dummies...")

pitcher_dummies = pd.get_dummies(df_baseline['pitcher_id'], prefix='pitcher', drop_first=True)
print(f"  Created {len(pitcher_dummies.columns)} dummy variables")

# Prepare regressors
year_dummy_cols = [f'year_{y}' for y in years_non_base]
all_regressors = year_dummy_cols + interaction_cols

X_ppml = pd.concat([
    df_baseline[all_regressors],
    pitcher_dummies
], axis=1)

X_ppml = X_ppml.astype(np.float64)
X_ppml = sm.add_constant(X_ppml, has_constant='add')

# Endog: Attempt counts
y_ppml = df_baseline[att_col].values

# Exposure: Opportunities
exposure = df_baseline[opp_col].values

print(f"\nFitting Poisson GLM with offset...")
print(f"  Outcome: {att_col} (attempt counts)")
print(f"  Exposure: {opp_col} (opportunities)")
print(f"  Family: Poisson")
print(f"  Regressors: {len(all_regressors)} (years + interactions)")
print(f"  Pitcher FE: {len(pitcher_dummies.columns)} dummies")
print(f"  Cluster SE: pitcher ({n_pitchers} clusters)")

try:
    ppml_model = sm.GLM(
        endog=y_ppml,
        exog=X_ppml,
        family=Poisson(),
        exposure=exposure
    )
    
    ppml_result = ppml_model.fit(
        cov_type='cluster',
        cov_kwds={'groups': df_baseline['pitcher_id'].values}
    )
    
    print(f"✓ Model converged")
    print(f"  Log-likelihood: {ppml_result.llf:.1f}")
    print(f"  Deviance: {ppml_result.deviance:.1f}")
    print(f"  Observations: {ppml_result.nobs:.0f}")
    
except Exception as e:
    print(f"✗ Model failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# PRE-TREND TEST
# ============================================================================

print("\n" + "=" * 70)
print("PRE-TREND TEST (CLUSTER-ROBUST)")
print("=" * 70)

pre_years = [y for y in years_non_base if y < 2023]
pre_year_cols = [f'year_{y}' for y in pre_years]

print(f"\nPre-treatment years: {pre_years}")
print(f"Testing H0: All pre-year coefficients = 0")

if len(pre_year_cols) > 0:
    param_names = list(ppml_result.params.index)
    pre_indices = [i for i, name in enumerate(param_names) if name in pre_year_cols]
    
    R = np.zeros((len(pre_indices), len(param_names)))
    for i, idx in enumerate(pre_indices):
        R[i, idx] = 1
    
    beta = ppml_result.params.values
    vcov = ppml_result.cov_params().values
    
    Rbeta = R @ beta
    RVR = R @ vcov @ R.T
    
    wald_stat = Rbeta.T @ np.linalg.inv(RVR) @ Rbeta
    df = len(pre_indices)
    p_value = 1 - stats.chi2.cdf(wald_stat, df)
    
    print(f"\nWald Test Results:")
    print(f"  Chi² statistic: {wald_stat:.3f}")
    print(f"  Degrees of freedom: {df}")
    print(f"  P-value: {p_value:.4f}")
    
    if p_value > 0.10:
        print(f"  ✓ Pre-trends jointly insignificant (p > 0.10)")
    elif p_value > 0.05:
        print(f"  ⚠ Weak evidence of pre-trends (0.05 < p < 0.10)")
    else:
        print(f"  ✗ Significant pre-trends detected (p < 0.05)")
else:
    print("\nNo pre-treatment years to test.")
    wald_stat, p_value = None, None

# ============================================================================
# EXTRACT COEFFICIENTS & RATE RATIOS
# ============================================================================

print("\n" + "-" * 70)
print("EXTRACTING COEFFICIENTS")
print("-" * 70)

params = ppml_result.params
se = ppml_result.bse
pvals = ppml_result.pvalues

# Build coefficient table
coef_data = []

for year in years:
    for tercile in ['T2', 'T1', 'T3']:
        if year == BASE_YEAR:
            coef_data.append({
                'year': year,
                'tercile': tercile,
                'coef': 0.0,
                'se': 0.0,
                'pval': np.nan,
                'rr': 1.0,
                'pct_change': 0.0
            })
        else:
            year_col = f'year_{year}'
            
            if tercile == 'T2':
                coef = params.get(year_col, 0.0)
                se_val = se.get(year_col, 0.0)
                pval = pvals.get(year_col, np.nan)
            else:
                int_col = f'y{year}_x_{tercile}'
                coef_year = params.get(year_col, 0.0)
                coef_int = params.get(int_col, 0.0)
                coef = coef_year + coef_int
                se_val = se.get(int_col, 0.0)
                pval = pvals.get(int_col, np.nan)
            
            rr = np.exp(coef)
            pct_change = (rr - 1) * 100
            
            coef_data.append({
                'year': year,
                'tercile': tercile,
                'coef': coef,
                'se': se_val,
                'pval': pval,
                'rr': rr,
                'pct_change': pct_change
            })

df_coef = pd.DataFrame(coef_data)

# Add CIs
df_coef['ci_lower'] = df_coef['coef'] - 1.96 * df_coef['se']
df_coef['ci_upper'] = df_coef['coef'] + 1.96 * df_coef['se']

print(f"\nExtracted {len(df_coef)} coefficients")

# ============================================================================
# KEY TREATMENT EFFECTS
# ============================================================================

print("\n" + "=" * 70)
print("KEY TREATMENT EFFECTS (T2, 2023-2024)")
print("=" * 70)

print(f"\n{'Year':<6} {'Coef':<10} {'RR':<8} {'% Change':<10} {'SE':<8} {'p-value':<10}")
print("-" * 70)

for year in [2023, 2024]:
    row = df_coef[(df_coef['year'] == year) & (df_coef['tercile'] == 'T2')].iloc[0]
    sig = "***" if row['pval'] < 0.01 else ("**" if row['pval'] < 0.05 else ("*" if row['pval'] < 0.10 else ""))
    
    print(f"{year:<6} {row['coef']:+.4f}{sig:<3} {row['rr']:.3f}  {row['pct_change']:+.1f}%     {row['se']:.4f}  {row['pval']:.4f}")

# ============================================================================
# SAVE OUTPUTS
# ============================================================================

print("\n" + "-" * 70)
print("SAVING OUTPUTS")
print("-" * 70)

# Coefficients
output_coef = OUTPUT_DIR / "c3_ppml_coefficients.csv"
df_coef.to_csv(output_coef, index=False)
print(f"\nSaved: {output_coef}")

# Summary
output_summary = OUTPUT_DIR / "c3_ppml_summary.txt"
with open(output_summary, 'w', encoding='utf-8') as f:
    f.write("C3 ATTEMPT RATE ANALYSIS (PPML)\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Model: FE-PPML with log(opportunities) offset\n")
    f.write(f"Sample: Pitchers with ≥50 pitches in 2022\n")
    if EXCLUDE_2020:
        f.write(f"Robustness: 2020 (COVID) excluded\n")
    f.write(f"Base year: {BASE_YEAR}\n")
    f.write(f"Baseline rate: {baseline_rate:.4f}\n\n")
    
    f.write(f"Sample statistics:\n")
    f.write(f"  Observations: {len(df_baseline):,}\n")
    f.write(f"  Pitchers: {n_pitchers}\n\n")
    
    f.write("-" * 70 + "\n")
    f.write("PRE-TREND TEST\n")
    f.write("-" * 70 + "\n")
    if wald_stat is not None:
        f.write(f"Wald Chi² = {wald_stat:.3f} (df={df})\n")
        f.write(f"P-value = {p_value:.4f}\n")
        if p_value > 0.10:
            f.write("✓ Pre-trends jointly insignificant\n\n")
        else:
            f.write("⚠ Evidence of pre-trends\n\n")
    
    f.write("-" * 70 + "\n")
    f.write("TREATMENT EFFECTS (T2, 2023-2024)\n")
    f.write("-" * 70 + "\n")
    for year in [2023, 2024]:
        row = df_coef[(df_coef['year'] == year) & (df_coef['tercile'] == 'T2')].iloc[0]
        f.write(f"\n{year}:\n")
        f.write(f"  Coefficient: {row['coef']:+.4f} (p={row['pval']:.4f})\n")
        f.write(f"  Rate Ratio: {row['rr']:.3f}\n")
        f.write(f"  % Change: {row['pct_change']:+.1f}%\n")

print(f"Saved: {output_summary}")

# ============================================================================
# PLOTS
# ============================================================================

print("\n" + "-" * 70)
print("CREATING PLOTS")
print("-" * 70)

# Event study plot
fig, ax = plt.subplots(figsize=(10, 6))

for tercile in ['T1', 'T2', 'T3']:
    df_plot = df_coef[df_coef['tercile'] == tercile].sort_values('year')
    ax.plot(df_plot['year'], df_plot['coef'], 
            marker='o', linewidth=2, label=tercile)
    ax.fill_between(df_plot['year'], 
                     df_plot['ci_lower'], 
                     df_plot['ci_upper'], 
                     alpha=0.2)

ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax.axvline(BASE_YEAR, color='red', linestyle=':', linewidth=1.5, alpha=0.5, label='Base')
ax.axvline(2023, color='green', linestyle=':', linewidth=1.5, alpha=0.5, label='Treatment')
ax.set_xlabel('Year', fontsize=11)
ax.set_ylabel('Log Rate Ratio', fontsize=11)
ax.set_title('C3: Attempt Rate (FE-PPML)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='best')
ax.grid(alpha=0.3)

plt.tight_layout()
output_plot = OUTPUT_DIR / "c3_ppml_event_study.png"
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
print(f"\nSaved: {output_plot}")
plt.close()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("C3 ATTEMPT RATE (PPML) COMPLETE")
print("=" * 70)

print(f"\nSample: {len(df_baseline):,} pitcher-season obs")
print(f"Baseline attempt rate (2022): {baseline_rate:.4f}")

row_2023 = df_coef[(df_coef['year'] == 2023) & (df_coef['tercile'] == 'T2')].iloc[0]
row_2024 = df_coef[(df_coef['year'] == 2024) & (df_coef['tercile'] == 'T2')].iloc[0]

print(f"\n2023 Treatment Effect (T2):")
print(f"  Rate Ratio: {row_2023['rr']:.3f} ({row_2023['pct_change']:+.1f}%)")
print(f"  p-value: {row_2023['pval']:.4f}")

print(f"\n2024 Effect (T2):")
print(f"  Rate Ratio: {row_2024['rr']:.3f} ({row_2024['pct_change']:+.1f}%)")
print(f"  p-value: {row_2024['pval']:.4f}")

if wald_stat is not None:
    if p_value > 0.10:
        print(f"\n✓ Pre-trends: jointly insignificant (p={p_value:.4f})")
    else:
        print(f"\n⚠ Pre-trends: p={p_value:.4f}")

print(f"\nOutputs:")
print(f"  1. c3_ppml_coefficients.csv")
print(f"  2. c3_ppml_summary.txt")
print(f"  3. c3_ppml_event_study.png")

print("\n" + "=" * 70)
print("Comparison to original C3:")
print("=" * 70)
print("Original C3: FE-OLS/WLS on attempt_rate (linear pp)")
print("This version: FE-PPML on attempt_counts (Rate Ratios)")
print("Both are valid - PPML enables RR-based decomposition with C5")
print("=" * 70)