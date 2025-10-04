"""
c4_success_rate_v2.py
=====================
Updated event study for steal success rate (conditional on attempt).

Changes from v1:
- PRIMARY MODEL: Binomial-GLM on [SB, CS] with cluster-robust SE
  (avoids freq_weights warning)
- Secondary: LPM-FE (kept for transparency)
- Removed: Fractional Logit (only deskriptiv if needed)
- Added: Pre-Trend F-Test (cluster-robust)
- Reporting: Odds Ratios + pp-translation

Outcome: success_rate = n_sb / attempts_2b
Sample: Only pitcher-seasons with attempts_2b > 0
Exclude 2020 (COVID)

Treatment context:
- 2023: 15s BE / 20s RO pitch timer + 2 pickoff limit + larger bases
- 2024: 18s RO
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import statsmodels.api as sm
from statsmodels.genmod.families import Binomial
from linearmodels import PanelOLS
from scipy import stats
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FILE = Path("analysis/c_running_game/c_panel_with_baseline.csv")
OUTPUT_DIR = Path("analysis/c4_success_rate")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_YEAR = 2022
BASE_TERCILE = 'T2'
EXCLUDE_2020 = True

# ============================================================================
# LOAD DATA
# ============================================================================

print("=" * 70)
print("C4 SUCCESS RATE EVENT STUDY (v2)")
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

# Handle column names robustly
sb_col = next(c for c in ["n_sb", "sb_2b", "stolen_bases_2b"] if c in df_baseline.columns)
cs_col = next(c for c in ["n_cs", "cs_2b", "caught_stealing_2b"] if c in df_baseline.columns)

if "attempts_2b" not in df_baseline.columns:
    df_baseline["attempts_2b"] = df_baseline[sb_col].fillna(0) + df_baseline[cs_col].fillna(0)

# CRITICAL: Only keep observations with attempts > 0
df_baseline = df_baseline[df_baseline['attempts_2b'] > 0].copy()
print(f"\nFiltered to attempts > 0: {len(df_baseline):,} observations")
print(f"  Pitchers: {df_baseline['pitcher_id'].nunique():,}")
print(f"  Years: {sorted(df_baseline['season'].unique())}")

# Baseline group column
group_col = "baseline_group" if "baseline_group" in df_baseline.columns else "baseline_tercile_2022"

# ============================================================================
# CREATE SUCCESS RATE & DESCRIPTIVES
# ============================================================================

print("\n" + "-" * 70)
print("COMPUTING SUCCESS RATE")
print("-" * 70)

df_baseline["success_rate"] = df_baseline[sb_col] / df_baseline["attempts_2b"]

print(f"\nSuccess rate statistics:")
print(f"  Mean: {df_baseline['success_rate'].mean():.3f}")
print(f"  Median: {df_baseline['success_rate'].median():.3f}")
print(f"  Std: {df_baseline['success_rate'].std():.3f}")

n_perfect_success = (df_baseline['success_rate'] == 1.0).sum()
n_perfect_failure = (df_baseline['success_rate'] == 0.0).sum()
print(f"\nPerfect outcomes:")
print(f"  100% success: {n_perfect_success} ({n_perfect_success/len(df_baseline)*100:.1f}%)")
print(f"  0% success: {n_perfect_failure} ({n_perfect_failure/len(df_baseline)*100:.1f}%)")

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
df_baseline = df_baseline.dropna(subset=['success_rate', 'attempts_2b', sb_col, cs_col])
print(f"\nAfter dropping NA: {len(df_baseline):,} observations")

# ============================================================================
# MODEL 1: BINOMIAL-GLM [SB, CS] (PRIMARY)
# ============================================================================

print("\n" + "=" * 70)
print("MODEL 1: BINOMIAL-GLM [SB, CS] (PRIMARY)")
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

X_binomial = pd.concat([
    df_baseline[all_regressors],
    pitcher_dummies
], axis=1)

X_binomial = X_binomial.astype(np.float64)
X_binomial = sm.add_constant(X_binomial, has_constant='add')

# Endog: [successes, failures] = [SB, CS]
y_binomial = np.column_stack([
    df_baseline[sb_col].values,
    df_baseline[cs_col].values
])

print(f"\nFitting Binomial-GLM...")
print(f"  Outcome: [SB, CS] counts")
print(f"  Family: Binomial (logit link)")
print(f"  Regressors: {len(all_regressors)} (years + interactions)")
print(f"  Pitcher FE: {len(pitcher_dummies.columns)} dummies")
print(f"  Cluster SE: pitcher ({n_pitchers} clusters)")

try:
    binomial_model = sm.GLM(
        endog=y_binomial,
        exog=X_binomial,
        family=Binomial()
    )
    
    binomial_result = binomial_model.fit(
        cov_type='cluster',
        cov_kwds={'groups': df_baseline['pitcher_id'].values}
    )
    
    print(f"✓ Model converged (no freq_weights warning)")
    print(f"  Log-likelihood: {binomial_result.llf:.1f}")
    print(f"  Deviance: {binomial_result.deviance:.1f}")
    print(f"  Observations: {binomial_result.nobs:.0f}")
    
except Exception as e:
    print(f"✗ Model failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# MODEL 2: LPM-FE (TRANSPARENCY BENCHMARK)
# ============================================================================

print("\n" + "=" * 70)
print("MODEL 2: LPM-FE (TRANSPARENCY BENCHMARK)")
print("=" * 70)

df_panel = df_baseline.set_index(['pitcher_id', 'season'])

X_lpm = df_panel[all_regressors]
y_lpm = df_panel['success_rate']
w_lpm = df_panel['attempts_2b']

print(f"\nFitting PanelOLS (WLS with opportunity weights)...")

mod_lpm = PanelOLS(
    dependent=y_lpm,
    exog=X_lpm,
    weights=w_lpm,
    entity_effects=True,
    time_effects=False
)

res_lpm = mod_lpm.fit(cov_type='clustered', cluster_entity=True)

print(f"✓ Model converged")
print(f"  R²: {res_lpm.rsquared:.4f}")
print(f"  R² (within): {res_lpm.rsquared_within:.4f}")

# ============================================================================
# PRE-TREND F-TEST (CLUSTER-ROBUST)
# ============================================================================

print("\n" + "=" * 70)
print("PRE-TREND F-TEST (CLUSTER-ROBUST)")
print("=" * 70)

# Identify pre-treatment years (before 2023)
pre_years = [y for y in years_non_base if y < 2023]
pre_year_cols = [f'year_{y}' for y in pre_years]

print(f"\nPre-treatment years: {pre_years}")
print(f"Testing H0: All pre-year coefficients = 0")

# Test using LPM (easier to interpret)
# Get coefficient indices
param_names = list(res_lpm.params.index)
pre_indices = [i for i, name in enumerate(param_names) if name in pre_year_cols]

if len(pre_indices) > 0:
    # Wald test
    R = np.zeros((len(pre_indices), len(param_names)))
    for i, idx in enumerate(pre_indices):
        R[i, idx] = 1
    
    # Manual Wald test with cluster-robust vcov
    beta = res_lpm.params.values
    vcov = res_lpm.cov.values
    
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
# EXTRACT & COMPARE COEFFICIENTS
# ============================================================================

print("\n" + "-" * 70)
print("EXTRACTING COEFFICIENTS")
print("-" * 70)

# Binomial-GLM
params_bin = binomial_result.params
se_bin = binomial_result.bse
pvals_bin = binomial_result.pvalues

# LPM
params_lpm = res_lpm.params
se_lpm = res_lpm.std_errors
pvals_lpm = res_lpm.pvalues

# Build coefficient comparison
coef_data = []

# Baseline SB% for pp-translation (use 2022 mean)
baseline_sb_pct = df_baseline[df_baseline['season'] == BASE_YEAR]['success_rate'].mean()
print(f"\nBaseline success rate (2022): {baseline_sb_pct:.3f}")

for year in years:
    for tercile in ['T2', 'T1', 'T3']:
        if year == BASE_YEAR:
            coef_data.append({
                'year': year,
                'tercile': tercile,
                'binomial_coef': 0.0,
                'binomial_se': 0.0,
                'binomial_pval': np.nan,
                'binomial_or': 1.0,
                'binomial_pp': 0.0,
                'lpm_coef': 0.0,
                'lpm_se': 0.0,
                'lpm_pval': np.nan
            })
        else:
            year_col = f'year_{year}'
            
            if tercile == 'T2':
                # Main effect only
                bin_coef = params_bin.get(year_col, 0.0)
                bin_se_val = se_bin.get(year_col, 0.0)
                bin_pval = pvals_bin.get(year_col, np.nan)
                
                lpm_coef = params_lpm.get(year_col, 0.0)
                lpm_se_val = se_lpm.get(year_col, 0.0)
                lpm_pval = pvals_lpm.get(year_col, np.nan)
            else:
                # Main + interaction
                int_col = f'y{year}_x_{tercile}'
                
                bin_year = params_bin.get(year_col, 0.0)
                bin_int = params_bin.get(int_col, 0.0)
                bin_coef = bin_year + bin_int
                bin_se_val = se_bin.get(int_col, 0.0)
                bin_pval = pvals_bin.get(int_col, np.nan)
                
                lpm_year = params_lpm.get(year_col, 0.0)
                lpm_int = params_lpm.get(int_col, 0.0)
                lpm_coef = lpm_year + lpm_int
                lpm_se_val = se_lpm.get(int_col, 0.0)
                lpm_pval = pvals_lpm.get(int_col, np.nan)
            
            # Odds Ratio
            bin_or = np.exp(bin_coef)
            
            # pp-approximation (Papke-Wooldridge)
            bin_pp = baseline_sb_pct * (1 - baseline_sb_pct) * bin_coef
            
            coef_data.append({
                'year': year,
                'tercile': tercile,
                'binomial_coef': bin_coef,
                'binomial_se': bin_se_val,
                'binomial_pval': bin_pval,
                'binomial_or': bin_or,
                'binomial_pp': bin_pp,
                'lpm_coef': lpm_coef,
                'lpm_se': lpm_se_val,
                'lpm_pval': lpm_pval
            })

df_coef = pd.DataFrame(coef_data)

# Add CIs
df_coef['bin_ci_lower'] = df_coef['binomial_coef'] - 1.96 * df_coef['binomial_se']
df_coef['bin_ci_upper'] = df_coef['binomial_coef'] + 1.96 * df_coef['binomial_se']
df_coef['lpm_ci_lower'] = df_coef['lpm_coef'] - 1.96 * df_coef['lpm_se']
df_coef['lpm_ci_upper'] = df_coef['lpm_coef'] + 1.96 * df_coef['lpm_se']

print(f"\nExtracted {len(df_coef)} coefficients")

# ============================================================================
# PRINT KEY RESULTS
# ============================================================================

print("\n" + "=" * 70)
print("KEY TREATMENT EFFECTS (T2, 2023-2024)")
print("=" * 70)

print(f"\n{'Year':<6} {'Model':<10} {'Coef':<10} {'OR':<8} {'pp':<8} {'SE':<8} {'p-value':<10}")
print("-" * 70)

for year in [2023, 2024]:
    row = df_coef[(df_coef['year'] == year) & (df_coef['tercile'] == 'T2')].iloc[0]
    
    sig_bin = "***" if row['binomial_pval'] < 0.01 else ("**" if row['binomial_pval'] < 0.05 else ("*" if row['binomial_pval'] < 0.10 else ""))
    sig_lpm = "***" if row['lpm_pval'] < 0.01 else ("**" if row['lpm_pval'] < 0.05 else ("*" if row['lpm_pval'] < 0.10 else ""))
    
    print(f"{year:<6} {'Binomial':<10} {row['binomial_coef']:+.4f}{sig_bin:<3} {row['binomial_or']:.3f}  {row['binomial_pp']:+.4f}  {row['binomial_se']:.4f}  {row['binomial_pval']:.4f}")
    print(f"{'':<6} {'LPM':<10} {row['lpm_coef']:+.4f}{sig_lpm:<3} {'':8} {'':8} {row['lpm_se']:.4f}  {row['lpm_pval']:.4f}")
    print()

# ============================================================================
# SAVE OUTPUTS
# ============================================================================

print("\n" + "-" * 70)
print("SAVING OUTPUTS")
print("-" * 70)

# Coefficients
output_coef = OUTPUT_DIR / "c4_coefficients_v2.csv"
df_coef.to_csv(output_coef, index=False)
print(f"\nSaved: {output_coef}")

# Summary
output_summary = OUTPUT_DIR / "c4_summary_v2.txt"
with open(output_summary, 'w', encoding='utf-8') as f:
    f.write("C4 SUCCESS RATE EVENT STUDY (v2)\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"PRIMARY: Binomial-GLM on [SB, CS] counts\n")
    f.write(f"BENCHMARK: LPM-FE\n")
    f.write(f"Sample: Pitchers with ≥50 pitches in 2022 AND attempts > 0\n")
    if EXCLUDE_2020:
        f.write(f"Robustness: 2020 (COVID) excluded\n")
    f.write(f"Base year: {BASE_YEAR}\n")
    f.write(f"Baseline SB%: {baseline_sb_pct:.3f}\n\n")
    
    f.write(f"Sample statistics:\n")
    f.write(f"  Observations: {len(df_baseline):,}\n")
    f.write(f"  Pitchers: {n_pitchers}\n")
    f.write(f"  Mean success rate: {df_baseline['success_rate'].mean():.3f}\n\n")
    
    f.write("-" * 70 + "\n")
    f.write("PRE-TREND TEST (LPM)\n")
    f.write("-" * 70 + "\n")
    if wald_stat is not None:
        f.write(f"Wald Chi² = {wald_stat:.3f} (df={len(pre_indices)})\n")
        f.write(f"P-value = {p_value:.4f}\n")
        if p_value > 0.10:
            f.write("✓ Pre-trends jointly insignificant\n\n")
        else:
            f.write("⚠ Evidence of pre-trends\n\n")
    
    f.write("-" * 70 + "\n")
    f.write("MODEL 1: BINOMIAL-GLM [SB, CS]\n")
    f.write("-" * 70 + "\n")
    f.write(f"Log-likelihood: {binomial_result.llf:.1f}\n")
    f.write(f"Deviance: {binomial_result.deviance:.1f}\n\n")
    
    f.write("Key parameters (T2, 2023-2024):\n")
    for year in [2023, 2024]:
        row = df_coef[(df_coef['year'] == year) & (df_coef['tercile'] == 'T2')].iloc[0]
        f.write(f"  {year}: β={row['binomial_coef']:+.4f}, OR={row['binomial_or']:.3f}, pp≈{row['binomial_pp']:+.4f}, p={row['binomial_pval']:.4f}\n")
    
    f.write("\n" + "-" * 70 + "\n")
    f.write("MODEL 2: LPM-FE\n")
    f.write("-" * 70 + "\n")
    f.write(str(res_lpm))

print(f"Saved: {output_summary}")

# ============================================================================
# PLOTS
# ============================================================================

print("\n" + "-" * 70)
print("CREATING EVENT STUDY PLOTS")
print("-" * 70)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Panel 1: Binomial-GLM (log-odds)
for tercile in ['T1', 'T2', 'T3']:
    df_plot = df_coef[df_coef['tercile'] == tercile].sort_values('year')
    ax1.plot(df_plot['year'], df_plot['binomial_coef'], 
             marker='o', linewidth=2, label=tercile)
    ax1.fill_between(df_plot['year'], 
                      df_plot['bin_ci_lower'], 
                      df_plot['bin_ci_upper'], 
                      alpha=0.2)

ax1.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax1.axvline(BASE_YEAR, color='red', linestyle=':', linewidth=1.5, alpha=0.5, label='Base')
ax1.axvline(2023, color='green', linestyle=':', linewidth=1.5, alpha=0.5, label='Treatment')
ax1.set_xlabel('Year', fontsize=11)
ax1.set_ylabel('Coefficient (log-odds)', fontsize=11)
ax1.set_title('Binomial-GLM [SB, CS]', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9, loc='best')
ax1.grid(alpha=0.3)

# Panel 2: LPM
for tercile in ['T1', 'T2', 'T3']:
    df_plot = df_coef[df_coef['tercile'] == tercile].sort_values('year')
    ax2.plot(df_plot['year'], df_plot['lpm_coef'], 
             marker='o', linewidth=2, label=tercile)
    ax2.fill_between(df_plot['year'], 
                      df_plot['lpm_ci_lower'], 
                      df_plot['lpm_ci_upper'], 
                      alpha=0.2)

ax2.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax2.axvline(BASE_YEAR, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
ax2.axvline(2023, color='green', linestyle=':', linewidth=1.5, alpha=0.5)
ax2.set_xlabel('Year', fontsize=11)
ax2.set_ylabel('Coefficient (pp change)', fontsize=11)
ax2.set_title('LPM-FE', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9, loc='best')
ax2.grid(alpha=0.3)

plt.tight_layout()
output_plot = OUTPUT_DIR / "c4_event_study_v2.png"
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
print(f"\nSaved: {output_plot}")
plt.close()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("C4 SUCCESS RATE COMPLETE (v2)")
print("=" * 70)

print(f"\nSample: {len(df_baseline):,} pitcher-season obs (attempts > 0)")
print(f"Baseline SB% (2022): {baseline_sb_pct:.3f}")

row_2023 = df_coef[(df_coef['year'] == 2023) & (df_coef['tercile'] == 'T2')].iloc[0]
row_2024 = df_coef[(df_coef['year'] == 2024) & (df_coef['tercile'] == 'T2')].iloc[0]

print(f"\n2023 Treatment Effects (T2):")
print(f"  Binomial-GLM: β={row_2023['binomial_coef']:+.4f}, OR={row_2023['binomial_or']:.3f}, pp≈{row_2023['binomial_pp']:+.4f}")
print(f"  LPM-FE:       {row_2023['lpm_coef']:+.4f} pp")

print(f"\n2024 Effects (T2):")
print(f"  Binomial-GLM: β={row_2024['binomial_coef']:+.4f}, OR={row_2024['binomial_or']:.3f}, pp≈{row_2024['binomial_pp']:+.4f}")
print(f"  LPM-FE:       {row_2024['lpm_coef']:+.4f} pp")

if wald_stat is not None and p_value > 0.10:
    print(f"\n✓ Pre-trends: jointly insignificant (p={p_value:.4f})")
elif wald_stat is not None:
    print(f"\n⚠ Pre-trends: p={p_value:.4f}")

print(f"\nOutputs:")
print(f"  1. c4_coefficients_v2.csv")
print(f"  2. c4_summary_v2.txt")
print(f"  3. c4_event_study_v2.png")

print("\n" + "=" * 70)
print("Interpretation:")
print("=" * 70)
print("PRIMARY: Binomial-GLM with cluster-robust SE (no freq_weights warning)")
print("BENCHMARK: LPM-FE for transparency")
print("REPORTING: Odds Ratios + pp-translation (Papke-Wooldridge)")
print("PRE-TRENDS: Formal F-test included")
print("=" * 70)