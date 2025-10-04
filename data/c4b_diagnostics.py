"""
c4_diagnostics.py
=================
Diagnostics for C4 pre-trend problem.

Focus:
1. Sample composition across years
2. Re-estimate models excluding 2019
3. Compare coefficients
4. Visual comparison

Goal: Understand if pre-trend is driven by 2019 sample or systematic.
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
OUTPUT_DIR = Path("analysis/c4_success_rate/diagnostics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_YEAR = 2022
EXCLUDE_2020 = True

# ============================================================================
# LOAD & PREPARE DATA
# ============================================================================

print("=" * 70)
print("C4 DIAGNOSTICS - PRE-TREND ANALYSIS")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)
df_baseline = df[df['in_baseline_2022'] == 1].copy()

if EXCLUDE_2020:
    df_baseline = df_baseline[df_baseline['season'] != 2020].copy()

# Handle column names
sb_col = next(c for c in ["n_sb", "sb_2b", "stolen_bases_2b"] if c in df_baseline.columns)
cs_col = next(c for c in ["n_cs", "cs_2b", "caught_stealing_2b"] if c in df_baseline.columns)

if "attempts_2b" not in df_baseline.columns:
    df_baseline["attempts_2b"] = df_baseline[sb_col].fillna(0) + df_baseline[cs_col].fillna(0)

df_baseline = df_baseline[df_baseline['attempts_2b'] > 0].copy()
df_baseline["success_rate"] = df_baseline[sb_col] / df_baseline["attempts_2b"]

group_col = "baseline_group" if "baseline_group" in df_baseline.columns else "baseline_tercile_2022"

print(f"\nTotal sample: {len(df_baseline):,} observations")
print(f"Pitchers: {df_baseline['pitcher_id'].nunique()}")
print(f"Years: {sorted(df_baseline['season'].unique())}")

# ============================================================================
# DIAGNOSTIC 1: SAMPLE COMPOSITION BY YEAR
# ============================================================================

print("\n" + "=" * 70)
print("DIAGNOSTIC 1: SAMPLE COMPOSITION")
print("=" * 70)

comp = df_baseline.groupby('season').agg({
    'pitcher_id': 'count',
    'attempts_2b': ['mean', 'median', 'sum'],
    'success_rate': ['mean', 'std'],
    sb_col: 'sum',
    cs_col: 'sum'
}).round(3)

comp.columns = ['n_obs', 'attempts_mean', 'attempts_median', 'attempts_sum', 
                'sb_pct_mean', 'sb_pct_std', 'total_sb', 'total_cs']

print("\n" + comp.to_string())

# Check pitcher overlap with 2022
print("\n" + "-" * 70)
print("PITCHER OVERLAP WITH 2022")
print("-" * 70)

pitchers_2022 = set(df_baseline[df_baseline['season'] == 2022]['pitcher_id'])

for year in sorted(df_baseline['season'].unique()):
    if year == 2022:
        continue
    pitchers_year = set(df_baseline[df_baseline['season'] == year]['pitcher_id'])
    overlap = len(pitchers_year & pitchers_2022)
    only_year = len(pitchers_year - pitchers_2022)
    pct = (overlap / len(pitchers_year)) * 100 if len(pitchers_year) > 0 else 0
    print(f"{year}: {overlap:3d} overlap, {only_year:3d} unique ({pct:.1f}% overlap)")

# ============================================================================
# DIAGNOSTIC 2: VISUAL INSPECTION OF RAW MEANS
# ============================================================================

print("\n" + "=" * 70)
print("DIAGNOSTIC 2: RAW SUCCESS RATES BY YEAR/TERCILE")
print("=" * 70)

raw_means = df_baseline.groupby(['season', group_col]).agg({
    'success_rate': ['mean', 'std', 'count']
}).round(3)

print("\n" + raw_means.to_string())

# ============================================================================
# FUNCTION TO ESTIMATE MODEL
# ============================================================================

def estimate_models(df_input, label):
    """Estimate both Binomial-GLM and LPM-FE on given data."""
    
    years = sorted(df_input['season'].unique())
    years_non_base = [y for y in years if y != BASE_YEAR]
    
    # Create dummies
    for year in years_non_base:
        df_input[f'year_{year}'] = (df_input['season'] == year).astype(int)
    
    interaction_cols = []
    for year in years_non_base:
        for tercile in ['T1', 'T3']:
            int_col = f'y{year}_x_{tercile}'
            df_input[int_col] = (
                (df_input['season'] == year) & 
                (df_input[group_col] == tercile)
            ).astype(int)
            interaction_cols.append(int_col)
    
    year_dummy_cols = [f'year_{y}' for y in years_non_base]
    all_regressors = year_dummy_cols + interaction_cols
    
    # Binomial-GLM
    pitcher_dummies = pd.get_dummies(df_input['pitcher_id'], prefix='pitcher', drop_first=True)
    X_bin = pd.concat([df_input[all_regressors], pitcher_dummies], axis=1)
    X_bin = X_bin.astype(np.float64)
    X_bin = sm.add_constant(X_bin, has_constant='add')
    
    y_bin = np.column_stack([df_input[sb_col].values, df_input[cs_col].values])
    
    bin_model = sm.GLM(endog=y_bin, exog=X_bin, family=Binomial())
    bin_result = bin_model.fit(cov_type='cluster', cov_kwds={'groups': df_input['pitcher_id'].values})
    
    # LPM-FE
    df_panel = df_input.set_index(['pitcher_id', 'season'])
    X_lpm = df_panel[all_regressors]
    y_lpm = df_panel['success_rate']
    w_lpm = df_panel['attempts_2b']
    
    lpm_model = PanelOLS(dependent=y_lpm, exog=X_lpm, weights=w_lpm, 
                         entity_effects=True, time_effects=False)
    lpm_result = lpm_model.fit(cov_type='clustered', cluster_entity=True)
    
    # Pre-trend test (LPM)
    pre_years = [y for y in years_non_base if y < 2023]
    pre_year_cols = [f'year_{y}' for y in pre_years]
    
    if len(pre_year_cols) > 0:
        param_names = list(lpm_result.params.index)
        pre_indices = [i for i, name in enumerate(param_names) if name in pre_year_cols]
        
        R = np.zeros((len(pre_indices), len(param_names)))
        for i, idx in enumerate(pre_indices):
            R[i, idx] = 1
        
        beta = lpm_result.params.values
        vcov = lpm_result.cov.values
        
        Rbeta = R @ beta
        RVR = R @ vcov @ R.T
        
        wald_stat = Rbeta.T @ np.linalg.inv(RVR) @ Rbeta
        p_value = 1 - stats.chi2.cdf(wald_stat, len(pre_indices))
    else:
        wald_stat, p_value = None, None
    
    # Extract coefficients for T2
    results = {}
    baseline_sb_pct = df_input[df_input['season'] == BASE_YEAR]['success_rate'].mean()
    
    for year in years:
        if year == BASE_YEAR:
            results[year] = {
                'bin_coef': 0.0, 'bin_se': 0.0, 'bin_pval': np.nan,
                'lpm_coef': 0.0, 'lpm_se': 0.0, 'lpm_pval': np.nan
            }
        else:
            year_col = f'year_{year}'
            results[year] = {
                'bin_coef': bin_result.params.get(year_col, 0.0),
                'bin_se': bin_result.bse.get(year_col, 0.0),
                'bin_pval': bin_result.pvalues.get(year_col, np.nan),
                'lpm_coef': lpm_result.params.get(year_col, 0.0),
                'lpm_se': lpm_result.std_errors.get(year_col, 0.0),
                'lpm_pval': lpm_result.pvalues.get(year_col, np.nan)
            }
    
    return {
        'results': results,
        'wald_stat': wald_stat,
        'wald_pval': p_value,
        'n_obs': len(df_input),
        'baseline_sb_pct': baseline_sb_pct
    }

# ============================================================================
# DIAGNOSTIC 3: FULL SAMPLE VS EXCLUDE 2019
# ============================================================================

print("\n" + "=" * 70)
print("DIAGNOSTIC 3: MODEL COMPARISON (FULL VS NO-2019)")
print("=" * 70)

print("\nEstimating FULL sample...")
full_results = estimate_models(df_baseline.copy(), "Full")

print(f"Done. N={full_results['n_obs']}")
if full_results['wald_pval'] is not None:
    print(f"Pre-trend test: Chi²={full_results['wald_stat']:.3f}, p={full_results['wald_pval']:.4f}")

print("\nEstimating WITHOUT 2019...")
df_no2019 = df_baseline[df_baseline['season'] != 2019].copy()
no2019_results = estimate_models(df_no2019, "No2019")

print(f"Done. N={no2019_results['n_obs']}")
if no2019_results['wald_pval'] is not None:
    print(f"Pre-trend test: Chi²={no2019_results['wald_stat']:.3f}, p={no2019_results['wald_pval']:.4f}")

# ============================================================================
# COMPARISON TABLE
# ============================================================================

print("\n" + "=" * 70)
print("COEFFICIENT COMPARISON (T2 ONLY)")
print("=" * 70)

print(f"\n{'Year':<6} {'Sample':<10} {'Model':<10} {'Coef':<10} {'SE':<8} {'p-value':<10}")
print("-" * 70)

for year in sorted(df_baseline['season'].unique()):
    if year == BASE_YEAR:
        continue
    
    # Full sample
    fr = full_results['results'][year]
    print(f"{year:<6} {'Full':<10} {'Binomial':<10} {fr['bin_coef']:+.4f}    {fr['bin_se']:.4f}  {fr['bin_pval']:.4f}")
    print(f"{'':<6} {'':<10} {'LPM':<10} {fr['lpm_coef']:+.4f}    {fr['lpm_se']:.4f}  {fr['lpm_pval']:.4f}")
    
    # No 2019
    if year != 2019:
        nr = no2019_results['results'][year]
        print(f"{'':<6} {'No-2019':<10} {'Binomial':<10} {nr['bin_coef']:+.4f}    {nr['bin_se']:.4f}  {nr['bin_pval']:.4f}")
        print(f"{'':<6} {'':<10} {'LPM':<10} {nr['lpm_coef']:+.4f}    {nr['lpm_se']:.4f}  {nr['lpm_pval']:.4f}")
    
    print()

# ============================================================================
# KEY TREATMENT YEARS COMPARISON
# ============================================================================

print("\n" + "=" * 70)
print("CRITICAL: 2023/2024 STABILITY CHECK")
print("=" * 70)

for year in [2023, 2024]:
    print(f"\n{year} Treatment Effect (T2):")
    print("-" * 40)
    
    fr = full_results['results'][year]
    nr = no2019_results['results'][year]
    
    print(f"Full sample (LPM):    {fr['lpm_coef']:+.4f} (p={fr['lpm_pval']:.4f})")
    print(f"No-2019 (LPM):        {nr['lpm_coef']:+.4f} (p={nr['lpm_pval']:.4f})")
    
    diff = abs(fr['lpm_coef'] - nr['lpm_coef'])
    pct_change = (diff / abs(fr['lpm_coef'])) * 100 if fr['lpm_coef'] != 0 else 0
    
    print(f"Difference:           {diff:.4f} ({pct_change:.1f}% change)")
    
    if diff < 0.01 and abs(fr['lpm_pval'] - nr['lpm_pval']) < 0.05:
        print("✓ STABLE - 2019 not driving result")
    elif diff < 0.02:
        print("⚠ MODERATELY STABLE - small sensitivity")
    else:
        print("✗ UNSTABLE - 2019 may be critical")

# ============================================================================
# VISUAL COMPARISON
# ============================================================================

print("\n" + "-" * 70)
print("CREATING COMPARISON PLOTS")
print("-" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Extract data for plotting
years_plot = sorted([y for y in df_baseline['season'].unique() if y != BASE_YEAR])
full_bin = [full_results['results'][y]['bin_coef'] for y in years_plot]
full_lpm = [full_results['results'][y]['lpm_coef'] for y in years_plot]
no19_bin = [no2019_results['results'][y]['bin_coef'] if y != 2019 else np.nan for y in years_plot]
no19_lpm = [no2019_results['results'][y]['lpm_coef'] if y != 2019 else np.nan for y in years_plot]

# Panel 1: Binomial-GLM comparison
ax1 = axes[0, 0]
ax1.plot(years_plot, full_bin, 'o-', linewidth=2, label='Full sample', color='C0')
ax1.plot(years_plot, no19_bin, 's--', linewidth=2, label='Excl. 2019', color='C1')
ax1.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax1.axvline(BASE_YEAR, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
ax1.axvline(2023, color='green', linestyle=':', linewidth=1.5, alpha=0.5)
ax1.set_xlabel('Year')
ax1.set_ylabel('Coefficient (log-odds)')
ax1.set_title('Binomial-GLM: Full vs No-2019')
ax1.legend()
ax1.grid(alpha=0.3)

# Panel 2: LPM comparison
ax2 = axes[0, 1]
ax2.plot(years_plot, full_lpm, 'o-', linewidth=2, label='Full sample', color='C0')
ax2.plot(years_plot, no19_lpm, 's--', linewidth=2, label='Excl. 2019', color='C1')
ax2.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax2.axvline(BASE_YEAR, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
ax2.axvline(2023, color='green', linestyle=':', linewidth=1.5, alpha=0.5)
ax2.set_xlabel('Year')
ax2.set_ylabel('Coefficient (pp change)')
ax2.set_title('LPM-FE: Full vs No-2019')
ax2.legend()
ax2.grid(alpha=0.3)

# Panel 3: Sample sizes
ax3 = axes[1, 0]
sample_sizes = df_baseline.groupby('season').size()
ax3.bar(sample_sizes.index, sample_sizes.values, alpha=0.7, color='steelblue')
ax3.axvline(2019, color='red', linestyle='--', linewidth=2, label='2019 (excluded)')
ax3.set_xlabel('Year')
ax3.set_ylabel('N observations')
ax3.set_title('Sample Size by Year')
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# Panel 4: Raw success rates
ax4 = axes[1, 1]
raw_sr = df_baseline.groupby('season')['success_rate'].mean()
ax4.plot(raw_sr.index, raw_sr.values, 'o-', linewidth=2, color='darkgreen')
ax4.axhline(raw_sr[BASE_YEAR], color='red', linestyle=':', linewidth=1.5, alpha=0.5, label=f'2022 baseline')
ax4.axvline(2023, color='green', linestyle=':', linewidth=1.5, alpha=0.5, label='Treatment')
ax4.set_xlabel('Year')
ax4.set_ylabel('Mean Success Rate')
ax4.set_title('Raw Success Rate Trends')
ax4.legend()
ax4.grid(alpha=0.3)

plt.tight_layout()
output_plot = OUTPUT_DIR / "diagnostic_comparison.png"
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
print(f"\nSaved: {output_plot}")
plt.close()

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "-" * 70)
print("SAVING DIAGNOSTIC OUTPUTS")
print("-" * 70)

# Summary report
output_summary = OUTPUT_DIR / "diagnostic_summary.txt"
with open(output_summary, 'w', encoding='utf-8') as f:
    f.write("C4 DIAGNOSTIC REPORT - PRE-TREND ANALYSIS\n")
    f.write("=" * 70 + "\n\n")
    
    f.write("SAMPLE COMPOSITION\n")
    f.write("-" * 70 + "\n")
    f.write(comp.to_string() + "\n\n")
    
    f.write("PRE-TREND TESTS\n")
    f.write("-" * 70 + "\n")
    f.write(f"Full sample:  Chi²={full_results['wald_stat']:.3f}, p={full_results['wald_pval']:.4f}\n")
    f.write(f"No-2019:      Chi²={no2019_results['wald_stat']:.3f}, p={no2019_results['wald_pval']:.4f}\n\n")
    
    f.write("TREATMENT EFFECTS (2023/2024, T2, LPM)\n")
    f.write("-" * 70 + "\n")
    for year in [2023, 2024]:
        fr = full_results['results'][year]
        nr = no2019_results['results'][year]
        f.write(f"\n{year}:\n")
        f.write(f"  Full:     {fr['lpm_coef']:+.4f} (p={fr['lpm_pval']:.4f})\n")
        f.write(f"  No-2019:  {nr['lpm_coef']:+.4f} (p={nr['lpm_pval']:.4f})\n")
        diff = abs(fr['lpm_coef'] - nr['lpm_coef'])
        f.write(f"  Diff:     {diff:.4f}\n")
    
    f.write("\n" + "=" * 70 + "\n")
    f.write("RECOMMENDATION\n")
    f.write("-" * 70 + "\n")
    
    # Simple decision rule
    diff_2023 = abs(full_results['results'][2023]['lpm_coef'] - no2019_results['results'][2023]['lpm_coef'])
    diff_2024 = abs(full_results['results'][2024]['lpm_coef'] - no2019_results['results'][2024]['lpm_coef'])
    
    if diff_2023 < 0.01 and diff_2024 < 0.01:
        f.write("✓ RESULTS STABLE: 2019 not driving treatment effects.\n")
        f.write("  Proceed with publication. Report No-2019 as robustness check.\n")
    elif diff_2023 < 0.02 and diff_2024 < 0.02:
        f.write("⚠ MODERATE SENSITIVITY: Treatment effects slightly attenuated.\n")
        f.write("  Consider using No-2019 as primary specification.\n")
    else:
        f.write("✗ UNSTABLE RESULTS: 2019 critically affects estimates.\n")
        f.write("  Consider:\n")
        f.write("  1. Frame as descriptive rather than causal\n")
        f.write("  2. Focus on attempt rate (C3) if more robust\n")
        f.write("  3. Use alternative identification (MiLB experiments)\n")

print(f"\nSaved: {output_summary}")

# ============================================================================
# FINAL OUTPUT
# ============================================================================

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)

print(f"\nOutputs in: {OUTPUT_DIR}")
print("  1. diagnostic_summary.txt")
print("  2. diagnostic_comparison.png")

print("\n" + "=" * 70)
print("NEXT STEPS")
print("=" * 70)

diff_2023 = abs(full_results['results'][2023]['lpm_coef'] - no2019_results['results'][2023]['lpm_coef'])
diff_2024 = abs(full_results['results'][2024]['lpm_coef'] - no2019_results['results'][2024]['lpm_coef'])

if diff_2023 < 0.01 and diff_2024 < 0.01:
    print("\n✓ Good news: Results are stable without 2019")
    print("  → Use No-2019 specification as robustness check")
    print("  → Proceed with paper as planned")
elif diff_2023 < 0.02 and diff_2024 < 0.02:
    print("\n⚠ Mixed news: Moderate sensitivity to 2019")
    print("  → Consider No-2019 as primary specification")
    print("  → Honest discussion of pre-trends in paper")
else:
    print("\n✗ Problem: Results change substantially without 2019")
    print("  → Success rate analysis may not be causally identified")
    print("  → Check if C3 (attempt rate) has better pre-trends")
    print("  → Consider pivoting to MiLB quasi-experiments as primary")

print("\n" + "=" * 70)