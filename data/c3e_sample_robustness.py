"""
c3e_sample_robustness.py
=========================
Sample robustness checks for C3 event study.

Tests two alternative sample definitions:
1. Stricter baseline (≥100 pitches in 2022 instead of ≥50)
2. Balanced panel (pitchers present in ALL years 2018-2024, excl 2020)

Specification identical to C3v2:
- PanelOLS with pitcher FE + year FE
- Year dummies + Year×Tercile interactions
- Cluster SE by pitcher
- Base year: 2022

Expectation: Effects remain stable across samples
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

INPUT_FILE = Path("analysis/c_running_game/c_panel_with_baseline.csv")
OUTPUT_DIR = Path("analysis/c3_sample_robustness")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_YEAR = 2022
BASE_TERCILE = 'T2'
EXCLUDE_2020 = True

# ============================================================================
# LOAD DATA
# ============================================================================

print("=" * 70)
print("C3E SAMPLE ROBUSTNESS CHECKS")
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

# ============================================================================
# DEFINE ALTERNATIVE SAMPLES
# ============================================================================

print("\n" + "-" * 70)
print("DEFINING ALTERNATIVE SAMPLES")
print("-" * 70)

# Main sample (for comparison)
df_main = df[df['in_baseline_2022'] == 1].copy()
if EXCLUDE_2020:
    df_main = df_main[df_main['season'] != 2020]

print(f"\n1. MAIN SAMPLE (≥50 pitches in 2022):")
print(f"   Observations: {len(df_main):,}")
print(f"   Pitchers: {df_main['pitcher_id'].nunique():,}")

# Sample 2: Stricter baseline (≥150 opportunities in 2022)
# Use opportunities as proxy: ~150 opps ≈ 100+ pitches with runners on
df_2022 = df[df['season'] == 2022].copy()
strict_pitchers = df_2022[df_2022['opportunities_2b'] >= 150]['pitcher_id'].unique()

df_strict = df[df['pitcher_id'].isin(strict_pitchers)].copy()
if EXCLUDE_2020:
    df_strict = df_strict[df_strict['season'] != 2020]

print(f"\n2. STRICT BASELINE (≥100 pitches in 2022):")
print(f"   Observations: {len(df_strict):,}")
print(f"   Pitchers: {df_strict['pitcher_id'].nunique():,}")
print(f"   Dropped: {len(df_main) - len(df_strict):,} obs, {df_main['pitcher_id'].nunique() - df_strict['pitcher_id'].nunique():,} pitchers")

# Sample 3: Balanced panel (present in all years)
if EXCLUDE_2020:
    required_years = [2018, 2019, 2021, 2022, 2023, 2024]
else:
    required_years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]

# Count years per pitcher in main sample
pitcher_year_counts = df_main.groupby('pitcher_id')['season'].nunique()
balanced_pitchers = pitcher_year_counts[pitcher_year_counts == len(required_years)].index

df_balanced = df_main[df_main['pitcher_id'].isin(balanced_pitchers)].copy()

print(f"\n3. BALANCED PANEL (all years {min(required_years)}-{max(required_years)}):")
print(f"   Observations: {len(df_balanced):,}")
print(f"   Pitchers: {df_balanced['pitcher_id'].nunique():,}")
print(f"   Dropped: {len(df_main) - len(df_balanced):,} obs, {df_main['pitcher_id'].nunique() - df_balanced['pitcher_id'].nunique():,} pitchers")

# ============================================================================
# HELPER FUNCTION: RUN MODEL
# ============================================================================

def run_event_study(df_sample, sample_name):
    """Run C3v2 specification on given sample"""
    
    print(f"\n" + "-" * 70)
    print(f"SAMPLE: {sample_name}")
    print("-" * 70)
    
    # Prepare data
    years = sorted(df_sample['season'].unique())
    years_non_base = [y for y in years if y != BASE_YEAR]
    
    # Create year dummies
    for year in years_non_base:
        df_sample[f'year_{year}'] = (df_sample['season'] == year).astype(int)
    
    # Create interactions
    interaction_cols = []
    for year in years_non_base:
        for tercile in ['T1', 'T3']:
            int_col = f'y{year}_x_{tercile}'
            df_sample[int_col] = (
                (df_sample['season'] == year) & 
                (df_sample['baseline_group'] == tercile)
            ).astype(int)
            interaction_cols.append(int_col)
    
    # Drop NA
    df_sample = df_sample.dropna(subset=['attempt_rate', 'opportunities_2b'])
    
    # Set panel index
    df_panel = df_sample.set_index(['pitcher_id', 'season'])
    
    # Prepare regressors
    year_dummy_cols = [f'year_{y}' for y in years_non_base]
    all_regressors = year_dummy_cols + interaction_cols
    
    X = df_panel[all_regressors]
    y = df_panel['attempt_rate']
    w = df_panel['opportunities_2b']
    
    # Fit OLS
    mod_ols = PanelOLS(
        dependent=y,
        exog=X,
        entity_effects=True,
        time_effects=False
    )
    res_ols = mod_ols.fit(cov_type='clustered', cluster_entity=True)
    
    # Fit WLS
    mod_wls = PanelOLS(
        dependent=y,
        exog=X,
        weights=w,
        entity_effects=True,
        time_effects=False
    )
    res_wls = mod_wls.fit(cov_type='clustered', cluster_entity=True)
    
    print(f"  N: {res_ols.nobs:.0f}, Pitchers: {df_panel.index.get_level_values(0).nunique()}")
    print(f"  OLS R²: {res_ols.rsquared_within:.4f}")
    print(f"  WLS R²: {res_wls.rsquared_within:.4f}")
    
    # Extract coefficients
    params_ols = res_ols.params
    params_wls = res_wls.params
    se_ols = res_ols.std_errors
    se_wls = res_wls.std_errors
    pvals_ols = res_ols.pvalues
    pvals_wls = res_wls.pvalues
    
    coef_data = []
    
    for year in years:
        # T2
        if year == BASE_YEAR:
            coef_data.append({
                'year': year,
                'tercile': 'T2',
                'coef_ols': 0.0,
                'se_ols': 0.0,
                'pval_ols': np.nan,
                'coef_wls': 0.0,
                'se_wls': 0.0,
                'pval_wls': np.nan
            })
        else:
            year_col = f'year_{year}'
            coef_data.append({
                'year': year,
                'tercile': 'T2',
                'coef_ols': params_ols.get(year_col, 0.0),
                'se_ols': se_ols.get(year_col, 0.0),
                'pval_ols': pvals_ols.get(year_col, np.nan),
                'coef_wls': params_wls.get(year_col, 0.0),
                'se_wls': se_wls.get(year_col, 0.0),
                'pval_wls': pvals_wls.get(year_col, np.nan)
            })
        
        # T1, T3
        for tercile in ['T1', 'T3']:
            if year == BASE_YEAR:
                coef_data.append({
                    'year': year,
                    'tercile': tercile,
                    'coef_ols': 0.0,
                    'se_ols': 0.0,
                    'pval_ols': np.nan,
                    'coef_wls': 0.0,
                    'se_wls': 0.0,
                    'pval_wls': np.nan
                })
            else:
                year_col = f'year_{year}'
                int_col = f'y{year}_x_{tercile}'
                
                year_ols = params_ols.get(year_col, 0.0)
                year_wls = params_wls.get(year_col, 0.0)
                
                int_ols = params_ols.get(int_col, 0.0)
                int_wls = params_wls.get(int_col, 0.0)
                int_se_ols = se_ols.get(int_col, 0.0)
                int_se_wls = se_wls.get(int_col, 0.0)
                int_pval_ols = pvals_ols.get(int_col, np.nan)
                int_pval_wls = pvals_wls.get(int_col, np.nan)
                
                coef_data.append({
                    'year': year,
                    'tercile': tercile,
                    'coef_ols': year_ols + int_ols,
                    'se_ols': int_se_ols,
                    'pval_ols': int_pval_ols,
                    'coef_wls': year_wls + int_wls,
                    'se_wls': int_se_wls,
                    'pval_wls': int_pval_wls
                })
    
    df_coef = pd.DataFrame(coef_data)
    df_coef['sample'] = sample_name
    
    return df_coef, res_ols, res_wls

# ============================================================================
# RUN MODELS ON ALL SAMPLES
# ============================================================================

print("\n" + "=" * 70)
print("RUNNING MODELS")
print("=" * 70)

results = {}

# Main sample
coef_main, res_main_ols, res_main_wls = run_event_study(df_main.copy(), "Main (≥50)")
results['main'] = {
    'coef': coef_main,
    'res_ols': res_main_ols,
    'res_wls': res_main_wls
}

# Strict sample
coef_strict, res_strict_ols, res_strict_wls = run_event_study(df_strict.copy(), "Strict (≥100)")
results['strict'] = {
    'coef': coef_strict,
    'res_ols': res_strict_ols,
    'res_wls': res_strict_wls
}

# Balanced panel
coef_balanced, res_balanced_ols, res_balanced_wls = run_event_study(df_balanced.copy(), "Balanced Panel")
results['balanced'] = {
    'coef': coef_balanced,
    'res_ols': res_balanced_ols,
    'res_wls': res_balanced_wls
}

# Combine coefficients
df_all_coef = pd.concat([coef_main, coef_strict, coef_balanced], ignore_index=True)

# ============================================================================
# COMPARISON: KEY TREATMENT EFFECTS
# ============================================================================

print("\n" + "=" * 70)
print("TREATMENT EFFECT COMPARISON (T2, 2023-2024)")
print("=" * 70)

comparison_data = []

for year in [2023, 2024]:
    print(f"\n{year}:")
    print(f"{'Sample':<20} {'OLS Coef':<12} {'SE':<10} {'p-value':<10} {'WLS Coef':<12} {'SE':<10} {'p-value':<10}")
    print("-" * 90)
    
    for sample_name, sample_key in [("Main (≥50)", "main"), 
                                     ("Strict (≥100)", "strict"), 
                                     ("Balanced Panel", "balanced")]:
        row = results[sample_key]['coef'][(results[sample_key]['coef']['year'] == year) & 
                                           (results[sample_key]['coef']['tercile'] == 'T2')].iloc[0]
        
        sig_ols = "***" if row['pval_ols'] < 0.01 else ("**" if row['pval_ols'] < 0.05 else ("*" if row['pval_ols'] < 0.10 else ""))
        sig_wls = "***" if row['pval_wls'] < 0.01 else ("**" if row['pval_wls'] < 0.05 else ("*" if row['pval_wls'] < 0.10 else ""))
        
        print(f"{sample_name:<20} {row['coef_ols']:+.5f}{sig_ols:<3} {row['se_ols']:.5f}  {row['pval_ols']:.4f}    "
              f"{row['coef_wls']:+.5f}{sig_wls:<3} {row['se_wls']:.5f}  {row['pval_wls']:.4f}")
        
        comparison_data.append({
            'year': year,
            'sample': sample_name,
            'coef_ols': row['coef_ols'],
            'se_ols': row['se_ols'],
            'pval_ols': row['pval_ols'],
            'coef_wls': row['coef_wls'],
            'se_wls': row['se_wls'],
            'pval_wls': row['pval_wls']
        })

df_comparison = pd.DataFrame(comparison_data)

# Check stability
print("\n" + "-" * 70)
print("STABILITY CHECK")
print("-" * 70)

# Compare main vs strict
main_2023 = df_comparison[(df_comparison['year'] == 2023) & (df_comparison['sample'] == 'Main (≥50)')]['coef_ols'].values[0]
strict_2023 = df_comparison[(df_comparison['year'] == 2023) & (df_comparison['sample'] == 'Strict (≥100)')]['coef_ols'].values[0]
balanced_2023 = df_comparison[(df_comparison['year'] == 2023) & (df_comparison['sample'] == 'Balanced Panel')]['coef_ols'].values[0]

diff_strict = abs(main_2023 - strict_2023)
diff_balanced = abs(main_2023 - balanced_2023)

print(f"\n2023 OLS coefficients:")
print(f"  Main:     {main_2023:+.5f}")
print(f"  Strict:   {strict_2023:+.5f} (diff: {diff_strict:.5f})")
print(f"  Balanced: {balanced_2023:+.5f} (diff: {diff_balanced:.5f})")

if diff_strict < 0.001 and diff_balanced < 0.001:
    print(f"\n✓ EXCELLENT: Effects very stable across samples")
    stability = "excellent"
elif diff_strict < 0.003 and diff_balanced < 0.003:
    print(f"\n✓ GOOD: Effects reasonably stable")
    stability = "good"
else:
    print(f"\n⚠ MODERATE: Some variation across samples")
    stability = "moderate"

# ============================================================================
# SAVE OUTPUTS
# ============================================================================

print("\n" + "-" * 70)
print("SAVING OUTPUTS")
print("-" * 70)

# 1. All coefficients
output_coef = OUTPUT_DIR / "c3e_all_coefficients.csv"
df_all_coef.to_csv(output_coef, index=False)
print(f"\nSaved: {output_coef}")

# 2. Comparison table
output_comp = OUTPUT_DIR / "c3e_treatment_comparison.csv"
df_comparison.to_csv(output_comp, index=False)
print(f"Saved: {output_comp}")

# 3. Summary
output_summary = OUTPUT_DIR / "c3e_summary.txt"
with open(output_summary, 'w', encoding='utf-8') as f:
    f.write("C3E SAMPLE ROBUSTNESS SUMMARY\n")
    f.write("=" * 70 + "\n\n")
    
    f.write("SAMPLE DEFINITIONS\n")
    f.write("-" * 70 + "\n")
    f.write(f"1. Main (≥50):      {len(df_main):,} obs, {df_main['pitcher_id'].nunique():,} pitchers\n")
    f.write(f"2. Strict (≥100):   {len(df_strict):,} obs, {df_strict['pitcher_id'].nunique():,} pitchers\n")
    f.write(f"3. Balanced Panel:  {len(df_balanced):,} obs, {df_balanced['pitcher_id'].nunique():,} pitchers\n\n")
    
    f.write("2023 TREATMENT EFFECTS (T2)\n")
    f.write("-" * 70 + "\n")
    f.write(f"Main:     {main_2023:+.5f}\n")
    f.write(f"Strict:   {strict_2023:+.5f} (diff: {diff_strict:.5f})\n")
    f.write(f"Balanced: {balanced_2023:+.5f} (diff: {diff_balanced:.5f})\n\n")
    
    f.write("STABILITY ASSESSMENT\n")
    f.write("-" * 70 + "\n")
    if stability == "excellent":
        f.write("✓ EXCELLENT: Effects very stable across samples\n")
        f.write("Results not driven by sample selection or attrition.\n")
    elif stability == "good":
        f.write("✓ GOOD: Effects reasonably stable\n")
        f.write("Minor variations within expected range.\n")
    else:
        f.write("⚠ MODERATE: Some variation across samples\n")
        f.write("Review individual coefficients.\n")

print(f"Saved: {output_summary}")

# ============================================================================
# PLOTS
# ============================================================================

print("\n" + "-" * 70)
print("CREATING COMPARISON PLOTS")
print("-" * 70)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Row 1: OLS by sample
for i, (sample_name, sample_key) in enumerate([("Main (≥50)", "main"), 
                                                 ("Strict (≥100)", "strict"), 
                                                 ("Balanced Panel", "balanced")]):
    ax = axes[0, i]
    
    for tercile in ['T1', 'T2', 'T3']:
        df_plot = results[sample_key]['coef'][results[sample_key]['coef']['tercile'] == tercile].sort_values('year')
        ax.plot(df_plot['year'], df_plot['coef_ols'], 
               marker='o', linewidth=2, label=tercile)
    
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.axvline(BASE_YEAR, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
    ax.axvline(2023, color='green', linestyle=':', linewidth=1.5, alpha=0.5)
    ax.set_xlabel('Year', fontsize=10)
    ax.set_ylabel('OLS Coefficient', fontsize=10)
    ax.set_title(f'{sample_name}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

# Row 2: T2 comparison across samples
ax = axes[1, 0]
for sample_name, sample_key in [("Main (≥50)", "main"), 
                                 ("Strict (≥100)", "strict"), 
                                 ("Balanced", "balanced")]:
    df_plot = results[sample_key]['coef'][results[sample_key]['coef']['tercile'] == 'T2'].sort_values('year')
    ax.plot(df_plot['year'], df_plot['coef_ols'], 
           marker='o', linewidth=2, label=sample_name)

ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax.axvline(BASE_YEAR, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
ax.axvline(2023, color='green', linestyle=':', linewidth=1.5, alpha=0.5)
ax.set_xlabel('Year', fontsize=10)
ax.set_ylabel('OLS Coefficient', fontsize=10)
ax.set_title('T2: Sample Comparison (OLS)', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Treatment effect bars (2023)
ax = axes[1, 1]
samples = ['Main\n(≥50)', 'Strict\n(≥100)', 'Balanced']
coefs_2023 = [main_2023, strict_2023, balanced_2023]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

bars = ax.bar(samples, coefs_2023, color=colors, alpha=0.7, edgecolor='black')
ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax.set_ylabel('2023 Effect (pp)', fontsize=10)
ax.set_title('2023 Treatment Effect Comparison', fontsize=11, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

for i, (sample, coef) in enumerate(zip(samples, coefs_2023)):
    ax.text(i, coef + 0.0005 if coef > 0 else coef - 0.0005, 
           f'{coef:+.4f}', ha='center', fontsize=9, fontweight='bold')

# Sample sizes
ax = axes[1, 2]
n_obs = [len(df_main), len(df_strict), len(df_balanced)]
n_pitchers = [df_main['pitcher_id'].nunique(), 
              df_strict['pitcher_id'].nunique(), 
              df_balanced['pitcher_id'].nunique()]

x = np.arange(len(samples))
width = 0.35

ax.bar(x - width/2, n_obs, width, label='Observations', alpha=0.7, color='steelblue')
ax.bar(x + width/2, n_pitchers, width, label='Pitchers', alpha=0.7, color='coral')

ax.set_xticks(x)
ax.set_xticklabels(samples)
ax.set_ylabel('Count', fontsize=10)
ax.set_title('Sample Sizes', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
output_plot = OUTPUT_DIR / "c3e_sample_comparison.png"
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
print(f"\nSaved: {output_plot}")
plt.close()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("C3E SAMPLE ROBUSTNESS COMPLETE")
print("=" * 70)

print(f"\nSamples tested:")
print(f"  1. Main (≥50):      {len(df_main):,} obs")
print(f"  2. Strict (≥100):   {len(df_strict):,} obs")
print(f"  3. Balanced Panel:  {len(df_balanced):,} obs")

print(f"\nStability: {stability.upper()}")
print(f"  2023 effect range: [{min(coefs_2023):.5f}, {max(coefs_2023):.5f}]")
print(f"  Max deviation: {max(diff_strict, diff_balanced):.5f}")

if stability == "excellent":
    print(f"\n✓ Results highly robust to sample definition")
elif stability == "good":
    print(f"\n✓ Results reasonably robust")
else:
    print(f"\n⚠ Review sample sensitivity")

print(f"\nOutputs:")
print(f"  1. c3e_all_coefficients.csv")
print(f"  2. c3e_treatment_comparison.csv")
print(f"  3. c3e_summary.txt")
print(f"  4. c3e_sample_comparison.png")

print("\n" + "=" * 70)
print("All C3 robustness checks complete!")
print("=" * 70)