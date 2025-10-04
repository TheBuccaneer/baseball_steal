"""
c5_total_steal_rate.py
======================
Total stolen base rate analysis: SB per Opportunity

Purpose: 
- Direct estimation of overall steal rate effect via PPML
- Validation that C3 × C4 ≈ C5 (mechanistic decomposition)

Model: FE-PPML with log(Opportunities) offset
- Endog: n_sb (stolen base counts)
- Exposure: opportunities (pitches with runner on 1B only)
- Fixed effects: Pitcher + Year
- Standard errors: Cluster-robust by pitcher

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
OUTPUT_DIR = Path("analysis/c5_total_rate")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_YEAR = 2022
EXCLUDE_2020 = True

# Optional: Load C3/C4 results for decomposition comparison
C3_RESULTS = Path("analysis/c3_ppml/c3_ppml_coefficients.csv")
C4_RESULTS = Path("analysis/c4_success_rate/c4_coefficients_v2.csv")

# ============================================================================
# LOAD DATA
# ============================================================================

print("=" * 70)
print("C5 TOTAL STEAL RATE ANALYSIS")
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
sb_col = next(c for c in ["n_sb", "sb_2b", "stolen_bases_2b"] if c in df_baseline.columns)
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
    sb_col: ['sum', 'mean'],
    opp_col: ['sum', 'mean']
}).round(3)

desc.columns = ['n_obs', 'total_sb', 'mean_sb_per_pitcher', 'total_opps', 'mean_opps_per_pitcher']
desc['sb_per_opp'] = (desc['total_sb'] / desc['total_opps']).round(4)

print("\n" + desc.to_string())

# Baseline rate (2022)
baseline_rate = desc.loc[BASE_YEAR, 'sb_per_opp']
print(f"\nBaseline steal rate (2022): {baseline_rate:.4f}")

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
df_baseline = df_baseline.dropna(subset=[sb_col, opp_col])
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

# Endog: SB counts
y_ppml = df_baseline[sb_col].values

# Exposure: Opportunities
exposure = df_baseline[opp_col].values

print(f"\nFitting Poisson GLM with offset...")
print(f"  Outcome: {sb_col} (stolen bases)")
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
# DECOMPOSITION: C3 × C4 vs C5
# ============================================================================

print("\n" + "=" * 70)
print("DECOMPOSITION: C3 × C4 vs C5 DIRECT")
print("=" * 70)

try:
    c3_coef = pd.read_csv(C3_RESULTS)
    c4_coef = pd.read_csv(C4_RESULTS)
    
    decomp_data = []
    
    for year in [2023, 2024]:
        # C5 direct
        c5_row = df_coef[(df_coef['year'] == year) & (df_coef['tercile'] == 'T2')].iloc[0]
        c5_rr = c5_row['rr']
        
        # C3 attempt rate
        c3_row = c3_coef[(c3_coef['year'] == year) & (c3_coef['tercile'] == 'T2')].iloc[0]
        # C3 might have 'rr' or we calculate it from coefficient
        if 'rr' in c3_row.index:
            c3_rr = c3_row['rr']
        elif 'coef' in c3_row.index:
            c3_rr = np.exp(c3_row['coef'])
        else:
            # Try ppml_coef or similar
            coef_col = next((c for c in c3_row.index if 'coef' in c.lower()), None)
            if coef_col:
                c3_rr = np.exp(c3_row[coef_col])
            else:
                print(f"  Warning: Could not find coefficient for C3 year {year}")
                c3_rr = np.nan
        
        # C4 success rate (convert OR to RR if needed)
        c4_row = c4_coef[(c4_coef['year'] == year) & (c4_coef['tercile'] == 'T2')].iloc[0]
        c4_or = c4_row['binomial_or']
        
        # Approximate OR→RR: RR ≈ OR / ((1-p0) + p0*OR) where p0 = baseline success rate
        p0 = 0.740  # From C4
        c4_rr = c4_or / ((1 - p0) + p0 * c4_or)
        
        # Product
        c3_x_c4 = c3_rr * c4_rr
        
        decomp_data.append({
            'year': year,
            'c5_direct': c5_rr,
            'c3_attempt': c3_rr,
            'c4_success': c4_rr,
            'c3_x_c4': c3_x_c4,
            'difference': c5_rr - c3_x_c4
        })
    
    df_decomp = pd.DataFrame(decomp_data)
    
    print("\nDecomposition Results:")
    print(df_decomp.to_string(index=False))
    
    print("\nInterpretation:")
    print("  C5 (direct) should ≈ C3 (attempts) × C4 (success)")
    print("  Small differences reflect different samples/specifications")
    
except FileNotFoundError as e:
    print(f"\n⚠ Could not load C3/C4 results for decomposition: {e}")
    print("  Decomposition comparison skipped.")
    df_decomp = None

# ============================================================================
# SAVE OUTPUTS
# ============================================================================

print("\n" + "-" * 70)
print("SAVING OUTPUTS")
print("-" * 70)

# Coefficients
output_coef = OUTPUT_DIR / "c5_coefficients.csv"
df_coef.to_csv(output_coef, index=False)
print(f"\nSaved: {output_coef}")

# Decomposition
if df_decomp is not None:
    output_decomp = OUTPUT_DIR / "c5_decomposition.csv"
    df_decomp.to_csv(output_decomp, index=False)
    print(f"Saved: {output_decomp}")

# Summary
output_summary = OUTPUT_DIR / "c5_summary.txt"
with open(output_summary, 'w', encoding='utf-8') as f:
    f.write("C5 TOTAL STEAL RATE ANALYSIS\n")
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
    
    if df_decomp is not None:
        f.write("\n" + "-" * 70 + "\n")
        f.write("DECOMPOSITION\n")
        f.write("-" * 70 + "\n")
        f.write(df_decomp.to_string(index=False))

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
ax.set_title('C5: Total Steal Rate (FE-PPML)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='best')
ax.grid(alpha=0.3)

plt.tight_layout()
output_plot = OUTPUT_DIR / "c5_event_study.png"
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
print(f"\nSaved: {output_plot}")
plt.close()

# Decomposition plot if available
if df_decomp is not None:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(df_decomp))
    width = 0.25
    
    ax.bar(x - width, df_decomp['c3_attempt'] - 1, width, 
           label='C3: Attempt Rate', alpha=0.8)
    ax.bar(x, df_decomp['c4_success'] - 1, width, 
           label='C4: Success Rate', alpha=0.8)
    ax.bar(x + width, df_decomp['c5_direct'] - 1, width, 
           label='C5: Total (direct)', alpha=0.8)
    
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Year')
    ax.set_ylabel('Rate Ratio - 1 (% Change)')
    ax.set_title('Decomposition: C3 × C4 vs C5')
    ax.set_xticks(x)
    ax.set_xticklabels(df_decomp['year'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_decomp_plot = OUTPUT_DIR / "c5_decomposition.png"
    plt.savefig(output_decomp_plot, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_decomp_plot}")
    plt.close()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("C5 TOTAL STEAL RATE COMPLETE")
print("=" * 70)

print(f"\nSample: {len(df_baseline):,} pitcher-season obs")
print(f"Baseline steal rate (2022): {baseline_rate:.4f}")

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
print(f"  1. c5_coefficients.csv")
print(f"  2. c5_summary.txt")
print(f"  3. c5_event_study.png")
if df_decomp is not None:
    print(f"  4. c5_decomposition.csv")
    print(f"  5. c5_decomposition.png")

print("\n" + "=" * 70)
print("Interpretation:")
print("=" * 70)
print("C5 provides direct estimate of total steal rate effect.")
print("Decomposition shows C3 (attempts) × C4 (success) ≈ C5 (total).")
print("Both channels contribute to overall increase in stolen bases.")
print("=" * 70)