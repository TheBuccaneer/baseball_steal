"""
b2_07_robustness_suite.py
=========================
Three key robustness checks for main event-study results.

Tests:
1. WLS vs OLS: Weighted (pitch counts) vs unweighted (pitcher-average)
2. Balanced Panels: BP_22_23 and BP_22_23_24 (within-pitcher only)
3. Quartiles: Q1 vs Q4 (extreme groups) instead of terciles

All tests use T2 vs T3 comparison (T1 excluded per strategy).

Outputs:
- b2_robustness_wls_vs_ols.csv
- b2_robustness_balanced_panels.csv
- b2_robustness_quartiles.csv
- b2_robustness_comparison_table.csv
- b2_robustness_plot.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_PANEL = "./analysis/analysis_pitcher_panel_relative.csv"
INPUT_BASELINE = "./analysis/b2_baseline/b2_baseline_groups.csv"
OUTPUT_DIR = Path("./analysis/b2_robustness")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_PITCHES = 50
REFERENCE_YEAR = 2022

# ============================================================================
# LOAD DATA
# ============================================================================

print("=" * 70)
print("B2-07: ROBUSTNESS SUITE (3 CHECKS)")
print("=" * 70)

df_panel = pd.read_csv(INPUT_PANEL)
df_baseline = pd.read_csv(INPUT_BASELINE)

df = df_panel.merge(df_baseline[["pitcher_id", "baseline_group"]], 
                    on="pitcher_id", how="inner")
df = df[df["pitches_with_runners_on_base"] >= MIN_PITCHES].copy()

print(f"\n✓ Working sample: {len(df):,} pitcher-season observations")
print(f"  Pitchers: {df['pitcher_id'].nunique():,}")
print(f"  Seasons: {sorted(df['season'].unique())}")

# Prepare variables
df["tempo"] = df["tempo_with_runners_on_base"]
df["weights"] = df["pitches_with_runners_on_base"]

# Focus on T2 vs T3
df = df[df["baseline_group"].isin(["T2", "T3"])].copy()
print(f"\n✓ Restricted to T2 and T3: {len(df):,} observations")

# Create interactions
for year in sorted(df["season"].unique()):
    if year == REFERENCE_YEAR:
        continue
    for g in ["T2", "T3"]:
        df[f"year{year}_x_{g}"] = ((df["season"] == year) & (df["baseline_group"] == g)).astype(int)

interaction_terms = [f"year{year}_x_{g}" 
                     for year in sorted(df["season"].unique()) if year != REFERENCE_YEAR
                     for g in ["T2", "T3"]]

formula = f"tempo ~ {' + '.join(interaction_terms)} + C(pitcher_id) + C(season)"

# ============================================================================
# CHECK 1: WLS vs OLS
# ============================================================================

print("\n" + "=" * 70)
print("CHECK 1: WLS vs OLS")
print("=" * 70)

print("\nFitting WLS (weighted by pitch counts)...")
model_wls = smf.wls(formula, data=df, weights=df["weights"])
results_wls = model_wls.fit(cov_type="cluster", cov_kwds={"groups": df["pitcher_id"]})

print(f"✓ WLS: R² = {results_wls.rsquared:.3f}")

print("\nFitting OLS (unweighted)...")
model_ols = smf.ols(formula, data=df)
results_ols = model_ols.fit(cov_type="cluster", cov_kwds={"groups": df["pitcher_id"]})

print(f"✓ OLS: R² = {results_ols.rsquared:.3f}")

# Extract 2023 coefficients
comparison_data = []

for g in ["T2", "T3"]:
    var = f"year2023_x_{g}"
    
    comparison_data.append({
        "tercile": g,
        "spec": "WLS",
        "beta_2023": results_wls.params[var],
        "se_2023": results_wls.bse[var],
        "p_2023": results_wls.pvalues[var]
    })
    
    comparison_data.append({
        "tercile": g,
        "spec": "OLS",
        "beta_2023": results_ols.params[var],
        "se_2023": results_ols.bse[var],
        "p_2023": results_ols.pvalues[var]
    })

df_wls_ols = pd.DataFrame(comparison_data)

print("\n2023 Effects Comparison:")
print(df_wls_ols.to_string(index=False))

# Test difference
diff_t2 = results_wls.params["year2023_x_T2"] - results_ols.params["year2023_x_T2"]
diff_t3 = results_wls.params["year2023_x_T3"] - results_ols.params["year2023_x_T3"]

print(f"\nDifference (WLS - OLS):")
print(f"  T2: {diff_t2:+.3f}s")
print(f"  T3: {diff_t3:+.3f}s")

if abs(diff_t2) < 0.3 and abs(diff_t3) < 0.3:
    print("  ✓ Results robust to weighting choice")
else:
    print("  ⚠️  Notable sensitivity to weights")

output_file = OUTPUT_DIR / "b2_robustness_wls_vs_ols.csv"
df_wls_ols.to_csv(output_file, index=False)
print(f"\n✓ Saved: {output_file}")

# ============================================================================
# CHECK 2: BALANCED PANELS
# ============================================================================

print("\n" + "=" * 70)
print("CHECK 2: BALANCED PANELS")
print("=" * 70)

# Define balanced panels
pitchers_2022 = set(df[df["season"] == 2022]["pitcher_id"])
pitchers_2023 = set(df[df["season"] == 2023]["pitcher_id"])
pitchers_2024 = set(df[df["season"] == 2024]["pitcher_id"])

bp_22_23 = pitchers_2022 & pitchers_2023
bp_22_23_24 = pitchers_2022 & pitchers_2023 & pitchers_2024

print(f"\nBalanced Panel Sizes:")
print(f"  BP_22_23: {len(bp_22_23):,} pitchers")
print(f"  BP_22_23_24: {len(bp_22_23_24):,} pitchers")

bp_comparison = []

# Full sample (for reference)
for g in ["T2", "T3"]:
    var = f"year2023_x_{g}"
    bp_comparison.append({
        "panel": "Full",
        "n_pitchers": df["pitcher_id"].nunique(),
        "tercile": g,
        "beta_2023": results_wls.params[var],
        "se_2023": results_wls.bse[var]
    })

# BP_22_23
print("\nFitting BP_22_23...")
df_bp23 = df[df["pitcher_id"].isin(bp_22_23)]
model_bp23 = smf.wls(formula, data=df_bp23, weights=df_bp23["weights"])
results_bp23 = model_bp23.fit(cov_type="cluster", cov_kwds={"groups": df_bp23["pitcher_id"]})
print(f"✓ BP_22_23: R² = {results_bp23.rsquared:.3f}")

for g in ["T2", "T3"]:
    var = f"year2023_x_{g}"
    bp_comparison.append({
        "panel": "BP_22_23",
        "n_pitchers": len(bp_22_23),
        "tercile": g,
        "beta_2023": results_bp23.params[var],
        "se_2023": results_bp23.bse[var]
    })

# BP_22_23_24
print("\nFitting BP_22_23_24...")
df_bp24 = df[df["pitcher_id"].isin(bp_22_23_24)]
model_bp24 = smf.wls(formula, data=df_bp24, weights=df_bp24["weights"])
results_bp24 = model_bp24.fit(cov_type="cluster", cov_kwds={"groups": df_bp24["pitcher_id"]})
print(f"✓ BP_22_23_24: R² = {results_bp24.rsquared:.3f}")

for g in ["T2", "T3"]:
    var = f"year2023_x_{g}"
    bp_comparison.append({
        "panel": "BP_22_23_24",
        "n_pitchers": len(bp_22_23_24),
        "tercile": g,
        "beta_2023": results_bp24.params[var],
        "se_2023": results_bp24.bse[var]
    })

df_bp = pd.DataFrame(bp_comparison)

print("\nBalanced Panel Comparison:")
print(df_bp.to_string(index=False))

# Check consistency
for g in ["T2", "T3"]:
    full = df_bp[(df_bp["panel"] == "Full") & (df_bp["tercile"] == g)]["beta_2023"].values[0]
    bp23 = df_bp[(df_bp["panel"] == "BP_22_23") & (df_bp["tercile"] == g)]["beta_2023"].values[0]
    bp24 = df_bp[(df_bp["panel"] == "BP_22_23_24") & (df_bp["tercile"] == g)]["beta_2023"].values[0]
    
    print(f"\n{g}:")
    print(f"  Full vs BP_22_23: {abs(full - bp23):.3f}s difference")
    print(f"  Full vs BP_22_23_24: {abs(full - bp24):.3f}s difference")
    
    if abs(full - bp23) < 0.5 and abs(full - bp24) < 0.5:
        print(f"  ✓ {g} effects robust to attrition")

output_file = OUTPUT_DIR / "b2_robustness_balanced_panels.csv"
df_bp.to_csv(output_file, index=False)
print(f"\n✓ Saved: {output_file}")

# ============================================================================
# CHECK 3: QUARTILES
# ============================================================================

print("\n" + "=" * 70)
print("CHECK 3: QUARTILES (Q1 vs Q4)")
print("=" * 70)

# Get 2022 tempo for quartile calculation
df_2022_tempo = df_panel[(df_panel['season'] == 2022) & 
                         (df_panel['pitches_with_runners_on_base'] >= MIN_PITCHES)][
    ['pitcher_id', 'tempo_with_runners_on_base']
].copy()

# Calculate quartiles
quartiles = pd.qcut(df_2022_tempo['tempo_with_runners_on_base'], 
                    q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates='drop')
df_2022_tempo['quartile'] = quartiles

# Merge with full panel
df_quart = df_panel.merge(df_2022_tempo[['pitcher_id', 'quartile']], 
                          on='pitcher_id', how='inner')

# Keep only Q1 (fastest) and Q4 (slowest)
df_quart = df_quart[df_quart['quartile'].isin(["Q1", "Q4"])].copy()
df_quart = df_quart[df_quart['pitches_with_runners_on_base'] >= MIN_PITCHES].copy()

print(f"\n✓ Quartile sample: {len(df_quart):,} observations")
print(f"  Q1 (fastest): {(df_quart['quartile']=='Q1').sum():,}")
print(f"  Q4 (slowest): {(df_quart['quartile']=='Q4').sum():,}")

# Prepare variables
df_quart["tempo"] = df_quart["tempo_with_runners_on_base"]
df_quart["weights"] = df_quart["pitches_with_runners_on_base"]

# Create interactions
for year in sorted(df_quart["season"].unique()):
    if year == REFERENCE_YEAR:
        continue
    for q in ["Q1", "Q4"]:
        df_quart[f"year{year}_x_{q}"] = ((df_quart["season"] == year) & (df_quart["quartile"] == q)).astype(int)

interaction_terms_q = [f"year{year}_x_{q}" 
                       for year in sorted(df_quart["season"].unique()) if year != REFERENCE_YEAR
                       for q in ["Q1", "Q4"]]

formula_q = f"tempo ~ {' + '.join(interaction_terms_q)} + C(pitcher_id) + C(season)"

print("\nFitting quartile model...")
model_quart = smf.wls(formula_q, data=df_quart, weights=df_quart["weights"])
results_quart = model_quart.fit(cov_type="cluster", cov_kwds={"groups": df_quart["pitcher_id"]})

print(f"✓ Quartiles: R² = {results_quart.rsquared:.3f}")

# Extract effects
quart_comparison = []
for q in ["Q1", "Q4"]:
    var = f"year2023_x_{q}"
    quart_comparison.append({
        "group": q,
        "label": "fastest" if q == "Q1" else "slowest",
        "beta_2023": results_quart.params[var],
        "se_2023": results_quart.bse[var],
        "p_2023": results_quart.pvalues[var]
    })

df_quart_results = pd.DataFrame(quart_comparison)

print("\nQuartile Effects (2023):")
print(df_quart_results.to_string(index=False))

# Test Q4 vs Q1
contrast = [1 if v == "year2023_x_Q4" else -1 if v == "year2023_x_Q1" else 0 
            for v in results_quart.params.index]
t_test = results_quart.t_test(contrast)

diff = results_quart.params["year2023_x_Q4"] - results_quart.params["year2023_x_Q1"]
print(f"\nQ4 vs Q1 (2023):")
print(f"  Δβ = {diff:.3f}s")
print(f"  t = {t_test.tvalue[0][0]:.2f}")
print(f"  p = {t_test.pvalue:.3f}")

if t_test.pvalue < 0.05 and diff < 0:
    print("  ✓ Q4 (slowest) responds MORE than Q1 (fastest)")

output_file = OUTPUT_DIR / "b2_robustness_quartiles.csv"
df_quart_results.to_csv(output_file, index=False)
print(f"\n✓ Saved: {output_file}")

# ============================================================================
# COMPREHENSIVE COMPARISON TABLE
# ============================================================================

print("\n" + "=" * 70)
print("COMPREHENSIVE COMPARISON")
print("=" * 70)

# Build master table
master_table = []

# Main (WLS, T2 vs T3)
master_table.append({
    "specification": "Main (WLS)",
    "sample": "Full",
    "groups": "T2 vs T3",
    "n_pitchers": df["pitcher_id"].nunique(),
    "beta_slow_2023": results_wls.params["year2023_x_T3"],
    "beta_fast_2023": results_wls.params["year2023_x_T2"],
    "difference": results_wls.params["year2023_x_T3"] - results_wls.params["year2023_x_T2"]
})

# OLS
master_table.append({
    "specification": "OLS",
    "sample": "Full",
    "groups": "T2 vs T3",
    "n_pitchers": df["pitcher_id"].nunique(),
    "beta_slow_2023": results_ols.params["year2023_x_T3"],
    "beta_fast_2023": results_ols.params["year2023_x_T2"],
    "difference": results_ols.params["year2023_x_T3"] - results_ols.params["year2023_x_T2"]
})

# Balanced panels
master_table.append({
    "specification": "WLS",
    "sample": "BP_22_23",
    "groups": "T2 vs T3",
    "n_pitchers": len(bp_22_23),
    "beta_slow_2023": results_bp23.params["year2023_x_T3"],
    "beta_fast_2023": results_bp23.params["year2023_x_T2"],
    "difference": results_bp23.params["year2023_x_T3"] - results_bp23.params["year2023_x_T2"]
})

master_table.append({
    "specification": "WLS",
    "sample": "BP_22_23_24",
    "groups": "T2 vs T3",
    "n_pitchers": len(bp_22_23_24),
    "beta_slow_2023": results_bp24.params["year2023_x_T3"],
    "beta_fast_2023": results_bp24.params["year2023_x_T2"],
    "difference": results_bp24.params["year2023_x_T3"] - results_bp24.params["year2023_x_T2"]
})

# Quartiles
master_table.append({
    "specification": "WLS",
    "sample": "Full",
    "groups": "Q1 vs Q4",
    "n_pitchers": df_quart["pitcher_id"].nunique(),
    "beta_slow_2023": results_quart.params["year2023_x_Q4"],
    "beta_fast_2023": results_quart.params["year2023_x_Q1"],
    "difference": results_quart.params["year2023_x_Q4"] - results_quart.params["year2023_x_Q1"]
})

df_master = pd.DataFrame(master_table)

print("\nRobustness Summary Table:")
print(df_master.to_string(index=False))

output_file = OUTPUT_DIR / "b2_robustness_comparison_table.csv"
df_master.to_csv(output_file, index=False)
print(f"\n✓ Saved: {output_file}")

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\n" + "-" * 70)
print("CREATING ROBUSTNESS PLOT")
print("-" * 70)

fig, ax = plt.subplots(figsize=(12, 6))

specs = df_master['specification'] + " / " + df_master['sample'] + " / " + df_master['groups']
x = np.arange(len(specs))

# Plot differences with error bars
differences = df_master['difference'].values

# Calculate 95% CIs for differences
# Difference = beta_slow - beta_fast
# SE(difference) = sqrt(SE_slow^2 + SE_fast^2) assuming independence
se_diffs = []

for idx, row in df_master.iterrows():
    spec = row['specification']
    sample = row['sample']
    groups = row['groups']
    
    # Get SEs from appropriate source
    if spec == 'Main (WLS)':
        se_slow = results_wls.bse['year2023_x_T3']
        se_fast = results_wls.bse['year2023_x_T2']
    elif spec == 'OLS':
        se_slow = results_ols.bse['year2023_x_T3']
        se_fast = results_ols.bse['year2023_x_T2']
    elif 'BP_22_23_24' in sample:
        se_slow = results_bp24.bse['year2023_x_T3']
        se_fast = results_bp24.bse['year2023_x_T2']
    elif 'BP_22_23' in sample:
        se_slow = results_bp23.bse['year2023_x_T3']
        se_fast = results_bp23.bse['year2023_x_T2']
    elif 'Q1 vs Q4' in groups:
        se_slow = results_quart.bse['year2023_x_Q4']
        se_fast = results_quart.bse['year2023_x_Q1']
    else:
        se_slow = 0
        se_fast = 0
    
    # SE of difference (assuming independence)
    se_diff = np.sqrt(se_slow**2 + se_fast**2)
    se_diffs.append(se_diff)

se_diffs = np.array(se_diffs)

# 95% CI
ci_95 = 1.96 * se_diffs

# Color-code by spec type
colors = []
for s in specs:
    if 'Main' in s:
        colors.append('C0')
    elif 'OLS' in s:
        colors.append('C1')
    elif 'BP' in s:
        colors.append('C2')
    elif 'Q1 vs Q4' in s:
        colors.append('C3')
    else:
        colors.append('gray')

# Horizontal bars with error bars
bars = ax.barh(x, differences, color=colors, alpha=0.6, edgecolor='black', linewidth=1.5)

# Add error bars (95% CI)
ax.errorbar(differences, x, xerr=ci_95, fmt='none', ecolor='black', 
            capsize=5, capthick=2, linewidth=2, alpha=0.8)

# Add vertical line at main result
main_diff = df_master[df_master['specification'] == 'Main (WLS)']['difference'].values[0]
ax.axvline(main_diff, color='red', linestyle='--', linewidth=2, alpha=0.7, 
           label=f'Main: {main_diff:.2f}s', zorder=0)

ax.set_xlabel('Difference: Slow - Fast (seconds)', fontsize=11, fontweight='bold')
ax.set_ylabel('Specification', fontsize=11, fontweight='bold')
ax.set_title('Robustness of 2023 Treatment Effect\n(Slow Group - Fast Group Comparison with 95% CIs)', 
            fontsize=12, fontweight='bold')
ax.set_yticks(x)
ax.set_yticklabels([s.replace(' / ', '\n') for s in specs], fontsize=9)
ax.legend(loc='lower right')
ax.grid(axis='x', alpha=0.3)

# Add value labels with CIs
for i, (bar, val, se) in enumerate(zip(bars, differences, ci_95)):
    ax.text(val - 0.15, bar.get_y() + bar.get_height()/2, 
            f'{val:.2f}s\n±{se:.2f}', 
            va='center', ha='right', fontsize=8, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))

# Add note about Q1 vs Q4
ax.text(0.02, 0.98, 
        "Note: Q1 vs Q4 uses more extreme groups\n(larger effect expected)", 
        transform=ax.transAxes, fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))

plt.tight_layout()
output_file = OUTPUT_DIR / "b2_robustness_plot.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_file}")
plt.close()

# ============================================================================
# FINAL ASSESSMENT
# ============================================================================

print("\n" + "=" * 70)
print("ROBUSTNESS ASSESSMENT")
print("=" * 70)

main_diff = df_master[df_master['specification'] == 'Main (WLS)']['difference'].values[0]
all_diffs = df_master['difference'].values

# Range
min_diff = all_diffs.min()
max_diff = all_diffs.max()
range_diff = max_diff - min_diff

print(f"\nEffect Range Across Specifications:")
print(f"  Main: {main_diff:.2f}s")
print(f"  Min: {min_diff:.2f}s")
print(f"  Max: {max_diff:.2f}s")
print(f"  Range: {range_diff:.2f}s")

if range_diff < 0.5:
    print("\n✓✓✓ RESULTS HIGHLY ROBUST")
    print("  All specifications within 0.5s of each other")
elif range_diff < 1.0:
    print("\n✓✓ RESULTS REASONABLY ROBUST")
    print("  Specifications show some variation but consistent sign/magnitude")
else:
    print("\n⚠️  RESULTS SHOW SENSITIVITY")
    print("  Notable variation across specifications")

# Sign consistency
all_negative = all(d < 0 for d in all_diffs)
all_positive = all(d > 0 for d in all_diffs)

if all_negative:
    print("\n✓ SIGN CONSISTENCY: All specifications show slower group responds MORE")
elif all_positive:
    print("\n✗ SIGN CONSISTENCY: All specifications show slower group responds LESS (unexpected)")
else:
    print("\n⚠️  SIGN INCONSISTENCY: Mixed signs across specifications")

print(f"\nOutputs saved to: {OUTPUT_DIR}/")
print("\n" + "=" * 70)
print("B2 ROBUSTNESS CHECKS COMPLETE")
print("=" * 70)
print("\nNext Steps:")
print("  1. Review all outputs in b2_robustness/")
print("  2. Integrate key findings into paper")
print("  3. Begin C-schiene (Running Game analysis)")