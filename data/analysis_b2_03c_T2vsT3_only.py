"""
b2_03c_T2vsT3_only.py (CORRECTED)
==================================
Robustness: T2 vs. T3 only.

CRITICAL FIX: Pre-trends test now checks PARALLELISM (T3-T2=0 each year),
not whether each group is at zero.
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
OUTPUT_DIR = Path("./analysis/b2_eventstudy")

MIN_PITCHES = 50
REFERENCE_YEAR = 2022

# ============================================================================
# LOAD DATA
# ============================================================================

print("=" * 70)
print("B2-03c: T2 vs. T3 ONLY (CORRECTED)")
print("=" * 70)

df_panel = pd.read_csv(INPUT_PANEL)
df_baseline = pd.read_csv(INPUT_BASELINE)

df = df_panel.merge(df_baseline[["pitcher_id", "baseline_group"]], 
                    on="pitcher_id", how="inner")

# RESTRICT TO T2 AND T3
df = df[df["baseline_group"].isin(["T2", "T3"])].copy()
df = df[df["pitches_with_runners_on_base"] >= MIN_PITCHES].copy()

print(f"\n✓ Restricted sample: {len(df):,} pitcher-season observations")
print(f"  Included: T2 (mid), T3 (slow)")
print(f"  Excluded: T1 (fast)")

# ============================================================================
# VARIABLES
# ============================================================================

print("\n" + "-" * 70)
print("VARIABLE CONSTRUCTION")
print("-" * 70)

for g in ["T2", "T3"]:
    df[f"tercil_{g}"] = (df["baseline_group"] == g).astype(int)

for year in sorted(df["season"].unique()):
    if year == REFERENCE_YEAR:
        continue
    for g in ["T2", "T3"]:
        df[f"year{year}_x_{g}"] = ((df["season"] == year) & (df["baseline_group"] == g)).astype(int)

df["tempo"] = df["tempo_with_runners_on_base"]
df["weights"] = df["pitches_with_runners_on_base"]

print(f"✓ Variables constructed")

# ============================================================================
# MODEL
# ============================================================================

print("\n" + "=" * 70)
print("EVENT-STUDY MODEL (T2 vs. T3)")
print("=" * 70)

interaction_terms = [f"year{year}_x_{g}" 
                     for year in sorted(df["season"].unique()) if year != REFERENCE_YEAR
                     for g in ["T2", "T3"]]

formula = f"tempo ~ {' + '.join(interaction_terms)} + C(pitcher_id) + C(season)"

model_t2t3 = smf.wls(formula, data=df, weights=df["weights"])
results_t2t3 = model_t2t3.fit(cov_type="cluster", cov_kwds={"groups": df["pitcher_id"]})

print(f"\n✓ Model converged")
print(f"  N observations: {results_t2t3.nobs:,.0f}")
print(f"  R²: {results_t2t3.rsquared:.3f}")

# ============================================================================
# PRE-TRENDS TEST (CORRECTED: TEST PARALLELISM)
# ============================================================================

print("\n" + "-" * 70)
print("PRE-TRENDS TEST (PARALLELISM: T3-T2=0 each year)")
print("-" * 70)

pre_years = [y for y in sorted(df["season"].unique()) if y < REFERENCE_YEAR]

# Build hypotheses: For each pre-year, test T3 - T2 = 0
hypotheses = []
for year in pre_years:
    var_t3 = f"year{year}_x_T3"
    var_t2 = f"year{year}_x_T2"
    if var_t3 in results_t2t3.params.index and var_t2 in results_t2t3.params.index:
        hypotheses.append(f"{var_t3} - {var_t2} = 0")

if len(hypotheses) > 0:
    hypothesis_string = ", ".join(hypotheses)
    f_test = results_t2t3.f_test(hypothesis_string)
    
    print(f"\nJoint F-test on parallelism (T3-T2=0 for each pre-year):")
    print(f"  F({f_test.df_num}, {f_test.df_denom}) = {f_test.fvalue:.2f}")
    print(f"  p-value: {f_test.pvalue:.3f}")
    
    if f_test.pvalue < 0.05:
        print("  ⚠ T2 and T3 do NOT have parallel pre-trends (p<0.05)")
    else:
        print("  ✓ No evidence against parallel pre-trends")
else:
    print("\n⚠ Not enough pre-years to test parallelism")
    f_test = None

# ============================================================================
# EXTRACT EFFECTS
# ============================================================================

print("\n" + "-" * 70)
print("TREATMENT EFFECTS (2023 & 2024)")
print("-" * 70)

print("\n2023 Effects:")
for g in ["T2", "T3"]:
    var = f"year2023_x_{g}"
    if var in results_t2t3.params.index:
        coef = results_t2t3.params[var]
        se = results_t2t3.bse[var]
        p = results_t2t3.pvalues[var]
        stars = "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.1 else ""
        print(f"  {g}: β={coef:+.3f} (se={se:.3f}){stars}")

print("\n2024 Effects:")
for g in ["T2", "T3"]:
    var = f"year2024_x_{g}"
    if var in results_t2t3.params.index:
        coef = results_t2t3.params[var]
        se = results_t2t3.bse[var]
        p = results_t2t3.pvalues[var]
        stars = "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.1 else ""
        print(f"  {g}: β={coef:+.3f} (se={se:.3f}){stars}")

# ============================================================================
# DIFFERENCE TEST
# ============================================================================

print("\n" + "-" * 70)
print("DIFFERENCE TEST (T3 vs. T2)")
print("-" * 70)

# 2023
var1 = "year2023_x_T3"
var2 = "year2023_x_T2"
if var1 in results_t2t3.params.index and var2 in results_t2t3.params.index:
    contrast = [1 if v == var1 else -1 if v == var2 else 0 for v in results_t2t3.params.index]
    t_test = results_t2t3.t_test(contrast)
    print(f"2023: Δβ(T3-T2) = {results_t2t3.params[var1] - results_t2t3.params[var2]:.3f}, "
          f"t={t_test.tvalue[0][0]:.2f}, p={t_test.pvalue:.3f}")

# 2024
var1 = "year2024_x_T3"
var2 = "year2024_x_T2"
if var1 in results_t2t3.params.index and var2 in results_t2t3.params.index:
    contrast = [1 if v == var1 else -1 if v == var2 else 0 for v in results_t2t3.params.index]
    t_test = results_t2t3.t_test(contrast)
    print(f"2024: Δβ(T3-T2) = {results_t2t3.params[var1] - results_t2t3.params[var2]:.3f}, "
          f"t={t_test.tvalue[0][0]:.2f}, p={t_test.pvalue:.3f}")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "-" * 70)
print("SAVING RESULTS")
print("-" * 70)

results_table = []
for year in sorted(df["season"].unique()):
    if year == REFERENCE_YEAR:
        for g in ["T2", "T3"]:
            results_table.append({
                "year": year,
                "rel_year": 0,
                "tercile": g,
                "beta_t2t3": 0,
                "se_t2t3": 0,
                "p_t2t3": np.nan
            })
        continue
    
    for g in ["T2", "T3"]:
        var = f"year{year}_x_{g}"
        if var in results_t2t3.params.index:
            results_table.append({
                "year": year,
                "rel_year": year - REFERENCE_YEAR,
                "tercile": g,
                "beta_t2t3": results_t2t3.params[var],
                "se_t2t3": results_t2t3.bse[var],
                "p_t2t3": results_t2t3.pvalues[var]
            })

df_results = pd.DataFrame(results_table)
output_file = OUTPUT_DIR / "b2_eventstudy_T2T3_corrected_results.csv"
df_results.to_csv(output_file, index=False)
print(f"✓ Saved: {output_file}")

# Save pre-trends test result
if f_test is not None:
    pretrends_result = pd.DataFrame([{
        "test": "parallelism_T3_T2",
        "f_stat": f_test.fvalue,
        "df_num": f_test.df_num,
        "df_denom": f_test.df_denom,
        "p_value": f_test.pvalue,
        "interpretation": "Tests whether T3-T2 differs across pre-years"
    }])
    output_file = OUTPUT_DIR / "b2_pretrends_T2T3_corrected.csv"
    pretrends_result.to_csv(output_file, index=False)
    print(f"✓ Saved: {output_file}")

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\n" + "-" * 70)
print("CREATING PLOT")
print("-" * 70)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for i, g in enumerate(["T2", "T3"]):
    ax = axes[i]
    
    data_g = df_results[df_results["tercile"] == g].sort_values("rel_year")
    
    ax.errorbar(data_g["rel_year"], data_g["beta_t2t3"], 
                yerr=1.96*data_g["se_t2t3"],
                marker="o", capsize=5, linewidth=2, markersize=8, 
                color="C1" if g=="T2" else "C2")
    
    ax.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax.axvline(0, color="red", linestyle="--", alpha=0.3, label="2023 Timer")
    
    ax.set_xlabel("Years Relative to 2022", fontsize=11)
    if i == 0:
        ax.set_ylabel("Tempo Change (seconds)", fontsize=11)
    ax.set_title(f"{g} ({'mid' if g=='T2' else 'slow'})", 
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9)

if f_test is not None:
    fig.text(0.5, 0.02, f"Pre-trends test (parallelism): F={f_test.fvalue:.2f}, p={f_test.pvalue:.3f}", 
             ha="center", fontsize=10, 
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

plt.suptitle("Event-Study: T2 vs. T3 Only (Parallelism Test Corrected)", 
             fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0.05, 1, 0.96])

output_file = OUTPUT_DIR / "b2_eventstudy_T2T3_corrected_plot.png"
plt.savefig(output_file, dpi=300, bbox_inches="tight")
print(f"✓ Saved: {output_file}")
plt.close()

print(f"\n✓ Analysis complete with corrected pre-trends test.")