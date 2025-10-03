"""
b2_03b_detrended.py (CORRECTED)
================================
Event-study with tercile-specific linear time trends.

CRITICAL FIX: Trends estimated ONLY from pre-period (2018-2022),
then extrapolated to post-period. Avoids post-treatment contamination.
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
PRE_PERIOD_START = 2018

# ============================================================================
# LOAD DATA
# ============================================================================

print("=" * 70)
print("B2-03b: DETRENDED EVENT-STUDY (CORRECTED)")
print("=" * 70)

df_panel = pd.read_csv(INPUT_PANEL)
df_baseline = pd.read_csv(INPUT_BASELINE)

df = df_panel.merge(df_baseline[["pitcher_id", "baseline_group"]], 
                    on="pitcher_id", how="inner")
df = df[df["pitches_with_runners_on_base"] >= MIN_PITCHES].copy()

print(f"\n✓ Working sample: {len(df):,} pitcher-season observations")

# ============================================================================
# STEP 1: ESTIMATE TRENDS FROM PRE-PERIOD ONLY
# ============================================================================

print("\n" + "=" * 70)
print("STEP 1: ESTIMATE PRE-TRENDS (2018-2022 ONLY)")
print("=" * 70)

# Filter to pre-period
df_pre = df[df["season"] <= REFERENCE_YEAR].copy()
df_pre["time_trend"] = df_pre["season"] - PRE_PERIOD_START

print(f"\nPre-period sample: {len(df_pre):,} observations ({PRE_PERIOD_START}-{REFERENCE_YEAR})")

# Create tercile dummies
for g in ["T1", "T2", "T3"]:
    df_pre[f"tercil_{g}"] = (df_pre["baseline_group"] == g).astype(int)
    df_pre[f"trend_x_{g}"] = df_pre["time_trend"] * df_pre[f"tercil_{g}"]

# Outcome and weights
df_pre["tempo"] = df_pre["tempo_with_runners_on_base"]
df_pre["weights"] = df_pre["pitches_with_runners_on_base"]

# Estimate trend model (no year FE - we want to capture secular trends)
formula_pre = "tempo ~ trend_x_T1 + trend_x_T2 + trend_x_T3 + C(pitcher_id)"

model_pre = smf.wls(formula_pre, data=df_pre, weights=df_pre["weights"])
results_pre = model_pre.fit(cov_type="cluster", cov_kwds={"groups": df_pre["pitcher_id"]})

print(f"\n✓ Pre-period model converged")
print(f"  R²: {results_pre.rsquared:.3f}")

# Extract trend coefficients
trend_coefs = {}
for g in ["T1", "T2", "T3"]:
    var = f"trend_x_{g}"
    if var in results_pre.params.index:
        trend_coefs[g] = {
            "coef": results_pre.params[var],
            "se": results_pre.bse[var],
            "p": results_pre.pvalues[var]
        }
        stars = "***" if trend_coefs[g]["p"]<0.01 else "**" if trend_coefs[g]["p"]<0.05 else "*" if trend_coefs[g]["p"]<0.1 else ""
        print(f"\n{g}: γ={trend_coefs[g]['coef']:+.4f} s/year (se={trend_coefs[g]['se']:.4f}){stars}")

# ============================================================================
# STEP 2: CREATE DETRENDED OUTCOME (ALL YEARS)
# ============================================================================

print("\n" + "=" * 70)
print("STEP 2: DETREND OUTCOME USING PRE-PERIOD TRENDS")
print("=" * 70)

# Create time_trend for all years
df["time_trend"] = df["season"] - PRE_PERIOD_START

# Create tercile dummies
for g in ["T1", "T2", "T3"]:
    df[f"tercil_{g}"] = (df["baseline_group"] == g).astype(int)

# Compute predicted trend for each observation (extrapolated from pre-period)
df["trend_pred"] = 0.0
for g in ["T1", "T2", "T3"]:
    if g in trend_coefs:
        df.loc[df["baseline_group"] == g, "trend_pred"] = (
            trend_coefs[g]["coef"] * df.loc[df["baseline_group"] == g, "time_trend"]
        )

# Detrend outcome
df["tempo"] = df["tempo_with_runners_on_base"]
df["tempo_detrended"] = df["tempo"] - df["trend_pred"]
df["weights"] = df["pitches_with_runners_on_base"]

print(f"\n✓ Detrended outcome created")
print(f"  Example (T3, 2023): trend_pred = {trend_coefs['T3']['coef']:.4f} × {2023-PRE_PERIOD_START} = "
      f"{trend_coefs['T3']['coef'] * (2023-PRE_PERIOD_START):.2f}s")

# ============================================================================
# STEP 3: EVENT-STUDY ON DETRENDED OUTCOME
# ============================================================================

print("\n" + "=" * 70)
print("STEP 3: EVENT-STUDY ON DETRENDED OUTCOME")
print("=" * 70)

# Year × Tercil interactions (excluding 2022)
for year in sorted(df["season"].unique()):
    if year == REFERENCE_YEAR:
        continue
    for g in ["T1", "T2", "T3"]:
        df[f"year{year}_x_{g}"] = ((df["season"] == year) & (df["baseline_group"] == g)).astype(int)

# Build formula
interaction_terms = [f"year{year}_x_{g}" 
                     for year in sorted(df["season"].unique()) if year != REFERENCE_YEAR
                     for g in ["T1", "T2", "T3"]]

formula = f"tempo_detrended ~ {' + '.join(interaction_terms)} + C(pitcher_id) + C(season)"

print(f"\nFormula: Y_detrended ~ Year×Tercil + Pitcher-FE + Year-FE")

model_detrended = smf.wls(formula, data=df, weights=df["weights"])
results_detrended = model_detrended.fit(cov_type="cluster", cov_kwds={"groups": df["pitcher_id"]})

print(f"\n✓ Model converged")
print(f"  N observations: {results_detrended.nobs:,.0f}")
print(f"  R²: {results_detrended.rsquared:.3f}")

# ============================================================================
# EXTRACT EFFECTS
# ============================================================================

print("\n" + "-" * 70)
print("DETRENDED TREATMENT EFFECTS (2023 & 2024)")
print("-" * 70)

print("\n2023 Effects (deviation from extrapolated pre-trend):")
for g in ["T1", "T2", "T3"]:
    var = f"year2023_x_{g}"
    if var in results_detrended.params.index:
        coef = results_detrended.params[var]
        se = results_detrended.bse[var]
        p = results_detrended.pvalues[var]
        stars = "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.1 else ""
        print(f"  {g}: β={coef:+.3f} (se={se:.3f}){stars}")

print("\n2024 Effects:")
for g in ["T1", "T2", "T3"]:
    var = f"year2024_x_{g}"
    if var in results_detrended.params.index:
        coef = results_detrended.params[var]
        se = results_detrended.bse[var]
        p = results_detrended.pvalues[var]
        stars = "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.1 else ""
        print(f"  {g}: β={coef:+.3f} (se={se:.3f}){stars}")

# ============================================================================
# MONOTONICITY TEST
# ============================================================================

print("\n" + "-" * 70)
print("MONOTONICITY TEST (2023)")
print("-" * 70)

# T3 vs T2
var1 = "year2023_x_T3"
var2 = "year2023_x_T2"
contrast = [1 if v == var1 else -1 if v == var2 else 0 for v in results_detrended.params.index]
t_test = results_detrended.t_test(contrast)
print(f"T3 vs T2: Δβ={results_detrended.params[var1] - results_detrended.params[var2]:.3f}, "
      f"t={t_test.tvalue[0][0]:.2f}, p={t_test.pvalue:.3f}")

# T2 vs T1
var1 = "year2023_x_T2"
var2 = "year2023_x_T1"
contrast = [1 if v == var1 else -1 if v == var2 else 0 for v in results_detrended.params.index]
t_test = results_detrended.t_test(contrast)
print(f"T2 vs T1: Δβ={results_detrended.params[var1] - results_detrended.params[var2]:.3f}, "
      f"t={t_test.tvalue[0][0]:.2f}, p={t_test.pvalue:.3f}")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "-" * 70)
print("SAVING RESULTS")
print("-" * 70)

# Event-study results
results_table = []
for year in sorted(df["season"].unique()):
    if year == REFERENCE_YEAR:
        for g in ["T1", "T2", "T3"]:
            results_table.append({
                "year": year,
                "rel_year": 0,
                "tercile": g,
                "beta_detrended": 0,
                "se_detrended": 0,
                "p_detrended": np.nan
            })
        continue
    
    for g in ["T1", "T2", "T3"]:
        var = f"year{year}_x_{g}"
        if var in results_detrended.params.index:
            results_table.append({
                "year": year,
                "rel_year": year - REFERENCE_YEAR,
                "tercile": g,
                "beta_detrended": results_detrended.params[var],
                "se_detrended": results_detrended.bse[var],
                "p_detrended": results_detrended.pvalues[var]
            })

df_results = pd.DataFrame(results_table)
output_file = OUTPUT_DIR / "b2_eventstudy_detrended_corrected_results.csv"
df_results.to_csv(output_file, index=False)
print(f"✓ Saved: {output_file}")

# Trend coefficients
trend_results = []
for g in ["T1", "T2", "T3"]:
    if g in trend_coefs:
        trend_results.append({
            "tercile": g,
            "trend_coef": trend_coefs[g]["coef"],
            "trend_se": trend_coefs[g]["se"],
            "trend_p": trend_coefs[g]["p"],
            "interpretation": f"{trend_coefs[g]['coef']:.4f}s per year (pre-period only)"
        })

df_trends = pd.DataFrame(trend_results)
output_file = OUTPUT_DIR / "b2_time_trends_corrected.csv"
df_trends.to_csv(output_file, index=False)
print(f"✓ Saved: {output_file}")

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\n" + "-" * 70)
print("CREATING PLOT")
print("-" * 70)

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

for i, g in enumerate(["T1", "T2", "T3"]):
    ax = axes[i]
    
    data_g = df_results[df_results["tercile"] == g].sort_values("rel_year")
    
    ax.errorbar(data_g["rel_year"], data_g["beta_detrended"], 
                yerr=1.96*data_g["se_detrended"],
                marker="o", capsize=5, linewidth=2, markersize=8)
    
    ax.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax.axvline(0, color="red", linestyle="--", alpha=0.3, label="2023 Timer")
    
    ax.set_xlabel("Years Relative to 2022", fontsize=11)
    if i == 0:
        ax.set_ylabel("Detrended Tempo Change (seconds)", fontsize=11)
    
    if g in trend_coefs:
        trend = trend_coefs[g]["coef"]
        ax.text(0.05, 0.95, f"Pre-trend: {trend:+.3f}s/yr\n(2018-2022 only)", 
                transform=ax.transAxes, fontsize=9, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    ax.set_title(f"{g} ({'fast' if g=='T1' else 'mid' if g=='T2' else 'slow'})", 
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9)

plt.suptitle("Detrended Event-Study: Trends Estimated from Pre-Period Only", 
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()

output_file = OUTPUT_DIR / "b2_eventstudy_detrended_corrected_plot.png"
plt.savefig(output_file, dpi=300, bbox_inches="tight")
print(f"✓ Saved: {output_file}")
plt.close()

print(f"\n✓ Analysis complete. Trends estimated from pre-period only.")