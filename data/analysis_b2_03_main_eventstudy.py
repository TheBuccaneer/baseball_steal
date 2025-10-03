"""
b2_03_main_eventstudy.py
=========================
Main event-study analysis: Tempo response to pitch timer by baseline tercile.

Model Specification:
Y_it = Σ_{year≠2022} Σ_g β_{year,g} × 1[year=y] × 1[tercil=g] + α_i + τ_t + ε_it

Where:
- 2022 is omitted reference year
- α_i = pitcher fixed effects
- τ_t = year fixed effects
- SE clustered on pitcher_id

Outputs:
- Main: WLS (weights = pitches)
- Robustness: Unweighted, Balanced Panels, No Year-FE
- Tests: Monotonicity (Wald), Pre-trends (F-test), 2024 additional effect
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_PANEL = "./analysis/analysis_pitcher_panel_relative.csv"
INPUT_BASELINE = "./analysis/b2_baseline/b2_baseline_groups.csv"
INPUT_BP = "./analysis/b2_baseline/b2_attrition_balanced_panels.csv"
OUTPUT_DIR = Path("./analysis/b2_eventstudy")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_PITCHES = 50
REFERENCE_YEAR = 2022

# ============================================================================
# LOAD DATA
# ============================================================================

print("=" * 70)
print("B2-03: MAIN EVENT-STUDY ANALYSIS")
print("=" * 70)

# Load panel
df_panel = pd.read_csv(INPUT_PANEL)
print(f"\n✓ Loaded {len(df_panel):,} pitcher-season rows")

# Load baseline groups
df_baseline = pd.read_csv(INPUT_BASELINE)
print(f"✓ Loaded {len(df_baseline):,} baseline pitchers")

# Load balanced panel lists
df_bp_info = pd.read_csv(INPUT_BP)

# Merge baseline groups
df = df_panel.merge(df_baseline[["pitcher_id", "baseline_group"]], 
                    on="pitcher_id", how="inner")

# Filter to cohort with sufficient data
df = df[df["pitches_with_runners_on_base"] >= MIN_PITCHES].copy()
print(f"✓ Working sample: {len(df):,} pitcher-season observations")

# ============================================================================
# PREPARE VARIABLES
# ============================================================================

print("\n" + "-" * 70)
print("VARIABLE CONSTRUCTION")
print("-" * 70)

# Relative year (for tracking, not for formula)
df["rel_year"] = df["season"] - REFERENCE_YEAR
print(f"Years in sample: {sorted(df['season'].unique())}")
print(f"Reference year (omitted): {REFERENCE_YEAR}")

# Post indicators
df["post_2023"] = (df["season"] == 2023).astype(int)
df["post_2024"] = (df["season"] == 2024).astype(int)

# Tercile dummies
df["tercil_T1"] = (df["baseline_group"] == "T1").astype(int)
df["tercil_T2"] = (df["baseline_group"] == "T2").astype(int)
df["tercil_T3"] = (df["baseline_group"] == "T3").astype(int)

# Interactions for each year × tercil (excluding 2022 reference)
for year in sorted(df["season"].unique()):
    if year == REFERENCE_YEAR:
        continue  # Omit reference year
    for g in ["T1", "T2", "T3"]:
        df[f"year{year}_x_{g}"] = ((df["season"] == year) & (df["baseline_group"] == g)).astype(int)

# Post × tercil interactions
for g in ["T1", "T2", "T3"]:
    df[f"post2023_x_{g}"] = df["post_2023"] * df[f"tercil_{g}"]
    df[f"post2024_x_{g}"] = df["post_2024"] * df[f"tercil_{g}"]

# Outcome
df["tempo"] = df["tempo_with_runners_on_base"]

# Weights
df["weights"] = df["pitches_with_runners_on_base"]

print(f"✓ Variables constructed")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_coefs(results, pattern):
    """Extract coefficients matching pattern."""
    coefs = {}
    for name in results.params.index:
        if pattern in name:
            coefs[name] = {
                "beta": results.params[name],
                "se": results.bse[name],
                "t": results.tvalues[name],
                "p": results.pvalues[name],
                "ci_lo": results.conf_int().loc[name, 0],
                "ci_hi": results.conf_int().loc[name, 1]
            }
    return coefs

def monotonicity_test(results, year, terciles=["T1", "T2", "T3"]):
    """Test β_T3 < β_T2 < β_T1 for given year."""
    tests = []
    
    # T3 vs T2
    var1 = f"year{year}_x_T3"
    var2 = f"year{year}_x_T2"
    if var1 in results.params.index and var2 in results.params.index:
        contrast = [1 if v == var1 else -1 if v == var2 else 0 for v in results.params.index]
        t_test = results.t_test(contrast)
        tests.append({
            "comparison": "T3 vs T2",
            "year": year,
            "diff": results.params[var1] - results.params[var2],
            "t": t_test.tvalue[0][0],
            "p": t_test.pvalue
        })
    
    # T2 vs T1
    var1 = f"year{year}_x_T2"
    var2 = f"year{year}_x_T1"
    if var1 in results.params.index and var2 in results.params.index:
        contrast = [1 if v == var1 else -1 if v == var2 else 0 for v in results.params.index]
        t_test = results.t_test(contrast)
        tests.append({
            "comparison": "T2 vs T1",
            "year": year,
            "diff": results.params[var1] - results.params[var2],
            "t": t_test.tvalue[0][0],
            "p": t_test.pvalue
        })
    
    return tests

def pretrends_test(results, pre_years):
    """Joint F-test on all pre-period coefficients."""
    pre_vars = []
    for year in pre_years:
        for g in ["T1", "T2", "T3"]:
            var = f"year{year}_x_{g}"
            if var in results.params.index:
                pre_vars.append(var)
    
    if len(pre_vars) == 0:
        return None
    
    # Hypothesis: all pre-coefficients = 0
    # Build hypothesis string: "var1 = 0, var2 = 0, ..."
    hypothesis_string = ", ".join([f"{v} = 0" for v in pre_vars])
    f_test = results.f_test(hypothesis_string)
    
    return {
        "n_vars": len(pre_vars),
        "f_stat": f_test.fvalue,
        "df1": f_test.df_num,
        "df2": f_test.df_denom,
        "p": f_test.pvalue
    }

# ============================================================================
# MODEL 1: FULL EVENT-STUDY (MAIN)
# ============================================================================

print("\n" + "=" * 70)
print("MODEL 1: FULL EVENT-STUDY (WLS, PITCHER-FE, YEAR-FE)")
print("=" * 70)

# Build formula
interaction_terms = [f"year{year}_x_{g}" 
                     for year in sorted(df["season"].unique()) if year != REFERENCE_YEAR
                     for g in ["T1", "T2", "T3"]]

formula = f"tempo ~ {' + '.join(interaction_terms)} + C(pitcher_id) + C(season)"

print(f"\nFormula: Y ~ Year×Tercil + Pitcher-FE + Year-FE")
print(f"Parameters: {len(interaction_terms)} interactions + FEs")

# Fit weighted model
model_wls = smf.wls(formula, data=df, weights=df["weights"])
results_wls = model_wls.fit(cov_type="cluster", cov_kwds={"groups": df["pitcher_id"]})

print(f"\n✓ Model converged")
print(f"  N observations: {results_wls.nobs:,.0f}")
print(f"  R²: {results_wls.rsquared:.3f}")
print(f"  Adj R²: {results_wls.rsquared_adj:.3f}")

# Extract event-study coefficients
eventstudy_coefs = extract_coefs(results_wls, "year")

# ============================================================================
# TESTS: MONOTONICITY
# ============================================================================

print("\n" + "-" * 70)
print("MONOTONICITY TESTS (2023 & 2024)")
print("-" * 70)

monotonicity_results = []
for year in [2023, 2024]:
    tests = monotonicity_test(results_wls, year)
    monotonicity_results.extend(tests)
    for test in tests:
        print(f"{test['comparison']} ({test['year']}): "
              f"Δβ={test['diff']:.3f}, t={test['t']:.2f}, p={test['p']:.3f}")

df_monotonicity = pd.DataFrame(monotonicity_results)
output_file = OUTPUT_DIR / "b2_monotonicity_tests.csv"
df_monotonicity.to_csv(output_file, index=False)
print(f"\n✓ Saved: {output_file}")

# ============================================================================
# TESTS: PRE-TRENDS
# ============================================================================

print("\n" + "-" * 70)
print("PRE-TRENDS TEST")
print("-" * 70)

pre_years = [y for y in sorted(df["season"].unique()) if y < REFERENCE_YEAR]
pretrends = pretrends_test(results_wls, pre_years)
if pretrends:
    print(f"Joint F-test on pre-period coefficients:")
    print(f"  F({pretrends['df1']}, {pretrends['df2']}) = {pretrends['f_stat']:.2f}")
    print(f"  p-value: {pretrends['p']:.3f}")
    
    if pretrends['p'] < 0.05:
        print("  ⚠ Pre-trends differ from zero (p<0.05)")
    else:
        print("  ✓ No evidence of differential pre-trends")

# ============================================================================
# MODEL 2: POST-ONLY (2023 & 2024 SEPARATE)
# ============================================================================

print("\n" + "=" * 70)
print("MODEL 2: POST-ONLY (2023 & 2024 SEPARATE)")
print("=" * 70)

formula_post = ("tempo ~ post2023_x_T1 + post2023_x_T2 + post2023_x_T3 + "
                "post2024_x_T1 + post2024_x_T2 + post2024_x_T3 + "
                "C(pitcher_id) + C(season)")

model_post = smf.wls(formula_post, data=df, weights=df["weights"])
results_post = model_post.fit(cov_type="cluster", cov_kwds={"groups": df["pitcher_id"]})

print("\n2023 Effects:")
for g in ["T1", "T2", "T3"]:
    var = f"post2023_x_{g}"
    print(f"  {g}: β={results_post.params[var]:.3f} (se={results_post.bse[var]:.3f}, p={results_post.pvalues[var]:.3f})")

print("\n2024 Additional Effects:")
for g in ["T1", "T2", "T3"]:
    var = f"post2024_x_{g}"
    print(f"  {g}: γ={results_post.params[var]:.3f} (se={results_post.bse[var]:.3f}, p={results_post.pvalues[var]:.3f})")

# ============================================================================
# MODEL 3: WITHOUT YEAR-FE (ABSOLUTE EFFECTS)
# ============================================================================

print("\n" + "=" * 70)
print("MODEL 3: WITHOUT YEAR-FE (ABSOLUTE EFFECTS)")
print("=" * 70)

formula_noyearfe = f"tempo ~ {' + '.join(interaction_terms)} + C(pitcher_id)"

model_noyearfe = smf.wls(formula_noyearfe, data=df, weights=df["weights"])
results_noyearfe = model_noyearfe.fit(cov_type="cluster", cov_kwds={"groups": df["pitcher_id"]})

print(f"✓ R²: {results_noyearfe.rsquared:.3f}")
print("Note: Coefficients now show absolute tempo changes, not deviations from year-mean")

# ============================================================================
# MODEL 4: UNWEIGHTED (PITCHER-AVERAGE)
# ============================================================================

print("\n" + "=" * 70)
print("MODEL 4: UNWEIGHTED (PITCHER-AVERAGE EFFECT)")
print("=" * 70)

model_ols = smf.ols(formula, data=df)
results_ols = model_ols.fit(cov_type="cluster", cov_kwds={"groups": df["pitcher_id"]})

print(f"✓ R²: {results_ols.rsquared:.3f}")
print("Note: Equal weight per pitcher-season, regardless of pitches thrown")

# ============================================================================
# MODEL 5: BALANCED PANEL ROBUSTNESS
# ============================================================================

print("\n" + "=" * 70)
print("MODEL 5: BALANCED PANEL ROBUSTNESS")
print("=" * 70)

# Get BP pitcher lists
pitchers_2022 = set(df[df["season"] == 2022]["pitcher_id"])
pitchers_2023 = set(df[df["season"] == 2023]["pitcher_id"])
pitchers_2024 = set(df[df["season"] == 2024]["pitcher_id"])

bp_22_23 = pitchers_2022 & pitchers_2023
bp_22_23_24 = pitchers_2022 & pitchers_2023 & pitchers_2024

print(f"Balanced panels:")
print(f"  BP_22_23: {len(bp_22_23):,} pitchers")
print(f"  BP_22_23_24: {len(bp_22_23_24):,} pitchers")

# Fit on BP_22_23
df_bp23 = df[df["pitcher_id"].isin(bp_22_23)]
model_bp23 = smf.wls(formula, data=df_bp23, weights=df_bp23["weights"])
results_bp23 = model_bp23.fit(cov_type="cluster", cov_kwds={"groups": df_bp23["pitcher_id"]})
print(f"\n✓ BP_22_23: N={results_bp23.nobs:,.0f}, R²={results_bp23.rsquared:.3f}")

# Fit on BP_22_23_24
df_bp24 = df[df["pitcher_id"].isin(bp_22_23_24)]
model_bp24 = smf.wls(formula, data=df_bp24, weights=df_bp24["weights"])
results_bp24 = model_bp24.fit(cov_type="cluster", cov_kwds={"groups": df_bp24["pitcher_id"]})
print(f"✓ BP_22_23_24: N={results_bp24.nobs:,.0f}, R²={results_bp24.rsquared:.3f}")

# ============================================================================
# EXTRACT RESULTS TABLE
# ============================================================================

print("\n" + "-" * 70)
print("BUILDING RESULTS TABLE")
print("-" * 70)

# Extract coefficients for main years and terciles
results_table = []

for year in sorted(df["season"].unique()):
    if year == REFERENCE_YEAR:
        # Add reference year with zeros
        for g in ["T1", "T2", "T3"]:
            results_table.append({
                "year": year,
                "rel_year": 0,
                "tercile": g,
                "beta_main": 0,
                "se_main": 0,
                "p_main": np.nan,
                "beta_unweighted": 0,
                "se_unweighted": 0,
                "beta_bp23": 0,
                "se_bp23": 0
            })
        continue
    
    for g in ["T1", "T2", "T3"]:
        var = f"year{year}_x_{g}"
        
        row = {"year": year, "rel_year": year - REFERENCE_YEAR, "tercile": g}
        
        # Main (WLS)
        if var in results_wls.params.index:
            row["beta_main"] = results_wls.params[var]
            row["se_main"] = results_wls.bse[var]
            row["p_main"] = results_wls.pvalues[var]
        else:
            row["beta_main"] = np.nan
            row["se_main"] = np.nan
            row["p_main"] = np.nan
        
        # Unweighted
        if var in results_ols.params.index:
            row["beta_unweighted"] = results_ols.params[var]
            row["se_unweighted"] = results_ols.bse[var]
        else:
            row["beta_unweighted"] = np.nan
            row["se_unweighted"] = np.nan
        
        # BP_22_23
        if var in results_bp23.params.index:
            row["beta_bp23"] = results_bp23.params[var]
            row["se_bp23"] = results_bp23.bse[var]
        else:
            row["beta_bp23"] = np.nan
            row["se_bp23"] = np.nan
        
        results_table.append(row)

df_results = pd.DataFrame(results_table)
output_file = OUTPUT_DIR / "b2_eventstudy_results.csv"
df_results.to_csv(output_file, index=False)
print(f"✓ Saved: {output_file}")

# ============================================================================
# VISUALIZATION: EVENT-STUDY PLOT
# ============================================================================

print("\n" + "-" * 70)
print("CREATING EVENT-STUDY PLOT")
print("-" * 70)

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

for i, g in enumerate(["T1", "T2", "T3"]):
    ax = axes[i]
    
    # Extract data for this tercile
    data_g = df_results[df_results["tercile"] == g].sort_values("rel_year")
    
    # Plot
    ax.errorbar(data_g["rel_year"], data_g["beta_main"], 
                yerr=1.96*data_g["se_main"],
                marker="o", capsize=5, linewidth=2, markersize=8)
    
    # Zero line
    ax.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax.axvline(0, color="red", linestyle="--", alpha=0.3, label="2023 Timer")
    
    # Formatting
    ax.set_xlabel("Years Relative to 2022", fontsize=11)
    if i == 0:
        ax.set_ylabel("Tempo Change (seconds)", fontsize=11)
    ax.set_title(f"{g} ({'fast' if g=='T1' else 'mid' if g=='T2' else 'slow'})", 
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9)

plt.suptitle("Event-Study: Pitch Tempo Response by Baseline Tercile", 
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()

output_file = OUTPUT_DIR / "b2_eventstudy_plot.png"
plt.savefig(output_file, dpi=300, bbox_inches="tight")
print(f"✓ Saved: {output_file}")
plt.close()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"\nMain Results (2023):")
for g in ["T1", "T2", "T3"]:
    var = f"year2023_x_{g}"
    if var in results_wls.params.index:
        b = results_wls.params[var]
        se = results_wls.bse[var]
        p = results_wls.pvalues[var]
        stars = "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.1 else ""
        print(f"  {g}: β={b:+.3f} (se={se:.3f}){stars}")

print(f"\nMonotonicity (2023):")
if len(df_monotonicity[df_monotonicity["year"] == 2023]) > 0:
    for _, row in df_monotonicity[df_monotonicity["year"] == 2023].iterrows():
        sig = "✓" if row["p"] < 0.05 else "✗"
        print(f"  {row['comparison']}: {sig} (p={row['p']:.3f})")

if pretrends:
    sig_pre = "⚠" if pretrends['p'] < 0.05 else "✓"
    print(f"\nPre-trends: {sig_pre} F={pretrends['f_stat']:.2f}, p={pretrends['p']:.3f}")

print(f"\nFiles saved to: {OUTPUT_DIR}/")
print(f"\nNext step: Run b2_04_pretrends_test.py (detailed pre-period analysis)")