"""
b2_02_attrition_analysis.py
============================
Checks for selection bias in 2022 baseline cohort.

Four Checks:
1. Retention by tercile (2022→2023, 2022→2024)
2. Balanced panel flags (BP_22_23, BP_22_23_24)
3. Two-sample comparison (all 2023 vs. cohort 2023)
4. Exposure stability (Δpitches by tercile)

Outputs:
- b2_attrition_retention.csv
- b2_attrition_balanced_panels.csv
- b2_attrition_two_sample_2023.csv
- b2_attrition_exposure_delta.csv
- b2_attrition_retention_plot.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_PANEL = "./analysis/analysis_pitcher_panel_relative.csv"
INPUT_BASELINE = "./analysis/b2_baseline/b2_baseline_groups.csv"
OUTPUT_DIR = Path("./analysis/b2_baseline")

MIN_PITCHES = 50

# ============================================================================
# LOAD DATA
# ============================================================================

print("=" * 70)
print("B2-02: ATTRITION ANALYSIS")
print("=" * 70)

# Load full panel
df_panel = pd.read_csv(INPUT_PANEL)
print(f"\n✓ Loaded {len(df_panel):,} pitcher-season rows")

# Load baseline groups
df_baseline = pd.read_csv(INPUT_BASELINE)
print(f"✓ Loaded {len(df_baseline):,} baseline pitchers (2022)")

# Merge baseline groups into panel
df = df_panel.merge(df_baseline[["pitcher_id", "baseline_group"]], 
                    on="pitcher_id", how="left")

# Filter to cohort (pitchers in baseline)
df_cohort = df[df["baseline_group"].notna()].copy()
print(f"✓ Cohort panel: {len(df_cohort):,} rows")

# ============================================================================
# CHECK 1: RETENTION BY TERCILE
# ============================================================================

print("\n" + "-" * 70)
print("CHECK 1: RETENTION BY TERCILE")
print("-" * 70)

# Baseline counts
baseline_counts = df_baseline["baseline_group"].value_counts().sort_index()

# Retention in 2023
df_2023 = df_cohort[
    (df_cohort["season"] == 2023) & 
    (df_cohort["pitches_with_runners_on_base"] >= MIN_PITCHES)
]
retention_2023 = df_2023["baseline_group"].value_counts().sort_index()

# Retention in 2024
df_2024 = df_cohort[
    (df_cohort["season"] == 2024) & 
    (df_cohort["pitches_with_runners_on_base"] >= MIN_PITCHES)
]
retention_2024 = df_2024["baseline_group"].value_counts().sort_index()

# Build retention table
retention_data = []
for group in ["T1", "T2", "T3"]:
    n_2022 = baseline_counts.get(group, 0)
    n_2023 = retention_2023.get(group, 0)
    n_2024 = retention_2024.get(group, 0)
    
    retention_data.append({
        "tercile": group,
        "n_2022": n_2022,
        "n_2023": n_2023,
        "n_2024": n_2024,
        "retention_2023_pct": 100 * n_2023 / n_2022 if n_2022 > 0 else 0,
        "retention_2024_pct": 100 * n_2024 / n_2022 if n_2022 > 0 else 0
    })

df_retention = pd.DataFrame(retention_data)

# Chi-squared tests
observed_2023 = df_retention[["n_2023"]].values.flatten()
observed_2024 = df_retention[["n_2024"]].values.flatten()
expected_2023 = df_retention["n_2022"].values * (observed_2023.sum() / df_retention["n_2022"].sum())
expected_2024 = df_retention["n_2022"].values * (observed_2024.sum() / df_retention["n_2022"].sum())

chi2_2023, p_2023 = stats.chisquare(observed_2023, expected_2023)
chi2_2024, p_2024 = stats.chisquare(observed_2024, expected_2024)

# Add test results
test_row = {
    "tercile": "chi2_test",
    "n_2022": np.nan,
    "n_2023": np.nan,
    "n_2024": np.nan,
    "retention_2023_pct": p_2023,
    "retention_2024_pct": p_2024
}
df_retention = pd.concat([df_retention, pd.DataFrame([test_row])], ignore_index=True)

print("\nRetention by Tercile:")
print(df_retention.to_string(index=False))

# Save
output_file = OUTPUT_DIR / "b2_attrition_retention.csv"
df_retention.to_csv(output_file, index=False)
print(f"\n✓ Saved: {output_file}")

# ============================================================================
# CHECK 2: BALANCED PANEL FLAGS
# ============================================================================

print("\n" + "-" * 70)
print("CHECK 2: BALANCED PANEL FLAGS")
print("-" * 70)

# Get qualified pitchers per year
pitchers_2022 = set(df_cohort[
    (df_cohort["season"] == 2022) & 
    (df_cohort["pitches_with_runners_on_base"] >= MIN_PITCHES)
]["pitcher_id"])

pitchers_2023 = set(df_cohort[
    (df_cohort["season"] == 2023) & 
    (df_cohort["pitches_with_runners_on_base"] >= MIN_PITCHES)
]["pitcher_id"])

pitchers_2024 = set(df_cohort[
    (df_cohort["season"] == 2024) & 
    (df_cohort["pitches_with_runners_on_base"] >= MIN_PITCHES)
]["pitcher_id"])

# Define balanced panels
bp_22_23 = pitchers_2022 & pitchers_2023
bp_22_23_24 = pitchers_2022 & pitchers_2023 & pitchers_2024

# Count by tercile
bp_data = []

for panel_name, panel_set in [
    ("2022_only", pitchers_2022),
    ("BP_22_23", bp_22_23),
    ("BP_22_23_24", bp_22_23_24)
]:
    df_panel_subset = df_baseline[df_baseline["pitcher_id"].isin(panel_set)]
    counts = df_panel_subset["baseline_group"].value_counts().sort_index()
    
    bp_data.append({
        "panel_definition": panel_name,
        "n_total": len(panel_set),
        "n_T1": counts.get("T1", 0),
        "n_T2": counts.get("T2", 0),
        "n_T3": counts.get("T3", 0),
        "pct_of_baseline": 100 * len(panel_set) / len(df_baseline)
    })

df_bp = pd.DataFrame(bp_data)

print("\nBalanced Panels:")
print(df_bp.to_string(index=False))

output_file = OUTPUT_DIR / "b2_attrition_balanced_panels.csv"
df_bp.to_csv(output_file, index=False)
print(f"\n✓ Saved: {output_file}")

# ============================================================================
# CHECK 3: TWO-SAMPLE COMPARISON (2023)
# ============================================================================

print("\n" + "-" * 70)
print("CHECK 3: TWO-SAMPLE COMPARISON (2023)")
print("-" * 70)

# Sample A: All 2023 pitchers
df_all_2023 = df_panel[
    (df_panel["season"] == 2023) & 
    (df_panel["pitches_with_runners_on_base"] >= MIN_PITCHES)
].copy()

# Sample B: Only cohort pitchers in 2023
df_cohort_2023 = df_all_2023[df_all_2023["pitcher_id"].isin(df_baseline["pitcher_id"])]

# Compare tempo
tempo_all = df_all_2023["tempo_with_runners_on_base"].dropna()
tempo_cohort = df_cohort_2023["tempo_with_runners_on_base"].dropna()

# Welch t-test
t_stat, t_p = stats.ttest_ind(tempo_all, tempo_cohort, equal_var=False)

# Kolmogorov-Smirnov test
ks_stat, ks_p = stats.ks_2samp(tempo_all, tempo_cohort)

# Build comparison table
two_sample_data = [
    {
        "sample": "all_2023_pitchers",
        "n": len(tempo_all),
        "mean_tempo": tempo_all.mean(),
        "sd_tempo": tempo_all.std()
    },
    {
        "sample": "cohort_2023_only",
        "n": len(tempo_cohort),
        "mean_tempo": tempo_cohort.mean(),
        "sd_tempo": tempo_cohort.std()
    },
    {
        "sample": "difference",
        "n": np.nan,
        "mean_tempo": tempo_all.mean() - tempo_cohort.mean(),
        "sd_tempo": np.nan
    },
    {
        "sample": "welch_t_p",
        "n": np.nan,
        "mean_tempo": t_p,
        "sd_tempo": np.nan
    },
    {
        "sample": "ks_test_p",
        "n": np.nan,
        "mean_tempo": ks_p,
        "sd_tempo": np.nan
    }
]

df_two_sample = pd.DataFrame(two_sample_data)

print("\nTwo-Sample Comparison (2023):")
print(df_two_sample.to_string(index=False))

output_file = OUTPUT_DIR / "b2_attrition_two_sample_2023.csv"
df_two_sample.to_csv(output_file, index=False)
print(f"\n✓ Saved: {output_file}")

# ============================================================================
# CHECK 4: EXPOSURE STABILITY
# ============================================================================

print("\n" + "-" * 70)
print("CHECK 4: EXPOSURE STABILITY")
print("-" * 70)

# Get pitches for cohort in 2022 and 2023
df_22 = df_cohort[df_cohort["season"] == 2022][["pitcher_id", "pitches_with_runners_on_base"]]
df_22 = df_22.rename(columns={"pitches_with_runners_on_base": "pitches_2022"})

df_23 = df_cohort[df_cohort["season"] == 2023][["pitcher_id", "pitches_with_runners_on_base"]]
df_23 = df_23.rename(columns={"pitches_with_runners_on_base": "pitches_2023"})

# Merge and compute delta
df_exposure = df_baseline.merge(df_22, on="pitcher_id")
df_exposure = df_exposure.merge(df_23, on="pitcher_id", how="left")
df_exposure["delta_pitches"] = df_exposure["pitches_2023"] - df_exposure["pitches_2022"]

# Only keep pitchers in 2023
df_exposure = df_exposure[df_exposure["pitches_2023"].notna()]

# Stats by tercile
exposure_data = []
for group in ["T1", "T2", "T3"]:
    group_data = df_exposure[df_exposure["baseline_group"] == group]
    exposure_data.append({
        "tercile": group,
        "n": len(group_data),
        "mean_pitches_2022": group_data["pitches_2022"].mean(),
        "mean_pitches_2023": group_data["pitches_2023"].mean(),
        "mean_delta": group_data["delta_pitches"].mean(),
        "median_delta": group_data["delta_pitches"].median()
    })

df_exposure_stats = pd.DataFrame(exposure_data)

# ANOVA on delta
groups_delta = [
    df_exposure[df_exposure["baseline_group"] == "T1"]["delta_pitches"].dropna(),
    df_exposure[df_exposure["baseline_group"] == "T2"]["delta_pitches"].dropna(),
    df_exposure[df_exposure["baseline_group"] == "T3"]["delta_pitches"].dropna()
]
f_stat, anova_p = stats.f_oneway(*groups_delta)

# Add test result
test_row = {
    "tercile": "anova_p",
    "n": np.nan,
    "mean_pitches_2022": np.nan,
    "mean_pitches_2023": np.nan,
    "mean_delta": anova_p,
    "median_delta": np.nan
}
df_exposure_stats = pd.concat([df_exposure_stats, pd.DataFrame([test_row])], ignore_index=True)

print("\nExposure Stability (2022→2023):")
print(df_exposure_stats.to_string(index=False))

output_file = OUTPUT_DIR / "b2_attrition_exposure_delta.csv"
df_exposure_stats.to_csv(output_file, index=False)
print(f"\n✓ Saved: {output_file}")

# ============================================================================
# VISUALIZATION: RETENTION PLOT
# ============================================================================

print("\n" + "-" * 70)
print("CREATING RETENTION PLOT")
print("-" * 70)

fig, ax = plt.subplots(figsize=(10, 6))

# Plot retention rates
x = np.arange(3)
width = 0.35

retention_2023 = df_retention[df_retention["tercile"] != "chi2_test"]["retention_2023_pct"].values
retention_2024 = df_retention[df_retention["tercile"] != "chi2_test"]["retention_2024_pct"].values

ax.bar(x - width/2, retention_2023, width, label="2023", alpha=0.8)
ax.bar(x + width/2, retention_2024, width, label="2024", alpha=0.8)

ax.set_xlabel("Baseline Tercile (2022)", fontsize=12)
ax.set_ylabel("Retention Rate (%)", fontsize=12)
ax.set_title("Retention from 2022 Baseline by Tercile", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(["T1 (fast)", "T2 (mid)", "T3 (slow)"])
ax.legend()
ax.grid(axis="y", alpha=0.3)
ax.set_ylim([0, 100])

# Add p-values
ax.text(0.02, 0.98, f"2023: χ²={chi2_2023:.2f}, p={p_2023:.3f}\n2024: χ²={chi2_2024:.2f}, p={p_2024:.3f}",
        transform=ax.transAxes, fontsize=10, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

plt.tight_layout()
output_file = OUTPUT_DIR / "b2_attrition_retention_plot.png"
plt.savefig(output_file, dpi=300)
print(f"\n✓ Saved: {output_file}")
plt.close()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("ATTRITION SUMMARY")
print("=" * 70)

qc_checks = [
    ("Retention differs by tercile (2023)", p_2023 < 0.05),
    ("Retention differs by tercile (2024)", p_2024 < 0.05),
    ("Composition shift (2023)", t_p < 0.05),
    ("Exposure change differs by tercile", anova_p < 0.05),
]

for check_name, is_concerning in qc_checks:
    status = "⚠ YES" if is_concerning else "✓ NO"
    print(f"{status}: {check_name}")

any_issues = any(is_concerning for _, is_concerning in qc_checks)
if not any_issues:
    print("\n✓✓✓ NO MAJOR ATTRITION ISSUES - PROCEED WITH FULL SAMPLE ✓✓✓")
else:
    print("\n⚠⚠⚠ ATTRITION DETECTED - USE BALANCED PANELS FOR ROBUSTNESS ⚠⚠⚠")

print(f"\nBalanced panel sizes:")
print(f"  BP_22_23:    {len(bp_22_23):,} pitchers ({100*len(bp_22_23)/len(df_baseline):.1f}%)")
print(f"  BP_22_23_24: {len(bp_22_23_24):,} pitchers ({100*len(bp_22_23_24)/len(df_baseline):.1f}%)")

print(f"\nNext step: Run b2_03_main_eventstudy.py")