"""
b2_04a_t1_composition.py
========================
Diagnostic: Is T1 anomaly driven by compositional changes?

Tests whether the positive T1 coefficient in 2023 is due to:
- Different pitchers entering T1 in 2023 (roster turnover)
- Pitchers exiting T1 between 2022 and 2023

Key question: Are T1-2023 pitchers slower than T1-2022 pitchers?
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_PANEL = "./analysis/analysis_pitcher_panel_relative.csv"
INPUT_BASELINE = "./analysis/b2_baseline/b2_baseline_groups.csv"
OUTPUT_DIR = Path("./analysis/b2_diagnostics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_PITCHES = 50

# ============================================================================
# LOAD DATA
# ============================================================================

print("=" * 70)
print("B2-04a: T1 COMPOSITION DIAGNOSTIC")
print("=" * 70)

df_panel = pd.read_csv(INPUT_PANEL)
df_baseline = pd.read_csv(INPUT_BASELINE)

df = df_panel.merge(df_baseline[["pitcher_id", "baseline_group"]], 
                    on="pitcher_id", how="inner")
df = df[df["pitches_with_runners_on_base"] >= MIN_PITCHES].copy()

print(f"\n✓ Full sample loaded: {len(df):,} observations")

# ============================================================================
# IDENTIFY T1 PITCHERS BY YEAR
# ============================================================================

print("\n" + "-" * 70)
print("STEP 1: IDENTIFY T1 PITCHERS (2022 vs 2023)")
print("-" * 70)

# T1 pitchers in each year
t1_2022_ids = set(df[(df["season"] == 2022) & (df["baseline_group"] == "T1")]["pitcher_id"])
t1_2023_ids = set(df[(df["season"] == 2023) & (df["baseline_group"] == "T1")]["pitcher_id"])

# Categorize
retained = t1_2022_ids & t1_2023_ids  # In both years
exited = t1_2022_ids - t1_2023_ids    # In 2022, not in 2023
new = t1_2023_ids - t1_2022_ids       # In 2023, not in 2022

print(f"\nT1 Composition:")
print(f"  2022 total: {len(t1_2022_ids)} pitchers")
print(f"  2023 total: {len(t1_2023_ids)} pitchers")
print(f"\n  Retained (in both): {len(retained)} ({100*len(retained)/len(t1_2022_ids):.1f}%)")
print(f"  Exited (2022→out): {len(exited)} ({100*len(exited)/len(t1_2022_ids):.1f}%)")
print(f"  New (in→2023): {len(new)} ({100*len(new)/len(t1_2023_ids):.1f}%)")

# ============================================================================
# COMPARE TEMPO: RETAINED vs NEW (2023)
# ============================================================================

print("\n" + "-" * 70)
print("STEP 2: TEMPO COMPARISON (2023)")
print("-" * 70)

df_t1_2023 = df[(df["season"] == 2023) & (df["baseline_group"] == "T1")].copy()

# Mark each pitcher
df_t1_2023["group"] = "unknown"
df_t1_2023.loc[df_t1_2023["pitcher_id"].isin(retained), "group"] = "retained"
df_t1_2023.loc[df_t1_2023["pitcher_id"].isin(new), "group"] = "new"

# Summary stats
tempo_retained = df_t1_2023[df_t1_2023["group"] == "retained"]["tempo_with_runners_on_base"]
tempo_new = df_t1_2023[df_t1_2023["group"] == "new"]["tempo_with_runners_on_base"]

print(f"\n2023 Tempo (T1):")
print(f"  Retained pitchers: {tempo_retained.mean():.2f}s (SD={tempo_retained.std():.2f}, n={len(tempo_retained)})")
print(f"  New pitchers:      {tempo_new.mean():.2f}s (SD={tempo_new.std():.2f}, n={len(tempo_new)})")
print(f"  Difference:        {tempo_new.mean() - tempo_retained.mean():+.2f}s")

# t-test
if len(tempo_retained) > 0 and len(tempo_new) > 0:
    t_stat, p_val = stats.ttest_ind(tempo_new, tempo_retained, equal_var=False)
    print(f"\n  Welch t-test: t={t_stat:.2f}, p={p_val:.3f}")
    if p_val < 0.05:
        print("  → New pitchers are significantly different from retained")
    else:
        print("  → No significant difference")

# ============================================================================
# COMPARE TEMPO: 2022 vs 2023 (RETAINED ONLY)
# ============================================================================

print("\n" + "-" * 70)
print("STEP 3: WITHIN-PITCHER CHANGE (RETAINED ONLY)")
print("-" * 70)

df_retained = df[df["pitcher_id"].isin(retained) & 
                 (df["baseline_group"] == "T1") &
                 (df["season"].isin([2022, 2023]))].copy()

# Pivot to get 2022 and 2023 tempo for each pitcher
tempo_pivot = df_retained.pivot_table(
    index="pitcher_id",
    columns="season",
    values="tempo_with_runners_on_base"
)

if 2022 in tempo_pivot.columns and 2023 in tempo_pivot.columns:
    tempo_pivot = tempo_pivot.dropna()
    tempo_pivot["delta"] = tempo_pivot[2023] - tempo_pivot[2022]
    
    print(f"\nWithin-Pitcher Changes (n={len(tempo_pivot)} retained T1):")
    print(f"  Mean Δtempo (2023-2022): {tempo_pivot['delta'].mean():+.3f}s")
    print(f"  Median Δtempo: {tempo_pivot['delta'].median():+.3f}s")
    print(f"  SD: {tempo_pivot['delta'].std():.3f}s")
    
    # t-test against zero
    t_stat, p_val = stats.ttest_1samp(tempo_pivot["delta"], 0)
    print(f"\n  One-sample t-test (H0: Δ=0): t={t_stat:.2f}, p={p_val:.3f}")
    if p_val < 0.05:
        direction = "slower" if tempo_pivot["delta"].mean() > 0 else "faster"
        print(f"  → Retained T1 pitchers got significantly {direction} in 2023")
    else:
        print("  → No significant within-pitcher change")

# ============================================================================
# COMPARE BASELINE TEMPO: EXITED vs RETAINED (2022)
# ============================================================================

print("\n" + "-" * 70)
print("STEP 4: SELECTION BIAS (2022 BASELINE)")
print("-" * 70)

df_t1_2022 = df[(df["season"] == 2022) & (df["baseline_group"] == "T1")].copy()

tempo_2022_retained = df_t1_2022[df_t1_2022["pitcher_id"].isin(retained)]["tempo_with_runners_on_base"]
tempo_2022_exited = df_t1_2022[df_t1_2022["pitcher_id"].isin(exited)]["tempo_with_runners_on_base"]

print(f"\n2022 Baseline Tempo (T1):")
print(f"  Retained→2023: {tempo_2022_retained.mean():.2f}s (n={len(tempo_2022_retained)})")
print(f"  Exited by 2023: {tempo_2022_exited.mean():.2f}s (n={len(tempo_2022_exited)})")
print(f"  Difference: {tempo_2022_exited.mean() - tempo_2022_retained.mean():+.2f}s")

if len(tempo_2022_retained) > 0 and len(tempo_2022_exited) > 0:
    t_stat, p_val = stats.ttest_ind(tempo_2022_exited, tempo_2022_retained, equal_var=False)
    print(f"\n  Welch t-test: t={t_stat:.2f}, p={p_val:.3f}")
    if p_val < 0.05:
        print("  → Exited pitchers had different baseline tempo")
    else:
        print("  → No selection bias in attrition")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "-" * 70)
print("SAVING RESULTS")
print("-" * 70)

# Summary table
summary_data = [
    {"metric": "t1_2022_total", "value": len(t1_2022_ids)},
    {"metric": "t1_2023_total", "value": len(t1_2023_ids)},
    {"metric": "retained", "value": len(retained)},
    {"metric": "exited", "value": len(exited)},
    {"metric": "new", "value": len(new)},
    {"metric": "retention_rate", "value": 100*len(retained)/len(t1_2022_ids)},
    {"metric": "tempo_2023_retained_mean", "value": tempo_retained.mean()},
    {"metric": "tempo_2023_new_mean", "value": tempo_new.mean()},
    {"metric": "tempo_diff_new_vs_retained", "value": tempo_new.mean() - tempo_retained.mean()},
]

if 2022 in tempo_pivot.columns and 2023 in tempo_pivot.columns:
    summary_data.extend([
        {"metric": "within_pitcher_delta_mean", "value": tempo_pivot["delta"].mean()},
        {"metric": "within_pitcher_delta_median", "value": tempo_pivot["delta"].median()}
    ])

df_summary = pd.DataFrame(summary_data)
output_file = OUTPUT_DIR / "t1_composition_summary.csv"
df_summary.to_csv(output_file, index=False)
print(f"✓ Saved: {output_file}")

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\n" + "-" * 70)
print("CREATING PLOTS")
print("-" * 70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Tempo distribution 2023 (retained vs new)
ax = axes[0]
ax.hist(tempo_retained, bins=20, alpha=0.6, label="Retained", edgecolor="black")
ax.hist(tempo_new, bins=20, alpha=0.6, label="New", edgecolor="black")
ax.axvline(tempo_retained.mean(), color="blue", linestyle="--", linewidth=2, 
           label=f"Retained mean: {tempo_retained.mean():.2f}s")
ax.axvline(tempo_new.mean(), color="orange", linestyle="--", linewidth=2,
           label=f"New mean: {tempo_new.mean():.2f}s")
ax.set_xlabel("2023 Tempo (seconds)", fontsize=11)
ax.set_ylabel("Number of Pitchers", fontsize=11)
ax.set_title("T1 Tempo Distribution (2023): Retained vs. New", fontsize=12, fontweight="bold")
ax.legend()
ax.grid(axis="y", alpha=0.3)

# Plot 2: Within-pitcher changes (retained only)
if 2022 in tempo_pivot.columns and 2023 in tempo_pivot.columns:
    ax = axes[1]
    ax.hist(tempo_pivot["delta"], bins=20, edgecolor="black", alpha=0.7)
    ax.axvline(0, color="red", linestyle="--", linewidth=2, label="No change")
    ax.axvline(tempo_pivot["delta"].mean(), color="blue", linestyle="--", linewidth=2,
               label=f"Mean: {tempo_pivot['delta'].mean():+.3f}s")
    ax.set_xlabel("Δ Tempo 2023-2022 (seconds)", fontsize=11)
    ax.set_ylabel("Number of Pitchers", fontsize=11)
    ax.set_title("Within-Pitcher Changes (Retained T1 only)", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
output_file = OUTPUT_DIR / "t1_composition_plots.png"
plt.savefig(output_file, dpi=300)
print(f"✓ Saved: {output_file}")
plt.close()

# ============================================================================
# INTERPRETATION
# ============================================================================

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

print("\nComposition Effect:")
if tempo_new.mean() > tempo_retained.mean() + 0.5:
    print("  ⚠ NEW pitchers in 2023 are SLOWER than retained")
    print("  → Composition explains part of T1 anomaly")
elif tempo_new.mean() < tempo_retained.mean() - 0.5:
    print("  ⚠ NEW pitchers in 2023 are FASTER than retained")
    print("  → Composition worsens the anomaly")
else:
    print("  ✓ NEW and RETAINED pitchers are similar")
    print("  → Composition does NOT explain T1 anomaly")

if 2022 in tempo_pivot.columns and 2023 in tempo_pivot.columns:
    print("\nWithin-Pitcher Effect:")
    if tempo_pivot["delta"].mean() > 0.5:
        print("  ⚠ RETAINED pitchers got slower 2022→2023")
        print("  → Within-pitcher effect confirms anomaly")
    elif tempo_pivot["delta"].mean() < -0.5:
        print("  ✓ RETAINED pitchers got faster 2022→2023")
        print("  → Within-pitcher effect is normal")
    else:
        print("  ~ RETAINED pitchers show minimal change")
        print("  → Anomaly might be composition + trends")

print(f"\nNext: Run b2_04b_t1_balanced.py for formal within-pitcher test")