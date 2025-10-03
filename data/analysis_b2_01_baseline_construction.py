"""
b2_01_baseline_construction.py
===============================
Constructs pitcher baseline tempo groups (terciles) from 2022 data.

Critical Design Decisions:
- Minimum 50 pitches with runners on base in 2022 (robustness: 100)
- Weighted mean: tempo22_w = Σ(tempo × pitches) / Σ(pitches)
- Terciles: T1=fast, T2=mid, T3=slow

Outputs:
- b2_baseline_groups.csv: pitcher_id, tempo22_w, pitches22, baseline_group
- b2_baseline_summary.csv: summary stats by group + filtering info
- b2_baseline_histogram.png: distribution with tercile cutpoints

QC Flags:
- Coverage ≥90% of pitches retained
- No NA/duplicates
- Stable terciles at 100-pitch cutoff
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FILE = "./analysis/analysis_pitcher_panel_relative.csv"
OUTPUT_DIR = Path("analysis/b2_baseline")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Critical thresholds
MIN_PITCHES_MAIN = 50      # Conservative Savant-style qualifier
MIN_PITCHES_ROBUST = 100   # Robustness check
BASELINE_YEAR = 2022

# Required columns
REQUIRED_COLS = [
    "pitcher_id",
    "season", 
    "tempo_with_runners_on_base",
    "pitches_with_runners_on_base"
]

# ============================================================================
# LOAD & VALIDATE
# ============================================================================

print("=" * 70)
print("B2-01: BASELINE CONSTRUCTION (2022 TERCILES)")
print("=" * 70)

# Load data
df = pd.read_csv(INPUT_FILE)
print(f"\n✓ Loaded {len(df):,} pitcher-season rows from {INPUT_FILE}")

# Validate required columns
missing_cols = [col for col in REQUIRED_COLS if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")
print(f"✓ All required columns present: {REQUIRED_COLS}")

# ============================================================================
# FILTER TO 2022
# ============================================================================

df_2022 = df[df["season"] == BASELINE_YEAR].copy()
print(f"\n✓ Filtered to {BASELINE_YEAR}: {len(df_2022):,} pitchers")

# Check for duplicates
dupes = df_2022["pitcher_id"].duplicated().sum()
if dupes > 0:
    print(f"⚠ WARNING: {dupes} duplicate pitcher_ids in 2022 - keeping first")
    df_2022 = df_2022.drop_duplicates(subset="pitcher_id", keep="first")

# ============================================================================
# COMPUTE WEIGHTED MEAN TEMPO
# ============================================================================

print("\n" + "-" * 70)
print("WEIGHTED MEAN COMPUTATION")
print("-" * 70)

# Drop rows with missing tempo or pitches
n_before = len(df_2022)
df_2022 = df_2022.dropna(subset=["tempo_with_runners_on_base", 
                                  "pitches_with_runners_on_base"])
n_after = len(df_2022)
if n_before > n_after:
    print(f"⚠ Dropped {n_before - n_after} rows with NA tempo/pitches")

# Compute weighted mean: Σ(tempo × pitches) / Σ(pitches)
df_2022["tempo22_w"] = df_2022["tempo_with_runners_on_base"]
df_2022["pitches22"] = df_2022["pitches_with_runners_on_base"]

print(f"\n✓ Tempo metric: weighted mean (exposure-weighted)")
print(f"  Formula: tempo22_w = Σ(tempo × pitches) / Σ(pitches)")
print(f"  Statcast definition: Median release-to-release (same batter, after takes)")

# ============================================================================
# APPLY MINIMUM PITCH CUTOFF
# ============================================================================

print("\n" + "-" * 70)
print(f"MINIMUM PITCH FILTER: ≥{MIN_PITCHES_MAIN} pitches")
print("-" * 70)

# Total pitches before filtering
total_pitches_all = df_2022["pitches22"].sum()
n_pitchers_all = len(df_2022)

# Apply main cutoff
df_main = df_2022[df_2022["pitches22"] >= MIN_PITCHES_MAIN].copy()
n_pitchers_main = len(df_main)
total_pitches_main = df_main["pitches22"].sum()

pct_pitchers_kept = 100 * n_pitchers_main / n_pitchers_all
pct_pitches_kept = 100 * total_pitches_main / total_pitches_all

print(f"\nPitchers kept: {n_pitchers_main:,} / {n_pitchers_all:,} "
      f"({pct_pitchers_kept:.1f}%)")
print(f"Pitches covered: {total_pitches_main:,} / {total_pitches_all:,} "
      f"({pct_pitches_kept:.1f}%)")

# QC Check: Coverage ≥90%
if pct_pitches_kept < 90:
    print(f"\n⚠ WARNING: Coverage {pct_pitches_kept:.1f}% < 90% threshold")
else:
    print(f"\n✓ QC PASS: Coverage {pct_pitches_kept:.1f}% ≥ 90%")

# ============================================================================
# CREATE TERCILES
# ============================================================================

print("\n" + "-" * 70)
print("TERCILE ASSIGNMENT")
print("-" * 70)

# Compute tercile cutpoints
tercile_cuts = df_main["tempo22_w"].quantile([1/3, 2/3]).values
print(f"\nTercile cutpoints:")
print(f"  T1 (fast):  tempo22_w < {tercile_cuts[0]:.2f}s")
print(f"  T2 (mid):   {tercile_cuts[0]:.2f}s ≤ tempo22_w < {tercile_cuts[1]:.2f}s")
print(f"  T3 (slow):  tempo22_w ≥ {tercile_cuts[1]:.2f}s")

# Assign groups
df_main["baseline_group"] = pd.cut(
    df_main["tempo22_w"],
    bins=[-np.inf, tercile_cuts[0], tercile_cuts[1], np.inf],
    labels=["T1", "T2", "T3"]
)

# Verify approximately equal sizes
group_sizes = df_main["baseline_group"].value_counts().sort_index()
print(f"\nGroup sizes:")
for group, n in group_sizes.items():
    print(f"  {group}: {n:,} pitchers ({100*n/len(df_main):.1f}%)")

# ============================================================================
# ROBUSTNESS: 100-PITCH CUTOFF
# ============================================================================

print("\n" + "-" * 70)
print(f"ROBUSTNESS: ≥{MIN_PITCHES_ROBUST} pitches cutoff")
print("-" * 70)

df_robust = df_2022[df_2022["pitches22"] >= MIN_PITCHES_ROBUST].copy()
tercile_cuts_robust = df_robust["tempo22_w"].quantile([1/3, 2/3]).values

print(f"\nRobust tercile cutpoints:")
print(f"  T1: < {tercile_cuts_robust[0]:.2f}s (Main: {tercile_cuts[0]:.2f}s)")
print(f"  T2/T3: {tercile_cuts_robust[1]:.2f}s (Main: {tercile_cuts[1]:.2f}s)")

# Check stability
diff_33 = abs(tercile_cuts[0] - tercile_cuts_robust[0])
diff_67 = abs(tercile_cuts[1] - tercile_cuts_robust[1])

if diff_33 < 0.5 and diff_67 < 0.5:
    print(f"\n✓ QC PASS: Cutpoints stable (diff < 0.5s)")
else:
    print(f"\n⚠ WARNING: Cutpoints differ by {max(diff_33, diff_67):.2f}s")

# ============================================================================
# SAVE MAIN OUTPUT
# ============================================================================

print("\n" + "-" * 70)
print("SAVING OUTPUTS")
print("-" * 70)

# 1. Baseline groups CSV
output_groups = df_main[["pitcher_id", "tempo22_w", "pitches22", 
                         "baseline_group"]].copy()
output_file_groups = OUTPUT_DIR / "b2_baseline_groups.csv"
output_groups.to_csv(output_file_groups, index=False)
print(f"\n✓ Saved: {output_file_groups}")
print(f"  Columns: pitcher_id, tempo22_w, pitches22, baseline_group")
print(f"  Rows: {len(output_groups):,}")

# 2. Summary statistics CSV
summary_data = []

# Overall stats
summary_data.append({
    "metric": "total_pitchers_2022",
    "value": n_pitchers_all,
    "notes": "Before minimum pitch filter"
})
summary_data.append({
    "metric": "total_pitches_2022",
    "value": total_pitches_all,
    "notes": "Before minimum pitch filter"
})
summary_data.append({
    "metric": "min_pitch_cutoff",
    "value": MIN_PITCHES_MAIN,
    "notes": "Main analysis threshold"
})
summary_data.append({
    "metric": "pitchers_retained",
    "value": n_pitchers_main,
    "notes": f"{pct_pitchers_kept:.1f}% of total"
})
summary_data.append({
    "metric": "pitches_retained",
    "value": total_pitches_main,
    "notes": f"{pct_pitches_kept:.1f}% of total (QC: ≥90%)"
})

# Tercile cutpoints
summary_data.append({
    "metric": "tercile_cut_33",
    "value": round(tercile_cuts[0], 2),
    "notes": "T1/T2 boundary (seconds)"
})
summary_data.append({
    "metric": "tercile_cut_67",
    "value": round(tercile_cuts[1], 2),
    "notes": "T2/T3 boundary (seconds)"
})

# Group-level stats
for group in ["T1", "T2", "T3"]:
    group_data = df_main[df_main["baseline_group"] == group]
    summary_data.append({
        "metric": f"{group}_n_pitchers",
        "value": len(group_data),
        "notes": f"{100*len(group_data)/len(df_main):.1f}% of sample"
    })
    summary_data.append({
        "metric": f"{group}_mean_tempo",
        "value": round(group_data["tempo22_w"].mean(), 2),
        "notes": "Weighted mean (seconds)"
    })
    summary_data.append({
        "metric": f"{group}_median_tempo",
        "value": round(group_data["tempo22_w"].median(), 2),
        "notes": "Median (seconds)"
    })
    summary_data.append({
        "metric": f"{group}_total_pitches",
        "value": int(group_data["pitches22"].sum()),
        "notes": "Sum of exposure"
    })

df_summary = pd.DataFrame(summary_data)
output_file_summary = OUTPUT_DIR / "b2_baseline_summary.csv"
df_summary.to_csv(output_file_summary, index=False)
print(f"\n✓ Saved: {output_file_summary}")
print(f"  Metrics: {len(df_summary)} rows")

# ============================================================================
# CREATE VISUALIZATION
# ============================================================================

print("\n" + "-" * 70)
print("CREATING HISTOGRAM")
print("-" * 70)

fig, ax = plt.subplots(figsize=(10, 6))

# Histogram
ax.hist(df_main["tempo22_w"], bins=50, edgecolor="black", alpha=0.7)

# Add tercile lines
ax.axvline(tercile_cuts[0], color="red", linestyle="--", linewidth=2,
           label=f"T1/T2: {tercile_cuts[0]:.2f}s")
ax.axvline(tercile_cuts[1], color="red", linestyle="--", linewidth=2,
           label=f"T2/T3: {tercile_cuts[1]:.2f}s")

# Labels
ax.set_xlabel("2022 Weighted Mean Tempo (Runner On, seconds)", fontsize=12)
ax.set_ylabel("Number of Pitchers", fontsize=12)
ax.set_title(f"B2 Baseline Distribution (n={len(df_main):,}, ≥{MIN_PITCHES_MAIN} pitches)",
             fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)

# Save
output_file_hist = OUTPUT_DIR / "b2_baseline_histogram.png"
plt.tight_layout()
plt.savefig(output_file_hist, dpi=300)
print(f"\n✓ Saved: {output_file_hist}")
plt.close()

# ============================================================================
# FINAL QC SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("QC SUMMARY")
print("=" * 70)

qc_checks = [
    ("Coverage ≥90%", pct_pitches_kept >= 90),
    ("No NA values", df_main[["tempo22_w", "pitches22"]].isna().sum().sum() == 0),
    ("No duplicate pitcher_ids", df_main["pitcher_id"].duplicated().sum() == 0),
    ("Terciles stable (50 vs 100)", diff_33 < 0.5 and diff_67 < 0.5),
    ("Groups approximately equal", all(abs(n/len(df_main) - 1/3) < 0.05 
                                       for n in group_sizes.values))
]

for check_name, passed in qc_checks:
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"{status}: {check_name}")

all_passed = all(passed for _, passed in qc_checks)
if all_passed:
    print("\n" + "=" * 70)
    print("✓✓✓ ALL QC CHECKS PASSED - BASELINE READY FOR B2 ANALYSES ✓✓✓")
    print("=" * 70)
else:
    print("\n⚠⚠⚠ SOME QC CHECKS FAILED - REVIEW BEFORE PROCEEDING ⚠⚠⚠")

print(f"\nNext step: Run b2_02_attrition_analysis.py")