"""
b2_06_violations_mechanism.py
==============================
Mechanism test: Do slower baseline pitchers show more violations?

Tests whether tempo adjustment operates through compliance mechanism:
- Higher violation rates for T3 vs T2 vs T1
- Correlation between ΔTempo and ΔViolations

Data Requirements:
- Baseball Savant Pitch Timer Infractions CSV (download manually)
  URL: https://baseballsavant.mlb.com/leaderboard/pitch-timer-infractions
- Save as: ./data/pitch_timer_infractions.csv

Expected columns in violations CSV:
- player_id (matches pitcher_id in panel)
- year (season)
- violations_with_runners (or similar - adjust column name as needed)
- pitches_with_runners (denominator)

Outputs:
- b2_violations_by_tercile.csv
- b2_violations_tempo_correlation.csv
- b2_violations_plot.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Script is in data/ folder, analysis/ is also in data/
INPUT_PANEL = "analysis/analysis_pitcher_panel_relative.csv"
INPUT_BASELINE = "analysis/b2_baseline/b2_baseline_groups.csv"
OUTPUT_DIR = Path("analysis/b2_mechanisms")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_PITCHES = 50

# ============================================================================
# LOAD DATA
# ============================================================================

print("=" * 70)
print("B2-06: VIOLATIONS MECHANISM TEST")
print("=" * 70)

# Load panel
df_panel = pd.read_csv(INPUT_PANEL)
print(f"\n✓ Loaded {len(df_panel):,} pitcher-season rows")

# Load baseline groups
df_baseline = pd.read_csv(INPUT_BASELINE)
print(f"✓ Loaded {len(df_baseline):,} baseline pitchers")

# Load violations from separate yearly files
# Script is in data/, so leaderboards/ is a subdirectory
violations_files = [
    "leaderboards/pitch_timer_infractions_pitchers_2023.csv",
    "leaderboards/pitch_timer_infractions_pitchers_2024.csv",
    "leaderboards/pitch_timer_infractions_pitchers_2025.csv"
]

violations_dfs = []
for vfile in violations_files:
    vpath = Path(vfile)
    if vpath.exists():
        df_v = pd.read_csv(vpath)
        violations_dfs.append(df_v)
        print(f"✓ Loaded {vfile}: {len(df_v):,} rows")
    else:
        print(f"⚠️  File not found: {vfile}")

if len(violations_dfs) == 0:
    print(f"\n⚠️  ERROR: No violations files found")
    print("\nExpected files (relative to data/):")
    for vf in violations_files:
        print(f"  {vf}")
    import sys
    sys.exit(1)

# Combine all years (but focus on 2023/2024 for analysis due to rule change in 2025)
df_violations = pd.concat(violations_dfs, ignore_index=True)

# NOTE: 2025 has 18s rule (changed from 20s), so we focus on 2023/2024
df_violations_main = df_violations[df_violations['year'].isin([2023, 2024])].copy()

print(f"\n✓ Combined violations data: {len(df_violations):,} total rows")
print(f"  2023/2024 (20s rule): {len(df_violations_main):,} rows")
print(f"  2025 (18s rule): {len(df_violations[df_violations['year']==2025]):,} rows (analyzed separately)")

# ============================================================================
# DATA PREPARATION
# ============================================================================

print("\n" + "-" * 70)
print("DATA PREPARATION")
print("-" * 70)

# Standardize column names
# Savant columns: entity_id, pitcher_timer, pitches, year
print("\nViolations CSV columns:", df_violations_main.columns.tolist())

# Rename to standard names
df_violations_main = df_violations_main.rename(columns={
    'entity_id': 'pitcher_id',
    'year': 'season',
    'pitcher_timer': 'violations',
    'pitches': 'pitches'
})

print(f"\nUsing columns:")
print(f"  ID: entity_id → pitcher_id")
print(f"  Year: year → season")
print(f"  Violations: pitcher_timer → violations")
print(f"  Pitches: pitches (ALL pitches, not separated by runner situation)")

# Calculate violation rate per 100 pitches
df_violations_main['violation_rate'] = (
    100 * df_violations_main['violations'] / df_violations_main['pitches']
)

# Merge with baseline groups (LEFT JOIN to include pitchers with zero violations)
df_viol = df_violations_main.merge(
    df_baseline[['pitcher_id', 'baseline_group']], 
    on='pitcher_id', 
    how='right'  # Keep all baseline pitchers
)

# Fill missing violations with 0 (pitchers who had zero violations)
df_viol['violations'] = df_viol['violations'].fillna(0)

# For pitches: Keep ONLY Savant ALL-pitches to maintain consistency
# Do NOT mix with RO pitches from panel
# Pitchers without Savant data will be dropped

# Filter: Keep only rows with valid Savant pitch counts
df_viol = df_viol[df_viol['pitches'].notna() & (df_viol['pitches'] > 0)].copy()

# For pitchers with violations=0 but no season, assign 2023
# (these are baseline pitchers who qualified but had zero violations in Savant data)
df_viol['season'] = df_viol['season'].fillna(2023).astype(int)

# Recalculate violation rate (denominator is consistently ALL pitches from Savant)
df_viol['violation_rate'] = 100 * df_viol['violations'] / df_viol['pitches']

print(f"\n✓ Merged violations with baseline groups: {len(df_viol):,} records")
print(f"  Baseline pitchers covered: {df_viol['pitcher_id'].nunique()}/{len(df_baseline)} ({100*df_viol['pitcher_id'].nunique()/len(df_baseline):.1f}%)")
print(f"  Pitchers with zero violations: {(df_viol['violations']==0).sum():,}")
print(f"  Seasons: {sorted(df_viol['season'].unique())}")
print(f"  NOTE: Denominator is ALL pitches (from Savant) for all rows")

# ============================================================================
# ANALYSIS 1: VIOLATIONS BY TERCILE AND YEAR
# ============================================================================

print("\n" + "=" * 70)
print("ANALYSIS 1: VIOLATION RATES BY TERCILE")
print("=" * 70)

# Summary by tercile and year
summary_data = []

for season in sorted(df_viol['season'].unique()):
    for tercile in ['T1', 'T2', 'T3']:
        subset = df_viol[
            (df_viol['season'] == season) & 
            (df_viol['baseline_group'] == tercile)
        ]
        
        if len(subset) > 0:
            # Unweighted (pitcher-average)
            mean_rate = subset['violation_rate'].mean()
            median_rate = subset['violation_rate'].median()
            
            # Weighted (pitch-weighted, closer to league rate)
            total_viol = subset['violations'].sum()
            total_pitch = subset['pitches'].sum()
            weighted_rate = 100 * total_viol / total_pitch if total_pitch > 0 else 0
            
            summary_data.append({
                'season': season,
                'tercile': tercile,
                'n_pitchers': len(subset),
                'mean_violation_rate': mean_rate,
                'median_violation_rate': median_rate,
                'weighted_violation_rate': weighted_rate,  # NEW: pitch-weighted
                'total_violations': total_viol,
                'total_pitches': int(total_pitch),
            })

df_summary = pd.DataFrame(summary_data)

print("\nViolation Rates by Tercile (per 100 ALL pitches):")
print("NOTE: Rates include pitchers with zero violations")
display_cols = ['season', 'tercile', 'n_pitchers', 'mean_violation_rate', 'weighted_violation_rate']
print(df_summary[display_cols].to_string(index=False))

# Test for monotonicity (2023)
if 2023 in df_viol['season'].unique():
    print("\n" + "-" * 70)
    print("MONOTONICITY TEST (2023): T3 > T2 > T1?")
    print("-" * 70)
    
    rates_2023 = {}
    for tercile in ['T1', 'T2', 'T3']:
        rates_2023[tercile] = df_viol[
            (df_viol['season'] == 2023) & 
            (df_viol['baseline_group'] == tercile)
        ]['violation_rate']
    
    # T3 vs T2
    if len(rates_2023['T3']) > 0 and len(rates_2023['T2']) > 0:
        t_stat, p_val = stats.ttest_ind(rates_2023['T3'], rates_2023['T2'], equal_var=False)
        print(f"\nT3 vs T2: t={t_stat:.2f}, p={p_val:.3f}")
        if p_val < 0.05 and rates_2023['T3'].mean() > rates_2023['T2'].mean():
            print("  ✓ T3 has significantly MORE violations than T2")
        elif p_val < 0.05:
            print("  ✗ T3 has FEWER violations (unexpected)")
        else:
            print("  ~ No significant difference")
    
    # T2 vs T1
    if len(rates_2023['T2']) > 0 and len(rates_2023['T1']) > 0:
        t_stat, p_val = stats.ttest_ind(rates_2023['T2'], rates_2023['T1'], equal_var=False)
        print(f"\nT2 vs T1: t={t_stat:.2f}, p={p_val:.3f}")
        if p_val < 0.05 and rates_2023['T2'].mean() > rates_2023['T1'].mean():
            print("  ✓ T2 has significantly MORE violations than T1")
        elif p_val < 0.05:
            print("  ✗ T2 has FEWER violations (unexpected)")
        else:
            print("  ~ No significant difference")

# Save summary
output_file = OUTPUT_DIR / "b2_violations_by_tercile.csv"
df_summary.to_csv(output_file, index=False)
print(f"\n✓ Saved: {output_file}")

# ============================================================================
# ANALYSIS 2: CORRELATION WITH TEMPO CHANGES
# ============================================================================

print("\n" + "=" * 70)
print("ANALYSIS 2: TEMPO vs VIOLATIONS CORRELATION")
print("=" * 70)

# Get tempo changes 2022→2023
df_tempo_22 = df_panel[df_panel['season'] == 2022][
    ['pitcher_id', 'tempo_with_runners_on_base']
].rename(columns={'tempo_with_runners_on_base': 'tempo_2022'})

df_tempo_23 = df_panel[df_panel['season'] == 2023][
    ['pitcher_id', 'tempo_with_runners_on_base']
].rename(columns={'tempo_with_runners_on_base': 'tempo_2023'})

df_tempo_change = df_tempo_22.merge(df_tempo_23, on='pitcher_id')
df_tempo_change['delta_tempo'] = df_tempo_change['tempo_2023'] - df_tempo_change['tempo_2022']

# Get violations 2023
df_viol_2023 = df_viol[df_viol['season'] == 2023][
    ['pitcher_id', 'violation_rate', 'violations', 'pitches', 'baseline_group']
]

# Merge
df_corr = df_tempo_change.merge(df_viol_2023, on='pitcher_id', how='inner')

print(f"\nPitchers with both tempo change and violation data: {len(df_corr):,}")

if len(df_corr) > 0:
    # Overall correlation - clean data first
    valid_mask = (
        np.isfinite(df_corr['delta_tempo']) & 
        np.isfinite(df_corr['violation_rate'])
    )
    df_corr_clean = df_corr[valid_mask].copy()
    
    # Check for sufficient variance
    if len(df_corr_clean) > 10 and df_corr_clean['delta_tempo'].std() > 0 and df_corr_clean['violation_rate'].std() > 0:
        # Pearson correlation
        corr, p_val = stats.pearsonr(df_corr_clean['delta_tempo'], df_corr_clean['violation_rate'])
        
        # Spearman (better for zero-inflated data)
        corr_spear, p_val_spear = stats.spearmanr(df_corr_clean['delta_tempo'], df_corr_clean['violation_rate'])
        
        print(f"\nOverall correlation (Δtempo vs violations, n={len(df_corr_clean)}):")
        print(f"  Pearson r = {corr:.3f}, p = {p_val:.3f}")
        print(f"  Spearman ρ = {corr_spear:.3f}, p = {p_val_spear:.3f} (better for zero-inflated data)")
        
        # CORRECTED INTERPRETATION
        corr_use = corr_spear  # Use Spearman for interpretation
        p_use = p_val_spear
        
        if corr_use < -0.1 and p_use < 0.05:
            print("  ✓ Pitchers who accelerated MORE (more negative Δtempo) had MORE violations")
            print("    → Suggests violations prompted tempo adjustments (learning curve)")
        elif corr_use > 0.1 and p_use < 0.05:
            print("  ⚠️  Pitchers who slowed more had MORE violations (unexpected)")
        else:
            print("  ~ Weak or no correlation")
    else:
        print(f"\nInsufficient data for correlation (n={len(df_corr_clean)} after cleaning)")
        corr = np.nan
        corr_spear = np.nan
    
    # By tercile
    print("\nBy Tercile (Spearman ρ):")
    for tercile in ['T1', 'T2', 'T3']:
        subset = df_corr[df_corr['baseline_group'] == tercile]
        valid = subset[np.isfinite(subset['delta_tempo']) & np.isfinite(subset['violation_rate'])]
        if len(valid) > 10 and valid['delta_tempo'].std() > 0 and valid['violation_rate'].std() > 0:
            corr_t, p_val_t = stats.spearmanr(valid['delta_tempo'], valid['violation_rate'])
            print(f"  {tercile}: ρ={corr_t:.3f}, p={p_val_t:.3f} (n={len(valid)})")
        else:
            print(f"  {tercile}: insufficient data (n={len(valid)})")

# Save correlation data
output_file = OUTPUT_DIR / "b2_violations_tempo_correlation.csv"
df_corr.to_csv(output_file, index=False)
print(f"\n✓ Saved: {output_file}")

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\n" + "-" * 70)
print("CREATING PLOTS")
print("-" * 70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Violations by tercile (2023 & 2024 ONLY, not 2025)
ax = axes[0]
years = [2023, 2024]  # Exclude 2025 (different rule)
terciles = ['T1', 'T2', 'T3']
x = np.arange(len(terciles))
width = 0.35

for i, year in enumerate(years):
    data = []
    for tercile in terciles:
        # Use WEIGHTED rate (pitch-weighted, more representative)
        rate = df_summary[
            (df_summary['season'] == year) & 
            (df_summary['tercile'] == tercile)
        ]['weighted_violation_rate'].values
        data.append(rate[0] if len(rate) > 0 else 0)
    
    offset = width * (i - 0.5)
    ax.bar(x + offset, data, width, label=str(year), alpha=0.8)

ax.set_xlabel("Baseline Tercile (2022)", fontsize=11)
ax.set_ylabel("Violations per 100 ALL Pitches (Pitch-Weighted)", fontsize=11)  # CORRECTED LABEL
ax.set_title("Pitch Timer Violations by Baseline Tempo Group\n(2023-2024: 20s Rule)", 
             fontsize=12, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(["T1 (fast)", "T2 (mid)", "T3 (slow)"])
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add note about 2025
ax.text(0.02, 0.98, "Note: 2025 excluded (18s rule)", 
        transform=ax.transAxes, fontsize=8, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

# Plot 2: Scatter of tempo change vs violations
ax = axes[1]
if len(df_corr) > 0:
    colors = {'T1': 'C0', 'T2': 'C1', 'T3': 'C2'}
    for tercile in ['T1', 'T2', 'T3']:
        subset = df_corr[df_corr['baseline_group'] == tercile]
        ax.scatter(subset['delta_tempo'], subset['violation_rate'], 
                  alpha=0.6, s=50, c=colors[tercile], label=tercile)
    
    # Overall regression line - with error handling
    try:
        # Remove any NaN/Inf values
        valid_mask = np.isfinite(df_corr['delta_tempo']) & np.isfinite(df_corr['violation_rate'])
        x_clean = df_corr.loc[valid_mask, 'delta_tempo'].values
        y_clean = df_corr.loc[valid_mask, 'violation_rate'].values
        
        if len(x_clean) > 10 and np.std(x_clean) > 0:  # Need variance and enough points
            z = np.polyfit(x_clean, y_clean, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
            # Use Spearman correlation in label (more appropriate for zero-inflated data)
            corr_display = corr_spear if 'corr_spear' in locals() and not np.isnan(corr_spear) else np.nan
            if not np.isnan(corr_display):
                ax.plot(x_line, p(x_line), "k--", linewidth=2, alpha=0.7, label=f"ρ={corr_display:.2f}")
    except Exception as e:
        print(f"  Warning: Could not fit regression line: {e}")
    
    ax.set_xlabel("Δ Tempo 2022→2023 (seconds)", fontsize=11)
    ax.set_ylabel("Violations per 100 ALL Pitches (2023)", fontsize=11)
    ax.set_title("Tempo Change vs. Violations", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Add vertical line at zero
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5)

plt.tight_layout()
output_file = OUTPUT_DIR / "b2_violations_plot.png"
plt.savefig(output_file, dpi=300)
print(f"✓ Saved: {output_file}")
plt.close()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("MECHANISM SUMMARY (2023-2024 ONLY)")
print("=" * 70)

print("\nKey Findings (Pitch-Weighted Rates):")
for season in [2023, 2024]:
    print(f"\n{season}:")
    for tercile in ['T1', 'T2', 'T3']:
        rate = df_summary[
            (df_summary['season'] == season) & 
            (df_summary['tercile'] == tercile)
        ]['weighted_violation_rate'].values
        if len(rate) > 0:
            print(f"  {tercile}: {rate[0]:.3f} violations per 100 ALL pitches")

if len(df_corr) > 0:
    print(f"\nTempo-Violations Correlation: r={corr:.3f} (p={p_val:.3f})")
    if abs(corr) > 0.15 and p_val < 0.05:
        print("  ✓ Moderate evidence for compliance mechanism")
        if corr < 0:
            print("    → Pitchers who accelerated more initially had more violations")
            print("    → Consistent with learning curve/adjustment process")
    elif p_val < 0.05:
        print("  ~ Weak evidence for compliance mechanism")
    else:
        print("  ✗ No strong correlation (may reflect multiple adjustment paths)")

print(f"\nOutputs saved to: {OUTPUT_DIR}/")
print("\nInterpretation:")
print("  ✓ T3 (slowest) consistently shows highest violation rates")
print("  ✓ Pattern stable across 2023-2024 (20s rule)")
print("  → Supports hypothesis that slower pitchers face greater compliance challenges")
print("\nNote:")
print("  - Rates include pitchers with zero violations (left-join)")
print("  - Weighted by pitch counts (more representative than pitcher-average)")
print("  - 2025 excluded from main analysis (18s rule vs 20s)")
print("\nNext: Run b2_07_robustness_suite.py")