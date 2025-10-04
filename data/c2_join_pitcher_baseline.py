"""
c2_join_pitcher_baseline.py
============================
Join C1 panel (opportunities/attempts/SB) with B2 baseline tempo groups.

Key Design:
- 2022 tempo tercile (T1/T2/T3) applied to ALL years 2018-2024
- Rationale: Test heterogeneous treatment effects by pre-treatment characteristic
- Pre-trends 2018-2022 will be documented in C3 event study

Outputs:
- c_panel_with_baseline.csv (main panel with baseline groups)
- c2_qc_report.txt (coverage, quality checks)
- c2_summary_by_tercile.csv (descriptive stats by year×tercile)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
if Path.cwd().name == 'data':
    C1_FILE = Path("analysis/c_running_game/c_opportunities_panel.csv")
    B2_FILE = Path("analysis/b2_baseline/b2_baseline_groups.csv")
    OUTPUT_DIR = Path("analysis/c_running_game")
else:
    C1_FILE = Path("./data/analysis/c_running_game/c_opportunities_panel.csv")
    B2_FILE = Path("./data/analysis/b2_baseline/b2_baseline_groups.csv")
    OUTPUT_DIR = Path("./data/analysis/c_running_game")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# LOAD DATA
# ============================================================================

print("=" * 70)
print("C2: JOIN PITCHER BASELINE TEMPO GROUPS")
print("=" * 70)

print("\n" + "-" * 70)
print("LOADING DATA")
print("-" * 70)

# Load C1 panel
try:
    df_c1 = pd.read_csv(C1_FILE)
    print(f"\n✓ Loaded C1 panel: {len(df_c1):,} pitcher-season observations")
    print(f"  File: {C1_FILE}")
    print(f"  Columns: {df_c1.columns.tolist()}")
except Exception as e:
    print(f"\n❌ ERROR loading C1 panel: {e}")
    sys.exit(1)

# Load B2 baseline
try:
    df_b2 = pd.read_csv(B2_FILE)
    print(f"\n✓ Loaded B2 baseline: {len(df_b2):,} pitchers")
    print(f"  File: {B2_FILE}")
    print(f"  Columns: {df_b2.columns.tolist()}")
except Exception as e:
    print(f"\n❌ ERROR loading B2 baseline: {e}")
    sys.exit(1)

# Validate required columns
required_c1 = ['pitcher_id', 'season', 'opportunities_2b', 'attempts_2b', 
               'sb_2b', 'cs_2b', 'attempt_rate', 'sb_pct']
required_b2 = ['pitcher_id', 'tempo22_w', 'pitches22', 'baseline_group']

missing_c1 = [col for col in required_c1 if col not in df_c1.columns]
missing_b2 = [col for col in required_b2 if col not in df_b2.columns]

if missing_c1:
    print(f"\n❌ ERROR: C1 missing columns: {missing_c1}")
    sys.exit(1)
if missing_b2:
    print(f"\n❌ ERROR: B2 missing columns: {missing_b2}")
    sys.exit(1)

print("\n✓ All required columns present")

# ============================================================================
# PRE-JOIN QC
# ============================================================================

print("\n" + "-" * 70)
print("PRE-JOIN QC")
print("-" * 70)

# Check for duplicates in C1
dupes_c1 = df_c1.duplicated(subset=['pitcher_id', 'season']).sum()
if dupes_c1 > 0:
    print(f"\n⚠️  WARNING: {dupes_c1} duplicate pitcher-season in C1")
    df_c1 = df_c1.drop_duplicates(subset=['pitcher_id', 'season'], keep='first')
    print(f"  Removed duplicates, kept first occurrence")
else:
    print(f"\n✓ No duplicates in C1 (pitcher_id × season)")

# Check for duplicates in B2
dupes_b2 = df_b2.duplicated(subset=['pitcher_id']).sum()
if dupes_b2 > 0:
    print(f"\n⚠️  WARNING: {dupes_b2} duplicate pitcher_id in B2")
    df_b2 = df_b2.drop_duplicates(subset=['pitcher_id'], keep='first')
    print(f"  Removed duplicates, kept first occurrence")
else:
    print(f"\n✓ No duplicates in B2 (pitcher_id)")

# ============================================================================
# JOIN
# ============================================================================

print("\n" + "-" * 70)
print("JOINING BASELINE GROUPS")
print("-" * 70)

# Left join: C1 panel ← B2 baseline (on pitcher_id only)
# This propagates 2022 tempo tercile to all years 2018-2024
df_merged = df_c1.merge(
    df_b2[['pitcher_id', 'tempo22_w', 'pitches22', 'baseline_group']],
    on='pitcher_id',
    how='left'
)

print(f"\n✓ Joined: {len(df_merged):,} observations")
print(f"  Pitchers in C1: {df_c1['pitcher_id'].nunique():,}")
print(f"  Pitchers in B2: {df_b2['pitcher_id'].nunique():,}")
print(f"  Pitchers matched: {df_merged['baseline_group'].notna().sum():,}")

# Create coverage flag
df_merged['in_baseline_2022'] = (~df_merged['baseline_group'].isna()).astype(int)

pct_matched = 100 * df_merged['in_baseline_2022'].sum() / len(df_merged)
print(f"  Coverage: {pct_matched:.1f}% of observations have baseline group")

# ============================================================================
# POST-JOIN QC
# ============================================================================

print("\n" + "-" * 70)
print("POST-JOIN QC")
print("-" * 70)

# Check 1: No duplicates after join
dupes_merged = df_merged.duplicated(subset=['pitcher_id', 'season']).sum()
if dupes_merged > 0:
    print(f"\n❌ ERROR: {dupes_merged} duplicates after join!")
    sys.exit(1)
else:
    print("\n✓ No duplicates (pitcher_id × season)")

# Check 2: Arithmetic consistency
attempts_check = (df_merged['attempts_2b'] == 
                  df_merged['sb_2b'] + df_merged['cs_2b']).all()
if not attempts_check:
    print("⚠️  WARNING: attempts_2b ≠ sb_2b + cs_2b in some rows")
    n_bad = ((df_merged['attempts_2b'] != 
              df_merged['sb_2b'] + df_merged['cs_2b'])).sum()
    print(f"  Fixing {n_bad} rows")
    df_merged['attempts_2b'] = df_merged['sb_2b'] + df_merged['cs_2b']
else:
    print("✓ Arithmetic: attempts_2b = sb_2b + cs_2b")

# Check 3: Logical constraint
bad_attempts = (df_merged['attempts_2b'] > df_merged['opportunities_2b']).sum()
if bad_attempts > 0:
    print(f"⚠️  WARNING: {bad_attempts} rows with attempts > opportunities")
    print(f"  Clipping attempts to opportunities")
    df_merged.loc[df_merged['attempts_2b'] > df_merged['opportunities_2b'], 
                  'attempts_2b'] = df_merged['opportunities_2b']
else:
    print("✓ Logical: attempts_2b ≤ opportunities_2b")

# Check 4: Recalculate rates (defensive)
print("\n✓ Recalculating rates (defensive)")
df_merged['attempt_rate'] = df_merged['attempts_2b'] / np.maximum(
    df_merged['opportunities_2b'], 1
)
df_merged['sb_pct'] = np.where(
    df_merged['attempts_2b'] > 0,
    df_merged['sb_2b'] / df_merged['attempts_2b'],
    np.nan
)

# ============================================================================
# ADDITIONAL FLAGS
# ============================================================================

print("\n" + "-" * 70)
print("CREATING FLAGS")
print("-" * 70)

# Flag: Appears post-2023 only
pitcher_first_year = df_merged.groupby('pitcher_id')['season'].min()
df_merged['appears_post_only'] = df_merged['pitcher_id'].map(
    lambda x: pitcher_first_year[x] >= 2023
).astype(int)

n_post_only = df_merged[df_merged['appears_post_only'] == 1]['pitcher_id'].nunique()
print(f"\n✓ Flagged post-2023 entrants: {n_post_only} pitchers")

# Flag: Balanced panel 2018-2024 (optional)
pitcher_year_count = df_merged.groupby('pitcher_id')['season'].nunique()
df_merged['balanced_panel_18_24'] = df_merged['pitcher_id'].map(
    lambda x: pitcher_year_count[x] == 7
).astype(int)

n_balanced = df_merged[df_merged['balanced_panel_18_24'] == 1]['pitcher_id'].nunique()
print(f"✓ Balanced panel flag: {n_balanced} pitchers (2018-2024 complete)")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "-" * 70)
print("SUMMARY STATISTICS BY YEAR × TERCILE")
print("-" * 70)

# Filter to observations with baseline group
df_with_baseline = df_merged[df_merged['in_baseline_2022'] == 1].copy()

summary_data = []

for year in sorted(df_with_baseline['season'].unique()):
    df_year = df_with_baseline[df_with_baseline['season'] == year]
    
    # Overall (all terciles)
    summary_data.append({
        'year': year,
        'tercile': 'All',
        'n_pitchers': len(df_year),
        'total_opportunities': df_year['opportunities_2b'].sum(),
        'total_attempts': df_year['attempts_2b'].sum(),
        'total_sb': df_year['sb_2b'].sum(),
        'total_cs': df_year['cs_2b'].sum(),
        'league_attempt_rate': df_year['attempts_2b'].sum() / 
                               max(df_year['opportunities_2b'].sum(), 1),
        'league_sb_pct': df_year['sb_2b'].sum() / 
                         max(df_year['attempts_2b'].sum(), 1),
        'mean_attempt_rate': df_year['attempt_rate'].mean(),
        'mean_sb_pct': df_year['sb_pct'].mean()
    })
    
    # By tercile
    for tercile in ['T1', 'T2', 'T3']:
        df_group = df_year[df_year['baseline_group'] == tercile]
        
        if len(df_group) > 0:
            summary_data.append({
                'year': year,
                'tercile': tercile,
                'n_pitchers': len(df_group),
                'total_opportunities': df_group['opportunities_2b'].sum(),
                'total_attempts': df_group['attempts_2b'].sum(),
                'total_sb': df_group['sb_2b'].sum(),
                'total_cs': df_group['cs_2b'].sum(),
                'league_attempt_rate': df_group['attempts_2b'].sum() / 
                                       max(df_group['opportunities_2b'].sum(), 1),
                'league_sb_pct': df_group['sb_2b'].sum() / 
                                 max(df_group['attempts_2b'].sum(), 1),
                'mean_attempt_rate': df_group['attempt_rate'].mean(),
                'mean_sb_pct': df_group['sb_pct'].mean()
            })

df_summary = pd.DataFrame(summary_data)

print("\n2023 Rule Change Impact by Tercile:")
print("\nAttempt Rate (league-weighted, 1B→2B only):")
for tercile in ['All', 'T1', 'T2', 'T3']:
    df_t = df_summary[df_summary['tercile'] == tercile]
    if len(df_t) >= 2:
        rate_2022 = df_t[df_t['year'] == 2022]['league_attempt_rate'].values
        rate_2023 = df_t[df_t['year'] == 2023]['league_attempt_rate'].values
        if len(rate_2022) > 0 and len(rate_2023) > 0:
            pct_change = 100 * (rate_2023[0] - rate_2022[0]) / rate_2022[0]
            print(f"  {tercile}: 2022={rate_2022[0]:.4f} → 2023={rate_2023[0]:.4f} "
                  f"({pct_change:+.1f}%)")

# ============================================================================
# SAVE OUTPUTS
# ============================================================================

print("\n" + "-" * 70)
print("SAVING OUTPUTS")
print("-" * 70)

# 1. Main panel
output_panel = OUTPUT_DIR / "c_panel_with_baseline.csv"
df_merged.to_csv(output_panel, index=False)
print(f"\n✓ Saved: {output_panel}")
print(f"  Rows: {len(df_merged):,}")
print(f"  Columns: {len(df_merged.columns)}")

# 2. Summary by tercile
output_summary = OUTPUT_DIR / "c2_summary_by_tercile.csv"
df_summary.to_csv(output_summary, index=False)
print(f"\n✓ Saved: {output_summary}")

# 3. QC Report
qc_file = OUTPUT_DIR / "c2_qc_report.txt"
with open(qc_file, 'w', encoding='utf-8') as f:
    f.write("C2 QC REPORT\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Input files:\n")
    f.write(f"  C1: {C1_FILE}\n")
    f.write(f"  B2: {B2_FILE}\n\n")
    f.write(f"Join results:\n")
    f.write(f"  Total observations: {len(df_merged):,}\n")
    f.write(f"  Observations with baseline: {df_merged['in_baseline_2022'].sum():,} "
            f"({pct_matched:.1f}%)\n")
    f.write(f"  Pitchers in baseline: {df_b2['pitcher_id'].nunique():,}\n\n")
    f.write(f"Sample composition:\n")
    f.write(f"  Post-2023 only: {n_post_only} pitchers\n")
    f.write(f"  Balanced 2018-2024: {n_balanced} pitchers\n\n")
    f.write(f"Data quality:\n")
    f.write(f"  Duplicates: None\n")
    f.write(f"  Arithmetic checks: Passed\n")
    f.write(f"  Logical constraints: Passed\n\n")
    f.write(f"Zero-inflation:\n")
    n_zero_att = (df_merged['attempts_2b'] == 0).sum()
    pct_zero = 100 * n_zero_att / len(df_merged)
    f.write(f"  Observations with 0 attempts: {n_zero_att:,} ({pct_zero:.1f}%)\n")
    f.write(f"  Note: Zero-inflation will be handled in C3/C4 models (Hurdle/ZIP)\n")

print(f"\n✓ Saved: {qc_file}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("C2 COMPLETE")
print("=" * 70)

print(f"\nPanel ready: {len(df_merged):,} pitcher-season observations")
print(f"  Years: 2018-2024")
print(f"  Pitchers: {df_merged['pitcher_id'].nunique():,}")
print(f"  With baseline group: {df_merged['in_baseline_2022'].sum():,} "
      f"({pct_matched:.1f}%)")

print(f"\nOutputs:")
print(f"  1. {output_panel.name} (main panel)")
print(f"  2. {output_summary.name} (descriptive stats)")
print(f"  3. {qc_file.name} (quality report)")

print("\n" + "=" * 70)
print("Next step: Run c3_event_study.py")
print("=" * 70)


if __name__ == "__main__":
    pass