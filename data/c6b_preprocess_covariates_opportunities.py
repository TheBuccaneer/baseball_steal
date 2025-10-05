"""
c6_preprocess_covariates_opportunities.py
==========================================
Build pitcher-level composition covariates from OPPORTUNITIES (not events)

Purpose:
Extract pitcher-specific measures of runner/catcher quality from 2022 based on
ALL opportunities (runner on 1B, 2B open), not just realized steal attempts.

This avoids selection bias from event-weighting.

Approach:
- Load Statcast 2022 pitch-level data
- Filter to steal opportunities: on_1b.notna() & on_2b.isna()
- Extract pitcher-runner-catcher matchups
- Join with sprint_speed and pop_time leaderboards
- Aggregate to pitcher level (opportunity-weighted averages)

Input:
- statcast/statcast_2022/*.csv (pitch-level data)
- leaderboards/sprint_speed_2022.csv (runner quality)
- leaderboards/catcher_throwing_2022.csv or poptime_2022.csv (catcher quality)

Output:
- pitcher_covariates_2022_opportunities.csv 
  (pitcher_id, avg_speed_faced_2022, avg_poptime_behind_2022, n_opportunities_2022)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================

STATCAST_DIR = Path("statcast/statcast_2022")
LEADERBOARDS_DIR = Path("leaderboards")

OUTPUT_FILE = Path("analysis/intermediate/pitcher_covariates_2022_opportunities.csv")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Also load the event-weighted version for comparison
EVENT_WEIGHTED_FILE = Path("analysis/intermediate/pitcher_covariates_2022.csv")

# ============================================================================
# LOAD STATCAST DATA
# ============================================================================

print("=" * 70)
print("C6 PREPROCESSING: PITCHER COVARIATES 2022 (OPPORTUNITY-WEIGHTED)")
print("=" * 70)

print("\n" + "-" * 70)
print("LOADING STATCAST 2022 (Pitch-Level)")
print("-" * 70)

statcast_files = sorted(STATCAST_DIR.glob("*.csv"))

if not statcast_files:
    print(f"ERROR: No statcast files found in {STATCAST_DIR}")
    sys.exit(1)

print(f"Found {len(statcast_files)} monthly files")

# Load pitch-level data
df_list = []
for file in statcast_files:
    try:
        # Load needed columns only for memory efficiency
        df = pd.read_csv(file, usecols=[
            'game_pk', 'pitcher', 'on_1b', 'on_2b', 'on_3b', 'fielder_2'
        ])
        df_list.append(df)
        print(f"  Loaded: {file.name} ({len(df):,} pitches)")
    except Exception as e:
        print(f"  Warning: Failed to load {file.name}: {e}")

df_statcast = pd.concat(df_list, ignore_index=True)
print(f"\nTotal pitches: {len(df_statcast):,}")

# ============================================================================
# FILTER TO STEAL OPPORTUNITIES
# ============================================================================

print("\n" + "-" * 70)
print("FILTERING TO STEAL OPPORTUNITIES")
print("-" * 70)

# Opportunity = Runner on 1B, 2B open (steal attempt possible)
df_opportunities = df_statcast[
    (df_statcast['on_1b'].notna()) & 
    (df_statcast['on_2b'].isna())
].copy()

print(f"\nSteal opportunities (1B occupied, 2B open): {len(df_opportunities):,}")
print(f"  % of all pitches: {len(df_opportunities)/len(df_statcast)*100:.1f}%")

# Prepare IDs
df_opportunities['pitcher_id'] = df_opportunities['pitcher'].astype(int)
df_opportunities['runner_id'] = df_opportunities['on_1b'].astype(int)
df_opportunities['catcher_id'] = df_opportunities['fielder_2']

# Drop any rows with missing pitcher or runner
df_opportunities = df_opportunities.dropna(subset=['pitcher_id', 'runner_id'])

print(f"\nValid opportunities (with pitcher + runner): {len(df_opportunities):,}")
print(f"  Unique pitchers: {df_opportunities['pitcher_id'].nunique():,}")
print(f"  Unique runners: {df_opportunities['runner_id'].nunique():,}")
print(f"  Unique catchers: {df_opportunities['catcher_id'].nunique():,}")

# ============================================================================
# LOAD SPRINT SPEED
# ============================================================================

print("\n" + "-" * 70)
print("LOADING SPRINT SPEED (2022)")
print("-" * 70)

speed_file = LEADERBOARDS_DIR / "sprint_speed_2022.csv"

try:
    df_speed = pd.read_csv(speed_file)
    print(f"Loaded: {len(df_speed):,} players")
    
    # Find relevant columns
    speed_id_col = next((c for c in df_speed.columns if 'player' in c.lower() and 'id' in c.lower()), 'player_id')
    speed_val_col = next((c for c in df_speed.columns if 'sprint' in c.lower() and 'speed' in c.lower()), 'sprint_speed')
    
    df_speed = df_speed[[speed_id_col, speed_val_col]].rename(
        columns={speed_id_col: 'player_id', speed_val_col: 'sprint_speed'}
    )
    
    # Merge to opportunities
    df_opportunities = df_opportunities.merge(
        df_speed, 
        left_on='runner_id', 
        right_on='player_id',
        how='left'
    )
    
    n_with_speed = df_opportunities['sprint_speed'].notna().sum()
    print(f"Matched: {n_with_speed:,} / {len(df_opportunities):,} ({n_with_speed/len(df_opportunities)*100:.1f}%)")
    
except FileNotFoundError:
    print(f"Warning: {speed_file} not found")
    df_opportunities['sprint_speed'] = np.nan

# ============================================================================
# LOAD POP TIME
# ============================================================================

print("\n" + "-" * 70)
print("LOADING POP TIME (2022)")
print("-" * 70)

# Try multiple possible files
poptime_files = [
    LEADERBOARDS_DIR / "catcher_throwing_2022.csv",
    LEADERBOARDS_DIR / "poptime_2022.csv"
]

df_poptime = None
for pfile in poptime_files:
    try:
        df_poptime = pd.read_csv(pfile)
        print(f"Loaded: {pfile.name} ({len(df_poptime):,} catchers)")
        break
    except FileNotFoundError:
        continue

if df_poptime is not None:
    # Find relevant columns
    pop_id_col = next((c for c in df_poptime.columns if 'player' in c.lower() and 'id' in c.lower()), 
                      next((c for c in df_poptime.columns if 'entity' in c.lower() and 'id' in c.lower()), 'player_id'))
    
    pop_val_col = next((c for c in df_poptime.columns if 'pop' in c.lower() and 'time' in c.lower()), 'pop_time')
    
    # Use pop_2b_sba if available (pop time to 2nd base on steal attempts)
    if 'pop_2b_sba' in df_poptime.columns:
        pop_val_col = 'pop_2b_sba'
    
    df_poptime = df_poptime[[pop_id_col, pop_val_col]].rename(
        columns={pop_id_col: 'player_id', pop_val_col: 'pop_time'}
    )
    
    # Merge to opportunities
    df_opportunities = df_opportunities.merge(
        df_poptime, 
        left_on='catcher_id', 
        right_on='player_id',
        how='left',
        suffixes=('', '_catcher')
    )
    
    n_with_pop = df_opportunities['pop_time'].notna().sum()
    print(f"Matched: {n_with_pop:,} / {len(df_opportunities):,} ({n_with_pop/len(df_opportunities)*100:.1f}%)")
else:
    print("Warning: No pop time file found")
    df_opportunities['pop_time'] = np.nan

# ============================================================================
# AGGREGATE TO PITCHER LEVEL (OPPORTUNITY-WEIGHTED)
# ============================================================================

print("\n" + "-" * 70)
print("AGGREGATING TO PITCHER LEVEL (OPPORTUNITY-WEIGHTED)")
print("-" * 70)

# For each pitcher: mean over ALL opportunities (not just realized steal attempts)
pitcher_covariates_opp = df_opportunities.groupby('pitcher_id').agg({
    'sprint_speed': 'mean',
    'pop_time': 'mean'
}).reset_index()

pitcher_covariates_opp.columns = ['pitcher_id', 'avg_speed_faced_2022_opp', 'avg_poptime_behind_2022_opp']

# Count opportunities per pitcher
pitcher_n = df_opportunities.groupby('pitcher_id').size().reset_index(name='n_opportunities_2022')
pitcher_covariates_opp = pitcher_covariates_opp.merge(pitcher_n, on='pitcher_id')

print(f"\nPitchers with opportunity-weighted covariates: {len(pitcher_covariates_opp):,}")
print(f"\nCoverage:")
print(f"  Sprint speed: {pitcher_covariates_opp['avg_speed_faced_2022_opp'].notna().sum():,} pitchers")
print(f"  Pop time: {pitcher_covariates_opp['avg_poptime_behind_2022_opp'].notna().sum():,} pitchers")

# Summary stats
print(f"\nSummary statistics (OPPORTUNITY-WEIGHTED):")
print(pitcher_covariates_opp[['avg_speed_faced_2022_opp', 'avg_poptime_behind_2022_opp', 'n_opportunities_2022']].describe())

# ============================================================================
# COMPARISON WITH EVENT-WEIGHTED VERSION
# ============================================================================

print("\n" + "-" * 70)
print("COMPARISON: OPPORTUNITY-WEIGHTED vs EVENT-WEIGHTED")
print("-" * 70)

try:
    df_event = pd.read_csv(EVENT_WEIGHTED_FILE)
    
    # Merge the two versions
    df_compare = pitcher_covariates_opp.merge(
        df_event[['pitcher_id', 'avg_speed_faced_2022', 'avg_poptime_behind_2022', 'n_matchups_2022']],
        on='pitcher_id',
        how='outer',
        suffixes=('_opp', '_event')
    )
    
    # Compute differences
    df_compare['speed_diff'] = df_compare['avg_speed_faced_2022_opp'] - df_compare['avg_speed_faced_2022']
    df_compare['poptime_diff'] = df_compare['avg_poptime_behind_2022_opp'] - df_compare['avg_poptime_behind_2022']
    
    print(f"\nPitchers in both versions: {df_compare[['avg_speed_faced_2022_opp', 'avg_speed_faced_2022']].notna().all(axis=1).sum():,}")
    
    print(f"\nAverage differences (Opportunity - Event):")
    print(f"  Sprint Speed: {df_compare['speed_diff'].mean():.3f} ft/s (SD: {df_compare['speed_diff'].std():.3f})")
    print(f"  Pop Time: {df_compare['poptime_diff'].mean():.4f} s (SD: {df_compare['poptime_diff'].std():.4f})")
    
    print(f"\nSample size comparison:")
    print(f"  Opportunities (median): {df_compare['n_opportunities_2022'].median():.0f}")
    print(f"  Events (median): {df_compare['n_matchups_2022'].median():.0f}")
    print(f"  Ratio: {df_compare['n_opportunities_2022'].median() / df_compare['n_matchups_2022'].median():.1f}x more opportunities")
    
    # Correlation between the two measures
    speed_corr = df_compare[['avg_speed_faced_2022_opp', 'avg_speed_faced_2022']].corr().iloc[0, 1]
    pop_corr = df_compare[['avg_poptime_behind_2022_opp', 'avg_poptime_behind_2022']].corr().iloc[0, 1]
    
    print(f"\nCorrelations (Opportunity vs Event):")
    print(f"  Sprint Speed: {speed_corr:.3f}")
    print(f"  Pop Time: {pop_corr:.3f}")
    
    # Identify pitchers with largest differences
    df_compare['abs_speed_diff'] = df_compare['speed_diff'].abs()
    top_diff = df_compare.nlargest(5, 'abs_speed_diff')
    
    print(f"\nTop 5 pitchers with largest speed differences:")
    print(f"{'Pitcher':<10} {'Opp':<8} {'Event':<8} {'Diff':<8}")
    print("-" * 40)
    for _, row in top_diff.iterrows():
        print(f"{int(row['pitcher_id']):<10} {row['avg_speed_faced_2022_opp']:.2f}   {row['avg_speed_faced_2022']:.2f}   {row['speed_diff']:+.2f}")
    
except FileNotFoundError:
    print(f"\nWarning: Event-weighted file not found at {EVENT_WEIGHTED_FILE}")
    print("  Skipping comparison")

# ============================================================================
# SAVE OUTPUT
# ============================================================================

print("\n" + "-" * 70)
print("SAVING OUTPUT")
print("-" * 70)

pitcher_covariates_opp.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved: {OUTPUT_FILE}")
print(f"  Rows: {len(pitcher_covariates_opp):,}")
print(f"  Columns: {list(pitcher_covariates_opp.columns)}")

# ============================================================================
# VALIDATION
# ============================================================================

print("\n" + "-" * 70)
print("VALIDATION")
print("-" * 70)

# Check for extreme values
speed_valid = pitcher_covariates_opp['avg_speed_faced_2022_opp'].between(20, 35)
pop_valid = pitcher_covariates_opp['avg_poptime_behind_2022_opp'].between(1.5, 2.5)

n_speed_invalid = (~speed_valid & pitcher_covariates_opp['avg_speed_faced_2022_opp'].notna()).sum()
n_pop_invalid = (~pop_valid & pitcher_covariates_opp['avg_poptime_behind_2022_opp'].notna()).sum()

if n_speed_invalid > 0:
    print(f"Warning: {n_speed_invalid} pitchers with implausible sprint speed (<20 or >35 ft/s)")
if n_pop_invalid > 0:
    print(f"Warning: {n_pop_invalid} pitchers with implausible pop time (<1.5 or >2.5 s)")

# Check distribution
print(f"\nDistribution of opportunities per pitcher:")
print(pitcher_covariates_opp['n_opportunities_2022'].describe())

print(f"\nPitchers with <50 opportunities: {(pitcher_covariates_opp['n_opportunities_2022'] < 50).sum():,}")
print(f"  (May want higher threshold for opportunity-weighted than event-weighted)")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETE (OPPORTUNITY-WEIGHTED)")
print("=" * 70)

print(f"\nOutput: {OUTPUT_FILE}")
print(f"  {len(pitcher_covariates_opp):,} pitchers with opportunity-weighted composition measures")

print(f"\nMethodological advantage:")
print(f"  - Unbiased exposure (all opportunities, not just realized steals)")
print(f"  - Larger sample per pitcher (~10x more observations)")
print(f"  - Avoids selection bias from event-weighting")

print(f"\nNext steps:")
print(f"  1. Update c6_composition_robustness.py to use this file")
print(f"  2. Compare stability with opportunity-weighted vs event-weighted controls")
print(f"  3. Report both specifications in paper")

print("\n" + "=" * 70)