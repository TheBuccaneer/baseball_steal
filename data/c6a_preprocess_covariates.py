"""
c6_preprocess_covariates.py
===========================
Build pitcher-level composition covariates from event data

Purpose:
Extract pitcher-specific measures of runner/catcher quality from 2022:
- avg_speed_faced: Mean sprint speed of runners this pitcher faced
- avg_poptime_behind: Mean pop time of catchers behind this pitcher

These are used as pre-treatment (2022) controls in C6 composition robustness.

Input:
- mlb_runner_events_2022/*.csv (event-level data)
- sprint_speed_2022.csv (runner quality)
- catcher_throwing_2022.csv or poptime_2022.csv (catcher quality)

Output:
- pitcher_covariates_2022.csv (pitcher_id, avg_speed_faced, avg_poptime_behind)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================

MLB_STATS_DIR = Path("mlb_stats/mlb_stats_2022")
STATCAST_DIR = Path("statcast/statcast_2022")
LEADERBOARDS_DIR = Path("leaderboards")

OUTPUT_FILE = Path("analysis/intermediate/pitcher_covariates_2022.csv")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# ============================================================================
# LOAD MLB STATS DATA (has pitcher_id, runner_id)
# ============================================================================

print("=" * 70)
print("C6 PREPROCESSING: PITCHER COVARIATES 2022")
print("=" * 70)

print("\n" + "-" * 70)
print("LOADING MLB STATS (2022)")
print("-" * 70)

stats_files = sorted(MLB_STATS_DIR.glob("*.csv"))

if not stats_files:
    print(f"ERROR: No MLB stats files found in {MLB_STATS_DIR}")
    sys.exit(1)

print(f"Found {len(stats_files)} monthly files")

df_stats_list = []
for file in stats_files:
    try:
        df = pd.read_csv(file)
        # Filter to rows with runners (steal attempts)
        df = df[df['runner_id'].notna()].copy()
        df_stats_list.append(df)
        print(f"  Loaded: {file.name} ({len(df):,} runner events)")
    except Exception as e:
        print(f"  Warning: Failed to load {file.name}: {e}")

df_stats = pd.concat(df_stats_list, ignore_index=True)
print(f"\nTotal runner events: {len(df_stats):,}")

# ============================================================================
# LOAD STATCAST DATA (has fielder_2 = catcher)
# ============================================================================

print("\n" + "-" * 70)
print("LOADING STATCAST (2022)")
print("-" * 70)

statcast_files = sorted(STATCAST_DIR.glob("*.csv"))

if not statcast_files:
    print(f"WARNING: No statcast files found in {STATCAST_DIR}")
    df_statcast = None
else:
    print(f"Found {len(statcast_files)} monthly files")
    
    df_statcast_list = []
    for file in statcast_files:
        try:
            # Only load needed columns for memory efficiency
            df = pd.read_csv(file, usecols=['game_pk', 'inning', 'at_bat_number', 'fielder_2'])
            df_statcast_list.append(df)
            print(f"  Loaded: {file.name} ({len(df):,} pitches)")
        except Exception as e:
            print(f"  Warning: Failed to load {file.name}: {e}")
    
    df_statcast = pd.concat(df_statcast_list, ignore_index=True)
    print(f"\nTotal pitches: {len(df_statcast):,}")

# ============================================================================
# JOIN MLB STATS + STATCAST TO GET PITCHER/RUNNER/CATCHER
# ============================================================================

print("\n" + "-" * 70)
print("JOINING MLB STATS + STATCAST")
print("-" * 70)

# Prepare join keys
df_stats['game_pk'] = df_stats['game_id'].astype(int)
df_stats['inning'] = df_stats['inning'].astype(int)
df_stats['at_bat_number'] = df_stats['at_bat_index'].astype(int)

if df_statcast is not None:
    df_statcast['game_pk'] = df_statcast['game_pk'].astype(int)
    df_statcast['inning'] = df_statcast['inning'].astype(int)
    df_statcast['at_bat_number'] = df_statcast['at_bat_number'].astype(int)
    
    # Join on game_pk + inning + at_bat_number
    df_matchups = df_stats.merge(
        df_statcast[['game_pk', 'inning', 'at_bat_number', 'fielder_2']],
        on=['game_pk', 'inning', 'at_bat_number'],
        how='left'
    )
    
    print(f"Matched: {len(df_matchups):,} events")
    print(f"  With catcher: {df_matchups['fielder_2'].notna().sum():,} ({df_matchups['fielder_2'].notna().mean()*100:.1f}%)")
else:
    df_matchups = df_stats.copy()
    df_matchups['fielder_2'] = np.nan
    print(f"No statcast data - proceeding without catcher info")

# Keep only needed columns
df_matchups = df_matchups[['pitcher_id', 'runner_id', 'fielder_2']].copy()
df_matchups = df_matchups.dropna(subset=['pitcher_id', 'runner_id'])
df_matchups['catcher_id'] = df_matchups['fielder_2']

print(f"\nFinal matchups: {len(df_matchups):,}")
print(f"  Unique pitchers: {df_matchups['pitcher_id'].nunique():,}")
print(f"  Unique runners: {df_matchups['runner_id'].nunique():,}")
print(f"  Unique catchers: {df_matchups['catcher_id'].nunique():,}")

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
    
    # Merge to matchups
    df_matchups = df_matchups.merge(
        df_speed, 
        left_on='runner_id', 
        right_on='player_id',
        how='left'
    )
    
    n_with_speed = df_matchups['sprint_speed'].notna().sum()
    print(f"Matched: {n_with_speed:,} / {len(df_matchups):,} ({n_with_speed/len(df_matchups)*100:.1f}%)")
    
except FileNotFoundError:
    print(f"Warning: {speed_file} not found")
    df_matchups['sprint_speed'] = np.nan

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
    
    # Merge to matchups
    df_matchups = df_matchups.merge(
        df_poptime, 
        left_on='catcher_id', 
        right_on='player_id',
        how='left',
        suffixes=('', '_catcher')
    )
    
    n_with_pop = df_matchups['pop_time'].notna().sum()
    print(f"Matched: {n_with_pop:,} / {len(df_matchups):,} ({n_with_pop/len(df_matchups)*100:.1f}%)")
else:
    print("Warning: No pop time file found")
    df_matchups['pop_time'] = np.nan

# ============================================================================
# AGGREGATE TO PITCHER LEVEL
# ============================================================================

print("\n" + "-" * 70)
print("AGGREGATING TO PITCHER LEVEL")
print("-" * 70)

# For each pitcher: mean of faced runners' speed, mean of behind catchers' pop_time
pitcher_covariates = df_matchups.groupby('pitcher_id').agg({
    'sprint_speed': 'mean',
    'pop_time': 'mean'
}).reset_index()

pitcher_covariates.columns = ['pitcher_id', 'avg_speed_faced_2022', 'avg_poptime_behind_2022']

# Count how many observations per pitcher
pitcher_n = df_matchups.groupby('pitcher_id').size().reset_index(name='n_matchups_2022')
pitcher_covariates = pitcher_covariates.merge(pitcher_n, on='pitcher_id')

print(f"\nPitchers with covariates: {len(pitcher_covariates):,}")
print(f"\nCoverage:")
print(f"  Sprint speed: {pitcher_covariates['avg_speed_faced_2022'].notna().sum():,} pitchers")
print(f"  Pop time: {pitcher_covariates['avg_poptime_behind_2022'].notna().sum():,} pitchers")

# Summary stats
print(f"\nSummary statistics:")
print(pitcher_covariates[['avg_speed_faced_2022', 'avg_poptime_behind_2022', 'n_matchups_2022']].describe())

# ============================================================================
# SAVE OUTPUT
# ============================================================================

print("\n" + "-" * 70)
print("SAVING OUTPUT")
print("-" * 70)

pitcher_covariates.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved: {OUTPUT_FILE}")
print(f"  Rows: {len(pitcher_covariates):,}")
print(f"  Columns: {list(pitcher_covariates.columns)}")

# ============================================================================
# VALIDATION
# ============================================================================

print("\n" + "-" * 70)
print("VALIDATION")
print("-" * 70)

# Check for extreme values
speed_valid = pitcher_covariates['avg_speed_faced_2022'].between(20, 35)
pop_valid = pitcher_covariates['avg_poptime_behind_2022'].between(1.5, 2.5)

n_speed_invalid = (~speed_valid & pitcher_covariates['avg_speed_faced_2022'].notna()).sum()
n_pop_invalid = (~pop_valid & pitcher_covariates['avg_poptime_behind_2022'].notna()).sum()

if n_speed_invalid > 0:
    print(f"Warning: {n_speed_invalid} pitchers with implausible sprint speed (<20 or >35 ft/s)")
if n_pop_invalid > 0:
    print(f"Warning: {n_pop_invalid} pitchers with implausible pop time (<1.5 or >2.5 s)")

# Check distribution
print(f"\nDistribution of matchups per pitcher:")
print(pitcher_covariates['n_matchups_2022'].describe())

print(f"\nPitchers with <10 matchups: {(pitcher_covariates['n_matchups_2022'] < 10).sum():,}")
print(f"  (Consider excluding these for robustness)")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETE")
print("=" * 70)

print(f"\nOutput: {OUTPUT_FILE}")
print(f"  {len(pitcher_covariates):,} pitchers with composition measures")
print(f"  Ready to merge into c_panel_with_baseline.csv")

print(f"\nNext steps:")
print(f"  1. Update c6_composition_robustness.py to use this file")
print(f"  2. Merge on pitcher_id")
print(f"  3. Run composition robustness checks")

print("\n" + "=" * 70)