"""
Builds analysis_pitcher_panel.csv from pitcher running game + pitch tempo leaderboards
Analyzes pitcher control of running game and pitch tempo

Output: analysis_pitcher_panel.csv

Usage: python 10_build_pitcher_panel.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def load_all_years(folder_path, file_pattern):
    """Load and concatenate CSVs for all years"""
    files = sorted(Path(folder_path).glob(file_pattern))
    if not files:
        print(f"  ERROR: No files found: {folder_path}/{file_pattern}")
        return pd.DataFrame()
    
    dfs = []
    for f in files:
        year = int(f.stem.split('_')[-1])
        df = pd.read_csv(f)
        df['season'] = year
        dfs.append(df)
        print(f"  Loaded {len(df):4d} pitcher-records from {f.name}")
    
    return pd.concat(dfs, ignore_index=True)

def load_tempo_with_fix(folder_path):
    """Load pitch tempo files and fix duplicate/unclear column names"""
    files = sorted(Path(folder_path).glob("pitch_tempo_*.csv"))
    if not files:
        return pd.DataFrame()
    
    dfs = []
    for f in files:
        year = int(f.stem.split('_')[-1])
        df = pd.read_csv(f)
        
        # Robust column mapping - find onbase columns by pattern matching
        # Only rename FIRST match to avoid duplicates
        rename_map = {}
        onbase_found = False
        onbase_pitches_found = False
        
        for col in df.columns:
            col_lower = str(col).lower().replace(" ", "_")
            
            # Find median seconds with runners on - ONLY first match
            if "median" in col_lower and "second" in col_lower and not onbase_found:
                # Check for onbase indicators or pandas duplicate suffix
                if "on" in col_lower or "runner" in col_lower or ".1" in str(col):
                    rename_map[col] = "median_seconds_onbase"
                    onbase_found = True
            
            # Find total pitches onbase - ONLY first match
            if "total" in col_lower and "pitch" in col_lower and not onbase_pitches_found:
                if "on" in col_lower or "runner" in col_lower:
                    rename_map[col] = "total_pitches_onbase"
                    onbase_pitches_found = True
        
        df = df.rename(columns=rename_map)
        df['season'] = year
        dfs.append(df)
        print(f"  Loaded {len(df):4d} pitcher-tempo records from {f.name}")
    
    return pd.concat(dfs, ignore_index=True)

def main():
    print("=" * 80)
    print("BUILDING PITCHER PANEL (from pitcher_running_game + pitch_tempo)")
    print("=" * 80)
    print(f"\nCurrent directory: {Path.cwd()}")
    
    leaderboards_path = Path("leaderboards")
    
    if not leaderboards_path.exists():
        print(f"\nERROR: {leaderboards_path.absolute()} not found!")
        print("Please run this script from the data/ directory")
        sys.exit(1)
    
    # Load Pitcher Running Game data
    print("\nLoading pitcher_running_game leaderboards...")
    running = load_all_years(leaderboards_path, "pitcher_running_game_*.csv")
    
    if running.empty:
        print("  ERROR: No pitcher running game data")
        sys.exit(1)
    
    print(f"\n  Total loaded: {len(running):,} pitcher-records (includes 2B/3B splits)")
    
    # Filter to "All" bases only for season-level aggregate
    print("\nFiltering to 'All' bases aggregate...")
    running_all = running[running['key_target_base'] == 'All'].copy()
    print(f"  Kept {len(running_all):,} pitcher-seasons (filtered from {len(running):,} total)")
    
    if len(running_all) == 0:
        print("  ERROR: No 'All' base records found!")
        print("  Available key_target_base values:", running['key_target_base'].unique())
        sys.exit(1)
    
    # Load Pitch Tempo data with duplicate column fix
    print("\nLoading pitch_tempo leaderboards...")
    tempo = load_tempo_with_fix(leaderboards_path)
    
    if tempo.empty:
        print("  WARNING: No pitch tempo data found")
        tempo_available = False
    else:
        print(f"  Total loaded: {len(tempo):,} pitcher-tempo records")
        tempo_available = True
    
    # Load Pitch Timer Infractions (2023+ only)
    print("\nLoading pitch_timer_infractions leaderboards (2023+)...")
    infractions = load_all_years(leaderboards_path, "pitch_timer_infractions_pitchers_*.csv")
    
    if infractions.empty:
        print("  WARNING: No infractions data found (expected for 2023+)")
        infractions_available = False
    else:
        print(f"  Total loaded: {len(infractions):,} pitcher-infraction records")
        infractions_available = True
    
    # Rename columns for consistency
    print("\nRenaming columns...")
    running_all = running_all.rename(columns={
        'player_id': 'pitcher_id',
        'player_name': 'pitcher_name'
    })
    
    # Calculate derived metrics for running game
    print("\nCalculating running game metrics...")
    
    # Total attempts (SB + CS)
    running_all['total_sb_attempts'] = running_all['n_sb'] + running_all['n_cs']
    
    # SB success rate (runner perspective)
    running_all['sb_success_rate'] = running_all.apply(
        lambda row: row['n_sb'] / row['total_sb_attempts']
        if row['total_sb_attempts'] > 0 else np.nan,
        axis=1
    )
    
    # CS rate (pitcher perspective)
    running_all['cs_rate'] = running_all.apply(
        lambda row: row['n_cs'] / row['total_sb_attempts']
        if row['total_sb_attempts'] > 0 else np.nan,
        axis=1
    )
    
    # Pickoff rate (pickoffs per opportunity)
    running_all['pickoff_rate'] = running_all.apply(
        lambda row: row['n_pk'] / row['n_init']
        if row['n_init'] > 0 else np.nan,
        axis=1
    )
    
    # Lead distance difference (secondary - primary)
    running_all['lead_gain'] = running_all['r_secondary_lead'] - running_all['r_primary_lead']
    
    # Merge with Pitch Tempo
    if tempo_available:
        print("\nMerging pitch tempo data...")
        tempo = tempo.rename(columns={
            'entity_id': 'pitcher_id',
            'entity_name': 'pitcher_name'
        })
        
        # Select relevant tempo columns
        tempo_cols = ['pitcher_id', 'season', 'median_seconds_onbase', 'total_pitches_onbase']
        tempo_cols = [c for c in tempo_cols if c in tempo.columns]
        
        tempo_merge = tempo[tempo_cols].copy()
        tempo_merge = tempo_merge.rename(columns={
            'median_seconds_onbase': 'tempo_with_runners_on_base',
            'total_pitches_onbase': 'pitches_with_runners_on_base'
        })
        
        # Merge
        pitcher = running_all.merge(tempo_merge, on=['pitcher_id', 'season'], how='left')
        print(f"  Merged: {len(pitcher):,} rows")
        
        # Check if tempo column exists after merge
        if 'tempo_with_runners_on_base' in pitcher.columns:
            tempo_missing = int(pitcher['tempo_with_runners_on_base'].isna().sum())
            print(f"  Missing tempo data: {tempo_missing:,} ({tempo_missing/len(pitcher)*100:.1f}%)")
        else:
            print(f"  WARNING: tempo_with_runners_on_base not found after merge")
            print(f"  Tempo columns available: {[c for c in tempo_merge.columns]}")
            print(f"  Merged columns with 'tempo' or 'median': {[c for c in pitcher.columns if 'tempo' in c.lower() or 'median' in c.lower()]}")
    else:
        pitcher = running_all.copy()
    
    # Merge with Infractions (2023+ only)
    if infractions_available:
        print("\nMerging pitch timer infractions data...")
        
        # Note: infractions CSV has 'year' column, but load_all_years already added 'season'
        # We need to handle this carefully to avoid duplicates
        if 'year' in infractions.columns and 'season' in infractions.columns:
            # Drop the 'season' from load_all_years, keep 'year' and rename it
            infractions = infractions.drop(columns=['season'])
        
        infractions = infractions.rename(columns={
            'entity_id': 'pitcher_id',
            'entity_name': 'pitcher_name',
            'year': 'season'
        })
        
        # Select relevant infraction columns
        infraction_cols = ['pitcher_id', 'season', 'pitches', 'all_violations', 
                          'pitcher_timer', 'batter_timer']
        infraction_cols = [c for c in infraction_cols if c in infractions.columns]
        
        infractions_merge = infractions[infraction_cols].copy()
        
        # Calculate violation rate
        if 'all_violations' in infractions_merge.columns and 'pitches' in infractions_merge.columns:
            infractions_merge['violation_rate'] = infractions_merge.apply(
                lambda row: row['all_violations'] / row['pitches']
                if row['pitches'] > 0 else np.nan,
                axis=1
            )
        
        # Merge
        pitcher = pitcher.merge(infractions_merge, on=['pitcher_id', 'season'], how='left')
        print(f"  Merged: {len(pitcher):,} rows")
        
        infraction_coverage = pitcher['all_violations'].notna().sum()
        print(f"  Infraction data available: {infraction_coverage:,} rows (2023+ expected)")
    
    # Add treatment variable
    print("\nAdding treatment variables...")
    pitcher['post_2023'] = (pitcher['season'] >= 2023).astype(int)
    
    # Check for multi-team pitchers
    print("\nChecking for multi-team pitchers...")
    pitcher['multi_team'] = 0  # Placeholder - data is aggregated
    print("  Note: multi_team flag not available from aggregated data")
    
    # Finalize columns
    print("\nFinalizing dataset...")
    
    final_cols = [
        'season', 'pitcher_id', 'pitcher_name', 'team_name',
        'n_init', 'total_sb_attempts', 'n_sb', 'n_cs', 'n_pk', 'n_bk',
        'sb_success_rate', 'cs_rate', 'pickoff_rate',
        'rate_sbx',
        'r_primary_lead', 'r_secondary_lead', 'lead_gain',
        'r_primary_lead_sbx', 'r_secondary_lead_sbx',
        'runs_prevented_on_running_attr', 'n_pitcher_cs_aa',
        'n_plus', 'n_minus', 'net_attr_plus', 'net_attr_minus',
        'post_2023', 'multi_team'
    ]
    
    # Add tempo columns if available
    if 'tempo_with_runners_on_base' in pitcher.columns:
        final_cols.insert(-2, 'tempo_with_runners_on_base')
        final_cols.insert(-2, 'pitches_with_runners_on_base')
    
    # Add infraction columns if available
    if 'all_violations' in pitcher.columns:
        final_cols.insert(-2, 'all_violations')
        final_cols.insert(-2, 'pitcher_timer')
        final_cols.insert(-2, 'violation_rate')
    
    # Keep only columns that exist
    final_cols = [c for c in final_cols if c in pitcher.columns]
    pitcher_final = pitcher[final_cols].copy()
    pitcher_final = pitcher_final.sort_values(['season', 'pitcher_id'])
    
    # QA Checks
    print("\n" + "=" * 80)
    print("QA CHECKS")
    print("=" * 80)
    
    print(f"\nRows: {len(pitcher_final):,}")
    print(f"Seasons: {pitcher_final['season'].min()}-{pitcher_final['season'].max()}")
    print(f"Unique pitchers: {pitcher_final['pitcher_id'].nunique():,}")
    
    print("\nKey variables - Missing data:")
    key_vars = ['n_init', 'total_sb_attempts', 'cs_rate', 'pickoff_rate', 
                'r_primary_lead', 'tempo_with_runners_on_base', 'all_violations']
    for col in key_vars:
        if col in pitcher_final.columns:
            missing = pitcher_final[col].isna().sum()
            pct = (missing / len(pitcher_final)) * 100
            print(f"  {col:35s}: {missing:5,} ({pct:5.1f}%)")
    
    print("\nSummary stats:")
    summary_cols = ['n_init', 'total_sb_attempts', 'n_sb', 'n_cs', 'n_pk',
                    'sb_success_rate', 'cs_rate', 'pickoff_rate',
                    'r_primary_lead', 'r_secondary_lead', 'lead_gain']
    summary_cols = [c for c in summary_cols if c in pitcher_final.columns]
    print(pitcher_final[summary_cols].describe())
    
    # Check CS rate range
    print("\nCS Rate validation (pitcher perspective):")
    if 'cs_rate' in pitcher_final.columns:
        cs_valid = pitcher_final['cs_rate'].notna()
        if cs_valid.any():
            cs_data = pitcher_final.loc[cs_valid, 'cs_rate']
            print(f"  Range: {cs_data.min():.3f} - {cs_data.max():.3f} ({cs_data.min()*100:.1f}% - {cs_data.max()*100:.1f}%)")
            print(f"  Median: {cs_data.median():.3f} ({cs_data.median()*100:.1f}%)")
            print(f"  Mean: {cs_data.mean():.3f} ({cs_data.mean()*100:.1f}%)")
            print(f"  MLB Benchmark: ~25% pre-2023, ~18% post-2023")
    
    # Check pickoff rate
    print("\nPickoff Rate validation:")
    if 'pickoff_rate' in pitcher_final.columns:
        pk_valid = pitcher_final['pickoff_rate'].notna()
        if pk_valid.any():
            pk_data = pitcher_final.loc[pk_valid, 'pickoff_rate']
            print(f"  Range: {pk_data.min():.4f} - {pk_data.max():.4f} ({pk_data.min()*100:.2f}% - {pk_data.max()*100:.2f}%)")
            print(f"  Median: {pk_data.median():.4f} ({pk_data.median()*100:.2f}%)")
            print(f"  Mean: {pk_data.mean():.4f} ({pk_data.mean()*100:.2f}%)")
    
    # Check lead distances
    print("\nLead Distance validation:")
    if 'r_primary_lead' in pitcher_final.columns:
        lead_valid = pitcher_final['r_primary_lead'].notna()
        if lead_valid.any():
            lead_data = pitcher_final.loc[lead_valid, 'r_primary_lead']
            print(f"  Primary Lead Range: {lead_data.min():.2f} - {lead_data.max():.2f} ft")
            print(f"  Primary Lead Mean: {lead_data.mean():.2f} ft")
            print(f"  Typical range: 10-14 ft")
    
    if 'lead_gain' in pitcher_final.columns:
        gain_valid = pitcher_final['lead_gain'].notna()
        if gain_valid.any():
            gain_data = pitcher_final.loc[gain_valid, 'lead_gain']
            print(f"  Lead Gain Range: {gain_data.min():.2f} - {gain_data.max():.2f} ft")
            print(f"  Lead Gain Mean: {gain_data.mean():.2f} ft")
    
    # Check tempo (if available)
    if 'tempo_with_runners_on_base' in pitcher_final.columns:
        print("\nPitch Tempo validation (with runners on):")
        tempo_valid = pitcher_final['tempo_with_runners_on_base'].notna()
        if tempo_valid.any():
            tempo_data = pitcher_final.loc[tempo_valid, 'tempo_with_runners_on_base']
            print(f"  Range: {tempo_data.min():.2f} - {tempo_data.max():.2f} sec")
            print(f"  Median: {tempo_data.median():.2f} sec")
            print(f"  Mean: {tempo_data.mean():.2f} sec")
            print(f"  MLB Benchmark: ~17-18 sec pre-2023, ~15 sec post-2023 (pitch clock)")
    
    # Pre-2023 vs Post-2023 comparison
    print("\nRule Change Impact (2023):")
    if 'post_2023' in pitcher_final.columns:
        pre = pitcher_final[pitcher_final['post_2023'] == 0]
        post = pitcher_final[pitcher_final['post_2023'] == 1]
        
        if len(pre) > 0 and len(post) > 0:
            print(f"\nSample sizes: Pre={len(pre):,}, Post={len(post):,}")
            
            # CS Rate
            if 'cs_rate' in pitcher_final.columns:
                cs_pre = pre['cs_rate'].mean()
                cs_post = post['cs_rate'].mean()
                print(f"  CS Rate: {cs_pre:.3f} → {cs_post:.3f} ({(cs_pre-cs_post):.3f}, {(cs_pre-cs_post)*100:.1f} pp)")
            
            # Pickoff Rate
            if 'pickoff_rate' in pitcher_final.columns:
                pk_pre = pre['pickoff_rate'].mean()
                pk_post = post['pickoff_rate'].mean()
                print(f"  Pickoff Rate: {pk_pre:.4f} → {pk_post:.4f} ({(pk_pre-pk_post):.4f}, {(pk_pre-pk_post)*100:.2f} pp)")
            
            # Lead Distance
            if 'r_primary_lead' in pitcher_final.columns:
                lead_pre = pre['r_primary_lead'].mean()
                lead_post = post['r_primary_lead'].mean()
                print(f"  Primary Lead: {lead_pre:.2f} → {lead_post:.2f} ft ({(lead_post-lead_pre):+.2f} ft)")
            
            # Tempo
            if 'tempo_with_runners_on_base' in pitcher_final.columns:
                tempo_pre = pre['tempo_with_runners_on_base'].mean()
                tempo_post = post['tempo_with_runners_on_base'].mean()
                print(f"  Tempo: {tempo_pre:.2f} → {tempo_post:.2f} sec ({(tempo_post-tempo_pre):+.2f} sec)")
            
            print(f"\n  Expected changes post-2023:")
            print(f"    - Lower CS Rate (harder to control runners)")
            print(f"    - Lower Pickoff Rate (2-attempt limit)")
            print(f"    - Larger Lead Distances (runners more aggressive)")
            print(f"    - Faster Tempo (pitch clock)")
    
    # Check for pitchers with extreme metrics (small sample issues)
    print("\nSmall sample check:")
    small_sample = pitcher_final['n_init'] < 10
    n_small = small_sample.sum()
    print(f"  Pitchers with <10 opportunities: {n_small:,} ({n_small/len(pitcher_final)*100:.1f}%)")
    print(f"  Recommendation: Filter n_init >= 10 or 25 for robust analysis")
    
    # Save
    output_path = Path("analysis")
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "analysis_pitcher_panel.csv"
    
    pitcher_final.to_csv(output_file, index=False)
    
    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"\nSaved: {output_file}")
    print(f"Size: {output_file.stat().st_size / 1024:.1f} KB")
    print(f"\nPitcher panel ready for analysis!")
    print("\nKey features:")
    print("  ✓ Running game control metrics (CS rate, pickoff rate)")
    print("  ✓ Lead distance metrics (primary, secondary, gain)")
    print("  ✓ Pitch tempo with runners on base")
    print("  ✓ Pitch timer infractions (2023+)")
    print("  ✓ Pre/Post-2023 treatment variable")
    print("  ✓ Runs prevented attribution")

if __name__ == "__main__":
    main()