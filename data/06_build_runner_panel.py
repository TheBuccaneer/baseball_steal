"""
Builds analysis_runner_panel.csv
Merges runner leaderboards + schedule-weighted opponent metrics

Requires intermediate files from scripts 03-05:
  - team_poptime_2018_2025.csv
  - team_tempo_2018_2025.csv
  - runner_schedule_weights_2018_2025.csv

Usage: python 06_build_runner_panel.py
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
        
        # Standardize season column
        if 'season' not in df.columns:
            if 'year' in df.columns:
                df['season'] = df['year']
            elif 'start_year' in df.columns:
                df['season'] = df['start_year']
            else:
                df['season'] = year
        
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)

def compute_weighted_opponent_metrics(runner_panel, schedule_weights, team_poptime, team_tempo):
    """
    Compute schedule-weighted opponent pop time and tempo
    Uses on-base weighted exposure from schedule_weights
    """
    
    print("\nComputing schedule-weighted opponent metrics...")
    
    # Merge schedule weights with team poptime
    weights_poptime = schedule_weights.merge(
        team_poptime[['team_id', 'season', 'pop_time_2b_avg']],
        left_on=['opponent_team_id', 'season'],
        right_on=['team_id', 'season'],
        how='left'
    )
    
    # Merge schedule weights with team tempo
    weights_tempo = schedule_weights.merge(
        team_tempo[['team_id', 'season', 'tempo_onbase_avg']],
        left_on=['opponent_team_id', 'season'],
        right_on=['team_id', 'season'],
        how='left'
    )
    
    # Compute weighted averages per runner-season
    print("  Aggregating opponent pop time...")
    opponent_poptime = weights_poptime.groupby(['runner_id', 'season']).apply(
        lambda x: pd.Series({
            'opponent_avg_poptime': np.average(
                x['pop_time_2b_avg'].dropna(),
                weights=x.loc[x['pop_time_2b_avg'].notna(), 'weight']
            ) if len(x['pop_time_2b_avg'].dropna()) > 0 else np.nan
        })
    ).reset_index()
    
    print("  Aggregating opponent tempo...")
    opponent_tempo = weights_tempo.groupby(['runner_id', 'season']).apply(
        lambda x: pd.Series({
            'opponent_avg_tempo': np.average(
                x['tempo_onbase_avg'].dropna(),
                weights=x.loc[x['tempo_onbase_avg'].notna(), 'weight']
            ) if len(x['tempo_onbase_avg'].dropna()) > 0 else np.nan
        })
    ).reset_index()
    
    print(f"  Computed opponent metrics for {len(opponent_poptime):,} runner-seasons")
    
    # Merge into runner panel
    runner_panel = runner_panel.merge(
        opponent_poptime[['runner_id', 'season', 'opponent_avg_poptime']],
        on=['runner_id', 'season'],
        how='left'
    )
    
    runner_panel = runner_panel.merge(
        opponent_tempo[['runner_id', 'season', 'opponent_avg_tempo']],
        on=['runner_id', 'season'],
        how='left'
    )
    
    return runner_panel

def main():
    print("=" * 80)
    print("BUILDING RUNNER PANEL")
    print("=" * 80)
    print(f"\nCurrent directory: {Path.cwd()}")
    
    leaderboards_path = Path("leaderboards")
    intermediate_path = Path("analysis/intermediate")
    
    # Check paths exist
    if not leaderboards_path.exists():
        print(f"\nERROR: {leaderboards_path.absolute()} not found!")
        print("Please run this script from the data/ directory")
        sys.exit(1)
    
    if not intermediate_path.exists():
        print(f"\nERROR: {intermediate_path.absolute()} not found!")
        print("Please run scripts 03-05 first to generate intermediate files")
        sys.exit(1)
    
    # 1. Load Custom Stats (SB, CS, Success Rate)
    print("\n" + "=" * 80)
    print("LOADING RUNNER LEADERBOARDS")
    print("=" * 80)
    
    print("\n1. Custom Stats (SB, CS)...")
    custom = load_all_years(leaderboards_path, "custom_stats_*.csv")
    if custom.empty:
        print("  ERROR: No custom stats data")
        sys.exit(1)
    
    # Calculate derived fields
    custom['cs'] = (
        custom.get('r_caught_stealing_2b', 0).fillna(0) + 
        custom.get('r_caught_stealing_3b', 0).fillna(0) + 
        custom.get('r_caught_stealing_home', 0).fillna(0)
    )
    custom['sb'] = custom['r_total_stolen_base']
    custom['attempts'] = custom['sb'] + custom['cs']
    custom['success_rate'] = custom['r_stolen_base_pct']
    
    # Handle player names from custom_stats
    if 'first_name' in custom.columns and 'last_name' in custom.columns:
        custom['player_name'] = custom['first_name'] + ' ' + custom['last_name']
    else:
        custom['player_name'] = None
    
    custom['runner_id'] = custom['player_id']
    
    print(f"  Loaded {len(custom):,} runner-seasons")
    
    # 2. Load Baserunning Run Value
    print("\n2. Baserunning Run Value (BRV)...")
    brv = load_all_years(leaderboards_path, "baserunning_run_value_*.csv")
    if brv.empty:
        print("  ERROR: No BRV data")
        sys.exit(1)
    
    brv = brv[['player_id', 'season', 'runner_runs_tot', 'runner_runs_XB', 'runner_runs_SBX']].copy()
    brv.columns = ['runner_id', 'season', 'brv_total', 'brv_xb', 'brv_steal']
    
    print(f"  Loaded {len(brv):,} runner-seasons")
    
    # 3. Load Basestealing/Lead Distance
    print("\n3. Basestealing/Lead Distance...")
    stealing = load_all_years(leaderboards_path, "basestealing_running_game_*.csv")
    if stealing.empty:
        print("  ERROR: No basestealing data")
        sys.exit(1)
    
    # Get player names from basestealing if not in custom_stats
    stealing_cols = ['player_id', 'season', 'r_primary_lead', 'r_secondary_lead']
    if 'player_name' in stealing.columns:
        stealing_cols.append('player_name')
    
    stealing = stealing[stealing_cols].copy()
    stealing = stealing.rename(columns={'player_id': 'runner_id'})
    
    print(f"  Loaded {len(stealing):,} runner-seasons")
    
    # 4. Load Sprint Speed
    print("\n4. Sprint Speed (+ team, age)...")
    speed = load_all_years(leaderboards_path, "sprint_speed_*.csv")
    if speed.empty:
        print("  ERROR: No sprint speed data")
        sys.exit(1)
    
    speed = speed[['player_id', 'season', 'sprint_speed', 'team_id', 'team', 'age']].copy()
    speed.columns = ['runner_id', 'season', 'sprint_speed_fps', 'team_id', 'team_name', 'age']
    
    print(f"  Loaded {len(speed):,} runner-seasons")
    
    # 5. Merge runner data
    print("\n" + "=" * 80)
    print("MERGING RUNNER METRICS")
    print("=" * 80)
    
    # Start with custom stats
    runner = custom[['runner_id', 'season', 'player_name', 'sb', 'cs', 'attempts', 'success_rate']].copy()
    
    # Merge other metrics
    runner = runner.merge(brv, on=['runner_id', 'season'], how='left')
    runner = runner.merge(stealing, on=['runner_id', 'season'], how='left', suffixes=('', '_steal'))
    runner = runner.merge(speed, on=['runner_id', 'season'], how='left')
    
    # Use player_name from stealing if missing from custom
    if 'player_name_steal' in runner.columns:
        runner['player_name'] = runner['player_name'].fillna(runner['player_name_steal'])
        runner = runner.drop(columns=['player_name_steal'])
    
    print(f"\nMerged runner data: {len(runner):,} rows")
    print(f"Unique runners: {runner['runner_id'].nunique():,}")
    print(f"Seasons: {runner['season'].min()}-{runner['season'].max()}")
    
    # 6. Load intermediate files
    print("\n" + "=" * 80)
    print("LOADING INTERMEDIATE FILES")
    print("=" * 80)
    
    team_poptime_file = intermediate_path / "team_poptime_2018_2025.csv"
    team_tempo_file = intermediate_path / "team_tempo_2018_2025.csv"
    schedule_weights_file = intermediate_path / "runner_schedule_weights_2018_2025.csv"
    
    required_files = [team_poptime_file, team_tempo_file, schedule_weights_file]
    missing = [f for f in required_files if not f.exists()]
    
    if missing:
        print("\nERROR: Missing intermediate files:")
        for f in missing:
            print(f"  - {f}")
        print("\nPlease run scripts 03-05 first!")
        sys.exit(1)
    
    print("\nLoading team metrics...")
    team_poptime = pd.read_csv(team_poptime_file)
    team_tempo = pd.read_csv(team_tempo_file)
    schedule_weights = pd.read_csv(schedule_weights_file)
    
    print(f"  Team pop time: {len(team_poptime):,} rows")
    print(f"  Team tempo: {len(team_tempo):,} rows")
    print(f"  Schedule weights: {len(schedule_weights):,} rows")
    
    # 7. Compute opponent metrics
    print("\n" + "=" * 80)
    print("COMPUTING OPPONENT METRICS")
    print("=" * 80)
    
    runner = compute_weighted_opponent_metrics(
        runner, schedule_weights, team_poptime, team_tempo
    )
    
    # 8. Add treatment variable
    print("\nAdding treatment variables...")
    runner['post_2023'] = (runner['season'] >= 2023).astype(int)
    
    # 9. Check for multi-team players
    print("\nChecking for multi-team players...")
    # Simple check: if player has multiple team_ids in same season
    multi_team_check = runner.groupby(['runner_id', 'season'])['team_id'].nunique()
    multi_team_ids = multi_team_check[multi_team_check > 1].index
    runner['multi_team'] = 0
    for rid, season in multi_team_ids:
        runner.loc[(runner['runner_id'] == rid) & (runner['season'] == season), 'multi_team'] = 1
    
    multi_team_count = runner['multi_team'].sum()
    if multi_team_count > 0:
        print(f"  Found {multi_team_count} multi-team player-seasons")
    
    # 10. Finalize columns
    print("\nFinalizing dataset...")
    
    final_cols = [
        'season', 'runner_id', 'player_name',
        'team_id', 'team_name', 'age',
        'sb', 'cs', 'attempts', 'success_rate',
        'brv_total', 'brv_xb', 'brv_steal',
        'r_primary_lead', 'r_secondary_lead',
        'sprint_speed_fps',
        'opponent_avg_poptime', 'opponent_avg_tempo',
        'multi_team', 'post_2023'
    ]
    
    # Keep only columns that exist
    final_cols = [c for c in final_cols if c in runner.columns]
    runner_final = runner[final_cols].copy()
    runner_final = runner_final.sort_values(['season', 'runner_id'])
    
    # 11. QA Checks
    print("\n" + "=" * 80)
    print("QA CHECKS")
    print("=" * 80)
    
    print(f"\nRows: {len(runner_final):,}")
    print(f"Seasons: {runner_final['season'].min()}-{runner_final['season'].max()}")
    print(f"Unique players: {runner_final['runner_id'].nunique():,}")
    
    print("\nKey variables - Missing data:")
    key_vars = ['sb', 'cs', 'sprint_speed_fps', 'opponent_avg_poptime', 'opponent_avg_tempo']
    for col in key_vars:
        if col in runner_final.columns:
            missing = runner_final[col].isna().sum()
            pct = (missing / len(runner_final)) * 100
            print(f"  {col:25s}: {missing:5,} ({pct:5.1f}%)")
    
    print("\nSummary stats:")
    summary_cols = ['sb', 'cs', 'success_rate', 'sprint_speed_fps', 
                   'opponent_avg_poptime', 'opponent_avg_tempo']
    summary_cols = [c for c in summary_cols if c in runner_final.columns]
    print(runner_final[summary_cols].describe())
    
    # Check opponent metrics are reasonable
    if 'opponent_avg_poptime' in runner_final.columns:
        pop_valid = runner_final['opponent_avg_poptime'].notna()
        if pop_valid.any():
            pop_range = (runner_final.loc[pop_valid, 'opponent_avg_poptime'].min(), 
                        runner_final.loc[pop_valid, 'opponent_avg_poptime'].max())
            print(f"\nOpponent pop time range: {pop_range[0]:.3f} - {pop_range[1]:.3f} sec")
            if pop_range[0] < 1.8 or pop_range[1] > 2.2:
                print("  WARNING: Pop time outside expected range (1.8-2.2 sec)")
    
    if 'opponent_avg_tempo' in runner_final.columns:
        tempo_valid = runner_final['opponent_avg_tempo'].notna()
        if tempo_valid.any():
            tempo_range = (runner_final.loc[tempo_valid, 'opponent_avg_tempo'].min(),
                          runner_final.loc[tempo_valid, 'opponent_avg_tempo'].max())
            print(f"Opponent tempo range: {tempo_range[0]:.2f} - {tempo_range[1]:.2f} sec")
            if tempo_range[0] < 13 or tempo_range[1] > 30:
                print("  WARNING: Tempo outside expected range (13-30 sec)")
    
    # 12. Save
    output_path = Path("analysis")
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "analysis_runner_panel.csv"
    
    runner_final.to_csv(output_file, index=False)
    
    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"\nSaved: {output_file}")
    print(f"Size: {output_file.stat().st_size / 1024:.1f} KB")
    print(f"\nRunner panel ready for analysis!")

if __name__ == "__main__":
    main()