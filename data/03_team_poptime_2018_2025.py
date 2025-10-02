"""
Computes team-level Pop Time aggregated from individual catchers
Weighted by steal attempts for better accuracy

Output: team_poptime_2018_2025.csv

Usage: python 03_compute_team_poptime.py --data data/leaderboards --output data/analysis/intermediate
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def compute_team_poptime(leaderboards_path, output_path):
    """
    Aggregate catcher pop times to team-season level
    Weight by attempts (pop_2b_sba_count) for accuracy
    """
    
    print("=" * 80)
    print("COMPUTING TEAM-LEVEL POP TIME")
    print("=" * 80)
    
    leaderboards_path = Path(leaderboards_path)
    all_teams = []
    
    for year in range(2018, 2026):
        file = leaderboards_path / f"poptime_{year}.csv"
        
        if not file.exists():
            print(f"\nWarning: Missing {file.name}")
            continue
        
        print(f"\nProcessing {year}...")
        df = pd.read_csv(file)
        
        # Check required columns
        required = ['team_id', 'pop_2b_sba', 'pop_2b_sba_count']
        missing = [col for col in required if col not in df.columns]
        if missing:
            print(f"  Error: Missing columns {missing}")
            continue
        
        # Filter valid data
        df = df[
            (df['team_id'].notna()) & 
            (df['pop_2b_sba'].notna()) &
            (df['pop_2b_sba_count'] > 0)
        ].copy()
        
        if len(df) == 0:
            print(f"  No valid data")
            continue
        
        print(f"  {len(df)} catchers with valid pop time data")
        
        # Check for multi-team catchers (same entity_id, multiple team_ids in this year)
        if 'entity_id' in df.columns:
            multi_team = df.groupby('entity_id')['team_id'].nunique()
            multi_team_count = (multi_team > 1).sum()
            if multi_team_count > 0:
                print(f"  WARNING: {multi_team_count} catchers with multiple teams (mid-season trades)")
                print(f"           Attempts are attributed to current team_id in data")
        
        # Compute weighted average and std per team
        def safe_weighted_avg(group, value_col, weight_col):
            """Attempts-weighted average with zero-weight protection"""
            valid = (group[value_col].notna()) & (group[weight_col] > 0)
            if not valid.any() or group.loc[valid, weight_col].sum() == 0:
                return np.nan
            return np.average(
                group.loc[valid, value_col], 
                weights=group.loc[valid, weight_col]
            )
        
        def safe_weighted_std(group, value_col, weight_col):
            """Attempts-weighted standard deviation"""
            valid = (group[value_col].notna()) & (group[weight_col] > 0)
            if not valid.any() or group.loc[valid, weight_col].sum() == 0 or valid.sum() < 2:
                return np.nan
            
            values = group.loc[valid, value_col]
            weights = group.loc[valid, weight_col]
            avg = np.average(values, weights=weights)
            variance = np.average((values - avg)**2, weights=weights)
            return np.sqrt(variance)
        
        # Aggregate to team level
        team_data = []
        for team_id, team_group in df.groupby('team_id'):
            team_row = {
                'team_id': team_id,
                'season': year,
                'pop_time_2b_avg': safe_weighted_avg(team_group, 'pop_2b_sba', 'pop_2b_sba_count'),
                'pop_time_2b_std': safe_weighted_std(team_group, 'pop_2b_sba', 'pop_2b_sba_count'),
                'total_attempts_2b': team_group['pop_2b_sba_count'].sum(),
                'n_catchers': len(team_group)
            }
            
            # Add 3B pop time if available
            if 'pop_3b_sba' in df.columns and 'pop_3b_sba_count' in df.columns:
                team_row['pop_time_3b_avg'] = safe_weighted_avg(team_group, 'pop_3b_sba', 'pop_3b_sba_count')
                team_row['pop_time_3b_std'] = safe_weighted_std(team_group, 'pop_3b_sba', 'pop_3b_sba_count')
                team_row['total_attempts_3b'] = team_group['pop_3b_sba_count'].sum()
            else:
                team_row['pop_time_3b_avg'] = np.nan
                team_row['pop_time_3b_std'] = np.nan
                team_row['total_attempts_3b'] = 0
            
            team_data.append(team_row)
        
        team_agg = pd.DataFrame(team_data)
        
        # Sanity check: Should have exactly 30 teams
        n_teams = len(team_agg)
        if n_teams != 30:
            print(f"  WARNING: Expected 30 teams, got {n_teams}")
        
        # Flag low-reliability estimates
        low_attempts = team_agg['total_attempts_2b'] < 10
        if low_attempts.any():
            print(f"  WARNING: {low_attempts.sum()} teams with <10 attempts (low reliability)")
            print(f"           Team IDs: {team_agg.loc[low_attempts, 'team_id'].tolist()}")
        
        print(f"  {n_teams} teams")
        print(f"  Avg pop time: {team_agg['pop_time_2b_avg'].mean():.3f} sec (std: {team_agg['pop_time_2b_std'].mean():.3f})")
        
        all_teams.append(team_agg)
    
    if not all_teams:
        print("\nError: No data collected")
        return
    
    # Combine all years
    result = pd.concat(all_teams, ignore_index=True)
    
    # Flag unreliable estimates in final data
    result['low_reliability'] = (result['total_attempts_2b'] < 10).astype(int)
    
    # Sort
    result = result.sort_values(['season', 'team_id'])
    
    # Save
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_path / "team_poptime_2018_2025.csv"
    result.to_csv(output_file, index=False)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nTotal team-seasons: {len(result):,}")
    print(f"Unique teams: {result['team_id'].nunique()}")
    print(f"Seasons: {result['season'].min()}-{result['season'].max()}")
    
    # Check season coverage
    teams_per_season = result.groupby('season')['team_id'].nunique()
    print(f"\nTeams per season:")
    for season, count in teams_per_season.items():
        flag = "" if count == 30 else " ⚠️"
        print(f"  {season}: {count}{flag}")
    
    print(f"\nAvg catchers per team: {result['n_catchers'].mean():.1f}")
    print(f"Avg attempts per team: {result['total_attempts_2b'].mean():.0f}")
    
    low_rel_count = result['low_reliability'].sum()
    if low_rel_count > 0:
        print(f"\nLow reliability (<10 attempts): {low_rel_count} team-seasons ({low_rel_count/len(result)*100:.1f}%)")
    
    print(f"\nPop Time 2B:")
    print(f"  Mean: {result['pop_time_2b_avg'].mean():.3f} sec")
    print(f"  Std (avg across teams): {result['pop_time_2b_std'].mean():.3f} sec")
    print(f"  Range: {result['pop_time_2b_avg'].min():.3f} - {result['pop_time_2b_avg'].max():.3f} sec")
    
    print(f"\nSaved: {output_file}")
    print(f"Size: {output_file.stat().st_size / 1024:.1f} KB")
    print("\nComplete")

def main():
    parser = argparse.ArgumentParser(description='Compute team-level pop time metrics')
    parser.add_argument('--data', type=str, default='data/leaderboards',
                       help='Path to leaderboards folder')
    parser.add_argument('--output', type=str, default='data/analysis/intermediate',
                       help='Output folder for intermediate files')
    
    args = parser.parse_args()
    
    compute_team_poptime(args.data, args.output)

if __name__ == "__main__":
    main()