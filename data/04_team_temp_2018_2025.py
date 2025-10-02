"""
Computes team-level Pitch Tempo aggregated from individual pitchers
Weighted by pitches thrown for better accuracy

Output: team_tempo_2018_2025.csv (includes league means)

Usage: python 04_compute_team_tempo.py --data data/leaderboards --output data/analysis/intermediate
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def compute_team_tempo(leaderboards_path, output_path):
    """
    Aggregate pitcher tempo to team-season level
    Weight by pitches thrown (especially with runners on base)
    """
    
    print("=" * 80)
    print("COMPUTING TEAM-LEVEL PITCH TEMPO")
    print("=" * 80)
    
    leaderboards_path = Path(leaderboards_path)
    all_teams = []
    
    for year in range(2018, 2026):
        file = leaderboards_path / f"pitch_tempo_{year}.csv"
        
        if not file.exists():
            print(f"\nWarning: Missing {file.name}")
            continue
        
        print(f"\nProcessing {year}...")
        df = pd.read_csv(file)
        
        # Check for team_id column
        if 'team_id' not in df.columns:
            print(f"  Error: Missing team_id column")
            continue
        
        # Find tempo column (prioritize with runners on base)
        tempo_col = None
        weight_col = None
        
        if 'median_seconds_onbase' in df.columns:
            tempo_col = 'median_seconds_onbase'
            weight_col = 'total_pitches_onbase' if 'total_pitches_onbase' in df.columns else 'total_pitches'
        elif 'median_seconds_empty' in df.columns:
            tempo_col = 'median_seconds_empty'
            weight_col = 'total_pitches_empty' if 'total_pitches_empty' in df.columns else 'total_pitches'
        elif 'median_seconds' in df.columns:
            tempo_col = 'median_seconds'
            weight_col = 'total_pitches'
        else:
            print(f"  Error: No tempo columns found")
            print(f"  Available columns: {df.columns.tolist()}")
            continue
        
        if weight_col not in df.columns:
            print(f"  Error: Missing weight column {weight_col}")
            continue
        
        print(f"  Using: {tempo_col} weighted by {weight_col}")
        
        # Filter valid data
        df = df[
            (df['team_id'].notna()) & 
            (df[tempo_col].notna()) &
            (df[weight_col] > 0)
        ].copy()
        
        if len(df) == 0:
            print(f"  No valid data")
            continue
        
        print(f"  {len(df)} pitchers with valid tempo data")
        
        # Check for multi-team pitchers
        if 'entity_id' in df.columns:
            multi_team = df.groupby('entity_id')['team_id'].nunique()
            multi_team_count = (multi_team > 1).sum()
            if multi_team_count > 0:
                print(f"  WARNING: {multi_team_count} pitchers with multiple teams (mid-season trades)")
        
        # Compute weighted average and std per team
        def safe_weighted_avg(group, value_col, weight_col):
            """Pitches-weighted average with zero-weight protection"""
            valid = (group[value_col].notna()) & (group[weight_col] > 0)
            if not valid.any() or group.loc[valid, weight_col].sum() == 0:
                return np.nan
            return np.average(
                group.loc[valid, value_col], 
                weights=group.loc[valid, weight_col]
            )
        
        def safe_weighted_std(group, value_col, weight_col):
            """Pitches-weighted standard deviation"""
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
                'tempo_onbase_avg': safe_weighted_avg(team_group, tempo_col, weight_col),
                'tempo_onbase_std': safe_weighted_std(team_group, tempo_col, weight_col),
                'total_pitches': team_group[weight_col].sum(),
                'n_pitchers': len(team_group)
            }
            
            team_data.append(team_row)
        
        team_agg = pd.DataFrame(team_data)
        
        # Sanity check: Should have exactly 30 teams
        n_teams = len(team_agg)
        if n_teams != 30:
            print(f"  WARNING: Expected 30 teams, got {n_teams}")
        
        # Flag low-reliability estimates (arbitrary threshold: <1000 pitches)
        low_pitches = team_agg['total_pitches'] < 1000
        if low_pitches.any():
            print(f"  WARNING: {low_pitches.sum()} teams with <1000 pitches (low reliability)")
            print(f"           Team IDs: {team_agg.loc[low_pitches, 'team_id'].tolist()}")
        
        print(f"  {n_teams} teams")
        print(f"  Avg tempo: {team_agg['tempo_onbase_avg'].mean():.2f} sec (std: {team_agg['tempo_onbase_std'].mean():.2f})")
        
        all_teams.append(team_agg)
    
    if not all_teams:
        print("\nError: No data collected")
        return
    
    # Combine all years
    result = pd.concat(all_teams, ignore_index=True)
    
    # Flag unreliable estimates
    result['low_reliability'] = (result['total_pitches'] < 1000).astype(int)
    
    # Compute league means per season (pitches-weighted)
    print("\nComputing league means...")
    league_means = []
    
    for season in result['season'].unique():
        season_data = result[result['season'] == season].copy()
        
        # Use only reliable estimates for league mean
        reliable = season_data[season_data['low_reliability'] == 0]
        if len(reliable) == 0:
            reliable = season_data
        
        total_pitches = reliable['total_pitches'].sum()
        if total_pitches > 0:
            league_mean = np.average(
                reliable['tempo_onbase_avg'],
                weights=reliable['total_pitches']
            )
        else:
            league_mean = reliable['tempo_onbase_avg'].mean()
        
        league_means.append({
            'season': season,
            'league_tempo_onbase_mean': league_mean,
            'teams_in_season': len(season_data)
        })
        
        print(f"  {season}: {league_mean:.2f} sec (from {len(reliable)} teams)")
    
    league_df = pd.DataFrame(league_means)
    
    # Join league means to team data
    result = result.merge(league_df, on='season', how='left')
    
    # Compute relative tempo (team - league)
    result['tempo_diff'] = result['tempo_onbase_avg'] - result['league_tempo_onbase_mean']
    
    # Sort
    result = result.sort_values(['season', 'team_id'])
    
    # Reorder columns for clarity
    cols = [
        'season', 'team_id', 'tempo_onbase_avg', 'tempo_onbase_std',
        'league_tempo_onbase_mean', 'tempo_diff', 'teams_in_season',
        'total_pitches', 'n_pitchers', 'low_reliability'
    ]
    result = result[cols]
    
    # Save
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_path / "team_tempo_2018_2025.csv"
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
    
    print(f"\nAvg pitchers per team: {result['n_pitchers'].mean():.1f}")
    print(f"Avg pitches per team: {result['total_pitches'].mean():.0f}")
    
    low_rel_count = result['low_reliability'].sum()
    if low_rel_count > 0:
        print(f"\nLow reliability (<1000 pitches): {low_rel_count} team-seasons ({low_rel_count/len(result)*100:.1f}%)")
    
    print(f"\nTempo (with runners on base):")
    print(f"  Mean: {result['tempo_onbase_avg'].mean():.2f} sec")
    print(f"  Std (avg across teams): {result['tempo_onbase_std'].mean():.2f} sec")
    print(f"  Range: {result['tempo_onbase_avg'].min():.2f} - {result['tempo_onbase_avg'].max():.2f} sec")
    
    print(f"\nLeague means over time:")
    print(league_df.to_string(index=False))
    
    # Check for major tempo shift around 2023 (pitch clock introduction)
    if result['season'].max() >= 2023:
        pre_2023 = result[result['season'] < 2023]['tempo_onbase_avg'].mean()
        post_2023 = result[result['season'] >= 2023]['tempo_onbase_avg'].mean()
        reduction = pre_2023 - post_2023
        pct_change = (reduction / pre_2023) * 100
        
        print(f"\nPitch Clock Effect (2023 Rule Change):")
        print(f"  Pre-2023 avg: {pre_2023:.2f} sec")
        print(f"  2023+ avg: {post_2023:.2f} sec")
        print(f"  Reduction: {reduction:.2f} sec ({pct_change:.1f}%)")
        print(f"  Note: 20-second timer introduced 2023; enforced more strictly 2024+")
    
    print(f"\nSaved: {output_file}")
    print(f"Size: {output_file.stat().st_size / 1024:.1f} KB")
    print("\nComplete")

def main():
    parser = argparse.ArgumentParser(description='Compute team-level pitch tempo metrics')
    parser.add_argument('--data', type=str, default='data/leaderboards',
                       help='Path to leaderboards folder')
    parser.add_argument('--output', type=str, default='data/analysis/intermediate',
                       help='Output folder for intermediate files')
    
    args = parser.parse_args()
    
    compute_team_tempo(args.data, args.output)

if __name__ == "__main__":
    main()