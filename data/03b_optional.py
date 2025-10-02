"""
Computes season-relative pop time metrics
Adds league means and standardized differences

Input: team_poptime_2018_2025.csv
Output: team_poptime_2018_2025_relative.csv

Usage: python 03b_relative_poptime_metrics.py --input data/analysis/intermediate/team_poptime_2018_2025.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def compute_relative_metrics(input_file):
    """
    Compute league means and relative pop time metrics
    """
    
    print("=" * 80)
    print("COMPUTING RELATIVE POP TIME METRICS")
    print("=" * 80)
    
    # Load team data
    df = pd.read_csv(input_file)
    print(f"\nLoaded: {len(df)} team-seasons")
    
    # 1. Compute league mean per season (attempts-weighted)
    print("\nComputing league means (attempts-weighted)...")
    
    league_means = []
    for season in df['season'].unique():
        season_data = df[df['season'] == season].copy()
        
        # Filter out low reliability estimates for league mean
        reliable = season_data[season_data['low_reliability'] == 0]
        
        if len(reliable) == 0:
            print(f"  WARNING {season}: No reliable data, using all teams")
            reliable = season_data
        
        # Attempts-weighted league mean
        total_attempts = reliable['total_attempts_2b'].sum()
        if total_attempts > 0:
            league_mean_2b = np.average(
                reliable['pop_time_2b_avg'],
                weights=reliable['total_attempts_2b']
            )
        else:
            league_mean_2b = reliable['pop_time_2b_avg'].mean()
        
        # Same for 3B if available
        reliable_3b = reliable[reliable['pop_time_3b_avg'].notna()]
        if len(reliable_3b) > 0 and reliable_3b['total_attempts_3b'].sum() > 0:
            league_mean_3b = np.average(
                reliable_3b['pop_time_3b_avg'],
                weights=reliable_3b['total_attempts_3b']
            )
        else:
            league_mean_3b = np.nan
        
        league_means.append({
            'season': season,
            'league_mean_pop2b': league_mean_2b,
            'league_mean_pop3b': league_mean_3b,
            'n_teams': len(season_data),
            'n_reliable': len(reliable)
        })
        
        print(f"  {season}: {league_mean_2b:.3f} sec (from {len(reliable)} teams)")
    
    league_df = pd.DataFrame(league_means)
    
    # 2. Join league means to team data
    df = df.merge(league_df[['season', 'league_mean_pop2b', 'league_mean_pop3b']], 
                   on='season', how='left')
    
    # 3. Compute relative metrics
    print("\nComputing relative metrics...")
    
    # Difference from league mean
    df['pop2b_diff'] = df['pop_time_2b_avg'] - df['league_mean_pop2b']
    df['pop3b_diff'] = df['pop_time_3b_avg'] - df['league_mean_pop3b']
    
    # Z-score (season-standardized)
    for season in df['season'].unique():
        season_mask = df['season'] == season
        season_data = df[season_mask]
        
        # Use only reliable estimates for std calculation
        reliable = season_data[season_data['low_reliability'] == 0]
        
        if len(reliable) > 1:
            pop2b_std = reliable['pop_time_2b_avg'].std()
            pop3b_std = reliable['pop_time_3b_avg'].std()
        else:
            pop2b_std = season_data['pop_time_2b_avg'].std()
            pop3b_std = season_data['pop_time_3b_avg'].std()
        
        # Z-score = (team - league_mean) / league_std
        df.loc[season_mask, 'pop2b_zscore'] = df.loc[season_mask, 'pop2b_diff'] / pop2b_std if pop2b_std > 0 else 0
        df.loc[season_mask, 'pop3b_zscore'] = df.loc[season_mask, 'pop3b_diff'] / pop3b_std if pop3b_std > 0 else np.nan
    
    # Percentile rank within season
    df['pop2b_percentile'] = df.groupby('season')['pop_time_2b_avg'].rank(pct=True) * 100
    df['pop3b_percentile'] = df.groupby('season')['pop_time_3b_avg'].rank(pct=True) * 100
    
    # 4. Save
    output_file = Path(input_file).parent / "team_poptime_2018_2025_relative.csv"
    df.to_csv(output_file, index=False)
    
    # 5. Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print("\nLeague means over time:")
    print(league_df[['season', 'league_mean_pop2b', 'n_reliable']].to_string(index=False))
    
    print("\nRelative metrics distribution:")
    print(f"\npop2b_diff (team - league):")
    print(f"  Mean: {df['pop2b_diff'].mean():.4f} (should be ~0)")
    print(f"  Std: {df['pop2b_diff'].std():.4f}")
    print(f"  Range: {df['pop2b_diff'].min():.3f} to {df['pop2b_diff'].max():.3f} sec")
    
    print(f"\npop2b_zscore:")
    print(f"  Mean: {df['pop2b_zscore'].mean():.4f} (should be ~0)")
    print(f"  Std: {df['pop2b_zscore'].std():.4f} (should be ~1)")
    print(f"  Range: {df['pop2b_zscore'].min():.2f} to {df['pop2b_zscore'].max():.2f}")
    
    print(f"\nFastest teams (lowest pop time relative to league):")
    top_fast = df.nsmallest(5, 'pop2b_diff')[['season', 'team_id', 'pop_time_2b_avg', 'pop2b_diff', 'pop2b_zscore']]
    print(top_fast.to_string(index=False))
    
    print(f"\nSlowest teams (highest pop time relative to league):")
    top_slow = df.nlargest(5, 'pop2b_diff')[['season', 'team_id', 'pop_time_2b_avg', 'pop2b_diff', 'pop2b_zscore']]
    print(top_slow.to_string(index=False))
    
    print(f"\nSaved: {output_file}")
    print(f"Size: {output_file.stat().st_size / 1024:.1f} KB")
    print("\nComplete")

def main():
    parser = argparse.ArgumentParser(description='Compute relative pop time metrics')
    parser.add_argument('--input', type=str, 
                       default='data/analysis/intermediate/team_poptime_2018_2025.csv',
                       help='Input team pop time file')
    
    args = parser.parse_args()
    
    compute_relative_metrics(args.input)

if __name__ == "__main__":
    main()