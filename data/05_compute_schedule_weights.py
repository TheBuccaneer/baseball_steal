"""
Computes schedule-weighted opponent exposure for runners
Extracts which teams each runner played against and how often

Output: runner_schedule_weights_2018_2025.csv

Usage: python 05_compute_schedule_weights.py --data data/mlb_stats --output data/analysis/intermediate
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys

def get_mlb_team_mapping():
    """
    Official MLB team name -> team_id mapping
    Includes historical names (Cleveland Indians, Oakland Athletics)
    """
    return {
        'Arizona Diamondbacks': 109,
        'Atlanta Braves': 144,
        'Baltimore Orioles': 110,
        'Boston Red Sox': 111,
        'Chicago Cubs': 112,
        'Chicago White Sox': 145,
        'Cincinnati Reds': 113,
        'Cleveland Indians': 114,  # Pre-2022
        'Cleveland Guardians': 114,  # 2022+
        'Colorado Rockies': 115,
        'Detroit Tigers': 116,
        'Houston Astros': 117,
        'Kansas City Royals': 118,
        'Los Angeles Angels': 108,
        'Los Angeles Dodgers': 119,
        'Miami Marlins': 146,
        'Milwaukee Brewers': 158,
        'Minnesota Twins': 142,
        'New York Mets': 121,
        'New York Yankees': 147,
        'Oakland Athletics': 133,  # Pre-2025
        'Athletics': 133,  # 2025+ (no city)
        'Philadelphia Phillies': 143,
        'Pittsburgh Pirates': 134,
        'San Diego Padres': 135,
        'San Francisco Giants': 137,
        'Seattle Mariners': 136,
        'St. Louis Cardinals': 138,
        'Tampa Bay Rays': 139,
        'Texas Rangers': 140,
        'Toronto Blue Jays': 141,
        'Washington Nationals': 120
    }

def build_team_mapping():
    """Use hardcoded MLB team mapping"""
    print("\n" + "="*80)
    print("LOADING TEAM NAME → TEAM ID MAPPING")
    print("="*80)
    
    team_mapping = get_mlb_team_mapping()
    
    print(f"\nLoaded {len(team_mapping)} team name mappings")
    print(f"Includes historical names: Cleveland Indians/Guardians, Oakland/Athletics")
    
    return team_mapping
def extract_runner_opponents(mlb_stats_folder):
    """
    Extract opponent teams for each runner from MLB-Stats PBP data
    Uses half_inning logic: top=away_team, bottom=home_team
    """
    print(f"\nProcessing: {mlb_stats_folder.name}")
    
    csv_files = sorted(mlb_stats_folder.glob("*.csv"))
    
    if not csv_files:
        print(f"  No CSV files found")
        return pd.DataFrame()
    
    all_events = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, low_memory=False)
            all_events.append(df)
        except Exception as e:
            print(f"  Error reading {csv_file.name}: {e}")
    
    if not all_events:
        return pd.DataFrame()
    
    pbp = pd.concat(all_events, ignore_index=True)
    print(f"  Loaded {len(pbp):,} events")
    
    year = int(mlb_stats_folder.name.split('_')[-1])
    
    required = ['game_id', 'away_team', 'home_team', 'half_inning']
    missing = [col for col in required if col not in pbp.columns]
    if missing:
        print(f"  ERROR: Missing columns {missing}")
        return pd.DataFrame()
    
    pbp['opponent_team'] = pbp.apply(
        lambda row: row['home_team'] if row['half_inning'] == 'top' else row['away_team'],
        axis=1
    )
    
    runner_events = pbp[
        (pbp['runner_id'].notna()) & 
        (pbp['start_base'].notna())
    ].copy()
    
    print(f"  Found {len(runner_events):,} runner on-base events")
    
    if len(runner_events) == 0:
        return pd.DataFrame()
    
    onbase_counts = runner_events.groupby(
        ['runner_id', 'opponent_team']
    ).size().reset_index(name='onbase_count')
    
    game_counts = runner_events.groupby(
        ['runner_id', 'opponent_team']
    )['game_id'].nunique().reset_index(name='game_count')
    
    weights = onbase_counts.merge(game_counts, on=['runner_id', 'opponent_team'], how='outer')
    weights['season'] = year
    
    print(f"  Extracted {len(weights):,} runner-opponent pairs")
    
    return weights

def main():
    parser = argparse.ArgumentParser(description='Compute Schedule Weights')
    parser.add_argument('--data', type=str, default='mlb_stats',
                       help='Path to mlb_stats folder')
    parser.add_argument('--output', type=str, default='analysis/intermediate',
                       help='Output folder')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("COMPUTE SCHEDULE WEIGHTS")
    print("=" * 80)
    print(f"\nCurrent directory: {Path.cwd()}")
    
    # Build team mapping FIRST
    team_mapping = build_team_mapping()
    
    # Process MLB stats
    mlb_stats_path = Path(args.data)
    
    year_folders = sorted([d for d in mlb_stats_path.iterdir() 
                          if d.is_dir() and d.name.startswith('mlb_stats_')])
    
    if not year_folders:
        print(f"\nNo year folders found in {mlb_stats_path}")
        return
    
    print(f"\n{'='*80}")
    print(f"PROCESSING {len(year_folders)} YEAR FOLDERS")
    print(f"{'='*80}")
    
    all_weights = []
    
    for year_folder in year_folders:
        weights = extract_runner_opponents(year_folder)
        if not weights.empty:
            all_weights.append(weights)
    
    if not all_weights:
        print("\nNo data collected")
        return
    
    print("\nCombining all years...")
    combined = pd.concat(all_weights, ignore_index=True)
    
    print(f"Total runner-opponent-season pairs: {len(combined):,}")
    
    # Map team names to IDs
    print("\nMapping opponent names to team IDs...")
    combined['opponent_team_id'] = combined['opponent_team'].map(team_mapping)
    
    unmapped = combined[combined['opponent_team_id'].isna()]['opponent_team'].unique()
    if len(unmapped) > 0:
        print(f"\nWARNING: {len(unmapped)} team names couldn't be mapped:")
        for team in sorted(unmapped):
            count = (combined['opponent_team'] == team).sum()
            print(f"  - '{team}' ({count} occurrences)")
    else:
        print("  ✓ All team names successfully mapped!")
    
    # Compute weights
    print("\nComputing normalized weights...")
    
    combined['total_onbase_by_runner'] = combined.groupby(['runner_id', 'season'])['onbase_count'].transform('sum')
    combined['weight_onbase'] = combined['onbase_count'] / combined['total_onbase_by_runner']
    
    combined['total_games_by_runner'] = combined.groupby(['runner_id', 'season'])['game_count'].transform('sum')
    combined['weight_games'] = combined['game_count'] / combined['total_games_by_runner']
    
    combined['weight'] = combined['weight_onbase'].fillna(combined['weight_games'])
    
    combined = combined.sort_values(['season', 'runner_id', 'onbase_count'], ascending=[True, True, False])
    
    final_cols = [
        'runner_id', 'season', 'opponent_team', 'opponent_team_id',
        'onbase_count', 'game_count', 
        'weight_onbase', 'weight_games', 'weight'
    ]
    combined = combined[final_cols]
    
    # Save
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_path / "runner_schedule_weights_2018_2025.csv"
    combined.to_csv(output_file, index=False)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print(f"\nTotal rows: {len(combined):,}")
    print(f"Unique runners: {combined['runner_id'].nunique():,}")
    print(f"Unique opponents (names): {combined['opponent_team'].nunique()}")
    print(f"Unique opponents (IDs): {combined['opponent_team_id'].nunique()}")
    print(f"Seasons: {combined['season'].min()}-{combined['season'].max()}")
    
    mapped_pct = (combined['opponent_team_id'].notna().sum() / len(combined)) * 100
    print(f"\nTeam mapping success: {mapped_pct:.1f}%")
    
    print(f"\nOn-base events per opponent (avg): {combined['onbase_count'].mean():.1f}")
    print(f"Games per opponent (avg): {combined['game_count'].mean():.1f}")
    
    weight_sums = combined.groupby(['runner_id', 'season'])['weight'].sum()
    print(f"\nWeight sum check (should be 1.0):")
    print(f"  Mean: {weight_sums.mean():.6f}")
    
    print(f"\nSaved: {output_file}")
    print(f"Size: {output_file.stat().st_size / 1024:.1f} KB")
    print("\n✓ Complete")

if __name__ == "__main__":
    main()