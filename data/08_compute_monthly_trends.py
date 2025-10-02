"""
Computes monthly stolen base trends from play-by-play data
Direct aggregation from game_date - no disaggregation needed

Output: analysis_monthly_trends.csv

Usage: python 08_compute_monthly_trends.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def extract_monthly_sb(mlb_stats_folder):
    """
    Extract monthly SB/CS counts from PBP data
    Uses game_date for direct aggregation
    """
    print(f"\nProcessing: {mlb_stats_folder.name}")
    
    csv_files = sorted(mlb_stats_folder.glob("*.csv"))
    
    if not csv_files:
        print(f"  No CSV files found")
        return pd.DataFrame()
    
    all_events = []
    for csv_file in csv_files:
        try:
            # Only load needed columns for efficiency
            df = pd.read_csv(csv_file, usecols=['game_date', 'event', 'runner_id'], low_memory=False)
            all_events.append(df)
        except Exception as e:
            print(f"  Error reading {csv_file.name}: {e}")
    
    if not all_events:
        return pd.DataFrame()
    
    pbp = pd.concat(all_events, ignore_index=True)
    print(f"  Loaded {len(pbp):,} events")
    
    # Extract year from folder name
    year = int(mlb_stats_folder.name.split('_')[-1])
    
    # Parse game_date and extract month
    pbp['game_date'] = pd.to_datetime(pbp['game_date'], errors='coerce')
    pbp['month'] = pbp['game_date'].dt.month
    pbp['season'] = year
    
    # Filter to steal events only
    steal_events = pbp[
        (pbp['runner_id'].notna()) & 
        (pbp['event'].isin(['stolen_base_2b', 'stolen_base_3b', 'stolen_base_home', 
                           'caught_stealing_2b', 'caught_stealing_3b', 'caught_stealing_home']))
    ].copy()
    
    print(f"  Found {len(steal_events):,} steal events")
    
    if len(steal_events) == 0:
        return pd.DataFrame()
    
    # Classify as SB or CS
    steal_events['is_sb'] = steal_events['event'].str.startswith('stolen_base')
    steal_events['is_cs'] = steal_events['event'].str.startswith('caught_stealing')
    
    # Aggregate by runner-month
    monthly = steal_events.groupby(['runner_id', 'season', 'month']).agg(
        sb_count=('is_sb', 'sum'),
        cs_count=('is_cs', 'sum')
    ).reset_index()
    
    print(f"  Aggregated to {len(monthly):,} runner-month combinations")
    
    return monthly

def main():
    print("=" * 80)
    print("COMPUTE MONTHLY TRENDS")
    print("=" * 80)
    print(f"\nCurrent directory: {Path.cwd()}")
    
    mlb_stats_path = Path("mlb_stats")
    
    if not mlb_stats_path.exists():
        print(f"\nERROR: {mlb_stats_path.absolute()} not found!")
        print("Please run this script from the data/ directory")
        sys.exit(1)
    
    # Find all year folders
    year_folders = sorted([d for d in mlb_stats_path.iterdir() 
                          if d.is_dir() and d.name.startswith('mlb_stats_')])
    
    if not year_folders:
        print(f"\nNo year folders found in {mlb_stats_path}")
        return
    
    print(f"\nFound {len(year_folders)} year folders")
    
    # Process each year
    all_monthly = []
    
    for year_folder in year_folders:
        monthly = extract_monthly_sb(year_folder)
        if not monthly.empty:
            all_monthly.append(monthly)
    
    if not all_monthly:
        print("\nNo data collected")
        return
    
    # Combine all years
    print("\n" + "=" * 80)
    print("COMBINING & FINALIZING")
    print("=" * 80)
    
    combined = pd.concat(all_monthly, ignore_index=True)
    
    print(f"\nTotal runner-month combinations: {len(combined):,}")
    
    # Calculate derived fields
    combined['attempts'] = combined['sb_count'] + combined['cs_count']
    combined['success_rate'] = combined.apply(
        lambda row: row['sb_count'] / row['attempts'] if row['attempts'] > 0 else 0.0,
        axis=1
    )
    
    # Sort
    combined = combined.sort_values(['season', 'month', 'runner_id'])
    
    # Final columns
    final_cols = ['season', 'month', 'runner_id', 'sb_count', 'cs_count', 'attempts', 'success_rate']
    combined = combined[final_cols]
    
    # Save
    output_path = Path("analysis")
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "analysis_monthly_trends.csv"
    
    combined.to_csv(output_file, index=False)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print(f"\nTotal rows: {len(combined):,}")
    print(f"Unique runners: {combined['runner_id'].nunique():,}")
    print(f"Seasons: {combined['season'].min()}-{combined['season'].max()}")
    print(f"Months: {combined['month'].min()}-{combined['month'].max()}")
    
    # Monthly distribution
    print(f"\nEvents per month (avg):")
    monthly_dist = combined.groupby('month').agg({
        'sb_count': 'sum',
        'cs_count': 'sum',
        'attempts': 'sum'
    })
    monthly_dist['month_name'] = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    print(monthly_dist[['month_name', 'sb_count', 'cs_count', 'attempts']].to_string())
    
    # Yearly totals check
    print(f"\nYearly totals:")
    yearly = combined.groupby('season')[['sb_count', 'cs_count', 'attempts']].sum()
    print(yearly.to_string())
    
    print(f"\nSaved: {output_file}")
    print(f"Size: {output_file.stat().st_size / 1024:.1f} KB")
    print("\nMonthly trends ready!")

if __name__ == "__main__":
    main()