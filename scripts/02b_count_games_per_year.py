#!/usr/bin/env python3
"""
Count unique games per year from MLB PBP CSV files
"""

import argparse
from pathlib import Path
import pandas as pd

def main():
    parser = argparse.ArgumentParser(
        description='Count unique games per year from PBP CSV files'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Directory containing pbp_YYYY_MM.csv files'
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    
    if not input_dir.exists():
        print(f"Error: Directory not found: {input_dir}")
        return
    
    # Find all PBP CSV files
    csv_files = sorted(input_dir.glob('pbp_*.csv'))
    
    if not csv_files:
        print(f"No pbp_*.csv files found in {input_dir}")
        return
    
    print(f"\nFound {len(csv_files)} CSV files")
    print("=" * 50)
    
    # Collect all game_pks with their dates
    all_data = []
    
    for csv_file in csv_files:
        print(f"Reading {csv_file.name}...", end=' ')
        try:
            # Only read the columns we need
            df = pd.read_csv(csv_file, usecols=['game_pk', 'game_date'])
            print(f"({len(df):,} rows)")
            all_data.append(df)
        except Exception as e:
            print(f"ERROR: {e}")
    
    if not all_data:
        print("No data loaded")
        return
    
    # Combine all data
    combined = pd.concat(all_data, ignore_index=True)
    
    # Extract year from game_date
    combined['year'] = combined['game_date'].str[:4]
    
    # Get unique games per year
    games_per_year = combined.groupby('year')['game_pk'].nunique().sort_index()
    
    print("\n" + "=" * 50)
    print("UNIQUE GAMES PER YEAR")
    print("=" * 50)
    
    total_games = 0
    for year, count in games_per_year.items():
        print(f"  {year}: {count:4d} games")
        total_games += count
    
    print("=" * 50)
    print(f"  TOTAL: {total_games:4d} games")
    print("=" * 50)
    
    # Monthly breakdown for each year
    print("\n" + "=" * 50)
    print("MONTHLY BREAKDOWN")
    print("=" * 50)
    
    combined['month'] = combined['game_date'].str[5:7]
    monthly = combined.groupby(['year', 'month'])['game_pk'].nunique().reset_index()
    monthly.columns = ['Year', 'Month', 'Games']
    
    for year in sorted(combined['year'].unique()):
        year_data = monthly[monthly['Year'] == year]
        print(f"\n{year}:")
        for _, row in year_data.iterrows():
            print(f"  {row['Month']}: {row['Games']:4d} games")

if __name__ == '__main__':
    main()