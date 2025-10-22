```python
#!/usr/bin/env python3
"""
Statcast Monthly Downloader
Downloads MLB Statcast pitch-by-pitch data for a year,
filters to Regular Season, saves one CSV per month.
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from pybaseball import statcast, cache
from tqdm import tqdm

# Column Definitions
KEEP_COLS_CORE = [
    # Keys/Time
    "game_pk", "game_date", "game_year", "inning", "inning_topbot",
    "at_bat_number", "pitch_number",
    # Count/Context
    "balls", "strikes", "outs_when_up",
    # Teams/IDs (batter & pitcher are MLB Player IDs)
    "home_team", "away_team", "batter", "pitcher",
    # Basestates (before pitch)
    "on_1b", "on_2b", "on_3b",
    # Fielders (always included: C, 1B, 2B, 3B, SS)
    "fielder_2",  # Catcher
    "fielder_3",  # First Base
    "fielder_4",  # Second Base
    "fielder_5",  # Third Base
    "fielder_6",  # Shortstop
    # Outcomes/Labels
    "type", "events", "description", "des",  # 'des' = full play narrative
    # Score/Impact Context
    "home_score", "away_score", "bat_score", "fld_score",
    "delta_run_exp", "delta_home_win_exp",
    # Filter
    "game_type",
]

KEEP_COLS_PLUS = ["pitch_name", "release_speed", "stand", "p_throws"]

# Outfielders (optional via flag)
EXTRA_FIELDERS = ["fielder_7", "fielder_8", "fielder_9"]  # LF, CF, RF

CRITICAL_COLS = {"game_pk", "at_bat_number", "pitch_number"}


def get_month_ranges(year, test_mode=False):
    """Creates list of (month, start, end) tuples."""
    if test_mode:
        return [(4, datetime(2023, 4, 1), datetime(2023, 4, 10))]
    
    ranges = []
    for month in range(1, 13):
        start = datetime(year, month, 1)
        if month == 12:
            end = datetime(year, 12, 31)
        else:
            end = datetime(year, month + 1, 1) - timedelta(days=1)
        ranges.append((month, start, end))
    
    return ranges


def chunk_date_range(start, end, chunk_days):
    """Splits date range into chunks."""
    chunks = []
    current = start
    
    while current <= end:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end)
        chunks.append((current, chunk_end))
        current = chunk_end + timedelta(days=1)
    
    return chunks


def download_month(month, start_date, end_date, chunk_days, sleep_seconds):
    """Downloads data for one month in chunks."""
    chunks = chunk_date_range(start_date, end_date, chunk_days)
    month_data = []
    
    for chunk_start, chunk_end in tqdm(chunks, desc=f"Month {month:02d}", unit="chunk"):
        try:
            df_chunk = statcast(
                start_dt=chunk_start.strftime('%Y-%m-%d'),
                end_dt=chunk_end.strftime('%Y-%m-%d')
            )
            
            if df_chunk is not None and len(df_chunk) > 0:
                month_data.append(df_chunk)
            
            time.sleep(sleep_seconds)
            
        except Exception as e:
            print(f"Error downloading {chunk_start.strftime('%Y-%m-%d')}: {e}")
            continue
    
    if not month_data:
        return None
    
    return pd.concat(month_data, ignore_index=True)


def filter_regular_season(df):
    """Filters to Regular Season (game_type='R')."""
    if 'game_type' not in df.columns:
        return df
    
    return df[df['game_type'] == 'R'].copy()


def deduplicate(df):
    """Removes duplicates, keeps last version."""
    return df.drop_duplicates(
        subset=['game_pk', 'at_bat_number', 'pitch_number'],
        keep='last'
    )


def select_columns(df, include_outfielders):
    """Selects defined columns."""
    missing_critical = CRITICAL_COLS - set(df.columns)
    if missing_critical:
        raise ValueError(f"Critical columns missing: {missing_critical}")
    
    target_cols = KEEP_COLS_CORE + KEEP_COLS_PLUS
    if include_outfielders:
        target_cols += EXTRA_FIELDERS
    
    available_cols = [c for c in target_cols if c in df.columns]
    
    return df[available_cols].copy()


def save_month_csv(df, output_dir, year, month):
    """Saves monthly CSV."""
    filename = f"statcast_pitches_{year}_{month:02d}.csv"
    filepath = Path(output_dir) / filename
    
    df.to_csv(filepath, index=False, na_rep='')
    
    n_games = df['game_pk'].nunique()
    print(f"Saved {filename}: {len(df):,} rows, {n_games} games")
    
    return {
        'month': month,
        'rows': len(df),
        'games': n_games,
    }


def print_summary(stats):
    """Prints summary."""
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    total_rows = 0
    total_games = 0
    
    for stat in stats:
        print(f"Month {stat['month']:02d}: {stat['rows']:>8,} rows, {stat['games']:>4} games")
        total_rows += stat['rows']
        total_games += stat['games']
    
    print("-"*50)
    print(f"Total:    {total_rows:>8,} rows, {total_games:>4} games")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(description='Downloads MLB Statcast pitch-by-pitch data')
    
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--year', required=True, type=int, help='Year (YYYY)')
    parser.add_argument('--chunk-days', type=int, default=5, help='Days per chunk (default: 5)')
    parser.add_argument('--sleep-seconds', type=int, default=2, help='Pause between chunks (default: 2)')
    parser.add_argument('--test-mode', action='store_true', help='Test mode (April 2023, 10 days)')
    parser.add_argument('--include-outfielders', action='store_true', help='Include outfielder columns')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cache.enable()
    
    print(f"Downloading Statcast data for {args.year}")
    print(f"Output: {output_dir.absolute()}")
    
    month_ranges = get_month_ranges(args.year, args.test_mode)
    stats = []
    
    try:
        for month, start_date, end_date in month_ranges:
            df_month = download_month(month, start_date, end_date, args.chunk_days, args.sleep_seconds)
            
            if df_month is None:
                print(f"Month {month:02d}: No data")
                continue
            
            df_month = filter_regular_season(df_month)
            
            if len(df_month) == 0:
                print(f"Month {month:02d}: No regular season games")
                continue
            
            df_month = deduplicate(df_month)
            df_month = select_columns(df_month, args.include_outfielders)
            
            stat = save_month_csv(df_month, output_dir, args.year, month)
            stats.append(stat)
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    if stats:
        print_summary(stats)


if __name__ == '__main__':
    main()
```