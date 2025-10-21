#!/usr/bin/env python3
"""
Statcast Monthly Downloader
Downloads MLB Statcast pitch-by-pitch data for a year,
filters to Regular Season, saves one CSV per month.
"""

import argparse
import logging
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


def setup_logging(output_dir, year):
    """Configures logging to file + console."""
    log_path = Path(output_dir) / f"statcast_pull_{year}.log"
    
    handlers = [
        logging.FileHandler(log_path, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers,
        force=True
    )
    
    return logging.getLogger(__name__)


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


def download_month(month, start_date, end_date, chunk_days, sleep_seconds, logger):
    """Downloads data for one month in chunks."""
    logger.info(f"Month {month:02d}: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
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
                logger.debug(f"  Chunk {chunk_start.strftime('%m-%d')} to {chunk_end.strftime('%m-%d')}: {len(df_chunk)} rows")
            
            time.sleep(sleep_seconds)
            
        except Exception as e:
            logger.error(f"  Error in chunk {chunk_start.strftime('%Y-%m-%d')}: {e}")
            continue
    
    if not month_data:
        logger.warning(f"Month {month:02d}: No data found")
        return None
    
    # Concatenate
    df_month = pd.concat(month_data, ignore_index=True)
    logger.info(f"Month {month:02d}: {len(df_month)} rows before deduplication")
    
    return df_month


def filter_regular_season(df, logger):
    """Filters to Regular Season (game_type='R')."""
    if 'game_type' not in df.columns:
        logger.warning("Column 'game_type' not present - skipping filter")
        return df
    
    before = len(df)
    df_filtered = df[df['game_type'] == 'R'].copy()
    after = len(df_filtered)
    
    logger.info(f"Regular Season filter: {before} → {after} rows ({after/before*100:.1f}%)")
    
    return df_filtered


def deduplicate(df, logger):
    """Removes duplicates, keeps last version."""
    before = len(df)
    df_dedup = df.drop_duplicates(
        subset=['game_pk', 'at_bat_number', 'pitch_number'],
        keep='last'
    )
    after = len(df_dedup)
    
    if before > after:
        logger.info(f"Duplicates removed: {before} → {after} rows ({before - after} duplicates)")
    
    return df_dedup


def select_columns(df, include_outfielders, logger):
    """Selects defined columns."""
    # Check critical columns
    missing_critical = CRITICAL_COLS - set(df.columns)
    if missing_critical:
        raise ValueError(f"Critical columns missing: {missing_critical}")
    
    # Assemble columns
    target_cols = KEEP_COLS_CORE + KEEP_COLS_PLUS
    if include_outfielders:
        target_cols += EXTRA_FIELDERS
    
    # Only available columns
    available_cols = [c for c in target_cols if c in df.columns]
    missing = set(target_cols) - set(available_cols)
    
    if missing:
        logger.debug(f"Missing columns (skipped): {missing}")
    
    return df[available_cols].copy()


def save_month_csv(df, output_dir, year, month, logger):
    """Saves monthly CSV."""
    filename = f"statcast_pitches_{year}_{month:02d}.csv"
    filepath = Path(output_dir) / filename
    
    df.to_csv(filepath, index=False, na_rep='')
    
    # Statistics
    n_games = df['game_pk'].nunique()
    logger.info(f"Saved: {filename} ({len(df)} rows, {n_games} games)")
    
    return {
        'month': month,
        'rows': len(df),
        'games': n_games,
        'pct_regular': 100.0 if 'game_type' not in df.columns else 
                       (df['game_type'] == 'R').sum() / len(df) * 100
    }


def print_summary(stats):
    """Prints summary."""
    print("\n" + "="*70)
    print("DOWNLOAD SUMMARY")
    print("="*70)
    print(f"{'Month':<10} {'Rows':>10} {'Games':>10} {'% Regular':>12}")
    print("-"*70)
    
    total_rows = 0
    total_games = 0
    
    for stat in stats:
        print(f"{stat['month']:02d}         {stat['rows']:>10,} {stat['games']:>10,} {stat['pct_regular']:>11.1f}%")
        total_rows += stat['rows']
        total_games += stat['games']
    
    print("-"*70)
    print(f"{'TOTAL':10} {total_rows:>10,} {total_games:>10,}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Downloads MLB Statcast pitch-by-pitch data for one year (monthly)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--year', required=True, type=int, help='Year (YYYY)')
    parser.add_argument('--chunk-days', type=int, default=5, help='Days per chunk (default: 5)')
    parser.add_argument('--sleep-seconds', type=int, default=2, help='Pause between chunks (default: 2)')
    parser.add_argument('--test-mode', action='store_true', help='Test mode (April 2023, 10 days)')
    parser.add_argument('--include-outfielders', action='store_true', help='Include outfielder columns (fielder_7-9)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Enable logging
    logger = setup_logging(output_dir, args.year if not args.test_mode else 2023)
    
    # Enable cache
    cache.enable()
    logger.info("pybaseball cache enabled")
    
    # Log parameters
    logger.info(f"Year: {args.year}")
    logger.info(f"Output: {output_dir.absolute()}")
    logger.info(f"Chunk days: {args.chunk_days}")
    logger.info(f"Sleep: {args.sleep_seconds}s")
    logger.info(f"Test mode: {args.test_mode}")
    logger.info(f"Include outfielders: {args.include_outfielders}")
    
    # Month ranges
    month_ranges = get_month_ranges(args.year, args.test_mode)
    
    stats = []
    
    try:
        for month, start_date, end_date in month_ranges:
            # Download
            df_month = download_month(
                month, start_date, end_date,
                args.chunk_days, args.sleep_seconds, logger
            )
            
            if df_month is None:
                continue
            
            # Filter Regular Season
            df_month = filter_regular_season(df_month, logger)
            
            if len(df_month) == 0:
                logger.warning(f"Month {month:02d}: No Regular Season games")
                continue
            
            # Deduplication
            df_month = deduplicate(df_month, logger)
            
            # Column selection
            df_month = select_columns(df_month, args.include_outfielders, logger)
            
            # Save CSV
            stat = save_month_csv(df_month, output_dir, args.year, month, logger)
            stats.append(stat)
        
        # Summary
        if stats:
            print_summary(stats)
            logger.info("✓ Download completed successfully")
        else:
            logger.warning("No data downloaded")
    
    except KeyboardInterrupt:
        logger.warning("Download interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"✗ Critical error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()