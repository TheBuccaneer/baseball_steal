"""
Script 13: MiLB Boxscore Extraction & Team Running Game Stats
Extracts team-level stolen base/caught stealing stats from MiLB boxscores

Input: milb_data/milb_schedule.csv
Output: 
- milb_data/milb_team_game_stats.csv (team×game level)
- milb_data/milb_team_month_stats.csv (team×month aggregate)
- milb_data/milb_team_season_stats.csv (team×season aggregate)

Usage: python 13_extract_milb_boxscores.py [--start-index 0] [--max-games 100]
"""

import pandas as pd
import requests
import time
import json
from pathlib import Path
from datetime import datetime
import argparse

BASE_URL = "https://statsapi.mlb.com/api/v1"

def fetch_game_boxscore(game_pk, max_retries=3, debug=False):
    """Fetch boxscore for a single game"""
    url = f"{BASE_URL}/game/{game_pk}/boxscore"
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30)
            
            # DEBUG: Print status for first game
            if debug:
                print(f"\n  DEBUG gamePk {game_pk}:")
                print(f"    Status Code: {response.status_code}")
                print(f"    URL: {url}")
                if response.status_code != 200:
                    print(f"    Response: {response.text[:500]}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            if debug:
                print(f"    HTTP Error: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    print(f"    Response: {e.response.text[:500]}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            else:
                return None
        except Exception as e:
            if debug:
                print(f"    Exception: {type(e).__name__}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            else:
                return None

def extract_team_stats_from_boxscore(boxscore_data, game_pk, game_date):
    """Extract team-level SB/CS stats from boxscore"""
    records = []
    
    if boxscore_data is None:
        return records
    
    try:
        teams_data = boxscore_data.get('teams', {})
        
        # Process both teams
        for side in ['away', 'home']:
            team_data = teams_data.get(side, {})
            team_id = team_data.get('team', {}).get('id', None)
            team_name = team_data.get('team', {}).get('name', None)
            
            # Get batting stats (offensive perspective)
            batting_stats = team_data.get('teamStats', {}).get('batting', {})
            sb = batting_stats.get('stolenBases', 0)
            cs = batting_stats.get('caughtStealing', 0)
            
            # Get pitching stats (defensive perspective - runners against)
            pitching_stats = team_data.get('teamStats', {}).get('pitching', {})
            sb_allowed = pitching_stats.get('stolenBases', 0)
            cs_allowed = pitching_stats.get('caughtStealing', 0)
            
            records.append({
                'gamePk': game_pk,
                'game_date': game_date,
                'team_id': team_id,
                'team_name': team_name,
                'side': side,
                'sb': sb,
                'cs': cs,
                'attempts': sb + cs,
                'sb_allowed': sb_allowed,
                'cs_allowed': cs_allowed,
                'attempts_allowed': sb_allowed + cs_allowed
            })
    
    except Exception as e:
        print(f"    ERROR parsing boxscore {game_pk}: {e}")
        return []
    
    return records

def aggregate_to_team_month(team_game_df, schedule_df):
    """Aggregate game-level stats to team×month"""
    
    # Merge with schedule to get level info
    team_game_with_level = team_game_df.merge(
        schedule_df[['gamePk', 'season', 'level_id', 'level_name']],
        on='gamePk',
        how='left'
    )
    
    # Add month
    team_game_with_level['game_date'] = pd.to_datetime(team_game_with_level['game_date'])
    team_game_with_level['month'] = team_game_with_level['game_date'].dt.month
    
    # Aggregate by team×month
    month_agg = team_game_with_level.groupby(
        ['team_id', 'team_name', 'season', 'level_id', 'level_name', 'month']
    ).agg({
        'gamePk': 'count',  # Games played
        'sb': 'sum',
        'cs': 'sum',
        'attempts': 'sum',
        'sb_allowed': 'sum',
        'cs_allowed': 'sum',
        'attempts_allowed': 'sum'
    }).reset_index()
    
    month_agg = month_agg.rename(columns={'gamePk': 'games'})
    
    # Calculate success rates
    month_agg['success_rate'] = month_agg['sb'] / month_agg['attempts']
    month_agg['success_rate_allowed'] = month_agg['sb_allowed'] / month_agg['attempts_allowed']
    
    return month_agg

def aggregate_to_team_season(team_month_df):
    """Aggregate monthly data to season level"""
    
    season_agg = team_month_df.groupby(
        ['team_id', 'team_name', 'season', 'level_id', 'level_name']
    ).agg({
        'games': 'sum',
        'sb': 'sum',
        'cs': 'sum',
        'attempts': 'sum',
        'sb_allowed': 'sum',
        'cs_allowed': 'sum',
        'attempts_allowed': 'sum'
    }).reset_index()
    
    # Calculate success rates
    season_agg['success_rate'] = season_agg['sb'] / season_agg['attempts']
    season_agg['success_rate_allowed'] = season_agg['sb_allowed'] / season_agg['attempts_allowed']
    
    # Per-game rates
    season_agg['sb_per_game'] = season_agg['sb'] / season_agg['games']
    season_agg['attempts_per_game'] = season_agg['attempts'] / season_agg['games']
    
    return season_agg

def main():
    parser = argparse.ArgumentParser(description='Extract MiLB boxscore stats')
    parser.add_argument('--start-index', type=int, default=0, help='Start from this game index')
    parser.add_argument('--max-games', type=int, default=None, help='Process max this many games')
    parser.add_argument('--checkpoint-every', type=int, default=500, help='Save checkpoint every N games')
    args = parser.parse_args()
    
    print("=" * 80)
    print("MILB BOXSCORE EXTRACTION & TEAM STATS")
    print("=" * 80)
    
    data_path = Path("milb_data")
    
    if not data_path.exists():
        print(f"\nERROR: {data_path} not found!")
        print("Please run Script 12 first to extract schedule")
        return
    
    # Load inputs
    print("\nLoading schedule...")
    schedule_file = data_path / "milb_schedule.csv"
    
    if not schedule_file.exists():
        print(f"ERROR: {schedule_file} not found!")
        return
    
    schedule_df = pd.read_csv(schedule_file)
    print(f"  Loaded {len(schedule_df):,} games")
    
    # Apply filters
    start_idx = args.start_index
    end_idx = start_idx + args.max_games if args.max_games else len(schedule_df)
    games_to_process = schedule_df.iloc[start_idx:end_idx]
    
    print(f"\nProcessing games {start_idx:,} to {end_idx:,} ({len(games_to_process):,} games)")
    
    # Extract stats
    print("\n" + "=" * 80)
    print("EXTRACTING TEAM STATS FROM BOXSCORES")
    print("=" * 80)
    
    all_team_stats = []
    errors = []
    
    checkpoint_file = data_path / f"checkpoint_boxscore_{start_idx}_{end_idx}.csv"
    
    for idx, row in games_to_process.iterrows():
        game_pk = row['gamePk']
        level_name = row['level_name']
        season = row['season']
        game_date = row['game_date']
        
        if (idx - start_idx) % 50 == 0:
            progress = (idx - start_idx) / len(games_to_process) * 100
            print(f"  [{progress:5.1f}%] Game {idx - start_idx + 1}/{len(games_to_process)}: {game_pk} ({level_name} {season})")
        
        # Fetch boxscore with debug for first game
        debug_mode = (idx == start_idx)
        boxscore_data = fetch_game_boxscore(game_pk, debug=debug_mode)
        
        if boxscore_data is None:
            errors.append({'gamePk': game_pk, 'season': season, 'level': level_name, 'error': 'Failed to fetch'})
            continue
        
        # Extract team stats
        team_stats = extract_team_stats_from_boxscore(boxscore_data, game_pk, game_date)
        all_team_stats.extend(team_stats)
        
        # Checkpoint
        if (idx - start_idx) % args.checkpoint_every == 0 and idx > start_idx:
            team_df_checkpoint = pd.DataFrame(all_team_stats)
            team_df_checkpoint.to_csv(checkpoint_file, index=False)
            print(f"    Checkpoint: {len(all_team_stats):,} team-game records")
        
        # Rate limiting
        time.sleep(0.2)
    
    # Save team-game stats
    print(f"\n{len(all_team_stats):,} team-game records extracted")
    
    if errors:
        errors_df = pd.DataFrame(errors)
        errors_file = data_path / f"errors_boxscore_{start_idx}_{end_idx}.csv"
        errors_df.to_csv(errors_file, index=False)
        print(f"  Errors: {len(errors)} games failed, saved to {errors_file}")
    
    # Check if any stats were extracted
    if len(all_team_stats) == 0:
        print("\nWARNING: No stats extracted!")
        print("  All games failed - check errors file")
        return
    
    team_game_df = pd.DataFrame(all_team_stats)
    game_file = data_path / f"milb_team_game_stats_{start_idx}_{end_idx}.csv"
    team_game_df.to_csv(game_file, index=False)
    print(f"  Saved: {game_file}")
    
    # Aggregate
    print("\n" + "=" * 80)
    print("AGGREGATING TO TEAM×MONTH AND TEAM×SEASON")
    print("=" * 80)
    
    team_month_df = aggregate_to_team_month(team_game_df, schedule_df)
    team_season_df = aggregate_to_team_season(team_month_df)
    
    month_file = data_path / f"milb_team_month_stats_{start_idx}_{end_idx}.csv"
    season_file = data_path / f"milb_team_season_stats_{start_idx}_{end_idx}.csv"
    
    team_month_df.to_csv(month_file, index=False)
    team_season_df.to_csv(season_file, index=False)
    
    print(f"  Team×Month: {len(team_month_df):,} records -> {month_file}")
    print(f"  Team×Season: {len(team_season_df):,} records -> {season_file}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print("\nTeam×Season stolen base summary:")
    summary = team_season_df.groupby('level_name')[['sb', 'attempts', 'success_rate', 'sb_per_game']].mean()
    print(summary)
    
    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"\nFiles created:")
    print(f"  - milb_team_game_stats_{start_idx}_{end_idx}.csv ({len(team_game_df):,} rows)")
    print(f"  - milb_team_month_stats_{start_idx}_{end_idx}.csv ({len(team_month_df):,} rows)")
    print(f"  - milb_team_season_stats_{start_idx}_{end_idx}.csv ({len(team_season_df):,} rows)")
    print(f"\nTo process more games, run:")
    print(f"  python 13_extract_milb_boxscores.py --start-index {end_idx}")

if __name__ == "__main__":
    main()