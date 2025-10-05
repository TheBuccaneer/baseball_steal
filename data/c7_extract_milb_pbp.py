"""
Script 14: MiLB Play-by-Play Extraction for Pitcher Opportunities
Extracts pitch-level data with runner states to calculate pitcher opportunities

Input: milb_data/milb_schedule.csv
Output: 
- milb_data/milb_pitcher_opportunities.csv (pitcher×season level)
- milb_data/milb_pbp_events.csv (event-level SB/CS with context)

Strategy:
1. Fetch play-by-play for each game
2. Track runner state pitch-by-pitch
3. Identify opportunities (e.g., R1 only, R2 free)
4. Capture SB/CS events with pitcher/catcher context
5. Aggregate to pitcher×season

Usage: 
  # Pilot (AAA 2022 only)
  python 14_extract_milb_pbp.py --level AAA --seasons 2022 --max-games 100
  
  # Full run after pilot succeeds
  python 14_extract_milb_pbp.py --level AAA --seasons 2019,2021,2022
"""

import pandas as pd
import requests
import time
import json
from pathlib import Path
from datetime import datetime
import argparse
from collections import defaultdict

BASE_URL = "https://statsapi.mlb.com/api/v1"

# Runner state encoding
BASE_MAP = {
    'first': 1,
    'second': 2,
    'third': 3,
    '1B': 1,
    '2B': 2,
    '3B': 3
}

def fetch_game_pbp(game_pk, max_retries=3):
    """Fetch play-by-play data for a game"""
    url = f"{BASE_URL}/game/{game_pk}/playByPlay"
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None

def parse_runner_state(runners_data):
    """
    Parse runner positions into occupancy string
    Returns: "___", "1__", "_2_", "1_3", etc.
    
    Uses movement.originBase to identify EXISTING baserunners (not the batter)
    Handles both dict (runner_id: info) and list formats
    """
    state = ['_', '_', '_']  # [1B, 2B, 3B]
    
    if not runners_data:
        return '___'
    
    # Handle list format (API sometimes returns list of runner objects)
    if isinstance(runners_data, list):
        for runner in runners_data:
            movement = runner.get('movement', {})
            # originBase indicates where runner WAS before this play
            # None = batter, not a baserunner
            origin_base = movement.get('originBase', None)
            
            if origin_base in BASE_MAP:
                base_idx = BASE_MAP[origin_base] - 1
                state[base_idx] = str(BASE_MAP[origin_base])
    
    # Handle dict format (runner_id: runner_info)
    elif isinstance(runners_data, dict):
        for runner_id, runner_info in runners_data.items():
            movement = runner_info.get('movement', {})
            origin_base = movement.get('originBase', None)
            
            if origin_base in BASE_MAP:
                base_idx = BASE_MAP[origin_base] - 1
                state[base_idx] = str(BASE_MAP[origin_base])
    
    return ''.join(state)

def identify_opportunity(runner_state, base_to_steal='second'):
    """
    Check if this is a steal opportunity
    
    For SB of 2B: runner on 1B, 2B empty
    For SB of 3B: runner on 2B, 3B empty
    """
    if base_to_steal == 'second':
        # Need R1, no R2
        return runner_state[0] == '1' and runner_state[1] == '_'
    elif base_to_steal == 'third':
        # Need R2, no R3
        return runner_state[1] == '2' and runner_state[2] == '_'
    elif base_to_steal == 'home':
        # Need R3
        return runner_state[2] == '3'
    
    return False

def extract_steal_event(play_event):
    """
    Extract SB/CS from play event
    Returns: (event_type, base_to, runner_id) or None
    """
    # Check play events for stolen base or caught stealing
    play_events = play_event.get('playEvents', [])
    
    for event in play_events:
        event_type = event.get('details', {}).get('event', '')
        
        if event_type in ['Stolen Base 2B', 'Stolen Base 3B', 'Stolen Base Home']:
            # Extract which base
            if '2B' in event_type:
                base = 'second'
            elif '3B' in event_type:
                base = 'third'
            else:
                base = 'home'
            
            # Get runner
            runner = event.get('runner', {})
            runner_id = runner.get('id', None)
            
            return ('SB', base, runner_id)
        
        elif event_type in ['Caught Stealing 2B', 'Caught Stealing 3B', 'Caught Stealing Home']:
            if '2B' in event_type:
                base = 'second'
            elif '3B' in event_type:
                base = 'third'
            else:
                base = 'home'
            
            runner = event.get('runner', {})
            runner_id = runner.get('id', None)
            
            return ('CS', base, runner_id)
    
    return None

def process_game_pbp(game_pk, pbp_data, season, level_name):
    """
    Process play-by-play for one game
    Returns: (opportunities_list, events_list)
    """
    if not pbp_data or 'allPlays' not in pbp_data:
        return [], []
    
    opportunities = []
    events = []
    
    all_plays = pbp_data.get('allPlays', [])
    
    plays_checked = 0
    opps_identified = 0
    
    for play_idx, play in enumerate(all_plays):
        # Get pitcher
        matchup = play.get('matchup', {})
        pitcher_id = matchup.get('pitcher', {}).get('id', None)
        pitcher_name = matchup.get('pitcher', {}).get('fullName', None)
        batter_id = matchup.get('batter', {}).get('id', None)
        
        if not pitcher_id:
            continue
        
        # Get count
        count_data = play.get('count', {})
        balls = count_data.get('balls', 0)
        strikes = count_data.get('strikes', 0)
        outs = count_data.get('outs', 0)
        
        # Get runners at START of play
        runners = play.get('runners', [])  # CHANGED: default to [] not {}
        runner_state = parse_runner_state(runners)
        
        plays_checked += 1
        
        # DEBUG: print runner states for game 666415
        if game_pk == 666415 and runner_state != '___':
            print(f"      Play {play_idx}: state='{runner_state}'")
        
        # Check opportunities
        opp_1b_2b = identify_opportunity(runner_state, 'second')
        opp_2b_3b = identify_opportunity(runner_state, 'third')
        
        if opp_1b_2b or opp_2b_3b:
            opps_identified += 1
        
        # Check for SB/CS events
        steal_event = extract_steal_event(play)
        
        # Record opportunity
        if opp_1b_2b or opp_2b_3b:
            opp_record = {
                'gamePk': game_pk,
                'season': season,
                'level_name': level_name,
                'play_idx': play_idx,
                'pitcher_id': pitcher_id,
                'pitcher_name': pitcher_name,
                'batter_id': batter_id,
                'balls': balls,
                'strikes': strikes,
                'outs': outs,
                'runner_state': runner_state,
                'opp_1b_2b': int(opp_1b_2b),
                'opp_2b_3b': int(opp_2b_3b)
            }
            opportunities.append(opp_record)
        
        # Record event
        if steal_event:
            event_type, base_to, runner_id = steal_event
            
            event_record = {
                'gamePk': game_pk,
                'season': season,
                'level_name': level_name,
                'play_idx': play_idx,
                'pitcher_id': pitcher_id,
                'pitcher_name': pitcher_name,
                'batter_id': batter_id,
                'runner_id': runner_id,
                'balls': balls,
                'strikes': strikes,
                'outs': outs,
                'runner_state': runner_state,
                'event_type': event_type,
                'base_to': base_to
            }
            events.append(event_record)
    
    # DEBUG
    if game_pk == 666415:
        print(f"    DEBUG Game 666415: plays_checked={plays_checked}, opps_identified={opps_identified}, opps_recorded={len(opportunities)}")
    
    return opportunities, events

def aggregate_to_pitcher_season(opportunities_df, events_df):
    """Aggregate opportunities and events to pitcher×season level"""
    
    # Count opportunities by pitcher×season
    pitcher_opps = opportunities_df.groupby(['pitcher_id', 'pitcher_name', 'season', 'level_name']).agg({
        'opp_1b_2b': 'sum',
        'opp_2b_3b': 'sum',
        'gamePk': 'nunique'
    }).reset_index()
    
    pitcher_opps = pitcher_opps.rename(columns={'gamePk': 'games'})
    pitcher_opps['total_opps'] = pitcher_opps['opp_1b_2b'] + pitcher_opps['opp_2b_3b']
    
    # Count events by pitcher×season
    sb_counts = events_df[events_df['event_type'] == 'SB'].groupby(
        ['pitcher_id', 'pitcher_name', 'season', 'level_name']
    ).size().reset_index(name='sb')
    
    cs_counts = events_df[events_df['event_type'] == 'CS'].groupby(
        ['pitcher_id', 'pitcher_name', 'season', 'level_name']
    ).size().reset_index(name='cs')
    
    # Merge
    pitcher_stats = pitcher_opps.merge(
        sb_counts, 
        on=['pitcher_id', 'pitcher_name', 'season', 'level_name'],
        how='left'
    ).merge(
        cs_counts,
        on=['pitcher_id', 'pitcher_name', 'season', 'level_name'],
        how='left'
    )
    
    # Fill NAs
    pitcher_stats['sb'] = pitcher_stats['sb'].fillna(0).astype(int)
    pitcher_stats['cs'] = pitcher_stats['cs'].fillna(0).astype(int)
    pitcher_stats['attempts'] = pitcher_stats['sb'] + pitcher_stats['cs']
    
    # Rates
    pitcher_stats['attempt_rate'] = pitcher_stats['attempts'] / pitcher_stats['total_opps']
    pitcher_stats['success_rate'] = pitcher_stats['sb'] / pitcher_stats['attempts']
    pitcher_stats['sb_rate'] = pitcher_stats['sb'] / pitcher_stats['total_opps']
    
    return pitcher_stats

def main():
    parser = argparse.ArgumentParser(description='Extract MiLB play-by-play opportunities')
    parser.add_argument('--level', type=str, default='AAA', help='Level name (AAA, AA, High-A, A)')
    parser.add_argument('--seasons', type=str, default='2022', help='Comma-separated seasons (e.g., 2019,2021,2022)')
    parser.add_argument('--max-games', type=int, default=None, help='Max games to process (for pilot)')
    parser.add_argument('--start-index', type=int, default=0, help='Start from game index')
    args = parser.parse_args()
    
    seasons = [int(s.strip()) for s in args.seasons.split(',')]
    
    print("=" * 80)
    print("MILB PLAY-BY-PLAY EXTRACTION FOR PITCHER OPPORTUNITIES")
    print("=" * 80)
    print(f"\nLevel: {args.level}")
    print(f"Seasons: {seasons}")
    if args.max_games:
        print(f"Max games: {args.max_games} (PILOT MODE)")
    
    data_path = Path("milb_data")
    
    # Load schedule
    schedule_file = data_path / "milb_schedule.csv"
    if not schedule_file.exists():
        print(f"\nERROR: {schedule_file} not found!")
        print("Run Script 12 first to extract schedule")
        return
    
    schedule_df = pd.read_csv(schedule_file)
    
    # Filter
    filtered = schedule_df[
        (schedule_df['level_name'] == args.level) &
        (schedule_df['season'].isin(seasons))
    ]
    
    print(f"\nFiltered schedule: {len(filtered):,} games")
    
    # Apply limits
    if args.start_index > 0:
        filtered = filtered.iloc[args.start_index:]
    
    if args.max_games:
        filtered = filtered.head(args.max_games)
    
    print(f"Processing: {len(filtered):,} games")
    
    # Process games
    print("\n" + "=" * 80)
    print("EXTRACTING PLAY-BY-PLAY")
    print("=" * 80)
    
    all_opportunities = []
    all_events = []
    errors = []
    
    for idx, row in filtered.iterrows():
        game_pk = row['gamePk']
        season = row['season']
        level_name = row['level_name']
        
        if idx % 10 == 0:
            progress = idx / len(filtered) * 100
            print(f"  [{progress:5.1f}%] Game {idx}/{len(filtered)}: {game_pk} ({level_name} {season})")
        
        # Fetch PBP
        pbp_data = fetch_game_pbp(game_pk)
        
        if pbp_data is None:
            errors.append({
                'gamePk': game_pk,
                'season': season,
                'level': level_name,
                'error': 'Failed to fetch PBP'
            })
            continue
        
        # Process
        opps, evts = process_game_pbp(game_pk, pbp_data, season, level_name)
        
        # DEBUG
        if len(opps) > 0 or len(evts) > 0:
            print(f"    Game {game_pk}: {len(opps)} opps, {len(evts)} events")
        
        all_opportunities.extend(opps)
        all_events.extend(evts)
        
        # Rate limit
        time.sleep(0.3)
    
    # Save raw outputs
    print("\n" + "=" * 80)
    print("SAVING RAW DATA")
    print("=" * 80)
    
    if all_opportunities:
        opps_df = pd.DataFrame(all_opportunities)
        opps_file = data_path / f"milb_opportunities_raw_{args.level}_{min(seasons)}_{max(seasons)}.csv"
        opps_df.to_csv(opps_file, index=False)
        print(f"  Opportunities: {len(opps_df):,} records -> {opps_file.name}")
    else:
        print("  WARNING: No opportunities extracted!")
        opps_df = pd.DataFrame()
    
    if all_events:
        events_df = pd.DataFrame(all_events)
        events_file = data_path / f"milb_events_raw_{args.level}_{min(seasons)}_{max(seasons)}.csv"
        events_df.to_csv(events_file, index=False)
        print(f"  Events: {len(events_df):,} records -> {events_file.name}")
    else:
        print("  WARNING: No events extracted!")
        events_df = pd.DataFrame()
    
    if errors:
        errors_df = pd.DataFrame(errors)
        errors_file = data_path / f"errors_pbp_{args.level}_{min(seasons)}_{max(seasons)}.csv"
        errors_df.to_csv(errors_file, index=False)
        print(f"  Errors: {len(errors)} games -> {errors_file.name}")
    
    # Aggregate to pitcher×season
    if not opps_df.empty and not events_df.empty:
        print("\n" + "=" * 80)
        print("AGGREGATING TO PITCHER×SEASON")
        print("=" * 80)
        
        pitcher_stats = aggregate_to_pitcher_season(opps_df, events_df)
        pitcher_file = data_path / f"milb_pitcher_opportunities_{args.level}_{min(seasons)}_{max(seasons)}.csv"
        pitcher_stats.to_csv(pitcher_file, index=False)
        
        print(f"  Pitcher×Season: {len(pitcher_stats):,} records -> {pitcher_file.name}")
        
        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"\nPitcher×Season stats (n={len(pitcher_stats):,}):")
        print(pitcher_stats[['season', 'total_opps', 'attempts', 'sb', 'attempt_rate', 'success_rate']].describe())
        
        print("\n" + "=" * 80)
        print("PILOT COMPLETE")
        print("=" * 80)
        
        # Coverage check
        mean_opps = pitcher_stats['total_opps'].mean()
        print(f"\nCoverage assessment:")
        print(f"  Mean opportunities per pitcher: {mean_opps:.1f}")
        print(f"  Pitchers with >100 opps: {(pitcher_stats['total_opps'] > 100).sum()}")
        print(f"  Pitchers with >50 opps: {(pitcher_stats['total_opps'] > 50).sum()}")
        
        if mean_opps < 20:
            print("\n  ⚠️  WARNING: Low opportunity counts may indicate incomplete PBP coverage")
            print("  Check a sample of games manually before scaling up")
        else:
            print("\n  ✓ Coverage looks reasonable - safe to scale to more seasons/levels")
    
    else:
        print("\n" + "=" * 80)
        print("NO DATA EXTRACTED")
        print("=" * 80)
        print("  Check:")
        print("  1. Are the games in the schedule actually available?")
        print("  2. Does the play-by-play endpoint work for MiLB?")
        print("  3. Try a different season or level")

if __name__ == "__main__":
    main()