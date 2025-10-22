#!/usr/bin/env python3
"""
MLB StatsAPI Play-by-Play Monthly Extractor

Extracts pitch-level play-by-play data from MLB StatsAPI for a given year,
saving one CSV per month with SB/CS/PO flags and catcher information.
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Column order to maintain in output CSVs
KEEP_COLS = [
    # Keys/Time
    "game_pk", "game_date", "inning", "inning_topbot",
    "at_bat_index", "pitch_in_pa",
    # Count/Context before pitch
    "balls", "strikes", "outs",
    "home_team_id", "away_team_id",
    # Matchup info
    "batter_id", "pitcher_id",
    # Outcome labels (play level)
    "event_type", "event", "play_desc", "is_pa_last",
    # Runner flags per base
    "RUN1_SB_FL", "RUN2_SB_FL", "RUN3_SB_FL",
    "RUN1_CS_FL", "RUN2_CS_FL", "RUN3_CS_FL",
    "RUN1_PK_FL", "RUN2_PK_FL", "RUN3_PK_FL",
    # Detailed runner info (first runner only for simplicity, can be expanded)
    "runner_id", "start_base", "end_base", "is_out", "out_number",
    "runner_event", "runner_event_type", "movement_reason",
    "is_scoring_event", "rbi", "earned",
    # Catcher ID
    "catcher_id"
]


def create_session(max_retries: int = 3) -> requests.Session:
    """Create requests session with retry logic."""
    session = requests.Session()
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def exponential_backoff(func):
    """Decorator for exponential backoff on API calls."""
    def wrapper(*args, **kwargs):
        max_wait = 10
        wait = 1
        attempts = kwargs.get('max_retries', 3)
        
        for attempt in range(attempts):
            try:
                return func(*args, **kwargs)
            except requests.exceptions.RequestException as e:
                if attempt == attempts - 1:
                    raise
                print(f"  ⚠ Attempt {attempt + 1} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)
                wait = min(wait * 2, max_wait)
        
        raise Exception(f"Failed after {attempts} attempts")
    
    return wrapper


@exponential_backoff
def get_schedule(
    year: int,
    month: int,
    session: requests.Session,
    max_retries: int = 3,
    test_mode: bool = False
) -> List[Dict]:
    """
    Fetch regular season schedule for a given year and month.
    
    Returns list of game dictionaries with gamePk, officialDate, teams, etc.
    """
    if test_mode and month != 4:
        return []
    
    # Build date range for the month
    start_date = f"{year}-{month:02d}-01"
    if month == 12:
        end_date = f"{year}-{month:02d}-31"
    else:
        # Last day of month
        from calendar import monthrange
        last_day = monthrange(year, month)[1]
        end_date = f"{year}-{month:02d}-{last_day}"
    
    # Test mode: entire April month
    if test_mode and month == 4:
        end_date = f"{year}-04-30"
    
    url = "https://statsapi.mlb.com/api/v1/schedule"
    params = {
        'sportId': 1,  # MLB
        'startDate': start_date,
        'endDate': end_date,
        'gameType': 'R',  # Regular season only
        'hydrate': 'team,game(content(media(epg)))'
    }
    
    print(f"Fetching schedule for {year}-{month:02d} (Regular Season)...")
    
    response = session.get(url, params=params, timeout=30)
    response.raise_for_status()
    
    data = response.json()
    games = []
    
    # Valid completed game states
    # F = Final, O = Final (completed early), C = Completed
    # NOTE: 'S' (Suspended) games are NOT included - they have incomplete PBP data
    # until resumed. Example: Mets-Marlins 4/11/21 suspended, completed 8/31/21.
    # Such games will have officialDate in April but only appear as 'F' much later.
    completed_states = ['F', 'O', 'C']
    
    for date_entry in data.get('dates', []):
        for game in date_entry.get('games', []):
            # Filter 1: Only completed games
            status = game.get('status', {})
            game_state = status.get('codedGameState', '')
            if game_state not in completed_states:
                continue
            
            # Filter 2: Hard filter on official date (must be in target month)
            official_date = game.get('officialDate', '')
            if not official_date.startswith(f"{year}-{month:02d}"):
                continue
            
            games.append({
                'gamePk': game['gamePk'],
                'officialDate': official_date,
                'home_team_id': game['teams']['home']['team']['id'],
                'away_team_id': game['teams']['away']['team']['id'],
                'gameNumber': game.get('gameNumber', 1)
            })
    
    print(f"  → Found {len(games)} games")
    return games


@exponential_backoff
def get_game_feed(
    game_pk: int,
    session: requests.Session,
    max_retries: int = 3
) -> Optional[Dict]:
    """Fetch live game feed for a specific game."""
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
    
    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"  ⚠ Failed to fetch game {game_pk}: {e}")
        return None


def extract_catcher_id(
    boxscore: Dict,
    inning_topbot: str
) -> str:
    """
    Extract catcher ID (POS=2) from boxscore for the given half-inning.
    
    For top of inning: home team is fielding (catcher is from home team)
    For bottom of inning: away team is fielding (catcher is from away team)
    """
    try:
        # Determine fielding team
        if inning_topbot == "Top":
            fielding_team = boxscore.get('teams', {}).get('home', {})
        else:
            fielding_team = boxscore.get('teams', {}).get('away', {})
        
        players = fielding_team.get('players', {})
        
        # Find player with position code 'C' (Catcher) or position ID 2
        for player_id, player_data in players.items():
            position = player_data.get('position', {})
            if position.get('abbreviation') == 'C' or position.get('code') == '2':
                return str(player_data.get('person', {}).get('id', ''))
        
        return ""
    except Exception:
        return ""


def parse_runner_flags(runners: List[Dict]) -> Dict[str, str]:
    """
    Parse runner movements to extract SB/CS/PO flags per base.
    
    Returns dict with keys like 'RUN1_SB_FL', 'RUN2_CS_FL', etc.
    Values are 'T' for True or '' for False.
    """
    flags = {
        'RUN1_SB_FL': '', 'RUN2_SB_FL': '', 'RUN3_SB_FL': '',
        'RUN1_CS_FL': '', 'RUN2_CS_FL': '', 'RUN3_CS_FL': '',
        'RUN1_PK_FL': '', 'RUN2_PK_FL': '', 'RUN3_PK_FL': ''
    }
    
    base_map = {'1B': 'RUN1', '2B': 'RUN2', '3B': 'RUN3'}
    
    for runner in runners:
        movement = runner.get('movement', {})
        details = runner.get('details', {})
        
        start_base = movement.get('start')
        end_base = movement.get('end')
        is_out = movement.get('isOut', False)
        
        event_type = details.get('event', '').lower()
        runner_event_type = details.get('eventType', '').lower()
        
        # Map start base to runner prefix
        if start_base not in base_map:
            continue
        
        runner_prefix = base_map[start_base]
        
        # Stolen Base
        if 'stolen_base' in event_type or 'stolen_base' in runner_event_type:
            flags[f'{runner_prefix}_SB_FL'] = 'T'
            if is_out:
                flags[f'{runner_prefix}_CS_FL'] = 'T'
        
        # Caught Stealing (without successful steal)
        elif 'caught_stealing' in event_type or 'caught_stealing' in runner_event_type:
            flags[f'{runner_prefix}_CS_FL'] = 'T'
        
        # Pickoff
        elif 'pickoff' in event_type or 'pickoff' in runner_event_type:
            flags[f'{runner_prefix}_PK_FL'] = 'T'
            if is_out:
                # Pickoff caught stealing - mark both
                flags[f'{runner_prefix}_CS_FL'] = 'T'
    
    return flags


def parse_game_to_pitches(
    game_data: Dict,
    game_info: Dict
) -> List[Dict]:
    """
    Parse game feed into pitch-level rows.
    
    Each pitch gets: count before pitch, inning context, outcome flags,
    SB/CS/PO flags, matchup info, runner details, and catcher ID.
    """
    rows = []
    
    try:
        all_plays = game_data.get('liveData', {}).get('plays', {}).get('allPlays', [])
        boxscore = game_data.get('liveData', {}).get('boxscore', {})
        
        for play_idx, play in enumerate(all_plays):
            about = play.get('about', {})
            result = play.get('result', {})
            matchup = play.get('matchup', {})
            play_events = play.get('playEvents', [])
            runners = play.get('runners', [])
            
            # Extract inning info
            inning = about.get('inning', 0)
            is_top_inning = about.get('isTopInning', True)
            inning_topbot = "Top" if is_top_inning else "Bot"
            
            # Extract matchup info
            batter_id = matchup.get('batter', {}).get('id', '')
            pitcher_id = matchup.get('pitcher', {}).get('id', '')
            
            # Get catcher for this half-inning
            catcher_id = extract_catcher_id(boxscore, inning_topbot)
            
            # Parse runner flags for this play
            runner_flags = parse_runner_flags(runners)
            
            # Extract detailed runner info (first runner if multiple)
            runner_details = {}
            if runners:
                first_runner = runners[0]
                details = first_runner.get('details', {})
                movement = first_runner.get('movement', {})
                
                runner_details = {
                    'runner_id': details.get('runner', {}).get('id', ''),
                    'start_base': movement.get('start', ''),
                    'end_base': movement.get('end', ''),
                    'is_out': 'T' if movement.get('isOut', False) else '',
                    'out_number': movement.get('outNumber', ''),
                    'runner_event': details.get('event', ''),
                    'runner_event_type': details.get('eventType', ''),
                    'movement_reason': details.get('movementReason', ''),
                    'is_scoring_event': 'T' if details.get('isScoringEvent', False) else '',
                    'rbi': 'T' if details.get('rbi', False) else '',
                    'earned': 'T' if details.get('earned', False) else ''
                }
            else:
                runner_details = {
                    'runner_id': '', 'start_base': '', 'end_base': '',
                    'is_out': '', 'out_number': '', 'runner_event': '',
                    'runner_event_type': '', 'movement_reason': '',
                    'is_scoring_event': '', 'rbi': '', 'earned': ''
                }
            
            # Event-level info
            event_type = result.get('eventType', '')
            event = result.get('event', '')
            play_desc = result.get('description', '')
            
            # If no pitch events, create single row for the play
            if not play_events:
                row = {
                    'game_pk': game_info['gamePk'],
                    'game_date': game_info['officialDate'],
                    'inning': inning,
                    'inning_topbot': inning_topbot,
                    'at_bat_index': play_idx,
                    'pitch_in_pa': 0,
                    'balls': 0,
                    'strikes': 0,
                    'outs': about.get('outs', 0),
                    'home_team_id': game_info['home_team_id'],
                    'away_team_id': game_info['away_team_id'],
                    'batter_id': batter_id,
                    'pitcher_id': pitcher_id,
                    'event_type': event_type,
                    'event': event,
                    'play_desc': play_desc,
                    'is_pa_last': 'T',
                    'catcher_id': catcher_id,
                    **runner_flags,
                    **runner_details
                }
                rows.append(row)
                continue
            
            # Process each pitch in the PA
            num_pitches = len(play_events)
            for pitch_idx, pitch_event in enumerate(play_events):
                count = pitch_event.get('count', {})
                
                # Determine if this is the last pitch of the PA
                is_last_pitch = (pitch_idx == num_pitches - 1)
                
                # Only include runner flags on the last pitch of the PA to avoid duplication
                if is_last_pitch:
                    current_runner_flags = runner_flags
                    current_runner_details = runner_details
                else:
                    # Empty flags for non-final pitches
                    current_runner_flags = {k: '' for k in runner_flags.keys()}
                    current_runner_details = {k: '' for k in runner_details.keys()}
                
                row = {
                    'game_pk': game_info['gamePk'],
                    'game_date': game_info['officialDate'],
                    'inning': inning,
                    'inning_topbot': inning_topbot,
                    'at_bat_index': play_idx,
                    'pitch_in_pa': pitch_idx,
                    'balls': count.get('balls', 0),
                    'strikes': count.get('strikes', 0),
                    'outs': count.get('outs', 0),
                    'home_team_id': game_info['home_team_id'],
                    'away_team_id': game_info['away_team_id'],
                    'batter_id': batter_id,
                    'pitcher_id': pitcher_id,
                    'event_type': event_type if is_last_pitch else '',
                    'event': event if is_last_pitch else '',
                    'play_desc': play_desc if is_last_pitch else '',
                    'is_pa_last': 'T' if is_last_pitch else '',
                    'catcher_id': catcher_id,
                    **current_runner_flags,
                    **current_runner_details
                }
                rows.append(row)
        
    except Exception as e:
        print(f"  ⚠ Error parsing game {game_info['gamePk']}: {e}")
    
    return rows


def process_game(
    game_info: Dict,
    session: requests.Session,
    sleep_seconds: float,
    max_retries: int
) -> List[Dict]:
    """Process a single game and return pitch-level rows."""
    game_pk = game_info['gamePk']
    
    # Rate limiting
    time.sleep(sleep_seconds)
    
    # Fetch game feed
    game_data = get_game_feed(
        game_pk=game_pk,
        session=session,
        max_retries=max_retries
    )
    
    if game_data is None:
        return []
    
    # Parse to pitch rows
    rows = parse_game_to_pitches(game_data, game_info)
    
    return rows


def process_month(
    year: int,
    month: int,
    output_dir: Path,
    session: requests.Session,
    sleep_seconds: float,
    max_retries: int,
    max_workers: int,
    test_mode: bool
) -> int:
    """
    Process all games for a given month and save to CSV.
    
    Returns number of successfully processed games.
    """
    # Fetch schedule
    games = get_schedule(
        year=year,
        month=month,
        session=session,
        max_retries=max_retries,
        test_mode=test_mode
    )
    
    if not games:
        print(f"  → No games for this month, skipping")
        return 0
    
    # Process games in parallel
    all_rows = []
    successful_games = 0
    
    print(f"Processing {len(games)} games...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_game = {
            executor.submit(
                process_game,
                game_info,
                session,
                sleep_seconds,
                max_retries
            ): game_info for game_info in games
        }
        
        for future in as_completed(future_to_game):
            game_info = future_to_game[future]
            try:
                rows = future.result()
                if rows:
                    all_rows.extend(rows)
                    successful_games += 1
            except Exception as e:
                print(f"  ⚠ Game {game_info['gamePk']} failed: {e}")
    
    if not all_rows:
        print(f"  → No data extracted")
        return 0
    
    # Create DataFrame
    df = pd.DataFrame(all_rows)
    
    # Post-filter: Remove any games outside target month
    target_month_prefix = f"{year}-{month:02d}"
    df = df[df['game_date'].str.startswith(target_month_prefix)]
    
    if len(df) == 0:
        print(f"  → No data after date filtering")
        return 0
    
    # Ensure all expected columns exist
    for col in KEEP_COLS:
        if col not in df.columns:
            df[col] = ''
    
    # Reorder columns
    df = df[KEEP_COLS]
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['game_pk', 'at_bat_index', 'pitch_in_pa'])
    
    # Save to CSV
    output_file = output_dir / f"pbp_{year}_{month:02d}.csv"
    df.to_csv(output_file, index=False, na_rep='')
    
    # Should/Is comparison
    scheduled_game_pks = set(g['gamePk'] for g in games)
    extracted_game_pks = set(df['game_pk'].unique())
    missing_games = scheduled_game_pks - extracted_game_pks
    extra_games = extracted_game_pks - scheduled_game_pks
    
    print(f"  ✓ Saved {len(df)} pitch rows from {successful_games}/{len(games)} games to {output_file.name}")
    
    if missing_games:
        print(f"  ⚠ Missing {len(missing_games)} scheduled games: {sorted(missing_games)[:10]}{'...' if len(missing_games) > 10 else ''}")
    
    if extra_games:
        print(f"  ⚠ Found {len(extra_games)} extra games not in schedule: {sorted(extra_games)[:10]}{'...' if len(extra_games) > 10 else ''}")
    
    return successful_games


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Extract MLB play-by-play data from StatsAPI by month'
    )
    
    # Required arguments
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for CSV files'
    )
    parser.add_argument(
        '--year',
        type=int,
        required=True,
        help='Year to extract (e.g., 2023)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--sleep-seconds',
        type=float,
        default=1.5,
        help='Sleep duration between API requests (default: 1.5)'
    )
    parser.add_argument(
        '--retry',
        type=int,
        default=3,
        help='Number of retry attempts for failed requests (default: 3)'
    )
    parser.add_argument(
        '--test-mode',
        action='store_true',
        help='Test mode: only process entire April month'
    )
    
    args = parser.parse_args()
    
    # Validate year
    current_year = datetime.now().year
    if args.year < 2015 or args.year > current_year:
        print(f"Error: Year must be between 2015 and {current_year}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n=== MLB StatsAPI PBP Extractor ===")
    print(f"Year: {args.year}")
    print(f"Output: {output_dir.absolute()}")
    print(f"Test mode: {args.test_mode}\n")
    
    # Create session
    session = create_session(max_retries=args.retry)
    
    # Process each month
    total_games = 0
    months_to_process = [4] if args.test_mode else range(1, 13)
    
    for month in months_to_process:
        try:
            successful_games = process_month(
                year=args.year,
                month=month,
                output_dir=output_dir,
                session=session,
                sleep_seconds=args.sleep_seconds,
                max_retries=args.retry,
                max_workers=4,
                test_mode=args.test_mode
            )
            total_games += successful_games
        except Exception as e:
            print(f"\n⚠ Failed to process month {month}: {e}\n")
    
    print(f"\n=== Complete ===")
    print(f"Total games processed: {total_games}")
    print(f"Files saved to: {output_dir.absolute()}\n")


if __name__ == '__main__':
    main()