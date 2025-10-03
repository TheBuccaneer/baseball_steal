"""
Script 12: MiLB Level Mapping & Schedule Extraction
Extracts team-level assignments and game schedules for MiLB (2019-2022)

Outputs:
- milb_teams.csv: team_id, season, level_id, level_name, team_name, parent_org
- milb_schedule.csv: gamePk, season, level_id, game_date, home_team, away_team

Usage: python 12_extract_milb_schedule.py
"""

import pandas as pd
import requests
import time
from pathlib import Path
from datetime import datetime

# MiLB Level IDs (MLB Stats API)
LEVEL_IDS = {
    11: 'AAA',
    12: 'AA',
    13: 'High-A',
    14: 'A',
    16: 'Rookie'  # Optional
}

SEASONS = [2019, 2021, 2022]  # Skip 2020 (no regular MiLB season)

BASE_URL = "https://statsapi.mlb.com/api/v1"

def fetch_teams_for_level_season(sport_id, season):
    """Fetch all teams for a given level and season"""
    url = f"{BASE_URL}/teams"
    params = {
        'sportId': sport_id,
        'season': season
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        teams = []
        for team in data.get('teams', []):
            teams.append({
                'team_id': team['id'],
                'team_name': team['name'],
                'level_id': sport_id,
                'level_name': LEVEL_IDS[sport_id],
                'season': season,
                'parent_org': team.get('parentOrgName', None),
                'league_name': team.get('league', {}).get('name', None)
            })
        
        return teams
    
    except Exception as e:
        print(f"  ERROR fetching teams for sportId={sport_id}, season={season}: {e}")
        return []

def fetch_schedule_for_level_season(sport_id, season):
    """Fetch all games for a given level and season"""
    url = f"{BASE_URL}/schedule"
    
    # Get full season date range
    start_date = f"{season}-03-01"
    end_date = f"{season}-10-31"
    
    params = {
        'sportId': sport_id,
        'season': season,
        'startDate': start_date,
        'endDate': end_date,
        'gameType': 'R',  # Regular season only
        'hydrate': 'team'
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        games = []
        for date_entry in data.get('dates', []):
            game_date = date_entry['date']
            for game in date_entry.get('games', []):
                games.append({
                    'gamePk': game['gamePk'],
                    'season': season,
                    'level_id': sport_id,
                    'level_name': LEVEL_IDS[sport_id],
                    'game_date': game_date,
                    'home_team_id': game['teams']['home']['team']['id'],
                    'home_team_name': game['teams']['home']['team']['name'],
                    'away_team_id': game['teams']['away']['team']['id'],
                    'away_team_name': game['teams']['away']['team']['name'],
                    'game_type': game.get('gameType', 'R')
                })
        
        return games
    
    except Exception as e:
        print(f"  ERROR fetching schedule for sportId={sport_id}, season={season}: {e}")
        return []

def main():
    print("=" * 80)
    print("MILB LEVEL MAPPING & SCHEDULE EXTRACTION")
    print("=" * 80)
    
    output_path = Path("milb_data")
    output_path.mkdir(exist_ok=True)
    
    # ========================================================================
    # EXTRACT TEAMS
    # ========================================================================
    print("\n" + "=" * 80)
    print("EXTRACTING TEAMS BY LEVEL & SEASON")
    print("=" * 80)
    
    all_teams = []
    
    for season in SEASONS:
        print(f"\nSeason {season}:")
        for sport_id, level_name in LEVEL_IDS.items():
            print(f"  Fetching {level_name} (sportId={sport_id})...")
            teams = fetch_teams_for_level_season(sport_id, season)
            all_teams.extend(teams)
            print(f"    Found {len(teams)} teams")
            time.sleep(0.5)  # Rate limiting
    
    # Save teams
    teams_df = pd.DataFrame(all_teams)
    teams_file = output_path / "milb_teams.csv"
    teams_df.to_csv(teams_file, index=False)
    
    print(f"\n✓ Saved {len(teams_df):,} team-season records to {teams_file}")
    print(f"\nTeams by level:")
    print(teams_df.groupby(['level_name', 'season']).size())
    
    # ========================================================================
    # EXTRACT SCHEDULES
    # ========================================================================
    print("\n" + "=" * 80)
    print("EXTRACTING GAME SCHEDULES")
    print("=" * 80)
    
    all_games = []
    
    for season in SEASONS:
        print(f"\nSeason {season}:")
        for sport_id, level_name in LEVEL_IDS.items():
            print(f"  Fetching {level_name} schedule...")
            games = fetch_schedule_for_level_season(sport_id, season)
            all_games.extend(games)
            print(f"    Found {len(games):,} games")
            time.sleep(1.0)  # Rate limiting
    
    # Save schedule
    schedule_df = pd.DataFrame(all_games)
    schedule_file = output_path / "milb_schedule.csv"
    schedule_df.to_csv(schedule_file, index=False)
    
    print(f"\n✓ Saved {len(schedule_df):,} games to {schedule_file}")
    print(f"\nGames by level & season:")
    print(schedule_df.groupby(['level_name', 'season']).size())
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"\nFiles created in {output_path}/:")
    print(f"  - milb_teams.csv ({len(teams_df):,} records)")
    print(f"  - milb_schedule.csv ({len(schedule_df):,} games)")
    print(f"\nNext step: Run Script 13 to extract runner events from games")

if __name__ == "__main__":
    main()