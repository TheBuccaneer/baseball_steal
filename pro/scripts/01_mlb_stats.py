"""
We download play by play data. See README.md for more
"""

import statsapi
import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import time

# Regular Season Monate
SEASON_MONTHS = {
    3: 'march',
    4: 'april', 
    5: 'may',
    6: 'june',
    7: 'july',
    8: 'august',
    9: 'september'
}
"""
We download games month for month
"""
def get_month_games(year, month):

    # Erster und letzter Tag des Monats
    first_day = datetime(year, month, 1)
    if month == 12:
        last_day = datetime(year + 1, 1, 1) - timedelta(days=1)
    else:
        last_day = datetime(year, month + 1, 1) - timedelta(days=1)
    
    start_date = first_day.strftime("%Y-%m-%d")
    end_date = last_day.strftime("%Y-%m-%d")
    
    games = statsapi.schedule(start_date=start_date, end_date=end_date)
    
    regular_season = [g for g in games if g.get('game_type') == 'R' and g['status'] == 'Final']
    
    return regular_season

def extract_play_data(game_id, game_info):
    """Play-by-Play data"""
    try:
        playbyplay = statsapi.get('game_playByPlay', {'gamePk': game_id})
        plays_data = []
        
        for play in playbyplay.get('allPlays', []):
            about = play.get('about', {})
            result = play.get('result', {})
            matchup = play.get('matchup', {})
            count = play.get('count', {})
            runners = play.get('runners', [])
            
            runner_movements = []
            for runner in runners:
                movement = runner.get('movement', {})
                details = runner.get('details', {})
                runner_info = details.get('runner', {})
                
                if movement.get('start') and movement.get('end'):
                    runner_movements.append({
                        'runner_id': runner_info.get('id'),
                        'runner_name': runner_info.get('fullName'),
                        'start_base': movement.get('start'),
                        'end_base': movement.get('end'),
                        'is_out': movement.get('isOut', False),
                        'out_number': movement.get('outNumber')
                    })
            
            if not runner_movements:
                runner_movements = [{}]
            
            for runner_mov in runner_movements:
                play_row = {
                    'game_id': game_id,
                    'game_date': game_info['game_date'],
                    'away_team': game_info['away_name'],
                    'home_team': game_info['home_name'],
                    'away_score': game_info.get('away_score'),
                    'home_score': game_info.get('home_score'),
                    'inning': about.get('inning'),
                    'half_inning': about.get('halfInning'),
                    'at_bat_index': about.get('atBatIndex'),
                    'balls': count.get('balls'),
                    'strikes': count.get('strikes'),
                    'outs': count.get('outs'),
                    'batter_id': matchup.get('batter', {}).get('id'),
                    'batter_name': matchup.get('batter', {}).get('fullName'),
                    'batter_side': matchup.get('batSide', {}).get('code'),
                    'pitcher_id': matchup.get('pitcher', {}).get('id'),
                    'pitcher_name': matchup.get('pitcher', {}).get('fullName'),
                    'pitcher_hand': matchup.get('pitchHand', {}).get('code'),
                    'event': result.get('event'),
                    'event_type': result.get('eventType'),
                    'description': result.get('description'),
                    'rbi': result.get('rbi', 0),
                    'away_score_after': result.get('awayScore'),
                    'home_score_after': result.get('homeScore'),
                    'runner_id': runner_mov.get('runner_id'),
                    'runner_name': runner_mov.get('runner_name'),
                    'start_base': runner_mov.get('start_base'),
                    'end_base': runner_mov.get('end_base'),
                    'is_out': runner_mov.get('is_out'),
                    'out_number': runner_mov.get('out_number'),
                }
                plays_data.append(play_row)
        
        return plays_data
    
    except Exception as e:
        print(f"  Fehler bei Game {game_id}: {e}")
        return []

def download_month(year, month, output_dir):
    """we download given month in given year and save monthly csv-file"""
    
    month_name = SEASON_MONTHS[month]
    print(f"\n{'#'*50}")
    print(f"{month_name}, {year}")
    
    # get games
    games = get_month_games(year, month)
    
    if not games:
        print(f"no regular season games in {month_name}")
        return
    
    print(f"Spiele gefunden: {len(games)}")
    
    # Lade Play-by-Play für alle Spiele
    all_plays = []
    for i, game in enumerate(games, 1):
        game_id = game['game_id']
        
        
        print(f"  [{i}/{len(games)}] Game {game_id}: {game['away_name']} @ {game['home_name']}")
        
        plays = extract_play_data(game_id, game)
        all_plays.extend(plays)
        
        # Rate limit so mlb stats doesn't reject our request
        if i % 20 == 0:
            time.sleep(1)
    
    # Save as csv
    if all_plays:
        df = pd.DataFrame(all_plays)
        output_file = output_dir / f"mlb_stats_{year}_{month:02d}_{month_name}.csv"
        df.to_csv(output_file, index=False)
        
        print(f"Gespeichert: {output_file}")
        print(f"Plays: {len(df):,}")
        print(f"Größe: {output_file.stat().st_size / 1024:.1f} KB")
    else:
        print(f"Keine Daten für {month_name} {year}")

def main():
    parser = argparse.ArgumentParser(description='Download MLB-StatsAPI Season (monthly)')
    parser.add_argument('--year', type=int, required=True, help='Jahr (z.B. 2024)')
    parser.add_argument('--output', type=str, required=True, help='Output-Ordner')
    
    args = parser.parse_args()
    
    # Validierung
    current_year = datetime.now().year
    if args.year < 2008 or args.year > current_year:
        print(f"Jahr muss zwischen 2008 und {current_year} sein")
        return
    
    # create output dir
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'#'*50}")
    print(f"download year - {args.year}")
    print(f"\n{'#'*50}")
    print(f"Output folder: {output_dir}")
    
    start_time = time.time()
    
    for month in sorted(SEASON_MONTHS.keys()):
        download_month(args.year, month, output_dir)
    
    elapsed = time.time() - start_time
    
    # Zusammenfassung
    csv_files = list(output_dir.glob(f"mlb_stats_{args.year}_*.csv"))
    total_size = sum(f.stat().st_size for f in csv_files) / 1024 / 1024
    
    print(f"\n{'#'*50}")
    print(f"DONE \n")
    print(f"time: {elapsed/60:.1f} Minutes")
    print(f"\n files:")
    for f in sorted(csv_files):
        print(f"  - {f.name}")

if __name__ == "__main__":
    main()