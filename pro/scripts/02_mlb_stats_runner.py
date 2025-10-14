"""
purpose is to supplement data from frist script with runner information
"""

import statsapi
import pandas as pd
from datetime import datetime, timedelta
import argparse
from pathlib import Path
import time

SEASON_MONTHS = {
    3: 'march',
    4: 'april', 
    5: 'may',
    6: 'june',
    7: 'july',
    8: 'august',
    9: 'september'
}

def download_runner_events_for_month(year, month):
    """
    Download all runner events for a specific month
    """
    
    # Determine date range for month
    start_date = f"{year}-{month:02d}-01"
    
    if month == 12:
        end_year = year + 1
        end_month = 1
    else:
        end_year = year
        end_month = month + 1
    
    end_date = f"{end_year}-{end_month:02d}-01"
    
    print(f"\n  getting date: {start_date} to {end_date}")
    
    schedule = statsapi.schedule(start_date=start_date, end_date=end_date)
    
    if len(schedule) == 0:
        print(f"  no games!")
        return pd.DataFrame()
    
    print(f"  Found {len(schedule)} games")
    
    all_events = []
    
    for i, game in enumerate(schedule, 1):
        game_id = game['game_id']
        game_date = game['game_date']
        

        print(f"games {i}/{len(schedule)}...")
        
        try:
            game_data = statsapi.get('game', {'gamePk': game_id})
            
            if 'liveData' not in game_data or 'plays' not in game_data['liveData']:
                continue
            
            plays = game_data['liveData']['plays']['allPlays']
            
            for play_idx, play in enumerate(plays):
                if 'runners' not in play:
                    continue
                
                about = play.get('about', {})
                inning = about.get('inning')
                half_inning = about.get('halfInning', '').lower()  # top/bottom              
                matchup = play.get('matchup', {})
                batter_id = matchup.get('batter', {}).get('id')                
                count = play.get('count', {})
                balls = count.get('balls')
                strikes = count.get('strikes')
                outs = count.get('outs')
                
                for runner in play['runners']:
                    details = runner.get('details', {})
                    movement = runner.get('movement', {})
                    
                    event_row = {
                        'game_id': game_id,
                        'game_date': game_date,
                        'at_bat_index': play_idx,
                        'inning': inning,
                        'half_inning': half_inning,
                        'balls': balls,
                        'strikes': strikes,
                        'outs': outs,
                        'batter_id': batter_id,
                        'runner_id': details.get('runner', {}).get('id'),
                        'start_base': movement.get('start'),
                        'end_base': movement.get('end'),
                        'is_out': movement.get('isOut', False),
                        'out_number': movement.get('outNumber'),
                        'runner_event': details.get('event'),
                        'runner_event_type': details.get('eventType'),
                        'movement_reason': details.get('movementReason'),
                        'is_scoring_event': details.get('isScoringEvent', False),
                        'rbi': details.get('rbi', False),
                        'earned': details.get('earned', False)
                    }
                    
                    all_events.append(event_row)
            
            # we sleep to get some time rest, so the API won't deny lots of requests 
            time.sleep(0.1)
            
        except Exception as e:
            print(f"    Error processing game {game_id}: {e}")
            continue
    
    if not all_events:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_events)
    print(f"  Extracted {len(df):,} runner events")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Download runner event data')
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--output', type=str, required=True)
    
    args = parser.parse_args()
    
    print("#########################")
    print(f"year: {args.year}")
    
    output_path = Path(args.output)
    year_folder = output_path / f"mlb_runner_events_{args.year}"
    year_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput folder: {year_folder.absolute()}")
    
    # marc to october
    months = range(3, 11)  
    
    for month in months:
        month_name = datetime(args.year, month, 1).strftime('%B').lower()
        print("\n")
        print(f"monat: {month_name.upper()} {args.year}")
        
        df = download_runner_events_for_month(args.year, month)
        
        if df.empty:
            print(f"  No date {month_name}")
            continue
        
        # Save
        output_file = year_folder / f"mlb_{args.year}_{month:02d}_{month_name}.csv"
        df.to_csv(output_file, index=False)
        
        print(f"  Size: {output_file.stat().st_size / 1024:.1f} KB")
    
    print("\n" + "##################" )
    print("COMPLETE")
    print("\n")
    print(f"\nDownloaded runner events for {args.year}")
    print(f"Output: {year_folder.absolute()}")
    
    # Summary
    csv_files = list(year_folder.glob("*.csv"))
    if csv_files:
        total_rows = sum(len(pd.read_csv(f)) for f in csv_files)
        print(f"Total runner events: {total_rows:,}")
        print(f"Files created: {len(csv_files)}")

if __name__ == "__main__":
    main()