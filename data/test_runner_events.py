"""
Test download: April 2025 MLB data with runner event details
Checks if runner.details.eventType fields exist in API

Usage: python test_runner_events.py
"""

import statsapi
import pandas as pd
from datetime import datetime

print("="*80)
print("TEST: MLB RUNNER EVENT FIELDS")
print("="*80)

# Test: First week of April 2025
start_date = "2025-04-01"
end_date = "2025-04-07"

print(f"\nFetching games: {start_date} to {end_date}")

schedule = statsapi.schedule(start_date=start_date, end_date=end_date)

print(f"Found {len(schedule)} games")

if len(schedule) == 0:
    print("No games found - adjust date range")
    exit()

# Test first 3 games only
test_games = schedule[:3]

all_runner_events = []

for game in test_games:
    game_id = game['game_id']
    print(f"\nProcessing game {game_id}...")
    
    try:
        game_data = statsapi.get('game', {'gamePk': game_id})
        
        if 'liveData' not in game_data or 'plays' not in game_data['liveData']:
            continue
        
        plays = game_data['liveData']['plays']['allPlays']
        
        for play in plays:
            if 'runners' not in play:
                continue
            
            for runner in play['runners']:
                # Extract runner details
                details = runner.get('details', {})
                
                event_row = {
                    'game_id': game_id,
                    'runner_id': runner.get('details', {}).get('runner', {}).get('id'),
                    'start_base': runner.get('movement', {}).get('start'),
                    'end_base': runner.get('movement', {}).get('end'),
                    'is_out': runner.get('movement', {}).get('isOut', False),
                    'out_number': runner.get('movement', {}).get('outNumber'),
                    'runner_event': details.get('event'),
                    'runner_event_type': details.get('eventType'),
                    'movement_reason': details.get('movementReason'),
                    'runner_scored': runner.get('details', {}).get('isScoringEvent', False)
                }
                
                all_runner_events.append(event_row)
        
        print(f"  Extracted {len(all_runner_events)} runner events so far")
        
    except Exception as e:
        print(f"  Error: {e}")

# Create DataFrame
df = pd.DataFrame(all_runner_events)

# Save
output_file = "test_runner_events_april2025.csv"
df.to_csv(output_file, index=False)

print("\n" + "="*80)
print("RESULTS")
print("="*80)

print(f"\nTotal runner events: {len(df)}")

if len(df) > 0:
    print("\nColumn availability:")
    for col in ['runner_event', 'runner_event_type', 'movement_reason']:
        non_null = df[col].notna().sum()
        pct = (non_null / len(df)) * 100
        print(f"  {col:25s}: {non_null:4d} / {len(df):4d} ({pct:5.1f}%)")
    
    print("\nUnique runner_event_type values:")
    print(df['runner_event_type'].value_counts().to_string())
    
    print("\nSample rows with runner_event_type:")
    sample = df[df['runner_event_type'].notna()].head(10)
    print(sample[['runner_id', 'start_base', 'end_base', 'runner_event_type', 'is_out']].to_string(index=False))
    
    # Check for steals
    steals = df[df['runner_event_type'] == 'stolen_base']
    cs = df[df['runner_event_type'].str.contains('caught_stealing', na=False)]
    
    print(f"\nSteal events found:")
    print(f"  Stolen bases: {len(steals)}")
    print(f"  Caught stealing: {len(cs)}")

print(f"\nSaved: {output_file}")