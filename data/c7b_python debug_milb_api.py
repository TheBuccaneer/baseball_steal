"""
Debug ONE game thoroughly to see what's actually happening
"""

import requests
import json

BASE_URL = "https://statsapi.mlb.com/api/v1"
BASE_MAP = {'first': 1, 'second': 2, 'third': 3, '1B': 1, '2B': 2, '3B': 3}

def fetch_game_pbp(game_pk):
    url = f"{BASE_URL}/game/{game_pk}/playByPlay"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()

def parse_runner_state(runners):
    """Parse runner state from originBase"""
    state = ['_', '_', '_']
    
    if not runners:
        return '___'
    
    if isinstance(runners, list):
        for runner in runners:
            origin = runner.get('movement', {}).get('originBase', None)
            if origin in BASE_MAP:
                base_idx = BASE_MAP[origin] - 1
                state[base_idx] = str(BASE_MAP[origin])
    
    return ''.join(state)

def check_opportunity(runner_state):
    """Check if R1 only or R2 only"""
    # R1 only, R2 empty = steal 2B opportunity
    opp_2b = (runner_state[0] == '1' and runner_state[1] == '_')
    # R2 only, R3 empty = steal 3B opportunity  
    opp_3b = (runner_state[1] == '2' and runner_state[2] == '_')
    return opp_2b, opp_3b

# Game from debug that had a CS
game_pk = 666415

print(f"Analyzing game {game_pk}")
print("="*80)

pbp = fetch_game_pbp(game_pk)
all_plays = pbp.get('allPlays', [])

print(f"Total plays: {len(all_plays)}\n")

opportunities_found = 0
events_found = 0

for play_idx, play in enumerate(all_plays):
    runners = play.get('runners', [])
    
    if not runners:
        continue
    
    # Parse state
    state = parse_runner_state(runners)
    
    # Check for opportunities
    opp_2b, opp_3b = check_opportunity(state)
    
    # Check for SB/CS events
    has_steal_event = False
    for pe in play.get('playEvents', []):
        event = pe.get('details', {}).get('event', '')
        if 'Stolen' in event or 'Caught Stealing' in event:
            has_steal_event = True
            events_found += 1
            break
    
    # Print if we have opportunity OR steal event
    if opp_2b or opp_3b or has_steal_event:
        matchup = play.get('matchup', {})
        pitcher = matchup.get('pitcher', {})
        count = play.get('count', {})
        
        print(f"\nPlay {play_idx}:")
        print(f"  Pitcher: {pitcher.get('fullName', 'N/A')}")
        print(f"  Count: {count.get('balls', 0)}-{count.get('strikes', 0)}, {count.get('outs', 0)} out")
        print(f"  Runner state: {state}")
        print(f"  Opp 2B: {opp_2b}, Opp 3B: {opp_3b}")
        
        if has_steal_event:
            print(f"  >>> HAS STEAL EVENT")
        
        if opp_2b or opp_3b:
            opportunities_found += 1
            
        # Show runners detail
        print(f"  Runners ({len(runners)}):")
        for r in runners:
            origin = r.get('movement', {}).get('originBase', None)
            details = r.get('details', {})
            runner_name = details.get('runner', {}).get('fullName', 'N/A')
            print(f"    - {runner_name}: originBase={origin}")

print("\n" + "="*80)
print(f"SUMMARY")
print("="*80)
print(f"Opportunities found: {opportunities_found}")
print(f"Steal events found: {events_found}")
print()

if opportunities_found == 0:
    print("PROBLEM: No opportunities found!")
    print("Showing ALL plays with runners to diagnose:")
    print()
    
    for play_idx, play in enumerate(all_plays[:20]):  # First 20
        runners = play.get('runners', [])
        if runners:
            state = parse_runner_state(runners)
            print(f"Play {play_idx}: state={state}")