#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
c9_extract_pbp_and_did.py

Nimmt vorhandene game_pk CSVs und extrahiert:
- Play-by-Play für jedes Game
- Pitcher-level opportunities/SB/CS mit LEAGUE-TRENNUNG
- Aggregiert zu pitcher×season×league
- Rechnet zwei DiD-Analysen (D1: Pickoff, D2: Timer-Inkrement)

Usage:
  # Pilot (10 games pro CSV)
  python c9_extract_pbp_and_did.py --pilot
  
  # Full run
  python c9_extract_pbp_and_did.py
"""

import pandas as pd
import numpy as np
import requests
import time
import argparse
from pathlib import Path
from collections import defaultdict
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

BASE_URL = "https://statsapi.mlb.com/api/v1"

# ========== Team/League Mapping ==========

def load_team_league_mapping(data_dir):
    """
    Load all teams CSVs and create team_id -> league_name mapping
    This is CRITICAL for separating Low-A West vs East vs Southeast
    """
    
    data_path = Path(data_dir)
    team_csvs = sorted(data_path.glob('teams_*.csv'))
    
    if not team_csvs:
        raise FileNotFoundError(f"No teams CSVs found in {data_path}")
    
    print("\n" + "="*80)
    print("LOADING TEAM→LEAGUE MAPPING")
    print("="*80)
    
    team_to_league = {}
    
    for csv_path in team_csvs:
        df = pd.read_csv(csv_path)
        print(f"\n{csv_path.name}:")
        
        for _, row in df.iterrows():
            team_id = row['team_id']
            league_name = row['league_name']
            season = row['season']
            
            # Create composite key: (team_id, season)
            # Same team can be in different leagues across years
            key = (team_id, season)
            team_to_league[key] = league_name
            
        # Print unique leagues in this file
        unique_leagues = df['league_name'].unique()
        print(f"  Leagues: {', '.join(unique_leagues)}")
        print(f"  Teams: {len(df)}")
    
    print(f"\nTotal team×season mappings: {len(team_to_league)}")
    
    return team_to_league

def get_league_for_game(game_row, team_to_league):
    """
    Get league name for a game based on home/away team
    Both teams should be in same league (sanity check)
    """
    season = game_row['season']
    home_id = game_row['home_team_id']
    away_id = game_row['away_team_id']
    
    home_key = (home_id, season)
    away_key = (away_id, season)
    
    home_league = team_to_league.get(home_key)
    away_league = team_to_league.get(away_key)
    
    # Sanity check
    if home_league and away_league and home_league != away_league:
        print(f"  WARNING: Game {game_row['game_pk']} has teams from different leagues!")
        print(f"    Home: {home_league}, Away: {away_league}")
    
    # Return home league (or away if home missing)
    return home_league or away_league

# ========== PBP Extraction ==========

def fetch_game_pbp(game_pk, max_retries=3):
    """Fetch play-by-play for a game"""
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

def extract_pitcher_stats_from_pbp(game_pk, pbp_data):
    """Extract pitcher opportunities and SB/CS from play-by-play"""
    if not pbp_data or 'allPlays' not in pbp_data:
        return []
    
    pitcher_stats = defaultdict(lambda: {
        'total_opps': 0,
        'sb': 0,
        'cs': 0,
        'pitcher_name': None
    })
    
    for play in pbp_data.get('allPlays', []):
        matchup = play.get('matchup', {})
        pitcher_id = matchup.get('pitcher', {}).get('id')
        pitcher_name = matchup.get('pitcher', {}).get('fullName')
        
        if not pitcher_id:
            continue
        
        if pitcher_stats[pitcher_id]['pitcher_name'] is None:
            pitcher_stats[pitcher_id]['pitcher_name'] = pitcher_name
        
        # Check runners (opportunity = runner on 1B or 2B)
        runners = play.get('runners', [])
        has_runner = False
        for runner in runners:
            movement = runner.get('movement', {})
            origin_base = movement.get('originBase')
            if origin_base in ['1B', '2B']:
                has_runner = True
                break
        
        if has_runner:
            pitcher_stats[pitcher_id]['total_opps'] += 1
        
        # Check for SB/CS events
        play_events = play.get('playEvents', [])
        for event in play_events:
            event_type = event.get('details', {}).get('event', '')
            if 'Stolen Base' in event_type:
                pitcher_stats[pitcher_id]['sb'] += 1
            elif 'Caught Stealing' in event_type:
                pitcher_stats[pitcher_id]['cs'] += 1
    
    # Convert to list
    results = []
    for pitcher_id, stats in pitcher_stats.items():
        results.append({
            'pitcher_id': pitcher_id,
            'pitcher_name': stats['pitcher_name'],
            'total_opps': stats['total_opps'],
            'sb': stats['sb'],
            'cs': stats['cs'],
            'attempts': stats['sb'] + stats['cs']
        })
    
    return results

# ========== Main Processing ==========

def process_csv(csv_path, team_to_league, max_games=None):
    """Process one CSV: read games, fetch PBP, extract pitcher stats WITH LEAGUE"""
    
    print(f"\n{'='*80}")
    print(f"Processing: {csv_path.name}")
    print('='*80)
    
    # Read games
    df_games = pd.read_csv(csv_path)
    
    if max_games:
        df_games = df_games.head(max_games)
        print(f"  Pilot mode: limiting to {len(df_games)} games")
    
    print(f"  Total games: {len(df_games)}")
    
    # Extract season and level_key from filename
    parts = csv_path.stem.split('_')
    season = int(parts[1])
    level_key = '_'.join(parts[2:])
    
    # Process games
    all_pitcher_stats = []
    games_processed = 0
    games_with_data = 0
    
    for idx, row in df_games.iterrows():
        game_pk = row['game_pk']
        
        if idx % 50 == 0:
            print(f"  [{idx}/{len(df_games)}] Processing game {game_pk}...")
        
        # Get league for this game
        league_name = get_league_for_game(row, team_to_league)
        
        if league_name is None:
            print(f"    WARNING: No league found for game {game_pk}")
            continue
        
        # Fetch PBP
        pbp_data = fetch_game_pbp(game_pk)
        
        if pbp_data is None:
            continue
        
        games_processed += 1
        
        # Extract pitcher stats
        pitcher_stats = extract_pitcher_stats_from_pbp(game_pk, pbp_data)
        
        if pitcher_stats:
            games_with_data += 1
            for stats in pitcher_stats:
                stats.update({
                    'gamePk': game_pk,
                    'season': season,
                    'level_key': level_key,
                    'league_name': league_name  # CRITICAL: Include league name!
                })
                all_pitcher_stats.append(stats)
        
        # Rate limit
        time.sleep(0.3)
    
    print(f"  Processed: {games_processed} games, {games_with_data} with data")
    
    return all_pitcher_stats

def aggregate_to_pitcher_season(pitcher_stats_list):
    """Aggregate to pitcher×season×league"""
    df = pd.DataFrame(pitcher_stats_list)
    
    if df.empty:
        return df
    
    # Aggregate by pitcher×season×league (NOT just level_key!)
    agg = df.groupby(['pitcher_id', 'pitcher_name', 'season', 'level_key', 'league_name']).agg({
        'total_opps': 'sum',
        'sb': 'sum',
        'cs': 'sum',
        'attempts': 'sum',
        'gamePk': 'nunique'
    }).reset_index()
    
    agg = agg.rename(columns={'gamePk': 'games'})
    
    # Calculate rates
    agg['attempt_rate'] = np.where(agg['total_opps'] > 0, 
                                     agg['attempts'] / agg['total_opps'], 
                                     np.nan)
    agg['success_rate'] = np.where(agg['attempts'] > 0, 
                                     agg['sb'] / agg['attempts'], 
                                     np.nan)
    agg['sb_rate'] = np.where(agg['total_opps'] > 0, 
                               agg['sb'] / agg['total_opps'], 
                               np.nan)
    
    return agg

# ========== DiD Analysis ==========

def prepare_did_data(df, treatment_leagues, control_leagues, pre_year=2019, post_year=2021):
    """Prepare data for DiD using LEAGUE NAMES"""
    
    # Filter to relevant leagues and years
    df_did = df[
        (df['league_name'].isin(treatment_leagues + control_leagues)) &
        (df['season'].isin([pre_year, post_year]))
    ].copy()
    
    # Create treatment indicator
    df_did['treat'] = df_did['league_name'].isin(treatment_leagues).astype(int)
    
    # Create post indicator
    df_did['post'] = (df_did['season'] == post_year).astype(int)
    
    # Interaction
    df_did['treat_post'] = df_did['treat'] * df_did['post']
    
    # Filter to pitchers with opportunities
    df_did = df_did[df_did['total_opps'] > 0].copy()
    
    return df_did

def run_ppml_did(df_did, outcome='sb', name='DiD'):
    """Run PPML DiD with offset for opportunities"""
    
    print(f"\n  Running PPML DiD for outcome: {outcome}")
    print(f"  N observations: {len(df_did)}")
    print(f"  N pitchers: {df_did['pitcher_id'].nunique()}")
    
    # Create formula
    formula = f"{outcome} ~ treat + post + treat_post"
    
    # Fit PPML with offset
    try:
        model = smf.glm(
            formula=formula,
            data=df_did,
            family=sm.families.Poisson(),
            offset=np.log(df_did['total_opps'])
        ).fit(cov_type='cluster', cov_kwds={'groups': df_did['pitcher_id']})
        
        # Extract DiD coefficient
        coef = model.params['treat_post']
        se = model.bse['treat_post']
        pval = model.pvalues['treat_post']
        
        # Calculate % change
        pct_change = 100 * (np.exp(coef) - 1)
        ci_lower = 100 * (np.exp(coef - 1.96 * se) - 1)
        ci_upper = 100 * (np.exp(coef + 1.96 * se) - 1)
        
        results = {
            'name': name,
            'outcome': outcome,
            'coef': coef,
            'se': se,
            'pval': pval,
            'pct_change': pct_change,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_obs': len(df_did),
            'n_pitchers': df_did['pitcher_id'].nunique()
        }
        
        print(f"  DiD coefficient: {coef:.4f} (SE={se:.4f}, p={pval:.4f})")
        print(f"  % change: {pct_change:.2f}% [95% CI: {ci_lower:.2f}%, {ci_upper:.2f}%]")
        
        return results, model
    
    except Exception as e:
        print(f"  ERROR: {e}")
        return None, None

def run_did_d1(df):
    """D1: Pickoff-only effect (Low-A East+Southeast vs AA)"""
    
    print("\n" + "="*80)
    print("DiD D1: PICKOFF-ONLY EFFECT")
    print("Treatment: Low-A East + Southeast (Pickoff limit, NO timer)")
    print("Control: AA (no rule changes)")
    print("="*80)
    
    # 2019: Carolina League + Florida State League
    # 2021: Low-A East + Low-A Southeast
    treatment = ['Carolina League', 'Florida State League',  # 2019
                 'Low-A East', 'Low-A Southeast']            # 2021
    
    # 2019: Eastern/Southern/Texas League
    # 2021: Double-A Northeast/South/Central
    control = ['Eastern League', 'Southern League', 'Texas League',  # 2019
               'Double-A Northeast', 'Double-A South', 'Double-A Central']  # 2021
    
    df_did = prepare_did_data(df, treatment, control)
    
    # Print descriptives
    print("\nDescriptives:")
    desc = df_did.groupby(['treat', 'post']).agg({
        'sb': 'sum',
        'total_opps': 'sum',
        'pitcher_id': 'nunique'
    }).reset_index()
    desc['sb_rate'] = desc['sb'] / desc['total_opps']
    print(desc)
    
    print("\nBy league:")
    league_desc = df_did.groupby(['league_name', 'season']).agg({
        'sb': 'sum',
        'total_opps': 'sum',
        'pitcher_id': 'nunique'
    }).reset_index()
    league_desc['sb_rate'] = league_desc['sb'] / league_desc['total_opps']
    print(league_desc)
    
    # Run DiD
    results_sb, model_sb = run_ppml_did(df_did, outcome='sb', name='D1_pickoff')
    results_attempts, model_attempts = run_ppml_did(df_did, outcome='attempts', name='D1_pickoff')
    
    return {
        'sb': results_sb,
        'attempts': results_attempts,
        'data': df_did
    }

def run_did_d2(df):
    """D2: Timer increment effect (Low-A West vs East+Southeast)"""
    
    print("\n" + "="*80)
    print("DiD D2: TIMER INCREMENT EFFECT")
    print("Treatment: Low-A West (Pickoff + 15s timer)")
    print("Control: Low-A East + Southeast (Pickoff only, NO timer)")
    print("="*80)
    
    # 2019: California League (became Low-A West)
    # 2021: Low-A West
    treatment = ['California League',   # 2019
                 'Low-A West']          # 2021
    
    # 2019: Carolina + Florida State (became Low-A East/Southeast)
    # 2021: Low-A East + Southeast
    control = ['Carolina League', 'Florida State League',  # 2019
               'Low-A East', 'Low-A Southeast']            # 2021
    
    df_did = prepare_did_data(df, treatment, control)
    
    # Print descriptives
    print("\nDescriptives:")
    desc = df_did.groupby(['treat', 'post']).agg({
        'sb': 'sum',
        'total_opps': 'sum',
        'pitcher_id': 'nunique'
    }).reset_index()
    desc['sb_rate'] = desc['sb'] / desc['total_opps']
    print(desc)
    
    print("\nBy league:")
    league_desc = df_did.groupby(['league_name', 'season']).agg({
        'sb': 'sum',
        'total_opps': 'sum',
        'pitcher_id': 'nunique'
    }).reset_index()
    league_desc['sb_rate'] = league_desc['sb'] / league_desc['total_opps']
    print(league_desc)
    
    # Run DiD
    results_sb, model_sb = run_ppml_did(df_did, outcome='sb', name='D2_timer')
    results_attempts, model_attempts = run_ppml_did(df_did, outcome='attempts', name='D2_timer')
    
    return {
        'sb': results_sb,
        'attempts': results_attempts,
        'data': df_did
    }

# ========== Main ==========

def main():
    parser = argparse.ArgumentParser(description='Extract PBP and run DiD analysis')
    parser.add_argument('--pilot', action='store_true', help='Pilot mode (10 games per CSV)')
    parser.add_argument('--data-dir', type=str, default='data/c9', 
                       help='Directory with game/team CSVs')
    parser.add_argument('--out-dir', type=str, default='data/c9/analysis',
                       help='Output directory')
    args = parser.parse_args()
    
    data_path = Path(args.data_dir)
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    max_games = 10 if args.pilot else None
    
    print("="*80)
    print("LOW-A DiD ANALYSIS: PBP EXTRACTION + DiD")
    print("="*80)
    print(f"\nData directory: {data_path}")
    print(f"Output directory: {out_path}")
    print(f"Mode: {'PILOT (10 games/CSV)' if args.pilot else 'FULL'}")
    
    # Load team→league mapping
    team_to_league = load_team_league_mapping(data_path)
    
    # Find all game CSVs
    game_csvs = sorted(data_path.glob('games_*.csv'))
    
    if not game_csvs:
        raise FileNotFoundError(f"No game CSVs found in {data_path}")
    
    print(f"\nFound {len(game_csvs)} CSV files:")
    for csv in game_csvs:
        print(f"  - {csv.name}")
    
    # Process each CSV
    all_pitcher_stats = []
    
    for csv_path in game_csvs:
        stats = process_csv(csv_path, team_to_league, max_games)
        all_pitcher_stats.extend(stats)
    
    # Aggregate
    print("\n" + "="*80)
    print("AGGREGATING TO PITCHER×SEASON×LEAGUE")
    print("="*80)
    
    df_agg = aggregate_to_pitcher_season(all_pitcher_stats)
    
    print(f"\nAggregated data:")
    print(f"  Total pitchers: {df_agg['pitcher_id'].nunique():,}")
    print(f"  Total observations: {len(df_agg):,}")
    print(f"\nBy league×season:")
    print(df_agg.groupby(['league_name', 'season']).agg({
        'pitcher_id': 'nunique',
        'total_opps': 'sum',
        'sb': 'sum',
        'attempts': 'sum'
    }))
    
    # Save aggregated data
    agg_file = out_path / 'pitcher_season_aggregated.csv'
    df_agg.to_csv(agg_file, index=False)
    print(f"\nSaved aggregated data: {agg_file}")
    
    # Run DiD analyses
    print("\n" + "="*80)
    print("RUNNING DiD ANALYSES")
    print("="*80)
    
    # D1: Pickoff effect (East+Southeast vs AA)
    results_d1 = run_did_d1(df_agg)
    
    # D2: Timer increment (West vs East+Southeast)
    results_d2 = run_did_d2(df_agg)
    
    # Save results
    all_results = []
    if results_d1['sb']:
        all_results.append(results_d1['sb'])
    if results_d1['attempts']:
        all_results.append(results_d1['attempts'])
    if results_d2['sb']:
        all_results.append(results_d2['sb'])
    if results_d2['attempts']:
        all_results.append(results_d2['attempts'])
    
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_file = out_path / 'did_results.csv'
        results_df.to_csv(results_file, index=False)
        print(f"\nSaved DiD results: {results_file}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    if args.pilot:
        print("\nPilot successful! Review results, then run without --pilot.")
    else:
        print("\nFull analysis complete.")

if __name__ == "__main__":
    main()