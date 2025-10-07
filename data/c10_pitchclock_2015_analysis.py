#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
c10_pitchclock_2015_analysis.py

Analysiert den reinen Pitch-Clock Effekt (2015 in AA/AAA):
- Treatment: AA & AAA (bekamen 20s pitch clock in 2015)
- Control: High-A (keine pitch clock)
- Pre/Post: 2014 vs 2015

Outputs:
- Teams + Games CSVs
- Pitcher×Season×League aggregated data
- Two DiDs: (1) AA vs High-A, (2) AAA vs High-A

Usage:
  # Step 1: Download teams and games
  python c10_pitchclock_2015_analysis.py --step download --pilot
  
  # Step 2: Extract PBP and run DiD
  python c10_pitchclock_2015_analysis.py --step analyze --pilot
  
  # Full run (both steps)
  python c10_pitchclock_2015_analysis.py --step all
"""

import os
import sys
import time
import argparse
from pathlib import Path
from collections import defaultdict
import requests
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

BASE_URL = "https://statsapi.mlb.com/api/v1"
HEADERS = {"User-Agent": "research-script/1.0 (c10_2015_analysis)"}

# ========== CONFIGURATION ==========

SPORT_IDS = {
    'AAA': 11,
    'AA': 12,
    'High-A': 13,
    'Single-A': 14
}

SEASONS = [2014, 2015]

# ========== STEP 1: DOWNLOAD TEAMS & GAMES ==========

def _get(url, params):
    """HTTP GET with retries"""
    for attempt in range(5):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                return r.json()
            time.sleep(1.5 * (attempt + 1))
        except requests.RequestException:
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed GET {url} after retries")

def fetch_teams(sport_id, season):
    """Fetch all teams for a sport/season"""
    url = f"{BASE_URL}/teams"
    params = {"sportId": sport_id, "season": season, "activeStatus": "Y"}
    data = _get(url, params)
    
    teams = []
    for t in data.get("teams", []):
        teams.append({
            "season": season,
            "sport_id": sport_id,
            "team_id": t.get("id"),
            "team_name": t.get("name"),
            "team_abbrev": t.get("abbreviation"),
            "league_name": (t.get("league") or {}).get("name"),
            "division_name": (t.get("division") or {}).get("name"),
            "venue_id": (t.get("venue") or {}).get("id"),
            "venue_name": (t.get("venue") or {}).get("name"),
        })
    
    return pd.DataFrame(teams)

def fetch_schedule(team_id, season, sport_id):
    """Fetch regular season games for a team"""
    url = f"{BASE_URL}/schedule"
    params = {
        "teamId": team_id,
        "sportId": sport_id,
        "season": season,
        "gameType": "R",
        "hydrate": "team,linescore",
    }
    
    data = _get(url, params)
    games = []
    
    for d in data.get("dates", []):
        for g in d.get("games", []):
            games.append({
                "season": season,
                "sport_id": sport_id,
                "game_pk": g.get("gamePk"),
                "game_date": g.get("gameDate"),
                "status": (g.get("status") or {}).get("detailedState"),
                "home_team_id": ((g.get("teams") or {}).get("home") or {}).get("team", {}).get("id"),
                "home_team_name": ((g.get("teams") or {}).get("home") or {}).get("team", {}).get("name"),
                "away_team_id": ((g.get("teams") or {}).get("away") or {}).get("team", {}).get("id"),
                "away_team_name": ((g.get("teams") or {}).get("away") or {}).get("team", {}).get("name"),
                "venue_id": (g.get("venue") or {}).get("id"),
                "venue_name": (g.get("venue") or {}).get("name"),
            })
    
    return games

def download_teams_and_games(outdir, pilot=False):
    """Download teams and games for all levels/seasons"""
    
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("C10: DOWNLOADING 2014/2015 TEAMS & GAMES")
    print("="*80)
    print(f"\nLevels: {', '.join(SPORT_IDS.keys())}")
    print(f"Seasons: {SEASONS}")
    print(f"Mode: {'PILOT' if pilot else 'FULL'}\n")
    
    for level_name, sport_id in SPORT_IDS.items():
        for season in SEASONS:
            print(f"\n{'='*80}")
            print(f"{level_name} ({sport_id}) - {season}")
            print('='*80)
            
            # Fetch teams
            teams_df = fetch_teams(sport_id, season)
            
            if teams_df.empty:
                print(f"  WARNING: No teams found")
                continue
            
            print(f"  Teams: {len(teams_df)}")
            print(f"  Leagues: {', '.join(teams_df['league_name'].unique())}")
            
            # Save teams
            teams_file = outdir / f"teams_{season}_{level_name.replace('-', '')}.csv"
            teams_df.to_csv(teams_file, index=False)
            print(f"  Saved: {teams_file.name}")
            
            # Fetch games for each team
            all_games = []
            for team_id in teams_df['team_id'].unique():
                games = fetch_schedule(team_id, season, sport_id)
                all_games.extend(games)
                time.sleep(0.2)
            
            games_df = pd.DataFrame(all_games)
            
            if games_df.empty:
                print(f"  WARNING: No games found")
                continue
            
            # Deduplicate by game_pk
            games_df = games_df.drop_duplicates('game_pk').sort_values('game_pk')
            
            print(f"  Games: {len(games_df)}")
            
            # Save games
            games_file = outdir / f"games_{season}_{level_name.replace('-', '')}.csv"
            games_df.to_csv(games_file, index=False)
            print(f"  Saved: {games_file.name}")
    
    print("\n" + "="*80)
    print("DOWNLOAD COMPLETE")
    print("="*80)

# ========== STEP 2: PBP EXTRACTION & DiD ==========

def load_team_league_mapping(data_dir):
    """Load team→league mapping from teams CSVs"""
    
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
            key = (row['team_id'], row['season'])
            team_to_league[key] = {
                'league_name': row['league_name'],
                'sport_id': row['sport_id']
            }
        
        print(f"  Leagues: {', '.join(df['league_name'].unique())}")
        print(f"  Teams: {len(df)}")
    
    print(f"\nTotal mappings: {len(team_to_league)}")
    return team_to_league

def get_league_for_game(game_row, team_to_league):
    """Get league name for a game"""
    season = game_row['season']
    home_id = game_row['home_team_id']
    
    key = (home_id, season)
    info = team_to_league.get(key)
    
    if info:
        return info['league_name'], info['sport_id']
    
    return None, None

def fetch_game_pbp(game_pk, max_retries=3):
    """Fetch play-by-play for a game"""
    url = f"{BASE_URL}/game/{game_pk}/playByPlay"
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None

def extract_pitcher_stats_from_pbp(game_pk, pbp_data):
    """Extract pitcher opportunities and SB/CS"""
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
        
        # Check for runners on 1B or 2B
        runners = play.get('runners', [])
        has_runner = any(
            r.get('movement', {}).get('originBase') in ['1B', '2B']
            for r in runners
        )
        
        if has_runner:
            pitcher_stats[pitcher_id]['total_opps'] += 1
        
        # Check for SB/CS events
        for event in play.get('playEvents', []):
            event_type = event.get('details', {}).get('event', '')
            if 'Stolen Base' in event_type:
                pitcher_stats[pitcher_id]['sb'] += 1
            elif 'Caught Stealing' in event_type:
                pitcher_stats[pitcher_id]['cs'] += 1
    
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

def process_games_csv(csv_path, team_to_league, max_games=None):
    """Process games CSV: fetch PBP and extract stats"""
    
    print(f"\n{'='*80}")
    print(f"Processing: {csv_path.name}")
    print('='*80)
    
    df_games = pd.read_csv(csv_path)
    
    if max_games:
        df_games = df_games.head(max_games)
        print(f"  Pilot mode: {len(df_games)} games")
    
    print(f"  Total games: {len(df_games)}")
    
    all_pitcher_stats = []
    games_processed = 0
    games_with_data = 0
    
    for idx, row in df_games.iterrows():
        game_pk = row['game_pk']
        
        if idx % 50 == 0:
            print(f"  [{idx}/{len(df_games)}] Game {game_pk}...")
        
        # Get league
        league_name, sport_id = get_league_for_game(row, team_to_league)
        
        if not league_name:
            continue
        
        # Get level name from sport_id
        level_name = {v: k for k, v in SPORT_IDS.items()}.get(sport_id, 'Unknown')
        
        # Fetch PBP
        pbp_data = fetch_game_pbp(game_pk)
        
        if not pbp_data:
            continue
        
        games_processed += 1
        
        # Extract stats
        pitcher_stats = extract_pitcher_stats_from_pbp(game_pk, pbp_data)
        
        if pitcher_stats:
            games_with_data += 1
            for stats in pitcher_stats:
                stats.update({
                    'gamePk': game_pk,
                    'season': row['season'],
                    'level_name': level_name,
                    'league_name': league_name,
                    'sport_id': sport_id
                })
                all_pitcher_stats.append(stats)
        
        time.sleep(0.3)
    
    print(f"  Processed: {games_processed}, with data: {games_with_data}")
    
    return all_pitcher_stats

def aggregate_to_pitcher_season(pitcher_stats_list):
    """Aggregate to pitcher×season×level×league"""
    df = pd.DataFrame(pitcher_stats_list)
    
    if df.empty:
        return df
    
    agg = df.groupby(['pitcher_id', 'pitcher_name', 'season', 'level_name', 'league_name']).agg({
        'total_opps': 'sum',
        'sb': 'sum',
        'cs': 'sum',
        'attempts': 'sum',
        'gamePk': 'nunique'
    }).reset_index()
    
    agg = agg.rename(columns={'gamePk': 'games'})
    
    # Rates
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

# ========== DiD ANALYSIS ==========

def prepare_did_data(df, treatment_levels, control_levels):
    """Prepare DiD data"""
    df_did = df[
        df['level_name'].isin(treatment_levels + control_levels)
    ].copy()
    
    df_did['treat'] = df_did['level_name'].isin(treatment_levels).astype(int)
    df_did['post'] = (df_did['season'] == 2015).astype(int)
    df_did['treat_post'] = df_did['treat'] * df_did['post']
    
    df_did = df_did[df_did['total_opps'] > 0].copy()
    
    return df_did

def run_ppml_did(df_did, outcome='sb', name='DiD'):
    """Run PPML DiD"""
    print(f"\n  {name} - Outcome: {outcome}")
    print(f"  N={len(df_did)}, Pitchers={df_did['pitcher_id'].nunique()}")
    
    formula = f"{outcome} ~ treat + post + treat_post"
    
    try:
        model = smf.glm(
            formula=formula,
            data=df_did,
            family=sm.families.Poisson(),
            offset=np.log(df_did['total_opps'])
        ).fit(cov_type='cluster', cov_kwds={'groups': df_did['pitcher_id']})
        
        coef = model.params['treat_post']
        se = model.bse['treat_post']
        pval = model.pvalues['treat_post']
        
        pct_change = 100 * (np.exp(coef) - 1)
        ci_lower = 100 * (np.exp(coef - 1.96 * se) - 1)
        ci_upper = 100 * (np.exp(coef + 1.96 * se) - 1)
        
        print(f"  Effect: {pct_change:.2f}% [{ci_lower:.2f}%, {ci_upper:.2f}%], p={pval:.4f}")
        
        return {
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
    
    except Exception as e:
        print(f"  ERROR: {e}")
        return None

def run_did_aa_vs_higha(df):
    """D1: AA vs High-A"""
    print("\n" + "="*80)
    print("DiD D1: AA vs High-A (2014→2015)")
    print("Treatment: AA (got 20s pitch clock)")
    print("Control: High-A (no pitch clock)")
    print("="*80)
    
    df_did = prepare_did_data(df, ['AA'], ['High-A'])
    
    print("\nDescriptives:")
    print(df_did.groupby(['level_name', 'season']).agg({
        'sb': 'sum',
        'total_opps': 'sum',
        'pitcher_id': 'nunique'
    }))
    
    results = {
        'sb': run_ppml_did(df_did, 'sb', 'D1_AA_vs_HighA'),
        'attempts': run_ppml_did(df_did, 'attempts', 'D1_AA_vs_HighA')
    }
    
    return results

def run_did_aaa_vs_higha(df):
    """D2: AAA vs High-A"""
    print("\n" + "="*80)
    print("DiD D2: AAA vs High-A (2014→2015)")
    print("Treatment: AAA (got 20s pitch clock)")
    print("Control: High-A (no pitch clock)")
    print("="*80)
    
    df_did = prepare_did_data(df, ['AAA'], ['High-A'])
    
    print("\nDescriptives:")
    print(df_did.groupby(['level_name', 'season']).agg({
        'sb': 'sum',
        'total_opps': 'sum',
        'pitcher_id': 'nunique'
    }))
    
    results = {
        'sb': run_ppml_did(df_did, 'sb', 'D2_AAA_vs_HighA'),
        'attempts': run_ppml_did(df_did, 'attempts', 'D2_AAA_vs_HighA')
    }
    
    return results

def analyze_pbp_and_did(data_dir, out_dir, pilot=False):
    """Main analysis function"""
    
    data_path = Path(data_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    max_games = 10 if pilot else None
    
    print("="*80)
    print("C10: PBP EXTRACTION & DiD ANALYSIS")
    print("="*80)
    print(f"\nMode: {'PILOT' if pilot else 'FULL'}")
    
    # Load mappings
    team_to_league = load_team_league_mapping(data_path)
    
    # Find game CSVs
    game_csvs = sorted(data_path.glob('games_*.csv'))
    
    if not game_csvs:
        raise FileNotFoundError(f"No game CSVs in {data_path}")
    
    print(f"\nFound {len(game_csvs)} CSVs")
    
    # Process all games
    all_pitcher_stats = []
    
    for csv_path in game_csvs:
        stats = process_games_csv(csv_path, team_to_league, max_games)
        all_pitcher_stats.extend(stats)
    
    # Aggregate
    print("\n" + "="*80)
    print("AGGREGATING TO PITCHER×SEASON×LEVEL")
    print("="*80)
    
    df_agg = aggregate_to_pitcher_season(all_pitcher_stats)
    
    print(f"\nPitchers: {df_agg['pitcher_id'].nunique():,}")
    print(f"Observations: {len(df_agg):,}")
    print("\nBy level×season:")
    print(df_agg.groupby(['level_name', 'season']).agg({
        'pitcher_id': 'nunique',
        'total_opps': 'sum',
        'sb': 'sum'
    }))
    
    # Save
    agg_file = out_path / 'pitcher_season_aggregated_2015.csv'
    df_agg.to_csv(agg_file, index=False)
    print(f"\nSaved: {agg_file}")
    
    # Run DiDs
    print("\n" + "="*80)
    print("RUNNING DiD ANALYSES")
    print("="*80)
    
    results_d1 = run_did_aa_vs_higha(df_agg)
    results_d2 = run_did_aaa_vs_higha(df_agg)
    
    # Save results
    all_results = []
    for results in [results_d1, results_d2]:
        if results['sb']:
            all_results.append(results['sb'])
        if results['attempts']:
            all_results.append(results['attempts'])
    
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_file = out_path / 'did_results_2015.csv'
        results_df.to_csv(results_file, index=False)
        print(f"\nSaved: {results_file}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

# ========== MAIN ==========

def main():
    parser = argparse.ArgumentParser(description='C10: 2015 Pitch Clock Analysis')
    parser.add_argument('--step', choices=['download', 'analyze', 'all'], default='all',
                       help='Which step to run')
    parser.add_argument('--pilot', action='store_true', help='Pilot mode (10 games/CSV)')
    parser.add_argument('--data-dir', type=str, default='data/c10',
                       help='Data directory')
    parser.add_argument('--out-dir', type=str, default='data/c10/analysis',
                       help='Output directory')
    args = parser.parse_args()
    
    if args.step in ['download', 'all']:
        download_teams_and_games(args.data_dir, args.pilot)
    
    if args.step in ['analyze', 'all']:
        analyze_pbp_and_did(args.data_dir, args.out_dir, args.pilot)

if __name__ == "__main__":
    main()