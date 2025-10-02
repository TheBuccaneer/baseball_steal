"""
Computes league-wide context metrics per season
Provides baselines for season-relative comparisons

Output: league_context_2018_2025.csv

Usage: python 07_compute_league_context.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def compute_league_context():
    """
    Compute league-wide means and dispersion for pop time and tempo
    Weighted by attempts/pitches for accuracy
    """
    
    print("=" * 80)
    print("COMPUTING LEAGUE CONTEXT")
    print("=" * 80)
    print(f"\nCurrent directory: {Path.cwd()}")
    
    intermediate_path = Path("analysis/intermediate")
    
    # Check files exist
    team_poptime_file = intermediate_path / "team_poptime_2018_2025.csv"
    team_tempo_file = intermediate_path / "team_tempo_2018_2025.csv"
    
    if not team_poptime_file.exists() or not team_tempo_file.exists():
        print(f"\nERROR: Missing intermediate files")
        print(f"  Need: {team_poptime_file}")
        print(f"  Need: {team_tempo_file}")
        print("\nPlease run scripts 03-04 first!")
        sys.exit(1)
    
    # Load data
    print("\nLoading team metrics...")
    team_poptime = pd.read_csv(team_poptime_file)
    team_tempo = pd.read_csv(team_tempo_file)
    
    print(f"  Pop time: {len(team_poptime)} team-seasons")
    print(f"  Tempo: {len(team_tempo)} team-seasons")
    
    # Compute league context per season
    print("\n" + "=" * 80)
    print("COMPUTING LEAGUE MEANS (WEIGHTED)")
    print("=" * 80)
    
    league_context = []
    
    for season in sorted(team_poptime['season'].unique()):
        print(f"\n{season}:")
        
        season_pop = team_poptime[team_poptime['season'] == season].copy()
        season_tempo = team_tempo[team_tempo['season'] == season].copy()
        
        # Filter reliable estimates only
        season_pop_reliable = season_pop[season_pop['low_reliability'] == 0]
        season_tempo_reliable = season_tempo[season_tempo['low_reliability'] == 0]
        
        if len(season_pop_reliable) == 0:
            print(f"  WARNING: No reliable pop time data")
            season_pop_reliable = season_pop
        
        if len(season_tempo_reliable) == 0:
            print(f"  WARNING: No reliable tempo data")
            season_tempo_reliable = season_tempo
        
        # Pop Time (attempts-weighted)
        total_attempts = season_pop_reliable['total_attempts_2b'].sum()
        if total_attempts > 0:
            league_mean_pop = np.average(
                season_pop_reliable['pop_time_2b_avg'],
                weights=season_pop_reliable['total_attempts_2b']
            )
            # Weighted std
            pop_variance = np.average(
                (season_pop_reliable['pop_time_2b_avg'] - league_mean_pop) ** 2,
                weights=season_pop_reliable['total_attempts_2b']
            )
            league_std_pop = np.sqrt(pop_variance)
        else:
            league_mean_pop = season_pop_reliable['pop_time_2b_avg'].mean()
            league_std_pop = season_pop_reliable['pop_time_2b_avg'].std()
        
        # Tempo (pitches-weighted)
        total_pitches = season_tempo_reliable['total_pitches'].sum()
        if total_pitches > 0:
            league_mean_tempo = np.average(
                season_tempo_reliable['tempo_onbase_avg'],
                weights=season_tempo_reliable['total_pitches']
            )
            # Weighted std
            tempo_variance = np.average(
                (season_tempo_reliable['tempo_onbase_avg'] - league_mean_tempo) ** 2,
                weights=season_tempo_reliable['total_pitches']
            )
            league_std_tempo = np.sqrt(tempo_variance)
        else:
            league_mean_tempo = season_tempo_reliable['tempo_onbase_avg'].mean()
            league_std_tempo = season_tempo_reliable['tempo_onbase_avg'].std()
        
        # Count teams
        n_teams_pop = len(season_pop)
        n_teams_tempo = len(season_tempo)
        n_teams = min(n_teams_pop, n_teams_tempo)
        
        context = {
            'season': season,
            'league_mean_pop2b': league_mean_pop,
            'league_std_pop2b': league_std_pop,
            'league_mean_tempo_onbase': league_mean_tempo,
            'league_std_tempo_onbase': league_std_tempo,
            'n_teams': n_teams,
            'n_teams_pop': n_teams_pop,
            'n_teams_tempo': n_teams_tempo
        }
        
        league_context.append(context)
        
        print(f"  Pop time: {league_mean_pop:.3f} ± {league_std_pop:.3f} sec ({n_teams_pop} teams)")
        print(f"  Tempo:    {league_mean_tempo:.2f} ± {league_std_tempo:.2f} sec ({n_teams_tempo} teams)")
    
    # Create DataFrame
    league_df = pd.DataFrame(league_context)
    
    # Save
    output_path = Path("analysis")
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "league_context_2018_2025.csv"
    
    league_df.to_csv(output_file, index=False)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print(f"\nSeasons: {league_df['season'].min()}-{league_df['season'].max()}")
    print(f"Rows: {len(league_df)}")
    
    print("\nPop Time over time:")
    print(league_df[['season', 'league_mean_pop2b', 'league_std_pop2b']].to_string(index=False))
    
    print("\nTempo over time:")
    print(league_df[['season', 'league_mean_tempo_onbase', 'league_std_tempo_onbase']].to_string(index=False))
    
    # Check for pitch clock effect
    if league_df['season'].max() >= 2023:
        pre_2023_tempo = league_df[league_df['season'] < 2023]['league_mean_tempo_onbase'].mean()
        post_2023_tempo = league_df[league_df['season'] >= 2023]['league_mean_tempo_onbase'].mean()
        tempo_reduction = pre_2023_tempo - post_2023_tempo
        pct_reduction = (tempo_reduction / pre_2023_tempo) * 100
        
        print(f"\nPitch Clock Effect (2023 Rule Change):")
        print(f"  Pre-2023 avg tempo: {pre_2023_tempo:.2f} sec")
        print(f"  2023+ avg tempo:    {post_2023_tempo:.2f} sec")
        print(f"  Reduction:          {tempo_reduction:.2f} sec ({pct_reduction:.1f}%)")
        print(f"  Note: Timer rule = 20s (2023), 18s (2024+) with runners")
        print(f"        Tempo definition = release-to-release (different from timer)")
    
    print(f"\nSaved: {output_file}")
    print(f"Size: {output_file.stat().st_size / 1024:.1f} KB")
    print("\nLeague context ready!")

if __name__ == "__main__":
    compute_league_context()