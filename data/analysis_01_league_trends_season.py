"""
A1: League-Season Trends (SB, CS, Attempts, SR)
Aggregates runner panel to season-level for baseline analysis

Output: analysis/league_trends_season.csv
"""

import pandas as pd
from pathlib import Path

def main():
    print("=" * 60)
    print("LEAGUE SEASON TRENDS (2018-2025)")
    print("=" * 60)
    
    # Load data
    input_file = Path("analysis/analysis_runner_panel.csv")
    
    if not input_file.exists():
        print(f"\nERROR: {input_file} not found!")
        print("Please ensure runner panel exists")
        return
    
    runner = pd.read_csv(input_file)
    print(f"\nLoaded {len(runner):,} runner-seasons")
    
    # Aggregate by season
    league_trends = runner.groupby('season').agg({
        'sb': 'sum',
        'cs': 'sum', 
        'attempts': 'sum'
    }).reset_index()
    
    league_trends['sr'] = league_trends['sb'] / league_trends['attempts']
    
    # Display results
    print("\n" + "=" * 60)
    print("LEAGUE TOTALS BY SEASON")
    print("=" * 60)
    print(league_trends.to_string(index=False))
    
    # QC Checks (non-fatal)
    print("\n" + "=" * 60)
    print("QC CHECKS")
    print("=" * 60)
    
    # Check 1: All years present
    expected_years = set(range(2018, 2026))
    actual_years = set(league_trends['season'])
    
    if actual_years == expected_years:
        print("\n1. Years: All 2018-2025 present")
    else:
        missing = expected_years - actual_years
        extra = actual_years - expected_years
        if missing:
            print(f"\n1. Years: WARNING - Missing years: {missing}")
        if extra:
            print(f"\n1. Years: WARNING - Unexpected years: {extra}")
    
    # Check 2: Attempts consistency
    league_trends['diff'] = abs(league_trends['attempts'] - 
                                (league_trends['sb'] + league_trends['cs']))
    max_diff = league_trends['diff'].max()
    
    if max_diff <= 1:
        print(f"2. Attempts: Consistent (max diff = {max_diff})")
    else:
        print(f"2. Attempts: WARNING - Max difference = {max_diff}")
        print(league_trends[league_trends['diff'] > 1][['season', 'sb', 'cs', 'attempts', 'diff']])
    
    # Check 3: SR range
    sr_min = league_trends['sr'].min()
    sr_max = league_trends['sr'].max()
    
    if (sr_min >= 0.60) and (sr_max <= 0.95):
        print(f"3. SR Range: OK ({sr_min:.3f} - {sr_max:.3f})")
    else:
        print(f"3. SR Range: WARNING - {sr_min:.3f} to {sr_max:.3f} (expected 0.60-0.95)")
    
    # Check 4: Reality check - 2023 SR
    if 2023 in league_trends['season'].values:
        sr_2023 = league_trends.loc[league_trends['season'] == 2023, 'sr'].values[0]
        
        if sr_2023 >= 0.80:
            print(f"4. 2023 Benchmark: PASS (SR = {sr_2023:.3f}, expected >=0.80)")
        else:
            print(f"4. 2023 Benchmark: WARNING - SR = {sr_2023:.3f}, expected >=0.80")
            print("   ESPN/MLB reported ~80% SR in 2023")
    else:
        print("4. 2023 Benchmark: SKIPPED - 2023 data not found")
    
    # Save
    output_file = Path("analysis/league_trends_season.csv")
    output_cols = ['season', 'sb', 'cs', 'attempts', 'sr']
    league_trends[output_cols].to_csv(output_file, index=False)
    
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"\nSaved: {output_file}")
    print(f"Rows: {len(league_trends)}")

if __name__ == "__main__":
    main()