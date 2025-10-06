"""
Combine MiLB pitcher opportunities files into analysis-ready panel
Creates post2022 indicator and filters for C3-C5 models

Output: milb_data/milb_panel_c3c5.csv

Usage: python c7f_combine_milb_panel.py
"""

import pandas as pd
from pathlib import Path

def combine_milb_data():
    data_path = Path("milb_data")
    output_path = Path("analysis/c7_milb")
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load all pitcher opportunities files
    files = sorted(data_path.glob("milb_pitcher_opportunities_*.csv"))
    
    print("=" * 80)
    print("COMBINING MILB DATA FOR C3-C5 ANALYSIS")
    print("=" * 80)
    
    all_data = []
    
    for file in files:
        df = pd.read_csv(file)
        all_data.append(df)
        print(f"\nLoaded: {file.name}")
        print(f"  {len(df):,} pitchers, season {df['season'].iloc[0]}, level {df['level_name'].iloc[0]}")
    
    # Combine
    combined = pd.concat(all_data, ignore_index=True)
    
    print(f"\n\nCombined total: {len(combined):,} pitcher-season observations")
    
    # Create treatment indicator
    combined['post2022'] = (combined['season'] >= 2022).astype(int)
    
    # Create year relative to treatment
    combined['year_rel'] = combined['season'] - 2022
    
    # Filter: Keep 2019, 2021, 2022 only (exclude 2020 if exists, and 2023)
    analysis_years = [2019, 2021, 2022]
    panel = combined[combined['season'].isin(analysis_years)].copy()
    
    print(f"\nFiltered to analysis years {analysis_years}: {len(panel):,} observations")
    
    # Quality filters (similar to MLB analysis)
    print("\n" + "=" * 80)
    print("APPLYING QUALITY FILTERS")
    print("=" * 80)
    
    print(f"\nBefore filters: {len(panel):,} pitcher-seasons")
    
    # Minimum opportunities threshold
    min_opps = 10
    panel_filtered = panel[panel['total_opps'] >= min_opps].copy()
    print(f"After min {min_opps} opportunities: {len(panel_filtered):,} pitcher-seasons")
    
    # Summary by level and year
    print("\n" + "=" * 80)
    print("FINAL SAMPLE COMPOSITION")
    print("=" * 80)
    
    summary = panel_filtered.groupby(['level_name', 'season']).agg({
        'pitcher_id': 'count',
        'total_opps': 'sum',
        'attempts': 'sum',
        'sb': 'sum',
        'cs': 'sum'
    }).reset_index()
    
    summary = summary.rename(columns={'pitcher_id': 'pitchers'})
    summary['success_rate'] = summary['sb'] / summary['attempts']
    summary['attempt_rate'] = summary['attempts'] / summary['total_opps']
    
    print("\n", summary.to_string(index=False))
    
    # Save pooled
    output_pooled = output_path / "milb_panel_c3c5_pooled.csv"
    panel_filtered.to_csv(output_pooled, index=False)
    
    # Save separate by level
    output_aaa = output_path / "milb_panel_c3c5_AAA.csv"
    output_aa = output_path / "milb_panel_c3c5_AA.csv"
    
    panel_aaa = panel_filtered[panel_filtered['level_name'] == 'AAA'].copy()
    panel_aa = panel_filtered[panel_filtered['level_name'] == 'AA'].copy()
    
    panel_aaa.to_csv(output_aaa, index=False)
    panel_aa.to_csv(output_aa, index=False)
    
    print("\n" + "=" * 80)
    print("OUTPUT FILES")
    print("=" * 80)
    print(f"\nSaved to: {output_path}/")
    print(f"\n1. Pooled (AAA + AA): {output_pooled.name}")
    print(f"   Observations: {len(panel_filtered):,}")
    print(f"\n2. AAA only: {output_aaa.name}")
    print(f"   Observations: {len(panel_aaa):,}")
    print(f"\n3. AA only: {output_aa.name}")
    print(f"   Observations: {len(panel_aa):,}")
    print(f"\nColumns:")
    print(f"  - pitcher_id, pitcher_name, season, level_name")
    print(f"  - total_opps, attempts, sb, cs")
    print(f"  - attempt_rate, success_rate, sb_rate")
    print(f"  - post2022 (treatment indicator)")
    print(f"  - year_rel (relative to 2022)")
    
    print("\n" + "=" * 80)
    print("READY FOR C3-C5 MODELS")
    print("=" * 80)
    print("\nMain specification (pooled):")
    print("  outcome ~ post2022 + level_name + post2022:level_name")
    print("           + pitcher_id (FE) + offset(log(total_opps))")
    print("\nRobustness (separate by level):")
    print("  Run C3-C5 on AAA-only and AA-only files")
    print("\nModels:")
    print("  C3: Attempt rate (PPML)")
    print("  C4: Success | attempt (Binomial)")
    print("  C5: SB rate (PPML)")
    
    # Basic validation
    print("\n" + "=" * 80)
    print("PRE-POST COMPARISON")
    print("=" * 80)
    
    pre_post = panel_filtered.groupby('post2022').agg({
        'pitcher_id': 'count',
        'total_opps': 'sum',
        'attempts': 'sum',
        'sb': 'sum'
    })
    
    pre_post['attempt_rate'] = pre_post['attempts'] / pre_post['total_opps']
    pre_post['success_rate'] = pre_post['sb'] / pre_post['attempts']
    
    print("\nPre (2019, 2021):")
    print(f"  Pitchers: {int(pre_post.loc[0, 'pitcher_id']):,}")
    print(f"  Opportunities: {int(pre_post.loc[0, 'total_opps']):,}")
    print(f"  Attempt Rate: {pre_post.loc[0, 'attempt_rate']:.3f}")
    print(f"  Success Rate: {pre_post.loc[0, 'success_rate']:.3f}")
    
    print("\nPost (2022):")
    print(f"  Pitchers: {int(pre_post.loc[1, 'pitcher_id']):,}")
    print(f"  Opportunities: {int(pre_post.loc[1, 'total_opps']):,}")
    print(f"  Attempt Rate: {pre_post.loc[1, 'attempt_rate']:.3f}")
    print(f"  Success Rate: {pre_post.loc[1, 'success_rate']:.3f}")
    
    change_attempt = (pre_post.loc[1, 'attempt_rate'] - pre_post.loc[0, 'attempt_rate']) / pre_post.loc[0, 'attempt_rate'] * 100
    change_success = (pre_post.loc[1, 'success_rate'] - pre_post.loc[0, 'success_rate']) / pre_post.loc[0, 'success_rate'] * 100
    
    print(f"\nChange pre→post:")
    print(f"  Attempt Rate: {change_attempt:+.1f}%")
    print(f"  Success Rate: {change_success:+.1f}%")

if __name__ == "__main__":
    combine_milb_data()