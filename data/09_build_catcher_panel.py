"""
Builds analysis_catcher_panel.csv
Aggregates catcher pop time and throwing metrics

Output: analysis_catcher_panel.csv

Usage: python 09_build_catcher_panel.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def load_all_years(folder_path, file_pattern):
    """Load and concatenate CSVs for all years"""
    files = sorted(Path(folder_path).glob(file_pattern))
    if not files:
        print(f"  ERROR: No files found: {folder_path}/{file_pattern}")
        return pd.DataFrame()
    
    dfs = []
    for f in files:
        year = int(f.stem.split('_')[-1])
        df = pd.read_csv(f)
        df['season'] = year
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)

def main():
    print("=" * 80)
    print("BUILDING CATCHER PANEL")
    print("=" * 80)
    print(f"\nCurrent directory: {Path.cwd()}")
    
    leaderboards_path = Path("leaderboards")
    
    if not leaderboards_path.exists():
        print(f"\nERROR: {leaderboards_path.absolute()} not found!")
        print("Please run this script from the data/ directory")
        sys.exit(1)
    
    # Load Pop Time data
    print("\nLoading pop time leaderboards...")
    poptime = load_all_years(leaderboards_path, "poptime_*.csv")
    
    if poptime.empty:
        print("  ERROR: No pop time data")
        sys.exit(1)
    
    print(f"  Loaded {len(poptime):,} catcher-seasons")
    
    # Rename columns for consistency
    poptime = poptime.rename(columns={
        'entity_id': 'catcher_id',
        'entity_name': 'player_name',
        'maxeff_arm_2b_3b_sba': 'arm_strength',
        'exchange_2b_3b_sba': 'exchange_time'
    })
    
    # Calculate CS rate
    print("\nCalculating caught stealing rates...")
    
    # 2B CS rate
    poptime['cs_rate_2b'] = poptime.apply(
        lambda row: row['pop_2b_cs'] / (row['pop_2b_cs'] + row['pop_2b_sb']) 
        if (row['pop_2b_cs'] + row['pop_2b_sb']) > 0 else np.nan,
        axis=1
    )
    
    # 3B CS rate
    poptime['cs_rate_3b'] = poptime.apply(
        lambda row: row['pop_3b_cs'] / (row['pop_3b_cs'] + row['pop_3b_sb']) 
        if (row['pop_3b_cs'] + row['pop_3b_sb']) > 0 else np.nan,
        axis=1
    )
    
    # Overall CS rate (weighted by attempts)
    poptime['total_attempts'] = poptime['pop_2b_sba_count'] + poptime['pop_3b_sba_count']
    poptime['total_cs'] = poptime['pop_2b_cs'] + poptime['pop_3b_cs']
    poptime['total_sb'] = poptime['pop_2b_sb'] + poptime['pop_3b_sb']
    poptime['cs_rate_overall'] = poptime.apply(
        lambda row: row['total_cs'] / (row['total_cs'] + row['total_sb'])
        if (row['total_cs'] + row['total_sb']) > 0 else np.nan,
        axis=1
    )
    
    # Add treatment variable
    print("\nAdding treatment variables...")
    poptime['post_2023'] = (poptime['season'] >= 2023).astype(int)
    
    # Check for multi-team catchers
    print("\nChecking for multi-team catchers...")
    multi_team_check = poptime.groupby(['catcher_id', 'season'])['team_id'].nunique()
    multi_team_ids = multi_team_check[multi_team_check > 1].index
    poptime['multi_team'] = 0
    for cid, season in multi_team_ids:
        poptime.loc[(poptime['catcher_id'] == cid) & (poptime['season'] == season), 'multi_team'] = 1
    
    multi_team_count = poptime['multi_team'].sum()
    if multi_team_count > 0:
        print(f"  Found {multi_team_count} multi-team catcher-seasons")
    
    # Get team names (need to map from sprint_speed or other source)
    print("\nAdding team names...")
    # Simple approach: keep team_id, add team_name if available in data
    # Otherwise, leave as-is
    
    # Finalize columns
    print("\nFinalizing dataset...")
    
    final_cols = [
        'season', 'catcher_id', 'player_name',
        'team_id', 'age',
        'pop_2b_sba', 'pop_2b_cs', 'pop_2b_sb', 'pop_2b_sba_count',
        'pop_3b_sba', 'pop_3b_cs', 'pop_3b_sb', 'pop_3b_sba_count',
        'cs_rate_2b', 'cs_rate_3b', 'cs_rate_overall',
        'arm_strength', 'exchange_time',
        'total_attempts', 'total_cs', 'total_sb',
        'multi_team', 'post_2023'
    ]
    
    # Keep only columns that exist
    final_cols = [c for c in final_cols if c in poptime.columns]
    catcher_final = poptime[final_cols].copy()
    catcher_final = catcher_final.sort_values(['season', 'catcher_id'])
    
    # QA Checks
    print("\n" + "=" * 80)
    print("QA CHECKS")
    print("=" * 80)
    
    print(f"\nRows: {len(catcher_final):,}")
    print(f"Seasons: {catcher_final['season'].min()}-{catcher_final['season'].max()}")
    print(f"Unique catchers: {catcher_final['catcher_id'].nunique():,}")
    
    print("\nKey variables - Missing data:")
    key_vars = ['pop_2b_sba', 'cs_rate_overall', 'arm_strength', 'exchange_time']
    for col in key_vars:
        if col in catcher_final.columns:
            missing = catcher_final[col].isna().sum()
            pct = (missing / len(catcher_final)) * 100
            print(f"  {col:25s}: {missing:5,} ({pct:5.1f}%)")
    
    print("\nSummary stats:")
    summary_cols = ['pop_2b_sba', 'cs_rate_overall', 'arm_strength', 'exchange_time', 'total_attempts']
    summary_cols = [c for c in summary_cols if c in catcher_final.columns]
    print(catcher_final[summary_cols].describe())
    
    # Check CS rate range
    if 'cs_rate_overall' in catcher_final.columns:
        cs_valid = catcher_final['cs_rate_overall'].notna()
        if cs_valid.any():
            cs_range = (catcher_final.loc[cs_valid, 'cs_rate_overall'].min(),
                       catcher_final.loc[cs_valid, 'cs_rate_overall'].max())
            print(f"\nCS rate range: {cs_range[0]:.3f} - {cs_range[1]:.3f} ({cs_range[0]*100:.1f}% - {cs_range[1]*100:.1f}%)")
    
    # Check pop time range
    if 'pop_2b_sba' in catcher_final.columns:
        pop_valid = catcher_final['pop_2b_sba'].notna()
        if pop_valid.any():
            pop_range = (catcher_final.loc[pop_valid, 'pop_2b_sba'].min(),
                        catcher_final.loc[pop_valid, 'pop_2b_sba'].max())
            print(f"Pop time 2B range: {pop_range[0]:.3f} - {pop_range[1]:.3f} sec")
            if pop_range[0] < 1.7 or pop_range[1] > 2.3:
                print("  WARNING: Pop time outside typical range (1.7-2.3 sec)")
    
    # Save
    output_path = Path("analysis")
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "analysis_catcher_panel.csv"
    
    catcher_final.to_csv(output_file, index=False)
    
    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"\nSaved: {output_file}")
    print(f"Size: {output_file.stat().st_size / 1024:.1f} KB")
    print(f"\nCatcher panel ready for analysis!")

if __name__ == "__main__":
    main()