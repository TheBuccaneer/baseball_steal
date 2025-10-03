"""
A3: QC Benchmarks - Validate SB and SR against external sources
Uses flexible tolerance: PASS if |delta_%| <= 0.5 OR |delta_abs| <= 20

Input:  analysis/league_trends_season.csv
Output: analysis/qc_benchmarks_2018_2025.csv

Sources:
- 2023: https://www.mlb.com/news/mlb-records-3000th-stolen-base-in-2023
- 2024: https://www.mlb.com/news/mlb-rule-changes-for-2024
- Historical: Baseball Almanac totals
"""

import pandas as pd
from pathlib import Path
import sys


def main():
    print("=" * 70)
    print("A3: QC BENCHMARKS - VALIDATE SB & SR")
    print("=" * 70)
    
    # Input
    input_file = Path("analysis/league_trends_season.csv")
    if not input_file.exists():
        print(f"\nERROR: Input file not found: {input_file}")
        sys.exit(1)
    
    # Load data
    df = pd.read_csv(input_file)
    print(f"\nLoaded: {input_file}")
    print(f"Seasons: {df['season'].min()} - {df['season'].max()}")
    
    # Define benchmarks (external sources)
    benchmarks = {
        # SR benchmarks (Success Rate)
        'sr': {
            2023: 0.800,  # MLB.com: 80.0% success rate in 2023
        },
        # SB benchmarks (Total Stolen Bases)
        'sb': {
            2018: 2474,   # Baseball Almanac
            2019: 2280,   # Baseball Almanac
            2020: 884,    # Baseball Almanac (COVID-shortened)
            2021: 2213,   # Baseball Almanac
            2022: 2487,   # Baseball Almanac
            2023: 3503,   # MLB.com official
            2024: 3617,   # MLB.com official
        }
    }
    
    # Build QC DataFrame
    results = []
    
    for _, row in df.iterrows():
        season = int(row['season'])
        
        # --- SR Check ---
        if season in benchmarks['sr']:
            sr_ours = row['sr']
            sr_bench = benchmarks['sr'][season]
            sr_delta_pp = (sr_ours - sr_bench) * 100  # Percentage points
            
            # Status
            sr_status = "PASS" if abs(sr_delta_pp) <= 0.5 else "FAIL"
            
            results.append({
                'season': season,
                'metric': 'SR',
                'value_ours': round(sr_ours, 4),
                'value_bench': round(sr_bench, 4),
                'delta_abs': round(sr_ours - sr_bench, 4),
                'delta_pct': round(sr_delta_pp, 2),
                'status': sr_status,
                'source': 'MLB.com 2023 announcement'
            })
        
        # --- SB Check ---
        if season in benchmarks['sb']:
            sb_ours = int(row['sb'])
            sb_bench = benchmarks['sb'][season]
            sb_delta_abs = sb_ours - sb_bench
            sb_delta_pct = (sb_delta_abs / sb_bench) * 100
            
            # Flexible tolerance: PASS if |%| <= 0.5 OR |abs| <= 20
            sb_status = "PASS" if (abs(sb_delta_pct) <= 0.5 or abs(sb_delta_abs) <= 20) else "FAIL"
            
            # Source
            if season in [2023, 2024]:
                source = f"MLB.com official ({season})"
            else:
                source = "Baseball Almanac"
            
            results.append({
                'season': season,
                'metric': 'SB',
                'value_ours': sb_ours,
                'value_bench': sb_bench,
                'delta_abs': sb_delta_abs,
                'delta_pct': round(sb_delta_pct, 2),
                'status': sb_status,
                'source': source
            })
    
    # Create output DataFrame
    qc_df = pd.DataFrame(results)
    
    # Display
    print("\n" + "=" * 70)
    print("QC RESULTS")
    print("=" * 70)
    print(qc_df.to_string(index=False))
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    total_checks = len(qc_df)
    passed = len(qc_df[qc_df['status'] == 'PASS'])
    failed = len(qc_df[qc_df['status'] == 'FAIL'])
    
    print(f"Total checks:  {total_checks}")
    print(f"PASS:          {passed}")
    print(f"FAIL:          {failed}")
    
    if failed > 0:
        print("\n⚠️  WARNING: Some checks FAILED")
        print("\nFailed checks:")
        print(qc_df[qc_df['status'] == 'FAIL'][['season', 'metric', 'delta_abs', 'delta_pct']].to_string(index=False))
    else:
        print("\n✓ All checks PASSED")
    
    # Save
    output_file = Path("analysis/qc_benchmarks_2018_2025.csv")
    output_file.parent.mkdir(exist_ok=True)
    qc_df.to_csv(output_file, index=False)
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"\nSaved: {output_file}")
    print(f"Rows:  {len(qc_df)}")
    
    print("\n" + "=" * 70)
    print("NOTES")
    print("=" * 70)
    print("Flexible tolerance for SB:")
    print("  PASS if |delta_%| <= 0.5  OR  |delta_abs| <= 20")
    print("\nThis allows:")
    print("  - Small absolute errors (±20 SB) for COVID-2020")
    print("  - Strict percentage errors for full seasons")
    
    print("\nSources:")
    print("  2023 SR & SB: https://www.mlb.com/news/mlb-records-3000th-stolen-base-in-2023")
    print("  2024 SB:      https://www.mlb.com/news/mlb-rule-changes-for-2024")
    print("  Historical:   Baseball Almanac")


if __name__ == "__main__":
    main()