"""
Validate MiLB data against external benchmarks
Works for AAA and AA, multiple years (2019-2023)

Usage: python c7c_validate_milb_data.py
"""

import pandas as pd
from pathlib import Path
import glob

def load_and_validate():
    data_path = Path("milb_data")
    
    # Find all pitcher opportunities files
    files = sorted(data_path.glob("milb_pitcher_opportunities_*.csv"))
    
    if not files:
        print("ERROR: No pitcher opportunities files found!")
        return
    
    print("=" * 80)
    print("MILB DATA VALIDATION")
    print("=" * 80)
    
    # Load all files
    all_data = []
    for file in files:
        df = pd.read_csv(file)
        all_data.append(df)
        print(f"\nLoaded: {file.name}")
        print(f"  {len(df):,} pitchers, season {df['season'].iloc[0]}, level {df['level_name'].iloc[0]}")
    
    combined = pd.concat(all_data, ignore_index=True)
    
    # Group by level and year
    print("\n" + "=" * 80)
    print("AGGREGATE STATISTICS BY LEVEL × YEAR")
    print("=" * 80)
    
    results = []
    
    for (level, year), group in combined.groupby(['level_name', 'season']):
        total_sb = group['sb'].sum()
        total_cs = group['cs'].sum()
        total_attempts = group['attempts'].sum()
        success_rate = total_sb / total_attempts if total_attempts > 0 else 0
        
        results.append({
            'level': level,
            'year': year,
            'pitchers': len(group),
            'total_sb': total_sb,
            'total_cs': total_cs,
            'total_attempts': total_attempts,
            'success_rate': success_rate,
            'attempts_per_pitcher': total_attempts / len(group)
        })
        
        print(f"\n{level} {year}:")
        print(f"  Pitchers: {len(group):,}")
        print(f"  Total SB: {total_sb:,}")
        print(f"  Total CS: {total_cs:,}")
        print(f"  Total Attempts: {total_attempts:,}")
        print(f"  Success Rate: {success_rate:.1%}")
    
    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # Validation checks for AAA
    print("\n" + "=" * 80)
    print("VALIDATION CHECKS (AAA)")
    print("=" * 80)
    
    aaa_data = results_df[results_df['level'] == 'AAA'].copy()
    
    if len(aaa_data) >= 2:
        # Compare years
        for i in range(len(aaa_data) - 1):
            year1 = aaa_data.iloc[i]
            year2 = aaa_data.iloc[i + 1]
            
            pct_change = ((year2['total_attempts'] - year1['total_attempts']) / year1['total_attempts']) * 100
            
            print(f"\n{int(year1['year'])} → {int(year2['year'])}:")
            print(f"  Attempts: {int(year1['total_attempts']):,} → {int(year2['total_attempts']):,} ({pct_change:+.1f}%)")
            print(f"  Success: {year1['success_rate']:.1%} → {year2['success_rate']:.1%}")
            
            # Check if 2022 has expected increase
            if int(year1['year']) == 2021 and int(year2['year']) == 2022:
                if pct_change > 10:
                    print(f"  ✓ 2021→2022 shows treatment effect")
                else:
                    print(f"  ⚠️  2021→2022 change seems low")
    
    print("\n" + "=" * 80)
    print("NOTES")
    print("=" * 80)
    print("Expected patterns:")
    print("  - 2019→2022: Attempts increase ~15-25% (AAA)")
    print("  - Success rates: 78-83% (AAA has better players than lower minors)")
    print("  - 2021 should be between 2019 and 2022")

if __name__ == "__main__":
    load_and_validate()