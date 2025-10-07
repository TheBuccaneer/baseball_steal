#!/usr/bin/env python3
"""
Quick check: Are all required leagues in the teams CSVs?
Run this BEFORE the full extraction to avoid wasting time.
"""

import pandas as pd
from pathlib import Path

data_dir = Path('data/c9')

print("="*80)
print("CHECKING IF ALL REQUIRED LEAGUES ARE PRESENT")
print("="*80)

checks = [
    {
        'file': 'teams_2019_highA.csv',
        'expected': ['California League', 'Carolina League', 'Florida State League']
    },
    {
        'file': 'teams_2021_singleA.csv', 
        'expected': ['Low-A West', 'Low-A East', 'Low-A Southeast']
    },
    {
        'file': 'teams_2019_aa.csv',
        'expected': ['Eastern League', 'Southern League', 'Texas League']
    },
    {
        'file': 'teams_2021_aa.csv',
        'expected': ['Double-A Northeast', 'Double-A South', 'Double-A Central']
    }
]

all_good = True

for check in checks:
    file_path = data_dir / check['file']
    
    print(f"\n{check['file']}:")
    
    if not file_path.exists():
        print(f"  ❌ FILE NOT FOUND")
        all_good = False
        continue
    
    df = pd.read_csv(file_path)
    leagues = df['league_name'].unique().tolist()
    
    print(f"  Leagues found: {', '.join(leagues)}")
    print(f"  Total teams: {len(df)}")
    
    missing = [l for l in check['expected'] if l not in leagues]
    
    if missing:
        print(f"  ❌ MISSING: {', '.join(missing)}")
        all_good = False
    else:
        print(f"  ✓ All expected leagues present")

print("\n" + "="*80)
if all_good:
    print("✓ ALL CHECKS PASSED - Safe to run full extraction!")
else:
    print("❌ SOME LEAGUES MISSING - DO NOT RUN FULL EXTRACTION YET!")
print("="*80)