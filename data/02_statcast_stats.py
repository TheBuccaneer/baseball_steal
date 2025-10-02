"""
Statcast Season Downloader - Monatliche CSV-Dateien
Lädt Pitch-by-Pitch Daten für Regular Season (März-September)

Nutzung: python script.py --year 2024 --output data/statcast/
"""

import argparse
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from pybaseball import statcast
import time

# Regular Season Monate
SEASON_MONTHS = {
    3: 'march',
    4: 'april', 
    5: 'may',
    6: 'june',
    7: 'july',
    8: 'august',
    9: 'september'
}

def download_month_statcast(year, month):
    """Lädt alle Statcast-Daten eines Monats"""
    
    # Erster und letzter Tag des Monats
    first_day = datetime(year, month, 1)
    if month == 12:
        last_day = datetime(year + 1, 1, 1) - timedelta(days=1)
    else:
        last_day = datetime(year, month + 1, 1) - timedelta(days=1)
    
    # Download in 5-Tage-Chunks (API Limit)
    all_data = []
    current_date = first_day
    chunk_days = 5
    chunk_num = 0
    
    total_days = (last_day - first_day).days + 1
    total_chunks = (total_days // chunk_days) + 1
    
    while current_date <= last_day:
        chunk_num += 1
        chunk_end = min(current_date + timedelta(days=chunk_days-1), last_day)
        
        start_str = current_date.strftime('%Y-%m-%d')
        end_str = chunk_end.strftime('%Y-%m-%d')
        
        print(f"  [{chunk_num}/{total_chunks}] {start_str} bis {end_str}... ", end='', flush=True)
        
        try:
            df_chunk = statcast(start_dt=start_str, end_dt=end_str)
            
            if df_chunk is not None and len(df_chunk) > 0:
                all_data.append(df_chunk)
                print(f"{len(df_chunk):,} Pitches")
            else:
                print("keine Daten")
            
            # Rate limiting
            time.sleep(2)
            
        except Exception as e:
            print(f"Fehler: {e}")
        
        current_date = chunk_end + timedelta(days=1)
    
    # Kombiniere alle Chunks
    if all_data:
        df_month = pd.concat(all_data, ignore_index=True)
        return df_month
    else:
        return None

def optimize_datatypes(df):
    """Optimiert Datentypen für CSV-Export"""
    
    # IDs als Integer
    id_cols = ['game_pk', 'pitcher', 'batter', 'fielder_2', 'fielder_3', 
               'fielder_4', 'fielder_5', 'fielder_6', 'fielder_7', 
               'fielder_8', 'fielder_9']
    for col in id_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
    
    # Datum als datetime
    if 'game_date' in df.columns:
        df['game_date'] = pd.to_datetime(df['game_date'])
    
    # Jahr als Integer
    if 'game_year' in df.columns:
        df['game_year'] = df['game_year'].astype('Int64')
    
    # Kategorische Variablen als String (besser für CSV)
    cat_cols = ['pitch_type', 'pitch_name', 'stand', 'p_throws', 'home_team', 
                'away_team', 'inning_topbot', 'if_fielding_alignment', 
                'of_fielding_alignment', 'bb_type']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    # Sortiere chronologisch
    if 'game_date' in df.columns and 'game_pk' in df.columns:
        df = df.sort_values(['game_date', 'game_pk', 'at_bat_number', 'pitch_number'])
    
    return df

def download_and_save_month(year, month, output_dir):
    """Lädt einen Monat und speichert als CSV"""
    
    month_name = SEASON_MONTHS[month]
    
    print(f"\n{'='*80}")
    print(f"{month_name.upper()} {year}")
    print(f"{'='*80}")
    
    # Download
    df = download_month_statcast(year, month)
    
    if df is None or len(df) == 0:
        print(f"Keine Daten für {month_name} {year}")
        return
    
    print(f"\nGesamt Pitches: {len(df):,}")
    
    # Optimiere Datentypen
    print("Optimiere Datentypen...")
    df = optimize_datatypes(df)
    
    # Speichere als CSV
    output_file = output_dir / f"statcast_{year}_{month:02d}_{month_name}.csv"
    print(f"Speichere {output_file.name}...")
    df.to_csv(output_file, index=False, quoting=1, sep=',', encoding='utf-8-sig')
    
    file_size = output_file.stat().st_size / 1024 / 1024
    
    print(f"Gespeichert: {output_file}")
    print(f"Größe: {file_size:.1f} MB")
    print(f"Spalten: {len(df.columns)}")

def main():
    parser = argparse.ArgumentParser(description='Download Statcast Season (monatlich)')
    parser.add_argument('--year', type=int, required=True, help='Jahr (z.B. 2024)')
    parser.add_argument('--output', type=str, required=True, help='Output-Ordner')
    
    args = parser.parse_args()
    
    # Validierung
    current_year = datetime.now().year
    if args.year < 2008 or args.year > current_year:
        print(f"Jahr muss zwischen 2008 und {current_year} sein (Statcast ab 2008)")
        return
    
    # Erstelle Output-Ordner
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"STATCAST SEASON DOWNLOADER - {args.year}")
    print(f"{'='*80}")
    print(f"Output: {output_dir}")
    
    # Lade jeden Monat
    start_time = time.time()
    
    for month in sorted(SEASON_MONTHS.keys()):
        download_and_save_month(args.year, month, output_dir)
    
    elapsed = time.time() - start_time
    
    # Zusammenfassung
    csv_files = list(output_dir.glob(f"statcast_{args.year}_*.csv"))
    total_size = sum(f.stat().st_size for f in csv_files) / 1024 / 1024
    
    print(f"\n{'='*80}")
    print(f"FERTIG")
    print(f"{'='*80}")
    print(f"Dateien erstellt: {len(csv_files)}")
    print(f"Gesamtgröße: {total_size:.1f} MB")
    print(f"Dauer: {elapsed/60:.1f} Minuten")
    print(f"\nDateien:")
    for f in sorted(csv_files):
        print(f"  - {f.name}")

if __name__ == "__main__":
    main()