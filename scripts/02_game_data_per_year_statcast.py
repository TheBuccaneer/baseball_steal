"""
Statcast Season Downloader - Alle Pitch-by-Pitch Daten
Struktur optimiert für späteres Mergen mit Leaderboards & MLB-StatsAPI

Nutzung: python script.py --year 2024 --output data/statcast_2024.parquet
"""

import argparse
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from pybaseball import statcast
import time

def download_season_statcast(year, output_path):
    """
    Lädt alle Statcast-Daten einer Saison
    Chunks von 5 Tagen (wegen API Limit von 40.000 Rows)
    """
    
    print(f"\n{'='*80}")
    print(f"STATCAST SEASON DOWNLOADER - {year}")
    print(f"{'='*80}\n")
    
    # Saison-Zeitraum (MLB Regular Season: März - September)
    start_date = datetime(year, 3, 1)
    end_date = datetime(year, 10, 31)
    
    all_data = []
    current_date = start_date
    chunk_days = 5  # API Limit: max 40k rows pro Request
    
    total_chunks = (end_date - start_date).days // chunk_days + 1
    chunk_num = 0
    
    print(f"📅 Zeitraum: {start_date.date()} bis {end_date.date()}")
    print(f"📦 Chunks: {total_chunks} (je {chunk_days} Tage)\n")
    
    # Lade in Chunks
    while current_date <= end_date:
        chunk_num += 1
        chunk_end = min(current_date + timedelta(days=chunk_days-1), end_date)
        
        start_str = current_date.strftime('%Y-%m-%d')
        end_str = chunk_end.strftime('%Y-%m-%d')
        
        print(f"[{chunk_num}/{total_chunks}] {start_str} bis {end_str}... ", end='', flush=True)
        
        try:
            # Statcast Download
            df_chunk = statcast(start_dt=start_str, end_dt=end_str)
            
            if df_chunk is not None and len(df_chunk) > 0:
                all_data.append(df_chunk)
                print(f"✅ {len(df_chunk):,} Pitches")
            else:
                print("⚠️ Keine Daten")
            
            # Rate limiting (freundlich zur API)
            time.sleep(2)
            
        except Exception as e:
            print(f"❌ Fehler: {e}")
        
        current_date = chunk_end + timedelta(days=1)
    
    # Kombiniere alle Chunks
    if not all_data:
        print("\n❌ Keine Daten geladen!")
        return
    
    print(f"\n📊 Kombiniere {len(all_data)} Chunks...")
    df_full = pd.concat(all_data, ignore_index=True)
    
    # Optimiere Datentypen für Mergen
    print("🔧 Optimiere Datentypen für Merging...")
    
    # Wichtige IDs als Integer (für Mergen mit Leaderboards)
    id_cols = ['game_pk', 'pitcher', 'batter', 'fielder_2', 'fielder_3', 
               'fielder_4', 'fielder_5', 'fielder_6', 'fielder_7', 
               'fielder_8', 'fielder_9']
    for col in id_cols:
        if col in df_full.columns:
            df_full[col] = pd.to_numeric(df_full[col], errors='coerce').astype('Int64')
    
    # Datum als datetime
    if 'game_date' in df_full.columns:
        df_full['game_date'] = pd.to_datetime(df_full['game_date'])
    
    # Jahr als Integer
    if 'game_year' in df_full.columns:
        df_full['game_year'] = df_full['game_year'].astype('Int64')
    
    # Kategorische Variablen (spart Speicher)
    cat_cols = ['pitch_type', 'pitch_name', 'stand', 'p_throws', 'home_team', 
                'away_team', 'inning_topbot', 'if_fielding_alignment', 
                'of_fielding_alignment', 'bb_type']
    for col in cat_cols:
        if col in df_full.columns:
            df_full[col] = df_full[col].astype('category')
    
    # Sortiere chronologisch
    df_full = df_full.sort_values(['game_date', 'game_pk', 'at_bat_number', 'pitch_number'])
    
    # Speichere als Parquet
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"💾 Speichere als Parquet...")
    df_full.to_parquet(output_path, index=False, compression='snappy')
    
    # Statistik
    file_size_mb = output_path.stat().st_size / 1024 / 1024
    
    print(f"\n{'='*80}")
    print(f"✅ FERTIG!")
    print(f"{'='*80}")
    print(f"Gespeichert: {output_path}")
    print(f"Pitches: {len(df_full):,}")
    print(f"Spalten: {len(df_full.columns)}")
    print(f"Größe: {file_size_mb:.1f} MB")
    print(f"Zeitraum: {df_full['game_date'].min()} bis {df_full['game_date'].max()}")
    
    # Merge-Keys Übersicht
    print(f"\n📌 WICHTIGE MERGE-KEYS:")
    print(f"  • game_pk      - Spiel-ID (eindeutig)")
    print(f"  • pitcher      - Pitcher-ID (für Tempo, Running Game)")
    print(f"  • batter       - Batter-ID (für Lead Distance, Sprint Speed)")
    print(f"  • fielder_2    - Catcher-ID (für Pop Time)")
    print(f"  • game_date    - Datum (für Zeitreihen)")
    print(f"  • game_year    - Jahr")
    
    # Beispiel erste Zeilen
    print(f"\n📋 Erste 3 Zeilen (Key-Spalten):")
    key_cols = [c for c in ['game_pk', 'game_date', 'pitcher', 'batter', 
                             'fielder_2', 'events', 'description'] if c in df_full.columns]
    print(df_full[key_cols].head(3).to_string(index=False))

def main():
    parser = argparse.ArgumentParser(description='Download Statcast Season Data (Merge-Ready)')
    parser.add_argument('--year', type=int, required=True, 
                       help='Saison Jahr (z.B. 2024)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output Pfad (z.B. data/statcast_2024.parquet)')
    
    args = parser.parse_args()
    
    # Validierung
    current_year = datetime.now().year
    if args.year < 2008 or args.year > current_year:
        print(f"❌ Jahr muss zwischen 2008 und {current_year} sein")
        print("   (Statcast Daten gibt es ab 2008)")
        return
    
    # Download
    start_time = time.time()
    download_season_statcast(args.year, args.output)
    
    elapsed = time.time() - start_time
    print(f"\n⏱️  Dauer: {elapsed/60:.1f} Minuten")
    
    print(f"\n💡 Nächste Schritte:")
    print(f"   1. Leaderboards downloaden (Lead Distance, Pop Time, etc.)")
    print(f"   2. Mit game_pk/pitcher/batter/fielder_2 mergen")
    print(f"   3. Analysen starten!")

if __name__ == "__main__":
    main()