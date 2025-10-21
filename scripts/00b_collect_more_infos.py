#!/usr/bin/env python3
"""
Retrosheet/Chadwick/MLB StatsAPI Data Availability Checker
Scrapes field definitions from Retrosheet/Chadwick and checks MLB StatsAPI availability.
"""

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    import statsapi
    STATSAPI_AVAILABLE = True
except ImportError:
    STATSAPI_AVAILABLE = False

# Source URLs
SOURCES = {
    'cwevent': 'https://chadwick.readthedocs.io/en/latest/cwevent.html',
    'cwgame': 'https://chadwick.sourceforge.net/doc/cwgame.html',
    'plays': 'https://retrosheet.org/downloads/plays.html',
    'crosswalk': 'https://retrosheet.org/downloads/pbpcrosswalk.html',
    'daily_logs': 'https://www.retrosheet.org/downloads/csvcontents.html',
    'eventfile': 'https://www.retrosheet.org/eventfile.htm'
}


def setup_logging(output_dir):
    """Configures logging to file + console."""
    log_path = Path(output_dir) / 'data_availability.log'
    
    handlers = [
        logging.FileHandler(log_path, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers,
        force=True
    )
    
    return logging.getLogger(__name__)


def get_session():
    """Creates requests session with retry logic."""
    session = requests.Session()
    
    retry_strategy = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    session.headers.update({
        'User-Agent': 'RetroSheetHeaderScraper/1.0 (Research Project)'
    })
    
    return session


def fetch_page(url, session, logger):
    """Fetches page with error handling."""
    try:
        logger.info(f"Fetching: {url}")
        response = session.get(url, timeout=30)
        response.raise_for_status()
        response.encoding = 'utf-8'
        time.sleep(1)
        return response.text
    except Exception as e:
        logger.error(f"Failed to fetch {url}: {e}")
        return None


def clean_text(text):
    """Cleans and normalizes text."""
    if not text:
        return ""
    text = text.replace('\xa0', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def parse_cwevent(html, url, logger):
    """Parses cwevent field definitions."""
    soup = BeautifulSoup(html, 'lxml')
    fields = []
    
    tables = soup.find_all('table')
    
    for table in tables:
        rows = table.find_all('tr')
        
        for row in rows[1:]:
            cols = row.find_all('td')
            if len(cols) >= 3:
                field_num = clean_text(cols[0].get_text())
                description = clean_text(cols[1].get_text())
                header = clean_text(cols[2].get_text())
                
                if header and description:
                    group = 'extended' if 'extended' in description.lower() else 'standard'
                    
                    fields.append({
                        'source': 'cwevent',
                        'group': group,
                        'field_number': field_num if field_num.isdigit() else '',
                        'header': header,
                        'description': description,
                        'url': url
                    })
    
    logger.info(f"Parsed cwevent: {len(fields)} fields")
    return pd.DataFrame(fields)


def parse_cwgame(html, url, logger):
    """Parses cwgame field definitions."""
    soup = BeautifulSoup(html, 'lxml')
    fields = []
    
    tables = soup.find_all('table')
    
    for table in tables:
        rows = table.find_all('tr')
        
        for row in rows[1:]:
            cols = row.find_all('td')
            if len(cols) >= 3:
                field_num = clean_text(cols[0].get_text())
                description = clean_text(cols[1].get_text())
                header = clean_text(cols[2].get_text())
                
                if header and description:
                    group = 'extended' if 'extended' in description.lower() else 'standard'
                    
                    fields.append({
                        'source': 'cwgame',
                        'group': group,
                        'field_number': field_num if field_num.isdigit() else '',
                        'header': header,
                        'description': description,
                        'url': url
                    })
    
    logger.info(f"Parsed cwgame: {len(fields)} fields")
    return pd.DataFrame(fields)


def parse_plays(html, url, logger):
    """Parses plays.csv column definitions."""
    soup = BeautifulSoup(html, 'lxml')
    fields = []
    
    content = soup.get_text()
    lines = content.split('\n')
    
    in_fields_section = False
    for line in lines:
        line = clean_text(line)
        
        if 'contents of parsed play-by-play' in line.lower():
            in_fields_section = True
            continue
        
        if in_fields_section and line:
            match = re.match(r'^([A-Z_]+)\s+(.+)$', line)
            if match:
                header = match.group(1)
                description = match.group(2)
                
                fields.append({
                    'source': 'plays',
                    'header': header,
                    'description': description,
                    'url': url
                })
            
            if line.startswith('=') or 'notice' in line.lower():
                break
    
    logger.info(f"Parsed plays: {len(fields)} fields")
    return pd.DataFrame(fields)


def parse_crosswalk(html, url, logger):
    """Parses play-by-play crosswalk table."""
    soup = BeautifulSoup(html, 'lxml')
    rows_data = []
    
    tables = soup.find_all('table')
    
    for table in tables:
        rows = table.find_all('tr')
        
        for row in rows[1:]:
            cols = row.find_all('td')
            if len(cols) >= 3:
                bevent_col = clean_text(cols[0].get_text())
                cwevent_header = clean_text(cols[1].get_text())
                plays_column = clean_text(cols[2].get_text())
                description = clean_text(cols[3].get_text()) if len(cols) > 3 else ''
                
                if bevent_col or cwevent_header or plays_column:
                    rows_data.append({
                        'bevent_colnum': bevent_col,
                        'cwevent_header': cwevent_header,
                        'plays_column': plays_column,
                        'description': description,
                        'url': url
                    })
    
    logger.info(f"Parsed crosswalk: {len(rows_data)} mappings")
    return pd.DataFrame(rows_data)


def parse_daily_logs(html, url, logger):
    """Parses Daily Logs CSV field definitions."""
    soup = BeautifulSoup(html, 'lxml')
    fields = []
    
    content = soup.get_text()
    lines = content.split('\n')
    
    current_dataset = None
    
    for line in lines:
        line = clean_text(line)
        
        if '.csv' in line.lower() and len(line) < 50:
            current_dataset = line.split()[0] if line else None
            continue
        
        if current_dataset and line:
            match = re.match(r'^([A-Z_]+[0-9]*)\s+(.+)$', line)
            if match:
                header = match.group(1)
                description = match.group(2)
                
                fields.append({
                    'source': 'daily_logs',
                    'dataset': current_dataset,
                    'header': header,
                    'description': description,
                    'url': url
                })
    
    logger.info(f"Parsed daily logs: {len(fields)} fields")
    return pd.DataFrame(fields)


def parse_eventfile_notes(html, url, logger):
    """Parses eventfile notation/legend (optional)."""
    soup = BeautifulSoup(html, 'lxml')
    notes = []
    
    headers = soup.find_all(['h1', 'h2', 'h3', 'h4'])
    
    for header in headers:
        title = clean_text(header.get_text())
        
        next_elem = header.find_next_sibling()
        if next_elem and next_elem.name in ['ul', 'ol']:
            items = next_elem.find_all('li')
            for item in items:
                text = clean_text(item.get_text())
                if text:
                    notes.append({
                        'source': 'eventfile',
                        'header': title,
                        'description': text,
                        'url': url
                    })
    
    logger.info(f"Parsed eventfile notes: {len(notes)} entries")
    return pd.DataFrame(notes)


def check_mlb_statsapi_availability(start_year, end_year, logger):
    """Checks MLB StatsAPI data availability for year range."""
    if not STATSAPI_AVAILABLE:
        logger.warning("MLB-StatsAPI not installed. Skipping availability check.")
        logger.warning("Install with: pip install MLB-StatsAPI")
        return None
    
    logger.info(f"Checking MLB StatsAPI availability ({start_year}-{end_year})...")
    
    availability = []
    
    for year in range(start_year, end_year + 1):
        try:
            # Check regular season schedule
            schedule = statsapi.schedule(
                start_date=f'01/01/{year}',
                end_date=f'12/31/{year}',
                sportId=1
            )
            
            regular_season = [g for g in schedule if g.get('game_type') == 'R']
            spring_training = [g for g in schedule if g.get('game_type') == 'S']
            postseason = [g for g in schedule if g.get('game_type') in ['F', 'D', 'L', 'W']]
            
            availability.append({
                'year': year,
                'source': 'mlb_statsapi',
                'regular_season_games': len(regular_season),
                'spring_training_games': len(spring_training),
                'postseason_games': len(postseason),
                'total_games': len(schedule),
                'available': len(schedule) > 0,
                'checked_at': datetime.now(timezone.utc).isoformat()
            })
            
            logger.info(f"  {year}: {len(regular_season)} regular season games available")
            time.sleep(0.5)  # Be polite to API
            
        except Exception as e:
            logger.warning(f"  {year}: Error checking availability - {e}")
            availability.append({
                'year': year,
                'source': 'mlb_statsapi',
                'regular_season_games': 0,
                'spring_training_games': 0,
                'postseason_games': 0,
                'total_games': 0,
                'available': False,
                'checked_at': datetime.now(timezone.utc).isoformat()
            })
    
    df = pd.DataFrame(availability)
    logger.info(f"MLB StatsAPI check complete: {df['available'].sum()}/{len(df)} years available")
    
    return df


def document_mlb_statsapi_endpoints(logger):
    """Documents available MLB StatsAPI endpoints and methods."""
    if not STATSAPI_AVAILABLE:
        return None
    
    logger.info("Documenting MLB StatsAPI endpoints...")
    
    endpoints = [
        {
            'endpoint': 'schedule',
            'function': 'statsapi.schedule()',
            'description': 'Get game schedule for date range',
            'parameters': 'start_date, end_date, team, opponent, sportId',
            'returns': 'List of games with game_pk, game_type, teams, scores',
            'useful_for': 'Finding game IDs, filtering by game_type (R/S/P)'
        },
        {
            'endpoint': 'game',
            'function': 'statsapi.get("game", {"gamePk": gamePk})',
            'description': 'Get detailed game data',
            'parameters': 'gamePk',
            'returns': 'Complete game data including plays, at-bats, runners',
            'useful_for': 'Detailed play-by-play analysis'
        },
        {
            'endpoint': 'game_pace',
            'function': 'statsapi.game_pace()',
            'description': 'Get game pace metrics by year',
            'parameters': 'season',
            'returns': 'Average game length, pace metrics',
            'useful_for': 'Tempo analysis, rule change impact'
        },
        {
            'endpoint': 'people',
            'function': 'statsapi.get("people", {"personIds": ids})',
            'description': 'Get player biographical data',
            'parameters': 'personIds',
            'returns': 'Player names, positions, teams',
            'useful_for': 'Player ID to name mapping'
        },
        {
            'endpoint': 'teams',
            'function': 'statsapi.get("teams", {"sportId": 1})',
            'description': 'Get team information',
            'parameters': 'sportId, season',
            'returns': 'Team IDs, names, abbreviations',
            'useful_for': 'Team lookups'
        },
        {
            'endpoint': 'stats',
            'function': 'statsapi.player_stats()',
            'description': 'Get player statistics',
            'parameters': 'personId, type, season',
            'returns': 'Batting/pitching statistics',
            'useful_for': 'Player performance metrics'
        }
    ]
    
    df = pd.DataFrame(endpoints)
    logger.info(f"Documented {len(df)} MLB StatsAPI endpoints")
    
    return df


def create_master_catalog(dfs, scraped_at):
    """Creates master catalog from all dataframes."""
    master_rows = []
    
    if 'cwevent' in dfs and not dfs['cwevent'].empty:
        for _, row in dfs['cwevent'].iterrows():
            master_rows.append({
                'source': row.get('source', ''),
                'group': row.get('group', ''),
                'dataset': '',
                'field_number': row.get('field_number', ''),
                'header': row.get('header', ''),
                'plays_column': '',
                'description': row.get('description', ''),
                'url': row.get('url', ''),
                'scraped_at': scraped_at
            })
    
    if 'cwgame' in dfs and not dfs['cwgame'].empty:
        for _, row in dfs['cwgame'].iterrows():
            master_rows.append({
                'source': row.get('source', ''),
                'group': row.get('group', ''),
                'dataset': '',
                'field_number': row.get('field_number', ''),
                'header': row.get('header', ''),
                'plays_column': '',
                'description': row.get('description', ''),
                'url': row.get('url', ''),
                'scraped_at': scraped_at
            })
    
    if 'plays' in dfs and not dfs['plays'].empty:
        for _, row in dfs['plays'].iterrows():
            master_rows.append({
                'source': row.get('source', ''),
                'group': '',
                'dataset': '',
                'field_number': '',
                'header': row.get('header', ''),
                'plays_column': '',
                'description': row.get('description', ''),
                'url': row.get('url', ''),
                'scraped_at': scraped_at
            })
    
    if 'daily_logs' in dfs and not dfs['daily_logs'].empty:
        for _, row in dfs['daily_logs'].iterrows():
            master_rows.append({
                'source': row.get('source', ''),
                'group': '',
                'dataset': row.get('dataset', ''),
                'field_number': '',
                'header': row.get('header', ''),
                'plays_column': '',
                'description': row.get('description', ''),
                'url': row.get('url', ''),
                'scraped_at': scraped_at
            })
    
    if 'eventfile' in dfs and not dfs['eventfile'].empty:
        for _, row in dfs['eventfile'].iterrows():
            master_rows.append({
                'source': row.get('source', ''),
                'group': '',
                'dataset': '',
                'field_number': '',
                'header': row.get('header', ''),
                'plays_column': '',
                'description': row.get('description', ''),
                'url': row.get('url', ''),
                'scraped_at': scraped_at
            })
    
    df = pd.DataFrame(master_rows)
    
    if not df.empty:
        df = df.drop_duplicates(
            subset=['source', 'group', 'dataset', 'header'],
            keep='first'
        )
    
    return df


def save_notice(output_dir, logger):
    """Saves Retrosheet mandatory notice."""
    notice = """RETROSHEET NOTICE

The information used here was obtained free of charge from and is
copyrighted by Retrosheet. Interested parties may contact Retrosheet
at www.retrosheet.org.

This data is provided "as is" without any warranty of any kind.
Retrosheet does not make any representations or warranties about the
accuracy, completeness, timeliness, reliability, or suitability of
this data for any particular purpose.

Source: https://www.retrosheet.org/notice.txt
"""
    
    notice_path = Path(output_dir) / 'RETROSHEET_NOTICE.txt'
    with open(notice_path, 'w', encoding='utf-8') as f:
        f.write(notice)
    
    logger.info(f"Saved: RETROSHEET_NOTICE.txt")


def print_summary(dfs, mlb_available):
    """Prints summary table."""
    print("\n" + "="*70)
    print("DATA AVAILABILITY SUMMARY")
    print("="*70)
    print(f"{'Source':<30} {'Items':>15}")
    print("-"*70)
    
    total = 0
    
    # Retrosheet/Chadwick
    for name, df in dfs.items():
        if name not in ['mlb_availability', 'mlb_endpoints']:
            count = len(df) if df is not None and not df.empty else 0
            print(f"{name:<30} {count:>15,} fields")
            total += count
    
    print("-"*70)
    print(f"{'Retrosheet/Chadwick Total':<30} {total:>15,} fields")
    
    # MLB StatsAPI
    if mlb_available is not None and not mlb_available.empty:
        available_years = mlb_available[mlb_available['available'] == True]
        print(f"\n{'MLB StatsAPI':<30} {len(available_years):>15} years")
        print(f"  Year range: {available_years['year'].min()}-{available_years['year'].max()}")
        total_games = available_years['regular_season_games'].sum()
        print(f"  Regular season games: {total_games:,}")
    
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Checks data availability: Retrosheet/Chadwick + MLB StatsAPI',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--include-eventfile-notes', action='store_true',
                        help='Include eventfile notation/legend')
    parser.add_argument('--check-mlb-api', action='store_true',
                        help='Check MLB StatsAPI availability (requires MLB-StatsAPI package)')
    parser.add_argument('--mlb-start-year', type=int, default=2018,
                        help='Start year for MLB API check (default: 2018)')
    parser.add_argument('--mlb-end-year', type=int, default=2025,
                        help='End year for MLB API check (default: 2025)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    logger.info(f"Output directory: {output_dir.absolute()}")
    logger.info(f"Include eventfile notes: {args.include_eventfile_notes}")
    logger.info(f"Check MLB API: {args.check_mlb_api}")
    
    # Create session
    session = get_session()
    
    # Timestamp
    scraped_at = datetime.now(timezone.utc).isoformat()
    
    # Storage for dataframes
    dfs = {}
    mlb_availability = None
    
    try:
        # Parse Retrosheet/Chadwick sources
        logger.info("\n=== RETROSHEET/CHADWICK SCRAPING ===")
        
        html = fetch_page(SOURCES['cwevent'], session, logger)
        if html:
            dfs['cwevent'] = parse_cwevent(html, SOURCES['cwevent'], logger)
            if not dfs['cwevent'].empty:
                csv_path = output_dir / 'cwevent_fields.csv'
                dfs['cwevent'].to_csv(csv_path, index=False, encoding='utf-8')
                logger.info(f"Saved: cwevent_fields.csv")
        
        html = fetch_page(SOURCES['cwgame'], session, logger)
        if html:
            dfs['cwgame'] = parse_cwgame(html, SOURCES['cwgame'], logger)
            if not dfs['cwgame'].empty:
                csv_path = output_dir / 'cwgame_fields.csv'
                dfs['cwgame'].to_csv(csv_path, index=False, encoding='utf-8')
                logger.info(f"Saved: cwgame_fields.csv")
        
        html = fetch_page(SOURCES['plays'], session, logger)
        if html:
            dfs['plays'] = parse_plays(html, SOURCES['plays'], logger)
            if not dfs['plays'].empty:
                csv_path = output_dir / 'plays_columns.csv'
                dfs['plays'].to_csv(csv_path, index=False, encoding='utf-8')
                logger.info(f"Saved: plays_columns.csv")
        
        html = fetch_page(SOURCES['crosswalk'], session, logger)
        if html:
            dfs['crosswalk'] = parse_crosswalk(html, SOURCES['crosswalk'], logger)
            if not dfs['crosswalk'].empty:
                csv_path = output_dir / 'plays_crosswalk.csv'
                dfs['crosswalk'].to_csv(csv_path, index=False, encoding='utf-8')
                logger.info(f"Saved: plays_crosswalk.csv")
        
        html = fetch_page(SOURCES['daily_logs'], session, logger)
        if html:
            dfs['daily_logs'] = parse_daily_logs(html, SOURCES['daily_logs'], logger)
            if not dfs['daily_logs'].empty:
                csv_path = output_dir / 'daily_logs_columns.csv'
                dfs['daily_logs'].to_csv(csv_path, index=False, encoding='utf-8')
                logger.info(f"Saved: daily_logs_columns.csv")
        
        if args.include_eventfile_notes:
            html = fetch_page(SOURCES['eventfile'], session, logger)
            if html:
                dfs['eventfile'] = parse_eventfile_notes(html, SOURCES['eventfile'], logger)
                if not dfs['eventfile'].empty:
                    csv_path = output_dir / 'eventfile_notes.csv'
                    dfs['eventfile'].to_csv(csv_path, index=False, encoding='utf-8')
                    logger.info(f"Saved: eventfile_notes.csv")
        
        # Create master catalog
        master_df = create_master_catalog(dfs, scraped_at)
        if not master_df.empty:
            csv_path = output_dir / 'retrosheet_headers_catalog.csv'
            master_df.to_csv(csv_path, index=False, encoding='utf-8')
            logger.info(f"Saved: retrosheet_headers_catalog.csv ({len(master_df)} total fields)")
        
        # Save notice
        save_notice(output_dir, logger)
        
        # Check MLB StatsAPI
        if args.check_mlb_api:
            logger.info("\n=== MLB STATSAPI CHECK ===")
            
            mlb_availability = check_mlb_statsapi_availability(
                args.mlb_start_year,
                args.mlb_end_year,
                logger
            )
            
            if mlb_availability is not None and not mlb_availability.empty:
                csv_path = output_dir / 'mlb_statsapi_availability.csv'
                mlb_availability.to_csv(csv_path, index=False, encoding='utf-8')
                logger.info(f"Saved: mlb_statsapi_availability.csv")
            
            # Document endpoints
            endpoints_df = document_mlb_statsapi_endpoints(logger)
            if endpoints_df is not None and not endpoints_df.empty:
                csv_path = output_dir / 'mlb_statsapi_endpoints.csv'
                endpoints_df.to_csv(csv_path, index=False, encoding='utf-8')
                logger.info(f"Saved: mlb_statsapi_endpoints.csv")
        
        # Print summary
        print_summary(dfs, mlb_availability)
        logger.info("\n✓ Data availability check completed successfully")
        
    except KeyboardInterrupt:
        logger.warning("Check interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"✗ Critical error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()