#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
c9_download_lowa_data.py

Zweck
-----
Lädt Teams und Spielpläne für das Low-A/High-A-Design:
  • PRE (2019): High-A-Leagues  -> California, Carolina, Florida State League  (sportId=13)
  • POST (2021): Single-A-Leagues -> Low-A West, Low-A East, Low-A Southeast  (sportId=14)
  • Kontrolle AA 2019/2021: Eastern/Southern/Texas (2019) bzw. Double-A Northeast/South/Central (2021) (sportId=12)

Ausgaben
--------
CSV in <outdir>/c9/:
  teams_<key>.csv   und   games_<key>.csv
wo <key> ∈ {2019_highA, 2021_singleA, 2019_aa, 2021_aa}

Hinweise
--------
• sportId-Mapping (MLB Stats API): 11=AAA, 12=AA, 13=High-A, 14=Low-A. 
• 2021-Regeltests: Alle Low-A-Ligen Pickoff-Limit (max. 2), Low-A West zusätzlich Pitch-Clock.
• 2020: MiLB-Saison abgesagt → einziges Pre-Jahr ist 2019.

(Quellen: baseballr Doku zu sportId; MiLB/MLB Ankündigungen zu 2021-Regeln; MLB/MiLB Meldungen zur 2020-Absage.)
"""

import os
import sys
import time
import json
import math
import argparse
from typing import Dict, List, Tuple
import requests
import pandas as pd

STATSAPI = "https://statsapi.mlb.com/api/v1"
HEADERS = {"User-Agent": "research-script/1.0 (c9_lowA_download)"}

# -------- Konfiguration: Ziel-Ligen & sportId je Jahr --------

LEAGUE_CONFIG: Dict[str, Dict] = {
    # PRE 2019: High-A (A+) -> diese drei Ligen
    "2019_highA": {
        "season": 2019,
        "sport_id": 13,  # High-A
        "league_names": ["California League", "Carolina League", "Florida State League"],
    },
    # POST 2021: Single-A (Low-A) -> diese drei Ligen
    "2021_singleA": {
        "season": 2021,
        "sport_id": 14,  # Single-A
        "league_names": ["Low-A West", "Low-A East", "Low-A Southeast"],
    },
    # Kontrolle AA PRE 2019
    "2019_aa": {
        "season": 2019,
        "sport_id": 12,  # AA
        "league_names": ["Eastern League", "Southern League", "Texas League"],
    },
    # Kontrolle AA POST 2021 (neue Namen)
    "2021_aa": {
        "season": 2021,
        "sport_id": 12,  # AA
        "league_names": ["Double-A Northeast", "Double-A South", "Double-A Central"],
    },
}

# -------- HTTP Helper --------

def _get(url: str, params: Dict) -> Dict:
    for attempt in range(5):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                return r.json()
            # 429/5xx: kurz backoff
            time.sleep(1.5 * (attempt + 1))
        except requests.RequestException:
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed GET {url} after retries; last params={params}")

# -------- Core: Teams & Schedule --------

def fetch_teams_for_leagues(sport_id: int, season: int, league_names: List[str]) -> pd.DataFrame:
    """
    Ruft alle Teams für sport_id+season ab und filtert auf angegebene league_names.
    Gibt DataFrame mit teamId, name, leagueName, venueId, parentOrgId, usw. zurück.
    """
    url = f"{STATSAPI}/teams"
    params = {"sportId": sport_id, "season": season, "activeStatus": "Y"}
    data = _get(url, params)
    teams = data.get("teams", [])
    rows = []
    for t in teams:
        league = (t.get("league") or {}).get("name")
        if league in set(league_names):
            rows.append({
                "season": season,
                "sport_id": sport_id,
                "team_id": t.get("id"),
                "team_name": t.get("name"),
                "team_abbrev": t.get("abbreviation"),
                "league_name": league,
                "division_name": (t.get("division") or {}).get("name"),
                "venue_id": (t.get("venue") or {}).get("id"),
                "venue_name": (t.get("venue") or {}).get("name"),
                "parent_org_id": (t.get("parentOrg") or {}).get("id"),
                "parent_org_name": (t.get("parentOrg") or {}).get("name"),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(
            f"No teams found for season={season}, sport_id={sport_id}, leagues={league_names}. "
            "Prüfe sportId/Jahr/Leaguenamen (Reorg 2021!)"
        )
    return df


def fetch_schedule_for_team(team_id: int, season: int, sport_id: int) -> List[Dict]:
    """
    Holt REGULAR-SEASON-Spiele für ein Team (deduplizieren wir später über gamePk).
    """
    url = f"{STATSAPI}/schedule"
    params = {
        "teamId": team_id,
        "sportId": sport_id,
        "season": season,
        "gameType": "R",      # Regular Season
        "hydrate": "team,linescore",
    }
    data = _get(url, params)
    results = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            results.append({
                "season": season,
                "sport_id": sport_id,
                "game_pk": g.get("gamePk"),
                "game_date": g.get("gameDate"),
                "status": (g.get("status") or {}).get("detailedState"),
                "home_team_id": ((g.get("teams") or {}).get("home") or {}).get("team", {}).get("id"),
                "home_team_name": ((g.get("teams") or {}).get("home") or {}).get("team", {}).get("name"),
                "away_team_id": ((g.get("teams") or {}).get("away") or {}).get("team", {}).get("id"),
                "away_team_name": ((g.get("teams") or {}).get("away") or {}).get("team", {}).get("name"),
                "venue_id": (g.get("venue") or {}).get("id"),
                "venue_name": (g.get("venue") or {}).get("name"),
            })
    return results


def build_and_save(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    outdir = os.path.join(outdir, "c9")
    os.makedirs(outdir, exist_ok=True)

    for key, cfg in LEAGUE_CONFIG.items():
        season = cfg["season"]
        sport_id = cfg["sport_id"]
        league_names = cfg["league_names"]

        print(f"[{key}] Fetching teams: season={season}, sport_id={sport_id}, leagues={league_names}")
        teams_df = fetch_teams_for_leagues(sport_id, season, league_names)
        teams_path = os.path.join(outdir, f"teams_{key}.csv")
        teams_df.to_csv(teams_path, index=False)
        print(f"  -> saved {len(teams_df)} teams to {teams_path}")

        # Schedule für alle Teams holen
        all_games: List[Dict] = []
        for tid in teams_df["team_id"].unique().tolist():
            games = fetch_schedule_for_team(tid, season, sport_id)
            all_games.extend(games)
            time.sleep(0.2)  # höflicher kleiner Delay

        games_df = pd.DataFrame(all_games)
        if games_df.empty:
            raise ValueError(f"[{key}] schedule empty. Prüfe sportId/leagues/season.")
        # Dedupliziere per game_pk (da pro Team geladen)
        games_df = games_df.sort_values("game_pk").drop_duplicates("game_pk")

        games_path = os.path.join(outdir, f"games_{key}.csv")
        games_df.to_csv(games_path, index=False)
        print(f"  -> saved {len(games_df)} unique games to {games_path}")

    print("\nDONE. Dateien liegen unter:", outdir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download MiLB Low-A/High-A + AA Teams & Schedules (2019/2021).")
    p.add_argument(
        "--outdir",
        type=str,
        default="data",
        help="Basis-Output-Verzeichnis (Default: ./data). CSVs landen in <outdir>/c9/",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_and_save(args.outdir)

"""
Beispiel:
    python c9_download_lowa_data.py --outdir ./data

Erwartete Dateien:
    ./data/c9/teams_2019_highA.csv
    ./data/c9/games_2019_highA.csv
    ./data/c9/teams_2021_singleA.csv
    ./data/c9/games_2021_singleA.csv
    ./data/c9/teams_2019_aa.csv
    ./data/c9/games_2019_aa.csv
    ./data/c9/teams_2021_aa.csv
    ./data/c9/games_2021_aa.csv
"""
