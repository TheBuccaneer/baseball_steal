#!/usr/bin/env python3
"""
Statcast Column Extractor
Erstellt 00_statcast_columns.csv mit allen verfügbaren Spalten und Beschreibungen.
"""

import pandas as pd
from pybaseball import statcast, cache
from datetime import datetime, timedelta
import argparse
import sys
from pathlib import Path

cache.enable()


def get_field_descriptions():
    """Vollständiges Dictionary aller Statcast-Felder mit Beschreibungen."""
    return {
        # Pitch Identifiers
        'pitch_type': 'Pitch type code (FF=4-seam FB, SI=Sinker, FC=Cutter, SL=Slider, CU=Curve, CH=Change, FS=Splitter)',
        'game_date': 'Date of game (YYYY-MM-DD)',
        'release_speed': 'Pitch velocity at release (mph)',
        'release_pos_x': 'Horizontal release point (ft, catcher perspective)',
        'release_pos_z': 'Vertical release point (ft)',
        'release_pos_y': 'Distance from home plate at release (ft)',
        
        # Player IDs
        'player_name': 'Batter name',
        'batter': 'Batter MLB ID',
        'pitcher': 'Pitcher MLB ID',
        'catcher': 'Catcher MLB ID',
        'events': 'Play outcome (single, double, strikeout, etc.)',
        'description': 'Pitch result (called_strike, ball, foul, hit_into_play, etc.)',
        
        # Game Context
        'zone': 'Strike zone location (1-9 in zone, 11-14 outside)',
        'des': 'Text description of play',
        'game_type': 'Game type (R=Regular, P=Playoffs, S=Spring)',
        'stand': 'Batter stance (R/L)',
        'p_throws': 'Pitcher throwing hand (R/L)',
        'home_team': 'Home team abbreviation',
        'away_team': 'Away team abbreviation',
        'type': 'Pitch result type (S=Strike, B=Ball, X=In Play)',
        'hit_location': 'Fielding position where ball was hit (1-9)',
        'bb_type': 'Batted ball type (ground_ball, line_drive, fly_ball, popup)',
        
        # Count & Outs
        'balls': 'Balls in count',
        'strikes': 'Strikes in count',
        'game_year': 'Season year',
        'pfx_x': 'Horizontal movement (in, from pitcher perspective)',
        'pfx_z': 'Vertical movement (in, vs gravity)',
        'plate_x': 'Horizontal location at plate (ft, catcher view)',
        'plate_z': 'Vertical location at plate (ft)',
        'on_3b': 'Runner on 3B (MLB ID or NaN)',
        'on_2b': 'Runner on 2B (MLB ID or NaN)',
        'on_1b': 'Runner on 1B (MLB ID or NaN)',
        'outs_when_up': 'Outs when batter came up',
        'inning': 'Inning number',
        'inning_topbot': 'Top or Bot of inning',
        
        # Batted Ball Data
        'hc_x': 'Hit coordinate X (pixel, 2.5ft/pixel)',
        'hc_y': 'Hit coordinate Y (pixel, 2.5ft/pixel)',
        'fielder_2': 'Catcher fielding play (MLB ID)',
        'fielder_3': 'First baseman (MLB ID)',
        'fielder_4': 'Second baseman (MLB ID)',
        'fielder_5': 'Third baseman (MLB ID)',
        'fielder_6': 'Shortstop (MLB ID)',
        'fielder_7': 'Left fielder (MLB ID)',
        'fielder_8': 'Center fielder (MLB ID)',
        'fielder_9': 'Right fielder (MLB ID)',
        'umpire': 'Home plate umpire (MLB ID)',
        'sv_id': 'DEPRECATED pitch tracking ID',
        'vx0': 'Velocity X at y=50ft (ft/s)',
        'vy0': 'Velocity Y at y=50ft (ft/s)',
        'vz0': 'Velocity Z at y=50ft (ft/s)',
        'ax': 'Acceleration X (ft/s²)',
        'ay': 'Acceleration Y (ft/s²)',
        'az': 'Acceleration Z (ft/s²)',
        'sz_top': 'Top of strike zone (ft)',
        'sz_bot': 'Bottom of strike zone (ft)',
        
        # Advanced Metrics
        'hit_distance_sc': 'Projected hit distance (ft, Statcast)',
        'launch_speed': 'Exit velocity (mph)',
        'launch_angle': 'Launch angle (degrees)',
        'effective_speed': 'Perceived velocity accounting for extension (mph)',
        'release_spin_rate': 'Spin rate at release (rpm)',
        'release_extension': 'Distance from rubber to release point (ft)',
        'game_pk': 'Unique game ID',
        'estimated_ba_using_speedangle': 'xBA (expected batting average)',
        'estimated_woba_using_speedangle': 'xwOBA (expected weighted on-base average)',
        'woba_value': 'wOBA value of play outcome',
        'woba_denom': 'wOBA denominator (plate appearances)',
        'babip_value': 'BABIP value (1 if hit, 0 if out on ball in play)',
        'iso_value': 'ISO value (extra bases on hit)',
        'launch_speed_angle': 'Launch speed/angle bucket (1-6, 6=Barrel)',
        'at_bat_number': 'At-bat number in game',
        'pitch_number': 'Pitch number in at-bat',
        
        # Pitch Movement
        'spin_axis': 'Spin axis (degrees, 0°=12 o\'clock from catcher view)',
        'delta_home_win_exp': 'Change in home win probability',
        'delta_run_exp': 'Change in run expectancy',
        
        # Post-2023 Rule Changes
        'if_fielding_alignment': 'Infield alignment (Standard, Strategic)',
        'of_fielding_alignment': 'Outfield alignment (Standard, Strategic)',
        'post_away_score': 'Away score after play',
        'post_home_score': 'Home score after play',
        'post_bat_score': 'Batting team score after play',
        'post_fld_score': 'Fielding team score after play',
        'bat_score': 'Batting team score at pitch',
        'fld_score': 'Fielding team score at pitch',
        
        # Timing & Baserunning (2023+)
        'pitch_tempo': 'Time between pitches (seconds)',
        'timer_infractions': 'Timer violation (pitcher/batter)',
        'lead_distance': 'Runner lead distance at pitch release (ft)',
        'lead_distance_gained': 'Additional lead gained during pitch (ft)',
        'pop_time_2b': 'Catcher pop time to 2B (seconds)',
        'pop_time_3b': 'Catcher pop time to 3B (seconds)',
        'arm_strength_c': 'Catcher arm strength (mph)',
        'steal_attempt': 'Steal attempt flag (1/0)',
        'caught_stealing': 'Caught stealing flag (1/0)',
        'steal_success': 'Successful steal flag (1/0)',
        'pickoff_attempt': 'Pickoff attempt flag (1/0)',
        'disengagement_count': 'Number of disengagements in PA',
        'sprint_speed': 'Runner sprint speed (ft/s)',
        'home_to_first': 'Time from home to first base (seconds)',
        
        # Deprecated
        'spin_dir': 'DEPRECATED - Use spin_axis',
        'spin_rate_deprecated': 'DEPRECATED - Use release_spin_rate',
        'break_angle_deprecated': 'DEPRECATED',
        'break_length_deprecated': 'DEPRECATED',
        'tfs_deprecated': 'DEPRECATED',
        'tfs_zulu_deprecated': 'DEPRECATED',
        'pitcher_1': 'DEPRECATED fielding designation',
        'fielder_2_1': 'DEPRECATED fielding designation',
    }


def main():
    parser = argparse.ArgumentParser(description='Extrahiert alle Statcast-Spalten mit Beschreibungen')
    parser.add_argument('--output', required=True, help='Output-Verzeichnis')
    args = parser.parse_args()
    
    try:
        # Output-Pfad vorbereiten
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / '00_statcast_columns.csv'
        
        # Sample-Daten laden
        print("Lade Sample-Daten...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=2)
        
        df = statcast(
            start_dt=start_date.strftime('%Y-%m-%d'),
            end_dt=end_date.strftime('%Y-%m-%d')
        )
        
        # Spalten extrahieren
        columns = df.columns.tolist()
        dtypes = df.dtypes.to_dict()
        descriptions = get_field_descriptions()
        
        # CSV erstellen
        df_doc = pd.DataFrame([
            {
                'column': col,
                'dtype': str(dtypes.get(col, 'unknown')),
                'description': descriptions.get(col, 'No description available')
            }
            for col in columns
        ])
        
        df_doc.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"✓ Erfolgreich: {output_file} ({len(columns)} Spalten)")
        
    except Exception as e:
        print(f"✗ Fehler: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()