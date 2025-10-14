# Collect Data

## Requirements

### Python
Python version 3.13.7

### Packages Dependencies
**full list of packages in requirements.txt**

`pip install MLB-StatsAPI`
`pip install pandas`
`pip install pybaseball`
`pip install pybaseball numpy`

## Scripts

### manually-download

**purpose**
We use Baseball Savant [Link-Text](https://baseballsavant.mlb.com/leaderboard/) leaderboards to complete stats which
are downloaded in this section by the python scripts. 

**location**
/data/raw/leaderboard/

**output**
baserunning_run_value
basestealing_running_game
catcher_throwing
custom_stats
pitch_temp
pitch_timer_infractions_batters
pitcher_running_game_2B
pitcher_running_game
poptime
sprint_speed



### 01_mlb_stats.py 
**purpose**
Downloads play by play season for a given season in output directory

**location**
/data/raw/mlb_stats/

**input**
--year - season year (required)
--output -output path (required)

**output** 
`mlb_stats/mlb_stats_YYYY_MM_month.csv`

**how to use (example)**
`python 01_mlb_stats.py --year 2024 --output data/mlb_stats/`


**data in detail**
'game_date'
'away_team'
'home_team'
'away_score'
'home_score',
'inning'
'half_inning'
'at_bat_index'
'balls'
'strikes'
'outs'
'batter_id'
'batter_name'
'batter_side'
'pitcher_id'
'pitcher_name'
'pitcher_hand'
'event'
'event_type'
'description'
'rbi'
'away_score_after'
'home_score_after'
'runner_id'
'runner_name'
'start_base'
'end_base'
'is_out'
'out_number'

### 02_mlb_stats_runner.py 

**purpose**
Supplements the basic data with specific runner information

**location**
/data/raw/mlb_runner_events/

**input**
--year - season year (required)
--output -output path (required)

**output**
mlb_runner_events/mlb_runner_events_XXXX

**how to use (example)**
`python .\02_mlb_stats_runner.py --year 2024 --output .\data`

**data in detail**
'game_id'
'game_date'
'at_bat_index'
'inning'
'half_inning'
'balls'
'strikes'
'outs'
'batter_id'
'runner_id'
'start_base'
'end_base'
'is_out'
'out_number'
'runner_event'
'runner_event_type'
'movement_reason'
'is_scoring_event'
'rbi'
'earned'


### 03_statcast_stats.py

**purpose**
downloads statcast Pitch-by-Pitch data like data of 01_mlb_stats.py. Not all data is available on one source. 

**location**
/data/raw/statcast/

**input**
--year - season year (required)
--output -output path (required)

**output** 
`statcast/statcast_YYYY_MM_monat.csv`

**how to use (example)**
`python 03_statcast_stats.py --year 2024 --output data/statcast/`

**data in details**
'game_pk' 
'pitcher'
'batter'
'fielder_2'
'fielder_3'
'fielder_4'
'fielder_5'
'fielder_6'
'fielder_7'
'fielder_8'
'fielder_9

### 04_team_poptime_2018_2025.py

**purpose**
Aggregates catcher pop times at the team level (weighted by steal attempts).
Catcher A: 1.85 sec (50 Attempts)
Catcher B: 2.10 sec (3 Attempts)
Team Pop Time = (1.85 × 50 + 2.10 × 3) / (50 + 3) = 1.864 sec

In path analysis for the small ball return, pop time represents the defensive side:
Pitch Tempo → Lead Distance → Pop Time → Steal Attempt → Steal Success

**location**
/data/analysed/

**input**
--data -path to the poptime_XXXX files, created in [Manually Download](### manually-download) section
--output -path to output folder. 

**output**
`./outputpath/04_team_poptime_2018_2025.csv`

**how to use (example)**
`python .\04_team_poptime_2018_2025.py --data ./ --out ./../data/analysed`

### 04b_optional.py

**purpose**
Calculates season-relative pop time metrics. Season pop times change. 
1.95 sec avg in 2018 -> Team is better than league avg
1.95 sec avg in 2018 -> Team is worse than league avg
We address this to put team pop time in relative to league avg

**input**
--input - path + input file, which was created via the `04_team_poptime_2018_2025.py` script, which is `04_team_poptime_2018_2025.csv`

**output**
output is `/input_path/04_b_team_poptime_2018_2025_relative.csv`

**how to use (example)**
`python .\04b_optional.py --input .\..\data\analysed\04_team_poptime_2018_2025.csv`

**data in details**
The new columns are:
-League mean per season (weighted)
-pop2/3b_diff: Difference from the league average (e.g., +0.05 sec = slower)
-pop2/3b_zscore: Standardized deviation (comparable across years)
-pop2/3b_percentile: Ranking within the season (0-100%)




## Results

### `04_team_poptime_2018_2025`

see /data/analysed/04_team_poptime_2018_2025.csv for full results

Top 5 (2nd Base)

2025 Season
---------------------------------------
Rank Team  Pop Time  Attempts  Catchers 
---------------------------------------
1    PHI   1.866     74        2        
2    NYM   1.901     74        3        
3    SF    1.906     68        3        
4    OAK   1.914     95        3        
5    KC    1.921     23        3        

2024 Season
---------------------------------------
Rank Team  Pop Time  Attempts  Catchers 
---------------------------------------
1    PHI   1.862     69        4        
2    CWS   1.899     78        3        
3    SF    1.920     102       6        
4    OAK   1.928     88        2        
5    TB    1.930     88        5        

2023 Season
---------------------------------------
Rank Team  Pop Time  Attempts  Catchers 
---------------------------------------
1    PHI   1.841     68        2        
2    TB    1.896     74        3        
3    OAK   1.918     90        4        
4    HOU   1.922     94        3        
5    SF    1.926     85        4        

2022 Season
---------------------------------------
Rank Team  Pop Time  Attempts  Catchers 
---------------------------------------
1    PHI   1.845     61        2        
2    HOU   1.932     91        4        
3    OAK   1.934     55        4        
4    TB    1.943     61        4        
5    BOS   1.955     49        2        

2021 Season
---------------------------------------
Rank Team  Pop Time  Attempts  Catchers 
---------------------------------------
1    PHI   1.889     36        3        
2    OAK   1.916     79        4        
3    STL   1.942     37        2        
4    DET   1.945     49        4        
5    TB    1.946     45        2        

2020 Season
---------------------------------------
Rank Team  Pop Time  Attempts  Catchers 
---------------------------------------
1    PHI   1.867     18        3        
2    OAK   1.920     20        3        
3    COL   1.929     20        3        
4    HOU   1.933     22        3        
5    TB    1.941     19        3        

2019 Season
---------------------------------------
Rank Team  Pop Time  Attempts  Catchers 
---------------------------------------
1    PHI   1.910     52        2        
2    SD    1.944     34        3        
3    PIT   1.949     28        3        
4    MIA   1.965     40        4        
5    NYY   1.972     42        3        

2018 Season
---------------------------------------
Rank Team  Pop Time  Attempts  Catchers 
---------------------------------------
1    CLE   1.938     55        3        
2    MIA   1.957     51        4        
3    PHI   1.964     84        3        
4    KC    1.979     48        3        
5    TB    1.986     34        4        



### 04b_optional.py

Hier noch ein R Script, um alles zu berechnen

Audit-Ergebnis: ✓ Alle Berechnungen korrekt
Kernbefunde:
1. Datenstruktur

✓ 240 Zeilen (30 Teams × 8 Saisons 2018-2025)
✓ Alle Saisons haben genau 30 Teams

2. League Means

✓ Korrekt gewichtet nach Attempts (nur reliable teams)
✓ Trend: 2.013s (2018) → 1.955s (2025) = -58ms (Catcher werden schneller)

3. Relative Metriken

✓ pop2b_diff = team_avg - league_mean → korrekt
✓ pop2b_zscore = diff / sample_std (n-1) → korrekt (statistisch sauber)
✓ pop2b_percentile = Rang innerhalb Saison → korrekt

4. Wichtiger technischer Punkt:
Das Script verwendet die sample standard deviation (Nenner: n-1) statt population std (Nenner: n). Das ist die korrekte Wahl für Stichprobenstatistik und entspricht Standard-Praxis.
5. Data Quality:

0 Teams mit low reliability (<10 attempts)
Alle Teams haben ausreichend Daten

Beispiel-Validierung (Team 143, 2023):

Pop time: 1.841s (schnellste Defense)
-119ms unter Liga-Durchschnitt
Z-score: -3.03 (top 3%)

Fazit: Sowohl Logik als auch Ergebnisse sind einwandfrei. Die Scripts arbeiten korrekt.