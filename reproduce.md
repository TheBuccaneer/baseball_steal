# Python (packages)
`python --version Python 3.13.9`
`pip install pybaseball pandas tqdm MLB-StatsAPI requests urllib3`


# Manually downloaded
downloaded from <https://baseballsavant.mlb.com/statcast_leaderboard>

baserunning_run_value_YYYY.csv
basestealing_running_game_YYYY.csv
catcher_throwing_YYYY.csv
custom_stats_YYYY.csv
pitcher_running_game_YYYY.csv
pitcher_running_game_2B_YYYY.csv
pitch_tempo_YYYY.csv
pitch_timer_infractions_batters_YYYY.csv
pitch_timer_infractions_catchers_YYYY.csv
pitch_timer_infractions_pitchers_YYYY.csv
poptime_2018.csv
sprint_speed_2018.csv



# Scripts

## 00_collect_statcast_infos.py

**purpose**
collects all column field header for statcast api in python with description

**arguments**
--output (required) - outputpath

**input files**
no

**output**
`00_statcast_columns.csv`

**location**
`/data/raw/mlb/statcast/`

**example usage**
`python 00_collect_statcast_infos --output ./../data/raw/mlb/statcast/`


## 00b_collect_more_infos.py

**purpose**
Retrosheet/Chadwick/MLB StatsAPI Data Availability Checker

**arguments**
--output (required) - outputpath
--include-eventfile-notes (optional) - if set, we get additional eventfiles
--check-mlb-api (optional) - add mlb api fields with the script
--mlb-start-year (optional Default: 2018) - set start year
--mlb-end-year (optional Default: 2025) - set end year

**input files**
no

**output**
`cwevent_fields.csv`
`cwgame_fields.csv`
`mlb_statsapi_availability.csv`
`mlb_statsapi_endpoints.csv`
`retrosheet_headers_catalog.csv`


**location**
`/data/raw/mlb/statcast/`

**example usage**
`python 00_collect_statcast_infos --output ./../data/raw/mlb/statcast/`


## 01_download_statcast.py

**purpose**
downloads data on pitch level. with columns:
game_pk	game_date	game_year	inning	inning_topbot	at_bat_number	pitch_number	balls	strikes	outs_when_up	home_team	away_team	batter	pitcher	on_1b	on_2b	on_3b	fielder_2	fielder_3	fielder_4	fielder_5	fielder_6	type	events	description	des	home_score	away_score	bat_score	fld_score	delta_run_exp	delta_home_win_exp	game_type	pitch_name	release_speed	stand	p_throws

**arguments**
--output (required) - outputpath
--year (required) - outputyear
--chunk-days (optional Default: 5) - Days per API chunk (≤5 recommended due to row limits)
--sleep-seconds (option Default: 2) - Pause between chunks
--test-mode (optional) - Small timeframe only (April 2023, 10 days)
--include-outfielders (optional) - Include outfielder columns (fielder_7-9)

**input files**
no

**output**
`statcast_pitches_YYYY_MM.csv`

**location**
`/data/raw/mlb/statcast/`

**example usage**
`python .\01_download_statcast.py --output ./../data/raw/mlb/statcast/ --year 2018`




## 02_statsapi_pbp_monthly

**purpose**
downloads additional data for runner movement purposes. Data will later be merged with other data. We use mlb stat api, because season 2025 is already included.
game_pk	game_date	inning	inning_topbot	at_bat_index	pitch_in_pa	balls	strikes	outs	home_team_id	away_team_id	batter_id	pitcher_id	event_type	event	play_desc	is_pa_last	RUN1_SB_FL	RUN2_SB_FL	RUN3_SB_FL	RUN1_CS_FL	RUN2_CS_FL	RUN3_CS_FL	RUN1_PK_FL	RUN2_PK_FL	RUN3_PK_FL	runner_id	start_base	end_base	is_out	out_number	runner_event	runner_event_type	movement_reason	is_scoring_event	rbi	earned	catcher_id

**arguments**

--output (required) - outputpath
--year (required) - outputyear
--sleep-seconds (optional - Default: 1.5) - 'Sleep duration between API requests
--retry (optional - Default: 1.5) - Number of retry attempts for failed requests 
--test-mode (optional) - downloads test data of month of april

**input files**
no

**output**
`pbp_YYYY_MM.csv`

**location**
`/data/raw/mlb/mlb_stats/`

**example usage**
`python .\02_statsapi_pbp_monthly.py --output ./../data/raw/mlb/statcast/ --year 2018`
