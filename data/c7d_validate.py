import pandas as pd

schedule = pd.read_csv('milb_data/milb_schedule.csv')
games_by_year = schedule[schedule['level_name'] == 'AAA'].groupby('season').size()
print(games_by_year)