import pandas as pd

df = pd.read_parquet('statcast_2023.parquet')

# Suche in DESCRIPTION statt events
steals = df[df['description'].str.contains('steals|caught stealing', case=False, na=False)]
print(f"Steals in description: {len(steals):,}")