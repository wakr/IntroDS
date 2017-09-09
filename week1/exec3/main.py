import pandas as pd
import sqlite3
conn = sqlite3.connect('database.sqlite')

players = pd.read_sql_query("SELECT * FROM player", conn)
hall_of_fames = pd.read_sql_query("SELECT * FROM hall_of_fame", conn)
player_colleges = pd.read_sql_query("SELECT * FROM player_college", conn)

fames = pd.merge(players, hall_of_fames, on=['player_id'])
fames = fames[fames.inducted == 'Y']

fames = pd.merge(fames, player_colleges, on=['player_id'])[['player_id', 'college_id', 'year']]

grouped = fames.groupby('college_id').player_id.nunique()
grouped[grouped >= 3]

# usc with three
