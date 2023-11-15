import pandas as pd


csv = pd.read_csv("./data/dataset.csv")

csv = csv[['rally','ball_round','roundscore_A','roundscore_B','player','type',
            'player_location_x','player_location_y','opponent_location_x','opponent_location_y','flaw','set','match_id','rally_id']]

csv.to_csv('./data/datasetSlice.csv', index=False)