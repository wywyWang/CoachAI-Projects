import pandas as pd
import numpy as np


data = pd.read_csv('dicrete_cannotReach.csv')

groups = data.groupby(['rally_id']).filter(lambda x: len(x) >= 2).groupby(['rally_id'])

statistics = {}
for name, group in groups:
    d = group.head(2)

    if d['player_region'].iloc[0] == 5 and d['opponent_region'].iloc[0] == 6:
        land = d['landing_region'].iloc[0]
        ret = d['landing_region'].iloc[1]

        if land not in statistics:
            statistics[land] = {}
        
        if ret not in statistics[land]:
            statistics[land][ret] = 0
        statistics[land][ret] += 1

print(statistics)
 