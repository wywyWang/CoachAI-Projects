import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

statistics = {}

data = pd.read_csv('continous_cannotReach_trainingData.csv')
data = data[['match_id', 'set', 'rally_id', 'player', 'type']]
data = data[data['type']!='接不到']

matches = data.groupby('match_id')
statistics['matches_num'] = len(matches)
statistics['sets_num'] = 0
for name, match in matches:
    statistics['sets_num'] += len(match.groupby('set'))

statistics['strokes_num'] = len(data)
statistics['players_num'] = len(data.groupby('player'))
statistics['shot_types_num'] = len(data.groupby('type'))

rallies = data.groupby('rally_id')
statistics['rallies_num'] = len(rallies)
statistics['rally_avg_len'] = statistics['strokes_num'] / statistics['rallies_num'] 

print("# of matches:{matches_num}\n"
      "# of sets: {sets_num}\n"
      "# of rallies: {rallies_num}\n"
      "# of strokes: {strokes_num}\n"
      "# of players: {players_num}\n"
      "# of shot types: {shot_types_num}\n"
      "average length of rallies: {rally_avg_len:.4f}".format(**statistics))

rally_length_distribution = np.zeros(75, dtype=np.int32)
for name, rally in rallies:
    rally_length_distribution[len(rally)] += 1


plt.plot(figsize=(20, 20))
plt.bar(np.arange(len(rally_length_distribution)), rally_length_distribution)
plt.yticks(np.arange(0, 1100, 100))
plt.xlabel('Rally length')
plt.ylabel('Count')
plt.title('Rally length distribution')
plt.savefig('D:\文件\學校\專題\AAAI\dataset_rally_length_distribution.png', dpi=300)
plt.show()

