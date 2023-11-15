import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

matplotlib.rcParams.update({'font.size': 4})

data = pd.read_csv('MDP.csv')
used_column = ['type', 'player_region', 'opponent_region']
data = data[used_column]

data = data.groupby(['player_region', 'opponent_region'])['type'].value_counts().reset_index(name='type_prob')
pivot = data.pivot_table(index=['player_region','opponent_region'], columns='type', values=['type_prob'], fill_value=0, dropna=False)
table = pivot.to_numpy().reshape(10,10,10)
np.save('type_distribution.npy', table)


region = list(range(1, 11))
fig, axs = plt.subplots(len(region), len(region), figsize=(10, 12), dpi=200) # dpi default 100

# Add labels to the start of each row and column
axs[0, 0].text(-0.8, 1, f"    opponenet_region\nplayer_region", transform=axs[0, 0].transAxes, va='bottom', ha='center')
for player_region in range(1,11):
    axs[player_region-1, 0].text(-0.3, 0.5, f"{player_region}", transform=axs[player_region-1, 0].transAxes, va='center', ha='right')
for opponenet_region in range(1,11):
    axs[0, opponenet_region-1].text(0.5, 1.1, f"{opponenet_region}", transform=axs[0, opponenet_region-1].transAxes, va='bottom', ha='center')

#color
colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] # default color
label_color_dict = dict(zip(region, colors))
for player_region in range(1,11):
    for opponenet_region in range(1,11):
        data = pivot.loc[(player_region, opponenet_region)]
        labels = data.index.values
        labels = np.array([e[1] for e in labels])
        sizes = data['type_prob'].values
        if not np.any(sizes): # if all zero
            axs[player_region-1, opponenet_region-1].set_xticks([])
            axs[player_region-1, opponenet_region-1].set_yticks([])
            axs[player_region-1, opponenet_region-1].set_frame_on(False)
            continue
        pie_colors = [label_color_dict[i+1] for i in range(len(labels))]
        axs[player_region-1, opponenet_region-1].pie(sizes, colors=pie_colors)
        axs[player_region-1, opponenet_region-1].text(1.2, -1.2, f"{sizes.sum()}")
        #axs[player_region-1, opponenet_region-1].set_title(f"player={player_region}, opponenet={opponenet_region}")

handles = []
for label, color in label_color_dict.items():
    handles.append(plt.Rectangle((0, 0), 1, 1, fc=color))
plt.legend(handles=handles, labels=labels.tolist(), loc='lower left', title='type', bbox_to_anchor=(1, 0.5))
fig.subplots_adjust(wspace=0.1, hspace=0.1, left=0.05, right=0.95, bottom=0.02, top = 0.96)
plt.show()
#data_count = sum(count.values())
#fig, ax = plt.subplots()
#ax.pie(count.values(), labels = count.keys(), autopct='%1.1f%%')
#plt.title(column)
#plt.show()