import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from typing import List, Tuple, Dict
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.backends.backend_agg as agg

ax = plt.axes()
fig = plt.gcf()
ax.set_facecolor('#198964')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_aspect('equal', adjustable='box')
plt.xticks()
data = pd.read_csv('output.csv')

data_A = data[data['obs_player'] == 'A']
data_B = data[data['obs_player'] == 'B']

landing_point_A:Dict[Tuple[int,int,int],List[Tuple[float, float]]] = {}
landing_point_B:Dict[Tuple[int,int,int],List[Tuple[float, float]]] = {}

for landing_point, data in [(landing_point_A, data_A), (landing_point_B, data_B)]:
    for i, row in data.iterrows():
        state = ast.literal_eval(row['state'])
        landing_x = row['act_landing_location_x']
        landing_y = row['act_landing_location_y']

        if not state in landing_point:
            landing_point[state] = []
        landing_point[state].append((landing_x, landing_y))

x = 177.5
y = 480


#plt.plot([25-x, 330-x], [810-y, 810-y], color='w', linestyle='-', linewidth=1.5)
#plt.plot([25-x, 330-x], [756-y, 756-y], color='w', linestyle='-', linewidth=1.5)
#plt.plot([25-x, 330-x], [594-y, 594-y], color='w', linestyle='-', linewidth=1.5)
plt.plot([25-x, 330-x], [366-y, 366-y], color='black', linestyle='-', linewidth=1)
plt.plot([25-x, 330-x], [204-y, 204-y], color='black', linestyle='-', linewidth=1)
plt.plot([25-x, 330-x], [150-y, 150-y], color='black', linestyle='-', linewidth=1)
#plt.plot([25-x, 25-x],  [150-y, 810-y], color='black', linestyle='-', linewidth=1)
plt.plot([25-x, 25-x],  [480-y, 150-y], color='black', linestyle='-', linewidth=1.5)
#plt.plot([50-x, 50-x],  [150-y, 810-y], color='black', linestyle='-', linewidth=1)
plt.plot([50-x, 50-x],  [480-y, 150-y], color='black', linestyle='-', linewidth=1.5)
#plt.plot([177.5-x, 177.5-x], [150-y, 810-y], color='black', linestyle='-', linewidth=1)
plt.plot([177.5-x, 177.5-x], [366-y, 150-y], color='black', linestyle='-', linewidth=1.5)
#plt.plot([305-x, 305-x], [150-y, 810-y], color='black', linestyle='-', linewidth=1)
plt.plot([305-x, 305-x], [480-y, 150-y], color='black', linestyle='-', linewidth=1.5)
#plt.plot([330-x, 330-x], [150-y, 810-y], color='black', linestyle='-', linewidth=1)
plt.plot([330-x, 330-x], [480-y, 150-y], color='black', linestyle='-', linewidth=1.5)
plt.plot([25-x, 330-x],  [480-y, 480-y], color='black', linestyle='-', linewidth=1.5) 


plot_state = (5,5,5)
coords = np.array(landing_point_A[plot_state])
sns.kdeplot(x=coords[:,0], y = coords[:,1], color='orange')


canvas = agg.FigureCanvasAgg(fig)
canvas.draw()
renderer = canvas.get_renderer()
raw_data = renderer.tostring_rgb()
size = canvas.get_width_height()
img = np.frombuffer(raw_data, dtype=np.uint8).reshape(size[::-1] + (3,))

cv2.imshow('fig',img)
cv2.waitKey(0)