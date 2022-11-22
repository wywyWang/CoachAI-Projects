import pandas as pd
import csv
import os

# Directory
inputFile = '/home/ino/Projects/ai-adminton/Data/Train_Test/OT.csv'
outputPath = '/home/ino/Projects/ai-adminton/Data/VideoToFrame/video_frames/'
outputFile = 'Label.csv'

# Create path if necessary
if not os.path.exists(outputPath):
    os.makedirs(outputPath)

# Open previous label data
with open(inputFile) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    frames = []
    x, y = [], []
    for row in readCSV:
        frames += [int(row[0])]
        x += [int(row[1])]
        y += [int(row[2])]
visibility = [1 for _ in range(len(frames))]

# Create DataFrame
df_label = pd.DataFrame(columns=['Frame', 'Visibility', 'X', 'Y'])
df_label['Frame'], df_label['Visibility'], df_label['X'], df_label['Y'] = frames, visibility, x, y

# Compensate the non-labeled frames due to no visibility of badminton
for i in range(1, frames[-1]+1):
    if i in list(df_label['Frame']):
        pass
    else:
        df_label = df_label.append(pd.DataFrame(data = {'Frame':[i], 'Visibility':[0], 'X':[0], 'Y':[0]}), ignore_index=True)

# Sorting by 'Frame'
df_label = df_label.sort_values(by=['Frame'])
df_label.to_csv(outputPath + outputFile, encoding='utf-8', index=False)
