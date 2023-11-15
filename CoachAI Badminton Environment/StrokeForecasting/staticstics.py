import pandas as pd


csv = pd.read_csv('data\dataset.csv')

print(csv['type'].unique())