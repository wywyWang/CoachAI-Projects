import pandas as pd

filename = '../../clip_info_19ASI_CS'
df = pd.read_excel(filename + '.xlsx')
hitarea = []
for i in range(len(df)):
	if df['hit_y'][i] <= 0 and df['hit_x'][i] <= 32:
		hitarea += ["D4"]
	if df['hit_y'][i] <= 74 and df['hit_y'][i] > 0 and df['hit_x'][i] <= 32:
		hitarea += ["D3"]
	if df['hit_y'][i] <= 307 and df['hit_y'][i] > 74 and df['hit_x'][i] <= 32:
		hitarea += ["D2"]
	if df['hit_y'][i] <= 468 and df['hit_y'][i] > 307 and df['hit_x'][i] <= 32:
		hitarea += ["D1"]
	if df['hit_y'][i] <= 629 and df['hit_y'][i] > 468 and df['hit_x'][i] <= 32:
		hitarea += ["E1"]
	if df['hit_y'][i] <= 871 and df['hit_y'][i] > 629 and df['hit_x'][i] <= 32:
		hitarea += ["E2"]
	if df['hit_y'][i] <= 935 and df['hit_y'][i] > 871 and df['hit_x'][i] <= 32:
		hitarea += ["E3"]
	if df['hit_y'][i] > 935 and df['hit_x'][i] <= 32:
		hitarea += ["E4"]
	if df['hit_y'][i] <= 0 and df['hit_x'][i] > 32 and df['hit_x'][i] <= 108:
		hitarea += ["B4"]
	if df['hit_y'][i] <= 74 and df['hit_y'][i] > 0 and df['hit_x'][i] > 32 and df['hit_x'][i] <= 108:
		hitarea += ["B3"]
	if df['hit_y'][i] <= 307 and df['hit_y'][i] > 74 and df['hit_x'][i] > 32 and df['hit_x'][i] <= 108:
		hitarea += ["B2"]
	if df['hit_y'][i] <= 468 and df['hit_y'][i] > 307 and df['hit_x'][i] > 32 and df['hit_x'][i] <= 108:
		hitarea += ["B1"]
	if df['hit_y'][i] <= 629 and df['hit_y'][i] > 468 and df['hit_x'][i] > 32 and df['hit_x'][i] <= 108:
		hitarea += ["C1"]
	if df['hit_y'][i] <= 871 and df['hit_y'][i] > 629 and df['hit_x'][i] > 32 and df['hit_x'][i] <= 108:
		hitarea += ["C2"]
	if df['hit_y'][i] <= 935 and df['hit_y'][i] > 871 and df['hit_x'][i] > 32 and df['hit_x'][i] <= 108:
		hitarea += ["C3"]
	if df['hit_y'][i] > 935 and df['hit_x'][i] > 32 and df['hit_x'][i] <= 108:
		hitarea += ["C4"]
	if df['hit_y'][i] <= 0 and df['hit_x'][i] > 108 and df['hit_x'][i] <= 316:
		hitarea += ["A4"]
	if df['hit_y'][i] <= 74 and df['hit_y'][i] > 0 and df['hit_x'][i] > 108 and df['hit_x'][i] <= 316:
		hitarea += ["A3"]
	if df['hit_y'][i] <= 307 and df['hit_y'][i] > 74 and df['hit_x'][i] > 108 and df['hit_x'][i] <= 316:
		hitarea += ["A2"]
	if df['hit_y'][i] <= 468 and df['hit_y'][i] > 307 and df['hit_x'][i] > 108 and df['hit_x'][i] <= 316:
		hitarea += ["A1"]
	if df['hit_y'][i] <= 629 and df['hit_y'][i] > 468 and df['hit_x'][i] > 108 and df['hit_x'][i] <= 316:
		hitarea += ["A1"]
	if df['hit_y'][i] <= 871 and df['hit_y'][i] > 629 and df['hit_x'][i] > 108 and df['hit_x'][i] <= 316:
		hitarea += ["A2"]
	if df['hit_y'][i] <= 935 and  df['hit_y'][i] > 871  and df['hit_x'][i] > 108 and df['hit_x'][i] <= 316:
		hitarea += ["A3"]
	if df['hit_y'][i] > 935 and df['hit_x'][i] > 108 and df['hit_x'][i] <= 108:
		hitarea += ["A4"]
	if df['hit_y'][i] <= 0 and df['hit_x'][i] > 316 and df['hit_x'][i] <= 392:
		hitarea += ["C4"]
	if df['hit_y'][i] <= 74 and df['hit_y'][i] > 0 and df['hit_x'][i] > 316 and df['hit_x'][i] <= 392:
		hitarea += ["C3"]
	if df['hit_y'][i] <= 307 and df['hit_y'][i] > 74 and df['hit_x'][i] > 316 and df['hit_x'][i] <= 392:
		hitarea += ["C2"]
	if df['hit_y'][i] <= 468 and df['hit_y'][i] > 307 and df['hit_x'][i] > 316 and df['hit_x'][i] <= 392:
		hitarea += ["C1"]
	if df['hit_y'][i] <= 629 and df['hit_y'][i] > 468 and df['hit_x'][i] > 316 and df['hit_x'][i] <= 392:
		hitarea += ["B1"]
	if df['hit_y'][i] <= 871 and df['hit_y'][i] > 629 and df['hit_x'][i] > 316 and df['hit_x'][i] <= 392:
		hitarea += ["B2"]
	if df['hit_y'][i] <= 935 and df['hit_y'][i] > 871 and df['hit_x'][i] > 316 and df['hit_x'][i] <= 392:
		hitarea += ["B3"]
	if df['hit_y'][i] > 935 and df['hit_x'][i] > 316 and df['hit_x'][i] <= 392:
		hitarea += ["B4"]
	if df['hit_y'][i] <= 0 and df['hit_x'][i] > 392:
		hitarea += ["E4"]
	if df['hit_y'][i] <= 74 and df['hit_y'][i] > 0 and df['hit_x'][i] > 392:
		hitarea += ["E3"]
	if df['hit_y'][i] <= 307 and df['hit_y'][i] > 74 and df['hit_x'][i] > 392:
		hitarea += ["E2"]
	if df['hit_y'][i] <= 468 and df['hit_y'][i] > 307 and df['hit_x'][i] > 392:
		hitarea += ["E1"]
	if df['hit_y'][i] <= 629 and df['hit_y'][i] > 468 and df['hit_x'][i] > 392:
		hitarea += ["D1"]
	if df['hit_y'][i] <= 871 and df['hit_y'][i] > 629 and df['hit_x'][i] > 392:
		hitarea += ["D2"]
	if df['hit_y'][i] <= 935 and df['hit_y'][i] > 871 and df['hit_x'][i] > 392:
		hitarea += ["D3"]
	if df['hit_y'][i] > 935 and df['hit_x'][i] > 392:
		hitarea += ["D4"]
df['hit_area']=hitarea
export_excel = df.to_excel (r'../../clip_info_19ASI_CS.xlsx', index = None, header=True) #Don't forget to add '.xlsx' at the end of the path