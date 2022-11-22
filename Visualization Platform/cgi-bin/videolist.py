import os
import pandas as pd
import json

def export_json(filepath,data):
    with open(filepath, 'w') as outfile:
        json.dump(data, outfile, indent = 4, separators=(',', ':'))

def savelist2json():
    filepath = './preprocessing/Data/Output/videolist.json'

    video_folder = './uploadvideo/'
    uploadvideo_list = os.listdir(video_folder)
    uploadvideo_list = [item.split('.')[0] for item in uploadvideo_list]
    uploadvideo_dict = {'previous_tracknet': [value for value in uploadvideo_list]}

    tracknet_folder = './preprocessing/Data/TrainTest/'
    tracknet_result = os.listdir(tracknet_folder)
    tracknet_list = [item for item in tracknet_result if 'Badminton_label' in item]
    tracknet_list = [item.split('.')[0].split('Badminton_label_')[-1] for item in tracknet_list]
    tracknet_dict = {'previous_segmentation': [value for value in tracknet_list]}

    segmentation_folder = './preprocessing/Data/AccuracyResult/'
    segmentation_result = os.listdir(segmentation_folder)
    segmentation_list = [item for item in segmentation_result if 'record_segmentation_' in item]
    segmentation_list = [item.split('.')[0].split('record_segmentation_')[-1] for item in segmentation_list]
    segmentation_dict = {'previous_predict_balltype': [value for value in segmentation_list]}

    uploadvideo_dict.update(tracknet_dict)
    uploadvideo_dict.update(segmentation_dict)

    export_json(filepath,uploadvideo_dict)
    
if __name__ == '__main__':
    savelist2json()