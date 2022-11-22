import glob
import csv
import cv2
import numpy
import numpy as np
import argparse
import os

#parse parameters
parser = argparse.ArgumentParser()
parser.add_argument("--heatmap_path", type = str  )

args = parser.parse_args()
heatmap_path = args.heatmap_path
total = os.listdir(heatmap_path)

#sort file name based on number
def getnumber(x):
    return int(x.split('.')[0])
total.sort(key = getnumber)

i = 0
#setup parameter for houghcircle
mindist=10
param2=4
minradius=1
maxradius=12
recircle = []
frame = []

#Get coordinate from heatmap by using houghcircle
for filename in total:
    if i % 500 == 0:
        print('>> Status: i =', i)
        
    pic_name = heatmap_path + filename
    heatmap = cv2.imread(pic_name, 0)
    if heatmap is None:
        print('>> Loading heatmap fails: %s' %(pic_name))
        continue
        
     # Heatmap is converted into a binary image by threshold method
    ret,heatmap = cv2.threshold(heatmap,127,255,cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT,dp=1,minDist=mindist,param2=param2,minRadius=minradius,maxRadius=maxradius)
    recircle+=[circles]
    frame+=[filename.split('.')[0]]
        
    i+=1


# store coordinate and radius into csv file
visino = 0
visiyes = 0
predict_name = heatmap_path.split('/')[-2]
#### You may need to change the path ####
record_ballsize_file='/home/ino/Projects/ai-badminton/Data/AccuracyResult/record_circle_ballsize_'+predict_name + '.csv'
with open(record_ballsize_file,'w',encoding='utf-8') as f:
    c=csv.writer(f,lineterminator='\n')
    f.write('Frame,Visibility,X,Y,Radius\n')
    for i in range(len(frame)):
        tmp=[]
        tmp.append(frame[i].split('/')[-1])
        if recircle[i] is None :
            visino +=1
            tmp.append(0)
            tmp.append(0)
            tmp.append(0)
            tmp.append(0)
        else:
            visiyes +=1
            tmp.append(1)
            ball = recircle[i][0][0]
            tmp.append(ball[0])
            tmp.append(ball[1])
            tmp.append(ball[2])
        c.writerow(tmp)
        
print("============= Finish Storing ============= ")
print("Not found count : ",visino)
print("Yes found count : ",visiyes)