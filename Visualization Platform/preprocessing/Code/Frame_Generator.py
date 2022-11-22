import cv2
import csv
import os

# Directory
videoName = '/home/ino/Projects/Yao_tmp/TMP/Data/PredictVideo/video.mp4'
#inputPath = '/home/hychih/notebook/Badminton/Data/VideoToFrame/video_frames_/'
#inputFile = 'Label.csv'
outputPath = '/home/ino/Projects/Yao_tmp/TMP/Data/PredictVideo/lalala/'

# Create path if necessary
if not os.path.exists(outputPath):
    os.makedirs(outputPath)

# Read cvs (The labeled coordinates)
# with open(inputPath + inputFile) as csvfile:
#    readCSV = csv.reader(csvfile, delimiter=',')
#     frames = []
#     next(readCSV, None)
#     for row in readCSV:
#         frames += [int(row[0])]
        
# Read video (Absolute path is required for macOS)
cap = cv2.VideoCapture(videoName)
success, count = True, 1
success, image = cap.read()
while success:
    #if count in frames:
    cv2.imwrite(outputPath + '%d.jpg' %(count), image)
    count += 1
    success, image = cap.read()