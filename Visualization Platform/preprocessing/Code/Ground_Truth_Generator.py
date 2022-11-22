import csv
import numpy
from PIL import Image
import os 

size = 20
# Create gussian heatmap 
def gaussian_kernel(variance):
    x, y = numpy.mgrid[-size:size+1, -size:size+1]
    g = numpy.exp(-(x**2+y**2)/float(2*variance))
    return g 

# Make the Gaussian by calling the function
variance = 10
gaussian_kernel_array = gaussian_kernel(variance)
# Rescale the value to 0-255
gaussian_kernel_array =  gaussian_kernel_array * 255//gaussian_kernel_array[len(gaussian_kernel_array)//2][len(gaussian_kernel_array)//2]
# Change type as integer
gaussian_kernel_array = gaussian_kernel_array.astype(int)

# Directory
output_pics_path = "/home/ino/Projects/ai-badminton/Data/VideoToFrame/ground_truth_new/"
label_path = "/home/ino/Projects/ai-badminton/Data/TrainTest/Badminton_label.csv"

# with open("ground.csv") as csvfile:
#     readCSV = csv.reader(csvfile, delimiter=',')
#     frames = []
#     for row in readCSV:
#         frames += [row[0]]

# Create path if necessary
if not os.path.exists(output_pics_path ):
    os.makedirs(output_pics_path)

# Read csv file
with open(label_path, 'r') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',', quotechar='|')
    # Skip the headers
    next(readCSV, None)
    for row in readCSV:
        visibility = int(float(row[1]))
        FileName = row[0]
#         if FileName in frames:
            # If visibility == 0, the heatmap is a black image
        if visibility == 0:
            heatmap = Image.new("RGB", (1920, 1080))
            pix = heatmap.load()
            for i in range(1920):
                for j in range(1080):
                        pix[i,j] = (0,0,0)
        else:
            x = int(float(row[2]))
            y = int(float(row[3]))

            # Create a black image
            heatmap = Image.new("RGB", (1920,1080))
            pix = heatmap.load()
            for i in range(1920):
                for j in range(1080):
                        pix[i,j] = (0,0,0)

            # Copy the heatmap on it
            for i in range(-size,size+1):
                for j in range(-size,size+1):
                        if x+i<1920 and x+i>=0 and y+j<1080 and y+j>=0 :
                            temp = gaussian_kernel_array[i+size][j+size]
                            if temp > 0:
                                pix[x+i,y+j] = (temp,temp,temp)
        # Save image
        heatmap.save(output_pics_path + "/" + FileName.split('.')[-1] + ".png", "PNG")
        print('File %s.png is successfully saved.' %(FileName))
