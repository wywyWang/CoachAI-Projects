import sys
import getopt
import numpy as np
import os
from glob import glob
import piexif
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import *
from keras.layers import *
from TrackNet import TrackNet
import keras.backend as K
from keras import optimizers
import tensorflow as tf
import cv2
from os.path import isfile, join
BATCH_SIZE=1
HEIGHT=360
WIDTH=640
def custom_loss(y_true, y_pred):
    weight=y_true*0.95+0.05
    return K.sum(K.square(y_true-y_pred)*weight)

def genHeatMap(osizex, osizey, x, y, sigma, facx, facy, radius, mag):
    return gen2DGaussian(osizex, osizey, x+1, y+1, sigma, facx, facy, radius)*mag

def gen2DGaussian(outSizeX, outSizeY, centerX, centerY, sigma, facX, facY, radius):

    if centerX <= 0 or centerY <= 0: return np.zeros((outSizeY, outSizeX))
    
    mux, muy = centerX, centerY
    radius = np.deg2rad(180) - radius

    sigmax = facX * sigma
    sigmay = facY * sigma

    # width, height
    x, y = np.meshgrid(np.linspace(1, outSizeX, outSizeX), np.linspace(1, outSizeY, outSizeY))
    
    with np.errstate(divide='ignore', invalid='ignore'):
        a = ((np.cos(radius)**2) / (2*sigmax**2)) + ((np.sin(radius)**2) / (2*sigmay**2))
        b = -((np.sin(2*radius)) / (4*sigmax**2)) + ((np.sin(2*radius)) / (4*sigmay**2))
        c = ((np.sin(radius)**2) / (2*sigmax**2)) + ((np.cos(radius)**2) / (2*sigmay**2))

        gau2d = np.exp(-(a*(x - mux)**2 + 2*b*(x - mux)*(y - muy) + c*(y - muy)**2))
    
    gau2d[np.isnan(gau2d)] = 0
    gau2d[gau2d<10**-18] = 0

    return gau2d

def adjustPredHeatMap(pred, sigma, mag):
	threshold_ratio = 0.5
	M = np.amax(pred)
	if M < (mag * threshold_ratio):
		return np.zeros(pred.shape, dtype='float32') 
	pos_pred = np.unravel_index(np.argmax(pred, axis=None), pred.shape)
	return genHeatMap(pred.shape[1], pred.shape[0], pos_pred[1], pos_pred[0], sigma, 1, 1, np.deg2rad(180), mag)
 
#predBatch: batch*360*640
def adjustPredHeatMaps(predBatch, sigma, mag):
	a = []
	for i in range(predBatch.shape[0]):
		a.append(adjustPredHeatMap(predBatch[i], sigma, mag))
	a = np.asarray(a)
	return a

#time: in milliseconds
def custom_time(time):
	remain = int(time / 1000)
	ms = (time / 1000) - remain
	s = remain % 60
	s += ms
	remain = int(remain / 60)
	m = remain % 60
	remain = int(remain / 60)
	h = remain
	#generate custom time string
	cts = ''
	if len(str(h)) >= 2:
		cts += str(h)
	else:
		for i in range(2 - len(str(h))):
			cts += '0'
		cts += str(h)
	
	cts += ':'

	if len(str(m)) >= 2:
		cts += str(m)
	else:
		for i in range(2 - len(str(m))):
			cts += '0'
		cts += str(m)

	cts += ':'

	if len(str(int(s))) == 1:
		cts += '0'
	cts += str(s)

	return cts

def run(TrackNet_input, TrackNet_output):
    load_weights = './preprocessing/Data/TrackNetModel/TrackNet_weight_0906.h5'
    print(TrackNet_input)
    sigma = 5
    mag = 1

    f = open(TrackNet_output, 'w')
    f.write('Frame,Visibility,X,Y,Time\n')

    model = load_model(load_weights, custom_objects={'custom_loss':custom_loss})

    cap = cv2.VideoCapture(TrackNet_input)

    success, image1 = cap.read()
    success, image2 = cap.read()
    success, image3 = cap.read()

    ratio = int(image1.shape[0] / 360)

    count = 3

    while success:
        unit = []
        #adjust BGR format (cv2) to RGB format (PIL)
        x1 = image1[...,::-1]
        x2 = image2[...,::-1]
        x3 = image3[...,::-1]
        #convert np arrays to PIL images
        x1 = array_to_img(x1)
        x2 = array_to_img(x2)
        x3 = array_to_img(x3)
        #resize the images
        x1 = x1.resize(size = (WIDTH, HEIGHT))
        x2 = x2.resize(size = (WIDTH, HEIGHT))
        x3 = x3.resize(size = (WIDTH, HEIGHT))
        #convert images to np arrays and adjust to channels first
        x1 = np.moveaxis(img_to_array(x1), -1, 0)		
        x2 = np.moveaxis(img_to_array(x2), -1, 0)		
        x3 = np.moveaxis(img_to_array(x3), -1, 0)
        #create data
        unit.append(x1[0])
        unit.append(x1[1])
        unit.append(x1[2])
        unit.append(x2[0])
        unit.append(x2[1])
        unit.append(x2[2])
        unit.append(x3[0])
        unit.append(x3[1])
        unit.append(x3[2])
        unit=np.asarray(unit)	
        unit = unit.reshape((1, 9, HEIGHT, WIDTH))
        unit = unit.astype('float32')
        unit /= 255
        y_raw = model.predict(unit, batch_size=BATCH_SIZE)
        y_pred = adjustPredHeatMaps(y_raw, sigma, mag)
        time = custom_time(cap.get(cv2.CAP_PROP_POS_MSEC))
        if np.amax(y_pred[0]) <= 0:
            f.write(str(count)+',0,0,0,'+time+'\n')
        else:	
            pos_pred = np.unravel_index(np.argmax(y_pred[0], axis=None), y_pred[0].shape)
            f.write(str(count)+',1,'+str(pos_pred[1]*ratio)+','+str(pos_pred[0]*ratio)+','+time+'\n')
        image1 = image2
        image2 = image3
        success, image3 = cap.read()
        count += 1

    f.close()
