from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report


from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K


import pandas as pd
import numpy as np
import csv
from collections import Counter
import matplotlib.pyplot as plt
import argparse
import os





numFrame = 18241

filename="../Data/TrainTest/clip_info.csv"
record=[]
with open(filename, 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    #next(spamreader, None)  # skip the headers
    for row in spamreader:
        if row[4] and int(row[4])>=numFrame:
            break
        if row[4] and row[7]!='未擊球' and row[7]!='未過網': 
            record+=[int(row[4])]
            
df = pd.read_csv('../Data/TrainTest/Badminton_label.csv')
df = df[0:numFrame]

marked = [0 for _ in range(numFrame)]
cnt=0
for i in range(numFrame):
    if i+1 in record:
        if df['Visibility'][i]==0:
            marked[i-1]=1
        else:
            marked[i]=1
        cnt+=1
        
df['hitpoint']=marked
df = df[df.Visibility == 1].reset_index(drop=True)

# Absolute position
X = df['X']
Y = df['Y']

# Vector X and Y
vecX = [X[i+1]-X[i] for i in range(len(X)-1)]
vecY = [Y[i+1]-Y[i] for i in range(len(Y)-1)]
vecX.append(0)
vecY.append(0)
dis = [vecX[i]**2+vecY[i]**2 for i in range(len(Y)-1)]
dis.append(0)
df['vecX'] = vecX
df['vecY'] = vecY
df['dis'] = dis
X = df['vecX']
Y = df['vecY']
dX = [X[i+1]-X[i] for i in range(len(X)-1)]
dY = [Y[i+1]-Y[i] for i in range(len(Y)-1)]
dX.append(0)
dY.append(0)
df['dX'] = dX
df['dY'] = dY
muldxdy=[]
for i in range(len(Y)):
    if dX[i]*dY[i]>0:
        muldxdy +=[1]
    else:
        muldxdy +=[0]
df['muldxdy']=muldxdy

trainNum = int(len(df[df['hitpoint']==1])*0.7) #328
tmp = df[df['hitpoint']==1].iloc[trainNum]
trainidx = df['Frame'][df['Frame']==tmp['Frame']].index[0]
train = df[0:trainidx+1]
test = df[trainidx+1:]

#solution 2
# new = []
# for i in range(len(train)-3):
#     if train['hitpoint'].iloc[i]==1:
#         new +=[train.iloc[i-3,:]]
#         new +=[train.iloc[i-2,:]]
#         new +=[train.iloc[i-1,:]]
#         new +=[train.iloc[i,:]]
#         new +=[train.iloc[i+1,:]]
#         new +=[train.iloc[i+2,:]]
#         new +=[train.iloc[i+3,:]]
# train = pd.DataFrame(new).reset_index(drop=True)

train_feature = train[['X', 'Y','vecX','vecY','dX','dY','muldxdy','dis']].copy()
train_target  = train[['hitpoint']].copy()
test_feature  = test[['X', 'Y','vecX','vecY','dX','dY','muldxdy','dis']].copy()
test_target  = test[['hitpoint']].copy()
print(len(df[df['hitpoint']==1])-trainNum)
print(len(test_target))
one=0
zero=0
for i in range(len(train_target)):
    if train_target.values[i]==1:
        one+=1
    else:
        zero+=1
        
print(train_feature[0])





# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


# network parameters
original_dim = 8
input_shape = (original_dim, )
intermediate_dim = 512
batch_size = 128
latent_dim = 2
epochs = 50

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = Dense(intermediate_dim, activation='relu')(inputs)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = Dense(original_dim, activation='sigmoid')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use mse loss instead of binary cross entropy (default)"
    parser.add_argument("-m","--mse",help=help_, action='store_true')
    args = parser.parse_args()
    models = (encoder, decoder)
    
    # VAE loss = mse_loss or xent_loss + kl_loss
    if args.mse:
        reconstruction_loss = mse(inputs, outputs)
    else:
        reconstruction_loss = binary_crossentropy(inputs,outputs)

    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    vae.summary()
    
    vae.fit(train_feature.values,epochs=epochs,batch_size=batch_size,validation_data=(test_feature.values, None))
    vae.save_weights('vae_mlp_mnist.h5')