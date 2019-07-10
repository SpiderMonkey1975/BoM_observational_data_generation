##
## train.py
##
## Python script that controls the training of a simple autoencoder neural network

from tensorflow.keras.optimizers import Adam
from datetime import datetime
from tqdm import tqdm
from plotting_routines import plot_images

import numpy as np
import argparse

import neural_nets,sys
from neural_nets import autoencoder

import matplotlib.pyplot as plt

##
## Look for any user specified commandline arguments
##

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epoch', type=int, default=10, help="set number of epoch")
parser.add_argument('-b', '--batch_size', type=int, default=32, help="set batch size to the GPU")
parser.add_argument('-o', '--output_frequency', type=int, default=10, help="set output frequency for plots and model weights")
args = parser.parse_args()

##
## Read image data and construct feature and label datasets 
##

image_data = np.load( "../input/jan28-29_2019_precipitation.npy" )
image_data = image_data[ :,:1968,:,: ]

##
## Form the neural network
##

num_filters = 16 
num_layers = 2
model = neural_nets.autoencoder( num_filters, num_layers, image_data.shape[1], image_data.shape[2] )

model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.00001), metrics=['mae'])

##
## Perform model training
##

def model_checkpoint( model, real_obs, num_epoch ):
    x = real_obs[5,:,:] 
    predicted_obs = np.squeeze( model.predict( x[ np.newaxis,:,:,np.newaxis ] ) )

    plt.figure(figsize=(5,2))

    plt.subplot(1, 2, 1)
    plt.imshow( real_obs[6,:,:] )
    plt.axis('off')
    plt.text( 0, 0, 'Observed Rainfall', fontsize=14 )

    plt.subplot(1, 2, 2)
    plt.imshow( predicted_obs )
    plt.axis('off')

    plt.colorbar()
    plt.tight_layout()
    filename = 'rainfall_' + str(num_epoch) + 'epoch.png'
    plt.savefig(filename)
    plt.close('all')

    model.save_weights( 'model_weights.h5' )


batch_count = int(image_data.shape[0] / args.batch_size)
if batch_count*args.batch_size > image_data.shape[0]:
   batch_count = batch_count - 1
batch_count = batch_count - 1

t1 = datetime.now()
for e in tqdm(range(1,args.epoch+1 )):
    for n in range(batch_count):
        i1 = n*args.batch_size
        i2 = i1 + args.batch_size

        x = image_data[ i1:i2,:,:,0 ]
        x = x[ :,:,:,np.newaxis ]
        y = image_data[ i1+1:i2+1,:,:,0 ]
        y = y[ :,:,:,np.newaxis ]

        model.train_on_batch( x, y )

    if e % args.output_frequency == 0:
        model_checkpoint( model, image_data[ :7,:,:,0 ], e )

training_time = (datetime.now()-t1 ).total_seconds()
model_checkpoint( model, image_data[ :7,:,:,0 ], e )

print(" ")
print(" ")
print("=====================================================================================")
print("                          Rainfall Regression Network")
print("=====================================================================================")
print(" ")
print("   basic autoencoder used with %3d initial filters used in %1d layers" % (num_filters,num_layers))
print("   batch size of %3d images used" % args.batch_size)
print(" ")
print("   TRAINING OUTPUT")
print("       training lasted for %7.1f seconds" % training_time)
print(" ")

