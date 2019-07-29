from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, ConvLSTM2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, History
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import argparse, sys

##
## Parse any commandline arguments given by the user
##

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epoch', type=int, default=10, help="set number of epoch")
parser.add_argument('-b', '--batch_size', type=int, default=5, help="set batch size to the GPU")
args = parser.parse_args()

##
## Read in the high-resolution radar precipitation data
##

radar_data = np.load( '../input/jan28-29_2019_precipitation.npy' ) 

##
## Consruct the convolutional-LTSM neural network
##

lat = np.int( radar_data.shape[1] / 8)
lon = np.int( radar_data.shape[2] / 8)
num_filters = 8
kernel_dim = 2

seq = Sequential()
seq.add(ConvLSTM2D(filters=num_filters, kernel_size=kernel_dim, input_shape=(None, lat, lon, 1), padding='same', return_sequences=True))
seq.add(BatchNormalization())

for n in range(3):
    num_filters = num_filters * 2
    seq.add(ConvLSTM2D(filters=num_filters, kernel_size=kernel_dim, padding='same', return_sequences=True))
    seq.add(BatchNormalization())

seq.add(Conv3D(filters=1, kernel_size=kernel_dim, activation='sigmoid', padding='same', data_format='channels_last'))

seq.compile(loss='binary_crossentropy', optimizer='adadelta')
#seq.compile(loss='binary_crossentropy', optimizer='adagrad')

##
## Convert radar data into a 5D input tensor [ SAMPLES, SLICES, LAT, LON, CHANNELS ]
## where:
##    SAMPLES -> # of hourly samples find in radar input
##    SLICES  -> # of 10 minute slices in eacg hour sample (should be 6)
##    LAT     -> # of points in latitude direction 
##    LON     -> # of points in longitude direction 
##    CHANNELS-> # of channels in input data (should be 1)
##

slices = 6
samples = np.int( radar_data.shape[0] / slices )

input_data = np.empty((samples, slices, lat, lon, 1), dtype=np.float)
    
cnt = 0
for i in range(samples):
    for t in range(slices):
        input_data[ i,t,:,:,0 ] = np.squeeze( radar_data[ cnt,:lat,:lon,0 ] )
        cnt = cnt + 1

where_are_nans = np.isnan( input_data )
input_data[ where_are_nans ] = 0.0

##
## Perform model training
##

earlystop = EarlyStopping( min_delta=0.1,
                           patience=5,
                           mode='min' )
history = History()
my_callbacks = [earlystop, history]

t1 = datetime.now()
hist = seq.fit( input_data[:40], 
                input_data[1:41],
                batch_size=args.batch_size,
                epochs=500,
                verbose=2,
                validation_split=.05,
                callbacks=my_callbacks )
training_time = (datetime.now()-t1 ).total_seconds()



sys.exit()
# Testing the network on one movie
# feed it with the first 7 positions and then
# predict the new positions
which = 10
track = movies[which][:7, ::, ::, ::]

for j in range(16):
    new_pos = seq.predict(track[np.newaxis, ::, ::, ::, ::])
    new = new_pos[::, -1, ::, ::, ::]
    track = np.concatenate((track, new), axis=0)

# And then compare the predictions
# to the ground truth
track2 = movies[which][::, ::, ::, ::]
for i in range(15):
    fig = plt.figure(figsize=(10, 5))

    ax = fig.add_subplot(121)

    if i >= 7:
        ax.text(1, 3, 'Predictions !', fontsize=20, color='w')
    else:
        ax.text(1, 3, 'Initial trajectory', fontsize=20)

    toplot = track[i, ::, ::, 0]

    plt.imshow(toplot)
    ax = fig.add_subplot(122)
    plt.text(1, 3, 'Ground truth', fontsize=20)

    toplot = track2[i, ::, ::, 0]
    if i >= 2:
        toplot = shifted_movies[which][i - 1, ::, ::, 0]

    plt.imshow(toplot)
    plt.savefig('%i_animate.png' % (i + 1))
