from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, History
from datetime import datetime
from random import shuffle

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import glob, sys, argparse

sys.path.insert(0, '/home/ubuntu/BoM_observational_data_generation/neural_network_architecture/')
from basic_autoencoder import autoencoder
from unet import unet

sys.path.insert(0, '/home/ubuntu/BoM_observational_data_generation/plotting_routines')
from plotting_routines import plot_fc_model_errors 

##
## Look for any user specified commandline arguments
##

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--num_filter', type=int, default=16, help="set initial number of filters used in CNN layers for the neural networks")
parser.add_argument('-v', '--verbose', type=int, default=0, help="set to 1 if additional debugging info desired")
parser.add_argument('-n', '--neural_net', type=str, default='basic_autoencoder', help="set neural network design. Valid values are basic_autoencoder and unet")
parser.add_argument('-l', '--learn_rate', type=float, default=0.001, help="set learn rate for the optimizer")
parser.add_argument('-z', '--num_files', type=int, default=100, help="set number of input files to be read in per block of training")
parser.add_argument('-b', '--batch_size', type=int, default=10, help="set batch size used for training")
args = parser.parse_args()

if args.neural_net!='basic_autoencoder' and args.neural_net!='unet' and args.neural_net!='tiramisu':
   args.neural_net = 'basic_autoencoder'

##
## Form the neural network
##

image_dims = np.empty((3,),dtype=np.int)
image_dims[0] = 2050
image_dims[1] = 2450
image_dims[2] = 10 

if args.neural_net == 'basic_autoencoder':
    model = autoencoder( image_dims, args.num_filter ) 
else:
    model = unet( image_dims, args.num_filter, 1 )

model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learn_rate), metrics=['mae'])
#model.summary()

##
## Set up the training of the model
##

filename = "model_weights_" + args.neural_net + "_" + str(args.num_filter) + "filters.h5"
checkpoint = ModelCheckpoint( filename, 
                                 monitor='val_mean_absolute_error', 
                                 save_best_only=True, 
                                 mode='min' )

earlystop = EarlyStopping( min_delta=0.001,
                           monitor='val_mean_absolute_error', 
                           patience=5,
                           mode='min' )

history = History()

my_callbacks = [checkpoint, earlystop, history]

##
## Get a list of input data files
##

input_file_list = []
cmd_str = '/data/input*.nc'
for fn in glob.iglob(cmd_str, recursive=True):
    input_file_list.append( fn )

input_file_list = list(dict.fromkeys(input_file_list))
shuffle( input_file_list )
print('# of input files located: ', len(input_file_list))

##
## Read in input data 
##

features = np.empty((args.num_files,image_dims[0],image_dims[1],image_dims[2]), dtype=np.float32)
labels = np.empty((args.num_files,image_dims[0],image_dims[1]), dtype=np.float32)

total_io_time = 0.0
total_training_time = 0.0

idx = 0
for nn in range( 8 ):
    t1 = datetime.now()
    for n in range( idx,idx+args.num_files ):
        fid = nc.Dataset( input_file_list[n] )
        x = np.array( fid['brightness'] )
        y = np.array( fid['precipitation'] )
        fid.close()

        features[ n,:,:,: ] = x[ np.newaxis,:,:,: ]
        labels[ n,:,: ] = y[ np.newaxis,:,: ]
    io_time = (datetime.now()-t1 ).total_seconds()

    t1 = datetime.now()
    hist = model.fit( features, labels[ :,:,:,np.newaxis ], batch_size=args.batch_size, validation_split=0.2, epochs=250, callbacks=my_callbacks )
    training_time = (datetime.now()-t1 ).total_seconds()

    total_io_time = total_io_time + io_time
    total_training_time = total_training_time + training_time

    idx = idx + args.num_files

print("   training took %7.1f seconds" % total_training_time)
print("   I/O took %7.1f seconds (%4.1f percent of total runtime)" % (total_io_time,100.0*(total_io_time/(total_io_time+total_training_time))))









































sys.exit()





satellite_data = np.empty((10,image_dims[0],image_dims[1],image_dims[2],), dtype=np.float32)
radar_data = np.empty_like( satellite_data ) 

print(satellite_data.shape, radar_data.shape)


t1 = datetime.now()
for n in range( 10 ):
    satellite_data[ n,:,:,: ], radar_data[ n,:,:,0 ] = read_input_file( input_file_list[n] )
io_time = (datetime.now()-t1 ).total_seconds()
print('input data read')

for n in range( 9 ):
    radar_data[ :,:,:,n+1 ] = radar_data[ :,:,:,0 ]


bull = np.ones((10,2050,2450,10,), dtype=np.float32)
shit = np.zeros((10,2050,2450,), dtype=np.float32)


t1 = datetime.now()
hist = model.fit( satellite_data, radar_data, 
                  batch_size=args.batch_size, 
                  epochs=2, 
                  verbose=2, 
                  validation_split=.2 ) 
#                  callbacks=my_callbacks ) 
train_time = (datetime.now()-t1 ).total_seconds()

print("   training took %7.1f seconds" % train_time)
#print("   I/O took %7.1f seconds (%4.1f percent of total runtime)" % (io_time,100.0*(io_time/(io_time+train_time))))

