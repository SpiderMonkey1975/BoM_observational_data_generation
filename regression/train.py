from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, History
from datetime import datetime
from random import shuffle

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import glob, sys, argparse

root_dir = '/home/ubuntu'
data_dir = '/data'

dirpath = root_dir + '/BoM_observational_data_generation/neural_network_architecture/'
#sys.path.insert(0, '/group/director2107/mcheeseman/BoM_observational_data_generation/neural_network_architecture/')
sys.path.insert(0, dirpath)
from basic_autoencoder import autoencoder, autoencoder_multigpu

image_dims = np.empty((3,),dtype=np.int)
image_dims[0] = 2050
image_dims[1] = 2450
image_dims[2] = 10 

##
## Look for any user specified commandline arguments
##

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--num_filter', type=int, default=16, help="set initial number of filters used in CNN layers for the neural networks")
parser.add_argument('-v', '--verbose', type=int, default=0, help="set to 1 if additional debugging info desired")
parser.add_argument('-l', '--learn_rate', type=float, default=0.001, help="set learn rate for the optimizer")
parser.add_argument('-t', '--stopping_tolerance', type=float, default=0.001, help="set tolerance limit for early stopping callback in training")
parser.add_argument('-z', '--num_files', type=int, default=100, help="set number of input files to be read in per block of training")
parser.add_argument('-g', '--num_gpus', type=int, default=1, help="set number of GPUS used in training")
parser.add_argument('-b', '--batch_size', type=int, default=10, help="set batch size used for training")
args = parser.parse_args()

##
## Form the neural network
##

if args.num_gpus>1:
   model = autoencoder_multigpu( image_dims, args.num_filter, args.num_gpus ) 
else:
   model = autoencoder( image_dims, args.num_filter ) 
   model.summary()

model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learn_rate), metrics=['mae'])

##
## Set up the training of the model
##

filename = "model_weights_" + str(args.num_filter) + "filters.h5"
checkpoint = ModelCheckpoint( filename, 
                              monitor='val_mean_absolute_error', 
                              save_best_only=True, 
                              mode='min' )

earlystop = EarlyStopping( min_delta=args.stopping_tolerance,
                           monitor='val_mean_absolute_error', 
                           patience=5,
                           mode='min' )

history = History()

if args.num_gpus>1:
   my_callbacks = [earlystop, history]
else: 
   my_callbacks = [checkpoint, earlystop, history]

##
## Get a list of input data files
##

input_file_list = []
cmd_str = data_dir + '/input*.nc'
for fn in glob.iglob(cmd_str, recursive=True):
    input_file_list.append( fn )

input_file_list = list(dict.fromkeys(input_file_list))
shuffle( input_file_list )

print(' ')
print('*******************************************')
print('  DATAFILE STATS')
print('*******************************************')
print(' ')
print('    %3d input files located ' % len(input_file_list))
print('    800 input files used for training')
print('    100 input files used for testing')
print(' ')



##-------------------------------------------------------------------------------------------------
##  TRAINING LOOP
##-------------------------------------------------------------------------------------------------

features = np.empty((args.num_files,image_dims[0],image_dims[1],image_dims[2]), dtype=np.float32)
labels = np.empty((args.num_files,image_dims[0],image_dims[1]), dtype=np.float32)

total_io_time = 0.0
total_training_time = 0.0

num_rounds = int( np.floor(800.0 / float(args.num_files)) )
num_rounds = 1

idx = 0
for nn in range( num_rounds ):

    # read in input data for current block of input files
    t1 = datetime.now()
    for n in range( args.num_files ):
        fid = nc.Dataset( input_file_list[idx+n] )
        x = np.array( fid['brightness'] )
        y = np.array( fid['precipitation'] )
        fid.close()

        features[ n,:,:,: ] = x[ np.newaxis,:,:,: ]
        labels[ n,:,: ] = y[ np.newaxis,:,: ]
    io_time = (datetime.now()-t1 ).total_seconds()

    # perform training on the read input data
    t1 = datetime.now()
    hist = model.fit( features, labels[ :,:,:,np.newaxis ], batch_size=args.batch_size, validation_split=0.2, epochs=250, callbacks=my_callbacks )
    training_time = (datetime.now()-t1 ).total_seconds()

    # update timing counts
    total_io_time = total_io_time + io_time
    total_training_time = total_training_time + training_time

    # update global file offset count
    idx = idx + args.num_files

print(' ')
print('*******************************************')
print('  TRAINING STATS')
print('*******************************************')
print(' ')
print("    training took %7.1f seconds" % total_training_time)
print("    I/O took %7.1f seconds (%4.1f percent of total runtime)" % (total_io_time,100.0*(total_io_time/(total_io_time+total_training_time))))
print(' ')



##-------------------------------------------------------------------------------------------------
##  INFERENCE 
##-------------------------------------------------------------------------------------------------

idx = 800

# read in input data for current block of input files
t1 = datetime.now()
for n in range( args.num_files ):
    fid = nc.Dataset( input_file_list[idx+n] )
    x = np.array( fid['brightness'] )
    y = np.array( fid['precipitation'] )
    fid.close()

    features[ n,:,:,: ] = x[ np.newaxis,:,:,: ]
    labels[ n,:,: ] = y[ np.newaxis,:,: ]
io_time = (datetime.now()-t1 ).total_seconds()

# perform inference on the read input data
t1 = datetime.now()
output = model.predict( features, batch_size=args.batch_size )
inference_time = (datetime.now()-t1 ).total_seconds()

print(' ')
print('*******************************************')
print('  INFERENCE STATS')
print('*******************************************')
print(' ')
print("    inference took %5.3f seconds" % inference_time)
print("    I/O took %5.3f seconds (%4.1f percent of total runtime)" % (io_time,100.0*(io_time/(io_time+inference_time))))



##-------------------------------------------------------------------------------------------------
##  COMPARISON STATISTICS 
##-------------------------------------------------------------------------------------------------

num_hits = np.sum( np.isclose( np.squeeze(output), labels, rtol=0.0, atol=0.0001) )
num_values = args.num_files * image_dims[0] * image_dims[1] 
accuracy = 100.0 * (float( num_hits ) / float(num_values))

print("    prediction accuracy was %4.1f" % accuracy)
print(' ')

