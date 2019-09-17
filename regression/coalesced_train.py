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
sys.path.insert(0, dirpath)
from fully_connected import create_simple_net

##
## Set some useful run constants
##

image_dims = np.empty((2,),dtype=np.int)
image_dims[0] = 724016 
image_dims[1] = 10 

num_test_images = 100

##
## Look for any user specified commandline arguments
##

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', type=int, default=50, help="set the batch size used in training")
parser.add_argument('-g', '--num_gpus', type=int, default=1, help="set the number of GPUS to be used in training")
parser.add_argument('-t', '--stopping_tolerance', type=float, default=0.001, help="set the tolerance used for the early stopping callback")
parser.add_argument('-n', '--num_nodes', type=int, default=16, help="set the number of nodes in the first dense layer in the network.")
parser.add_argument('-l', '--num_layers', type=int, default=3, help="set the number of dense layers in the neural network.")
args = parser.parse_args()

##
## Form the neural network
##

model = create_simple_net( image_dims, args.num_gpus, args.num_layers, args.num_nodes )
model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001), metrics=['mae'])

##
## Set up the training of the model
##

earlystop = EarlyStopping( min_delta=args.stopping_tolerance,
                           monitor='val_mean_absolute_error', 
                           patience=5,
                           mode='min' )
history = History()

my_callbacks = [earlystop, history]
if args.num_gpus == 1:
   filename = "fc_model_weights_" + str(args.num_layers) + "layers_" + str(args.num_nodes) + "nodes.h5"
   checkpoint = ModelCheckpoint( filename, 
                                 monitor='val_mean_absolute_error', 
                                 save_best_only=True, 
                                 mode='min' )
   my_callbacks.append( checkpoint )

##
## Get a list of input data files
##

input_file_list = []
cmd_str = data_dir + '/input*.nc'
for fn in glob.iglob(cmd_str, recursive=True):
    input_file_list.append( fn )

input_file_list = list(dict.fromkeys(input_file_list))
shuffle( input_file_list )

num_training_images = len(input_file_list) - num_test_images

print(' ')
print('*******************************************')
print('  DATAFILE STATS')
print('*******************************************')
print(' ')
print('    %3d input files located ' % len(input_file_list) )
print('    %3d input files used for training' % num_training_images )
print('    %3d input files used for testing' % num_test_images )
print(' ')



##-------------------------------------------------------------------------------------------------
##  TRAINING LOOP
##-------------------------------------------------------------------------------------------------

def read_input_file( filename ):
    fid = nc.Dataset( filename )
    x = np.array( fid['coalesced_brightness'] )
    y = np.array( fid['coalesced_precipitation'] )
    fid.close()
    return x[ np.newaxis,:,: ], y[ np.newaxis,: ]


features = np.empty((num_training_images,image_dims[0],image_dims[1]), dtype=np.float32) 
labels = np.empty((num_training_images,image_dims[0]), dtype=np.float32)

# read in input data for current block of input files
t1 = datetime.now()
for n in range( num_training_images ):
    features[ n,:,: ],labels[ n,: ] = read_input_file( input_file_list[n] )
io_time = (datetime.now()-t1 ).total_seconds()

# perform training on the read input data
t1 = datetime.now()
hist = model.fit( features, labels, batch_size=args.batch_size, validation_split=0.2, epochs=250, callbacks=my_callbacks )
training_time = (datetime.now()-t1 ).total_seconds()

print(' ')
print('*******************************************')
print('  TRAINING STATS')
print('*******************************************')
print(' ')
print("    training took %5.3f seconds" % training_time)
print("    I/O took %5.3f seconds" % io_time)
print(' ')



##-------------------------------------------------------------------------------------------------
##  INFERENCE 
##-------------------------------------------------------------------------------------------------

features = np.empty((num_test_images,image_dims[0],image_dims[1]), dtype=np.float32) 
labels = np.empty((num_test_images,image_dims[0]), dtype=np.float32)

# read in input data for current block of input files
t1 = datetime.now()
for n in range( num_test_images ):
    features[ n,:,: ],labels[ n,: ] = read_input_file( input_file_list[num_training_images+n] )
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
print("    I/O took %5.3f seconds" % io_time)



##-------------------------------------------------------------------------------------------------
##  COMPARISON STATISTICS 
##-------------------------------------------------------------------------------------------------

num_hits = np.sum( np.isclose( np.squeeze(output), labels, rtol=0.0, atol=0.0001) )
num_values = num_test_images * image_dims[0] 
accuracy = 100.0 * (float( num_hits ) / float(num_values))

print("    prediction accuracy was %4.1f" % accuracy)
print(' ')

