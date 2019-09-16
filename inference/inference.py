from tensorflow.keras.optimizers import Adam
from datetime import datetime

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import glob, sys, argparse

sys.path.insert(0, '/home/ubuntu/BoM_observational_data_generation/neural_network_architecture/')
from fully_connected import simple_net

sys.path.insert(0, '/home/ubuntu/BoM_observational_data_generation/plotting_routines')
from plotting_routines import plot_images 

##
## Set some constants for the run
##

num_channels = 10
num_datapoints = 724016
nx = 2050
ny = 2450
num_test_points = 100 

##
## Look for any user specified commandline arguments
##

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', type=int, default=10, help="set the batch size used in training")
parser.add_argument('-n', '--num_nodes', type=int, default=8, help="set the number of nodes in the first dense layer in the network.")
parser.add_argument('-l', '--num_layers', type=int, default=2, help="set the number of dense layers in the neural network.")
args = parser.parse_args()

##
## Setup the neural network and model
##

input_dims = np.empty((2,),dtype=np.int)
input_dims[0] = num_datapoints 
input_dims[1] = num_channels

model = simple_net( input_dims, args.num_layers, args.num_nodes )
model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001), metrics=['mae'])
model.load_weights( '/home/ubuntu/BoM_observational_data_generation/regression/model_weights_fully_connected.h5' )

##
## Get a list of input data files
##

input_file_list = []
cmd_str = '/data/input_*.nc'
for fn in glob.iglob(cmd_str, recursive=True):
    input_file_list.append( fn )

input_file_list = list(dict.fromkeys(input_file_list))
test_input_file_list = input_file_list[ -num_test_points: ]

##
## Read in the input data 
##

def read_input_file( filename ):
    fid = nc.Dataset( filename, 'r' )
    x = np.array( fid['coalesced_brightness'] )
    y = np.array( fid['coalesced_precipitation'] )

    x = x[ np.newaxis,:,: ]
    y = y[ np.newaxis,: ]

    fid.close()
    return x, y

satellite_data = np.empty((num_test_points,input_dims[0],10))
true_precip = np.empty((num_test_points,input_dims[0]))

t1 = datetime.now()
for n in range(num_test_points):
    satellite_data[ n,:,: ], true_precip[ n,: ] = read_input_file( test_input_file_list[n] )
io_time = (datetime.now()-t1 ).total_seconds()

##
## Perform inference
##

t1 = datetime.now()
predicted_precip = model.predict( satellite_data, verbose=0 )
inference_time = (datetime.now()-t1 ).total_seconds()

print("   inference took %5.4f seconds" % inference_time)
print("   I/O took %5.4f seconds (%4.1f percent of total runtime)" % (io_time, 100.0*(io_time/(io_time+inference_time))))

##
## Create some comparision statistics
##

accuracies = []
for n in range(num_test_points):
    num_hits = np.sum(np.isclose(predicted_precip[n,:],true_precip[n,:],0.001,0.00001)) 
    accuracies.append( 100.0*float(num_hits)/float(num_datapoints) )

accuracy = np.asarray( accuracies, dtype=np.float32 )

print(' ')
print("Prediction accuracy for full Australian continent")
print(' minimum value was %4.1f' % np.amin(accuracy))
print(' maximum value was %4.1f' % np.amax(accuracy))
print(' average value was %4.1f' % np.mean(accuracy))
print(' standard deviation was %4.1f' % np.std(accuracy))

accuracies = []
for n in range(num_test_points):
    num_hits = np.sum(np.isclose(np.zeros((1,num_datapoints,),dtype=np.float32),true_precip[n,:],0.001,0.00001))
    accuracies.append( 100.0*float(num_hits)/float(num_datapoints) )

accuracy = np.asarray( accuracies, dtype=np.float32 )

print(' ')
print("Prediction accuracy for full Australian continent")
print(' minimum value was %4.1f' % np.amin(accuracy))
print(' maximum value was %4.1f' % np.amax(accuracy))
print(' average value was %4.1f' % np.mean(accuracy))
print(' standard deviation was %4.1f' % np.std(accuracy))

