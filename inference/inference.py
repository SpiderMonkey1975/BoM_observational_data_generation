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
num_test_points = 5

##
## Look for any user specified commandline arguments
##

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', type=int, default=10, help="set the batch size used in training")
parser.add_argument('-w', '--weights_file', type=str, default='model_weights_fully_connected.h5', help="set filename containing model weight values")
args = parser.parse_args()

##
## Setup the neural network and model
##

input_dims = np.empty((2,),dtype=np.int)
input_dims[0] = num_datapoints 
input_dims[1] = num_channels

model = simple_net( input_dims )
model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001), metrics=['mae'])
model.load_weights( args.weights_file )

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
    var = fid['coalesced_brightness']
    x = var[ :,: ]
    var = fid['precipitation'] 
    y = var[ :,: ]
  
    x = x[ np.newaxis,:,: ]
    y = y[ np.newaxis,:,: ]

    fid.close()
    return x, y

x = np.empty((num_test_points,input_dims[0],num_channels))
true_precip = np.empty((num_test_points,nx,ny))

t1 = datetime.now()
for n in range(num_test_points):
    x[ n,:,: ], true_precip[ n,:,: ] = read_input_file( test_input_file_list[n] )

fid = nc.Dataset( 'mask.nc', 'r' )
var = fid['index_1'] 
index_1 = var[:]
var = fid['index_2'] 
index_2 = var[:]
fid.close()
io_time = (datetime.now()-t1 ).total_seconds()

##
## Perform inference 
##

t1 = datetime.now()
output = model.predict( x, batch_size=args.batch_size, verbose=0 )
inference_time = (datetime.now()-t1 ).total_seconds()

print("   inference took %5.4f seconds" % inference_time)
print("   I/O took %5.4f seconds (%4.1f percent of total runtime)" % (io_time, 100.0*(io_time/(io_time+inference_time))))

##
## Re-construct the predicted precipitation field
##

predicted_precip = np.empty((num_test_points,nx,ny), np.float32)
predicted_precip[:,:,:] = -1

#for n in range(len(index_1)):
for n in range(724000):
    i = int(index_1[n])
    j = int(index_2[n])
    predicted_precip[ :,i,j ] = output[ :,n ]

##
## Perform a visual comparision between the observed and predicted precipitation fields
##

plot_images( true_precip, predicted_precip, 'fully_connected', -1 )

##
## Create some comparision statistics
##

for time_slice in range(num_test_points):
    print('Test Point %1d' % time_slice)

    num_hits = 0
    num_cases = 0
    for n in range( 724000 ):
        i = int(index_1[n])
        j = int(index_2[n])
        if true_precip[time_slice,i,j] > 0.0:
           num_cases = num_cases + 1
           tol = abs( output[ time_slice,n ] - true_precip[time_slice,i,j] )
           if tol < 0.01:
              num_hits = num_hits + 1

    print( "  prediction accuracy for non-zero observations was %4.1f percent" % (100.0*float(num_hits)/float(num_cases)))

    print(' maximum predicted prediction value was %4.3f' % np.amax(output[time_slice,:]))
    print(' maximum observed prediction value was %4.3f' % np.amax(true_precip[time_slice,:,:]))

