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
    x = np.array( fid['brightness'] )
    x = x[ np.newaxis,:,:,: ]
    y = np.array( fid['precipitation'] )
    y = y[ np.newaxis,:,: ]
    fid.close()
    return x,y

##
## Generate input indicies
##

global_index = np.empty((nx*ny,2,), dtype=np.int32)

n = 0
for i in range(nx):
    for j in range(ny):
        global_index[n,0] = i
        global_index[n,1] = j
        n = n + 1

##
## Perform inference
##

x = np.empty((num_test_points,num_datapoints,num_channels), dtype=np.float32)
satellite_data = np.empty((num_test_points,nx,ny,num_channels))
predicted_precip = np.empty((num_test_points,nx,ny), np.float32)
true_precip = np.empty((num_test_points,nx,ny))

t1 = datetime.now()
fid = nc.Dataset( 'mask.nc', 'r' )
var = fid['index_1'] 
index_1 = var[:]
var = fid['index_2'] 
index_2 = var[:]
fid.close()

for n in range(num_test_points):
    satellite_data[ n,:,:,: ], true_precip[ n,:,: ] = read_input_file( test_input_file_list[n] )
io_time = (datetime.now()-t1 ).total_seconds()


t1 = datetime.now()

for nn in range(6):
    idx = nn*num_datapoints

    for n in range(num_datapoints):
        i = global_index[n+idx,0]
        j = global_index[n+idx,1]
        x[ :,n,: ] = satellite_data[ :,i,j,: ]

    output = model.predict( x, verbose=0 )

    for n in range(num_datapoints):
        i = global_index[n+idx,0]
        j = global_index[n+idx,1]
        predicted_precip[ :,i,j ] = output[ :,n ]

inference_time = (datetime.now()-t1 ).total_seconds()

print("   inference took %5.4f seconds" % inference_time)
print("   I/O took %5.4f seconds (%4.1f percent of total runtime)" % (io_time, 100.0*(io_time/(io_time+inference_time))))

##
## Output the precipitation fields (file and plot)
##

fid = nc.Dataset('precipitation.nc', "w")
fid.createDimension("t", num_test_points)
fid.createDimension("x", nx)
fid.createDimension("y", ny)
observed_var = fid.createVariable( 'observed_precipitation', 'f', ('t','x','y') )
predicted_var = fid.createVariable( 'predicted_precipitation', 'f', ('t','x','y') )

var = fid['observed_precipitation']
var[:,:] = true_precip
var = fid['predicted_precipitation']
var[:] = predicted_precip
fid.close()

plot_images( true_precip, predicted_precip, 'fully_connected', -1 )

##
## Create some comparision statistics
##

min_accuracy = 100.0
max_accuracy = 0.0;
avg_accuracy = 0.0

for time_slice in range(num_test_points):

    num_hits = 0
    num_cases = 0
    for n in range( num_datapoints ):
        i = int(index_1[n])
        j = int(index_2[n])
        if true_precip[time_slice,i,j] > 0.0:
           num_cases = num_cases + 1
           tol = abs( output[ time_slice,n ] - true_precip[time_slice,i,j] )
           if tol < 0.05:
              num_hits = num_hits + 1

    tol = 100.0*float(num_hits)/float(num_cases)
    if tol < min_accuracy:
       min_accuracy = tol
    if tol > max_accuracy:
       max_accuracy = tol

    avg_accuracy = avg_accuracy + tol


avg_accuracy = avg_accuracy / float(num_test_points)
print(' ')
print("Prediction accuracy for non-zero observations (full Australian continent)")
print(' minimum value was %4.1f' % min_accuracy)
print(' maximum value was %4.1f' % max_accuracy)
print(' average value was %4.1f' % avg_accuracy)

