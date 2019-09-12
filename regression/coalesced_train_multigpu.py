from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, History
from datetime import datetime
from random import shuffle

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import glob, sys, argparse

sys.path.insert(0, '/group/director2107/mcheeseman/BoM_observational_data_generation/neural_network_architecture/')
from fully_connected import simple_net_multigpu

sys.path.insert(0, '/group/director2107/mcheeseman/BoM_observational_data_generation/plotting_routines')
from plotting_routines import plot_fc_model_errors, plot_images

##
## Set some constants for the run
##

num_channels = 10
num_datapoints = 724016
nx = 2050
ny = 2450
num_test_points = 5

##
## Define some required I/O functions
##

def read_test_input_file( filename ):
    fid = nc.Dataset( filename, 'r' )
    var = fid['coalesced_brightness']
    x = var[ :,: ]
    var = fid['precipitation']
    y = var[ :,: ]

    x = x[ np.newaxis,:,: ]
    y = y[ np.newaxis,:,: ]

    fid.close()
    return x, y

def read_training_input_file( filename ):

    fid = nc.Dataset( filename, 'r' )
    x = np.array( fid['coalesced_brightness'] )
    y = np.array( fid['coalesced_precipitation'] )
  
    x = x[ np.newaxis,:,: ]
    y = y[ np.newaxis,: ]

    fid.close()
    return x, y

def read_test_input_file_full( filename ):
    fid = nc.Dataset( filename, 'r' )
    x = np.array( fid['brightness'] )
    x = x[ np.newaxis,:,:,: ]

    fid.close()
    return x


def read_mask():
    fid = nc.Dataset( 'mask.nc', 'r' )
    var = fid['index_1']
    index_1 = var[:]
    var = fid['index_2']
    index_2 = var[:]
    fid.close()
    return index_1, index_2

##
## Look for any user specified commandline arguments
##

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', type=int, default=50, help="set the batch size used in training")
parser.add_argument('-g', '--num_gpu', type=int, default=2, help="set the number of GPUS to be used in training")
parser.add_argument('-t', '--tolerance', type=float, default=0.001, help="set the tolerance used for the early stopping callback")
args = parser.parse_args()

##
## Form the neural network
##

input_dims = np.empty((2,),dtype=np.int)
input_dims[0] = num_datapoints 
input_dims[1] = num_channels

model = simple_net_multigpu( input_dims, args.num_gpu )
model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001), metrics=['mae'])
print('Neural network and model created')

##
## Set up the training of the model
##

history = History()

earlystop = EarlyStopping( min_delta=args.tolerance,
                           monitor='val_mean_absolute_error', 
                           patience=3,
                           mode='min' )

my_callbacks = [earlystop, history]

##
## Get a list of input data files
##

input_file_list = []
cmd_str = '/group/director2107/mcheeseman/bom_data//input_*.nc'
for fn in glob.iglob(cmd_str, recursive=True):
    input_file_list.append( fn )

input_file_list = list(dict.fromkeys(input_file_list))
num_training_images = len(input_file_list) - num_test_points 
print('%5d input datafiles located' % len(input_file_list))

test_input_file_list = input_file_list[ num_training_images: ]
training_input_file_list = input_file_list[ :num_training_images ]
shuffle( training_input_file_list )

##
## Perform the training
##

x = np.empty((num_training_images,num_datapoints,num_channels))
y = np.empty((num_training_images,num_datapoints))

t1 = datetime.now()
for n in range(num_training_images):
    x[ n,:,: ], y[ n,: ] = read_training_input_file( training_input_file_list[n] )
io_time = (datetime.now()-t1 ).total_seconds()
print( 'input training datafiles read' )

hist = model.fit( x, y, 
                  batch_size=args.batch_size,
                  epochs=500, 
                  verbose=2, 
                  validation_split=.125,
                  callbacks=my_callbacks, 
                  shuffle=False )
training_time = (datetime.now()-t1 ).total_seconds()

print("   training took %7.1f seconds" % training_time)
print("   I/O took %7.1f seconds (%4.1f percent of total runtime)" % (io_time,100.0*(io_time/(io_time+training_time))))

plot_fc_model_errors( 'simple_fully_connected', hist )

##
## Perform the inference on the trained model
##

x = np.empty((num_test_points,num_datapoints,num_channels))
true_precip = np.empty((num_test_points,nx,ny))

t1 = datetime.now()
for n in range(num_test_points):
    x[ n,:,: ], true_precip[ n,:,: ] = read_test_input_file( test_input_file_list[n] )

index_1, index_2 = read_mask()
io_time = (datetime.now()-t1 ).total_seconds()

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

for n in range(num_datapoints):
    i = int(index_1[n])
    j = int(index_2[n])
    predicted_precip[ :,i,j ] = output[ :,n ]

##
## Output the precipitation fields (file and plot)
##

fid = nc.Dataset('precipitation_radar_only.nc', "w")
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

plot_images( true_precip, predicted_precip, 'fully_connected_radar_only', -1 )

##
## Create some comparision statistics
##

for time_slice in range(num_test_points):
    print('Test Point %1d' % time_slice)

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

    print( "  prediction accuracy for non-zero observations was %4.1f percent" % (100.0*float(num_hits)/float(num_cases)))

    print(' maximum predicted prediction value was %4.3f' % np.amax(output[time_slice,:]))
    print(' maximum observed prediction value was %4.3f' % np.amax(true_precip[time_slice,:,:]))

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

t1 = datetime.now()
for n in range(num_test_points):
    satellite_data[ n,:,:,: ] = read_test_input_file_full( test_input_file_list[n] )
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

for time_slice in range(num_test_points):
    print('Test Point %1d' % time_slice)

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

    print( "  prediction accuracy for non-zero observations was %4.1f percent" % (100.0*float(num_hits)/float(num_cases)))

    print(' maximum predicted prediction value was %4.3f' % np.amax(output[time_slice,:]))
    print(' maximum observed prediction value was %4.3f' % np.amax(true_precip[time_slice,:,:]))
