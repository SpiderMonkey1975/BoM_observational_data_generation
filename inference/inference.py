from tensorflow.keras.optimizers import Adam
from datetime import datetime
from random import shuffle

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import glob, sys, argparse

sys.path.insert(0, '/group/director2107/mcheeseman/BoM_observational_data_generation/neural_network_architecture/')
from fully_connected import simple_net

sys.path.insert(0, '/group/director2107/mcheeseman/BoM_observational_data_generation/plotting_routines')
from plotting_routines import plot_images 


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
input_dims[0] = 724016 
input_dims[1] = 10

model, ref_model = simple_net( input_dims, 1 )
model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001), metrics=['mae'])
model.load_weights( args.weights_file )

##
## Get a list of input data files
##

input_file_list = []
cmd_str = '/group/director2107/mcheeseman/bom_data/input_*.nc'
for fn in glob.iglob(cmd_str, recursive=True):
    input_file_list.append( fn )

input_file_list = list(dict.fromkeys(input_file_list))
shuffle( input_file_list )

test_input_file_list = input_file_list[ -5: ]

##
## Read in the input data 
##

def read_input_file( filename ):
    fid = nc.Dataset( filename, 'r' )
    var = fid['coalesced_brightness']
    x = var[ :,: ]
    var = fid['precipitation'] 
    y = var[ :,: ]
    var = fid['brightness']
    x_full = var[ :,:,: ]
  
    x = x[ np.newaxis,:,: ]
    y = y[ np.newaxis,:,: ]
    x_full = x[ np.newaxis,:,:,: ]

    fid.close()
    return x, y, x_full

x = np.empty((5,input_dims[0],10))
true_precip = np.empty((5,2050,2450))
x_full = np.empty((5,2050,2450,10))

t1 = datetime.now()
for n in range(5):
    x[ n,:,: ], true_precip[ n,:,: ], x_full[ n,:,:,: ] = read_input_file( test_input_file_list[n] )

fid = nc.Dataset( 'mask.nc', 'r' )
var = fid['index_1'] 
index_1 = var[:]
var = fid['index_2'] 
index_2 = var[:]
var = fid['prediction_index_1'] 
pindex_1 = var[:]
var = fid['prediction_index_2'] 
pindex_2 = var[:]
fid.close()
io_time = (datetime.now()-t1 ).total_seconds()

fig, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(1,5, sharey=True)
fig.suptitle( 'Observed Precipitation' )
ax1.imshow( np.squeeze(true_precip[0,:,:]) )
ax2.imshow( np.squeeze(true_precip[1,:,:]) )
ax3.imshow( np.squeeze(true_precip[2,:,:]) )
ax4.imshow( np.squeeze(true_precip[3,:,:]) )
ax5.imshow( np.squeeze(true_precip[4,:,:]) )
plt.savefig( 'true_precipitation.png' )
plt.close('all')

##
## Perform inference 
##

t1 = datetime.now()
output = model.predict( x, batch_size=args.batch_size, verbose=0 )

for n in range( len(pindex_1) ):
    for i in range(5):
        x[ i,n,: ] = x_full[ i,int(pindex_1[n]),int(pindex_2[n]),: ]
output2 = model.predict( x, batch_size=args.batch_size, verbose=0 )

inference_time = (datetime.now()-t1 ).total_seconds()

print("   inference took %5.4f seconds" % inference_time)
print("   I/O took %5.4f seconds" % io_time)

##
## Re-construct the predicted precipitation field
##

predicted_precip = np.empty((5,2050,2450), np.float32)
predicted_precip[:,:,:] = -1

for n in range(len(index_1)):
    i = int(index_1[n])
    j = int(index_2[n])
    predicted_precip[ :,i,j ] = output[ :,n ]
    i = int(pindex_1[n])
    j = int(pindex_2[n])
    predicted_precip[ :,i,j ] = output2[ :,n ]

fig, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(1,5, sharey=True)
fig.suptitle( 'Predicted Precipitation' )
ax1.imshow( np.squeeze(predicted_precip[0,:,:]) )
ax2.imshow( np.squeeze(predicted_precip[1,:,:]) )
ax3.imshow( np.squeeze(predicted_precip[2,:,:]) )
ax4.imshow( np.squeeze(predicted_precip[3,:,:]) )
ax5.imshow( np.squeeze(predicted_precip[4,:,:]) )
plt.savefig( 'predicted_precipitation.png' )
plt.close('all')


