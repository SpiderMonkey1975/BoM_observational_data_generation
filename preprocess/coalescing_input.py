import netCDF4 as nc
import numpy as np
import glob, sys, pathlib

##
## Get list of all available input files
##

input_files = []
cmd_str = '/group/director2107/mcheeseman/bom_data/input*.nc'
for fn in glob.iglob(cmd_str, recursive=True):
    input_files.append( fn )

input_files = list(dict.fromkeys(input_files))

##
## Create mask for input data collection
##

fid = nc.Dataset( input_files[0], 'r' )
var = fid["precipitation"]
radar_data = np.empty((var.shape[0], var.shape[1]), dtype=np.float32)
radar_data[:, :] = var[...] 
fid.close()

index_1 = []
index_2 = []
for i in range( radar_data.shape[0] ):
    for j in range( radar_data.shape[1] ):
        if radar_data[i,j] < 0:
           status = 1 
        else:
           index_1.append( i )
           index_2.append( j )

num_points = len(index_1)
#print( num_points )

##
## Construct a global masking array
##

mask = np.ones((radar_data.shape[0],radar_data.shape[1],), dtype=np.int)
for n in range( num_points ):
    mask[ index_1[n],index_2[n] ] = 0

##
## Choose another set of indicies for prediction use 
##

prediction_index_1 = []
prediction_index_2 = []
for i in range( mask.shape[0] ):
    for j in range( mask.shape[1] ):
        if mask[i,j] == 1:
           prediction_index_1.append( i )
           prediction_index_2.append( j )

prediction_index_1 = prediction_index_1[ :num_points ]
prediction_index_2 = prediction_index_2[ :num_points ]

fid = nc.Dataset('mask.nc', "w")
fid.createDimension("n", len(index_1))
index_1_var = fid.createVariable( 'index_1', 'f', ('n') )
index_2_var = fid.createVariable( 'index_2', 'f', ('n') )
pindex_1_var = fid.createVariable( 'prediction_index_1', 'f', ('n') )
pindex_2_var = fid.createVariable( 'prediction_index_2', 'f', ('n') )

var = fid['index_1']
var[:] = index_1
var = fid['index_2']
var[:] = index_2
var = fid['prediction_index_1']
var[:] = prediction_index_2
var = fid['prediction_index_2']
var[:] = prediction_index_2
fid.close()

sys.exit()

##
## Get the first pair of satellite and radar input datafiles
##

for filename in input_files:
    fid = nc.Dataset( filename, 'r+' )

##
## Read in data 
##

    var = fid["precipitation"]
    radar_data = np.zeros((var.shape[0], var.shape[1]), dtype=np.float32)
    radar_data[:, :] = var[...]

    var = fid["brightness"]
    satellite_data = np.zeros((var.shape[0], var.shape[1], var.shape[2]), dtype=np.float32)
    satellite_data[:, :] = var[...]

##
## Coalesce data that only corresponds to non-zero points in the radar data domain
##

    coalesced_radar_data = np.empty((len(index_1),), dtype=np.float32)
    coalesced_satellite_data = np.empty((len(index_1),10,), dtype=np.float32)

    for n in range( len(index_1) ):
        coalesced_radar_data[n] = radar_data[ index_1[n],index_2[n] ]
        coalesced_satellite_data[ n,: ] = satellite_data[ index_1[n],index_2[n],: ]

##
## Write coalesced radar and satellite data into output datafile
##

    fid.createDimension("n", len(index_1))

    radar_var = fid.createVariable( 'coalesced_precipitation', 'f', ('n') )
    satellite_var = fid.createVariable( 'coalesced_brightness', 'f', ('n','channels') )

    radar_var = fid['coalesced_precipitation']
    radar_var[...] = coalesced_radar_data[...]

    satellite_var = fid['coalesced_brightness']
    satellite_var[...] = coalesced_satellite_data[...]
    fid.close()

