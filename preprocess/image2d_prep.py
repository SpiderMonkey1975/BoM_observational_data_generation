##
## image2d_prep
##
## Python script that constructs a 3D NumPy array consisting of a stack of 2D images.
## 
##   Mark Cheeseman, CSIRO
##   July 10, 2019
##===================================================================================

import xarray as xr
import numpy as np
import os, agparse

def prepare_array( filename, zero_remove ):
    ''' Reads an input precipitation NetCDF file.  A 4D numpy array is constructed
        and returned with the following dimensions:
           <1, latitude, longitude, 1>

        Input: filename -> name of input NetCDF file

    '''
    df = xr.open_dataset( filename ).to_dataframe().dropna()

    if zero_remove == 1:
       df = df[df['precipitation']>0.0]
    
    precip_array = df['precipitation'].to_xarray().values
    return precip_array[ np.newaxis,:,:,: ]

##
## Parse any commandline arguments given by the user 
##

arser = argparse.ArgumentParser()
parser.add_argument('-z', '--zero_remove', type=int, default=0, help="set to 1 if user wishes precipitation values equal to 0.0 to be removed")
args = parser.parse_args()

##
## Read in the individual precipitation datasets
##

array_list = []
for number, filename in enumerate(sorted(os.listdir('./')), start=1):
    if filename[-3:] == '.nc':
       print( filename )
       array_list.append( prepare_array( filename, args.remove_zero ) )

##
## Construct a NumPy array containing all the timeslices
##

precipitation_array = np.concatenate( array_list, axis=0 )
precipitation_array = precipitation_array[ :,:,:,np.newaxis ]

print(" ")
print(" %3d images stored" % (precipitation_array.shape[0]))
print(" image dimensions are %3d by %3d" % (precipitation_array.shape[1],precipitation_array.shape[2]))

##
## Write the NumPy array to hard disk
##

if args.zero_remove == 1:
    np.save( 'jan28-29_2019_precipitation_no_zeros.npy', precipitation_array )
else:
    np.save( 'jan28-29_2019_precipitation.npy', precipitation_array )

