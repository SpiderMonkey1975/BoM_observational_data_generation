##
## image2d_prep
##
## Python script that constructs a 3D NumPy array consisting of a stack of 2D images.
## 
##   Mark Cheeseman, CSIRO
##   July 8, 2019
##===================================================================================

import xarray as xr
import numpy as np
import os

def prepare_array( filename ):
    ''' Reads the 2D precipitation field from an NetCDF file.  A 2D numpy array is
        constructed and returned.

        Input: filename -> name of input NetCDF file

    '''
    fid = xr.open_dataset( filename )
    return np.moveaxis( fid['precipitation'].values,0,-1 )

##
## Read in the individual precipitation datasets
##

array_list = []
for number, filename in enumerate(sorted(os.listdir('./')), start=1):
    if filename[-3:] == '.nc':
       array_list.append( prepare_array( filename ) )

##
## Construct a NumPy array containing all the timeslices
##

precipitation_array = np.concatenate( array_list, axis=2 )

##
## Write the NumPy array to hard disk
##

np.save( 'jan28-29_2019_precipitation.npy', precipitation_array )

