##
## signal_prep
##
## Python script that constructs a 3D NumPy array consisting of a stack of filtered
## 2D slices of original precipitation data
##
##   Mark Cheeseman, CSIRO
##   July 8, 2019
##===================================================================================

import xarray as xr
import numpy as np
import pandas as pd
import os

def prepare_dataframe( filename ):
    ''' Reads the 2D precipitation field from an NetCDF file. Remove all NaN and zero
        values in the iresulting dataframe.  Drop all columns except Precipitation. 

        Input: filename -> name of input NetCDF file

    '''
    column_list = ['start_time','valid_time','crs']
    df = xr.open_dataset( filename ).to_dataframe().drop(labels=column_list,axis=1).dropna()
    df = df.reset_index('time',drop=True)
    return df[df['precipitation']>0.0]

##
## Read in the individual precipitation datasets
##

df_list = []
for number, filename in enumerate(sorted(os.listdir('./')), start=1):
    if filename[-3:] == '.nc':
       df_list.append( prepare_dataframe( filename ) )

##
## Merge all inidividual dataframes into a single global one
##

precipitation_df = df_list[0]
for n in range(1,len(df_list)):
    precipitation_df = pd.merge( precipitation_df, df_list[n], how='left', on=['lat','lon'] )
precipitation_df.fillna( -1 )

#precipitation_df.columns = col_names

##
## Construct a NumPy array containing all the timeslices and write it to hard disk
##

tmp_array = precipitation_df.to_numpy()
print( tmp_array.shape )
np.save( 'jan28-29_2019_precipitation_signals.npy', precipitation_df.to_numpy() )

