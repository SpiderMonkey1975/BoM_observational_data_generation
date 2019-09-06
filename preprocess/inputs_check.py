import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import argparse

##
## Look for any user specified commandline arguments
##

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--num_channels', type=int, default=10, help="set number of channels for each satellite input image")
args = parser.parse_args()

if args.num_channels<1:
   args.num_channels=1
if args.num_channels>10:
   args.num_channels=10

##
## Set image dimensions
##

image_dims = np.empty((3,),dtype=np.int)
image_dims[0] = 2050
image_dims[1] = 2450
image_dims[2] = args.num_channels

##
## Read in input data 
##

filename = '/group/director2107/vvillani/reproj/output/2019/01/20/0520/201901200520_rf_ahi.nc'

fid = nc.Dataset( filename, 'r' )
satellite_data = np.empty((image_dims[0],image_dims[1],image_dims[2]))

idx = 7
for n in range(image_dims[2]):
    if idx<10:
       varname = 'channel_000' + str(idx) + '_brightness_temperature'
    else:
       varname = 'channel_00' + str(idx) + '_brightness_temperature'
    satellite_data[ :,:,n ] = np.array( fid[varname] )
    idx = idx + 1

radar_data = np.squeeze( np.array( fid['precipitation'] ))
fid.close()

print( radar_data.shape )
print( satellite_data.shape )

##
## Plot radar and satellite data 
##

filename = 'input_image_check.png'

plt.subplot(1, 2, 1)
plt.imshow(radar_data)
plt.axis('off')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow( np.squeeze(satellite_data[:,:,0]) )
plt.axis('off')
plt.colorbar()

plt.tight_layout()
plt.savefig(filename)
plt.close('all')


