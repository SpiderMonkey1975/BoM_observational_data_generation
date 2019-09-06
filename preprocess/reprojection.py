import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import glob, os, sys, datetime, subprocess, pathlib

def check_if_file_exists( filename ):
    fid = pathlib.Path( filename )
    if fid.exists():
       status = 1  
    else:
       status = 0
    return status 

def get_filenames( radar_datetime ):
    satellite_datetime = radar_datetime - datetime.timedelta(minutes=10)
    channel = 7
    datetime_str = satellite_datetime.strftime("%Y%m%d%H%M")
    satellite_basedir = '/scratch/director2107/vvillani/data/ahi/'
    satellite_filenames = []
    for channel in range(7,10):
        filename = "%s%s%s%s%s%s00-P1S-ABOM_OBS_B0%1d-PRJ_GEOS141_2000-HIMAWARI8-AHI.nc" % (satellite_basedir,
                                                                                            datetime_str[0:4],
                                                                                            datetime_str[4:6],
                                                                                            datetime_str[6:8],
                                                                                            datetime_str[8:10],
                                                                                            datetime_str[10:12],
                                                                                            channel)
        satellite_filenames.append( filename )
    for channel in range(10,17):
        filename = "%s%s%s%s%s%s00-P1S-ABOM_OBS_B%2d-PRJ_GEOS141_2000-HIMAWARI8-AHI.nc" % (satellite_basedir,
                                                                                           datetime_str[0:4],
                                                                                           datetime_str[4:6],
                                                                                           datetime_str[6:8],
                                                                                           datetime_str[8:10],
                                                                                           datetime_str[10:12],
                                                                                           channel)
        satellite_filenames.append( filename )
    
    output_basedir = '/group/director2107/mcheeseman/bom_data/'
    output_filename = "%sinput_%s%s%s_%s%s.nc" % (output_basedir,
                                                  datetime_str[0:4],
                                                  datetime_str[4:6],
                                                  datetime_str[6:8],
                                                  datetime_str[8:10],
                                                  datetime_str[10:12])
    return satellite_filenames,output_filename

##
## Set some necessary global variables
##

rainfields_proj4 = "+proj=aea +lat_1=-18 +lat_2=-36 +lon_0=132.0 +lat_0=0.0 +a=6378137 +b=6356752.31414 +units=km"
temp_filepath = 'reprojected_satellite_data.nc'
gdal_warp_string = "gdalwarp -overwrite -r cubic -of NETCDF -co \"WRITE_BOTTOMUP=NO\" "
gdal_warp_string += "-t_srs \"%s\" -te -2300 -5100 2600 -1000 -tr 2 2 -srcnodata 1e+20 -dstnodata -1 %s %s"

##
## Get list of all available radar input files
##

radar_files = []
cmd_str = '/scratch/director2107/vvillani/data/rainfields_3/310_*.nc'
for fn in glob.iglob(cmd_str, recursive=True):
    radar_files.append( fn )

radar_files = list(dict.fromkeys(radar_files))
    
##
## Get the first pair of satellite and radar input datafiles
##

for radar_filename in radar_files:

    radar_datetime = datetime.datetime.strptime( radar_filename[-27:-14], "%Y%m%d_%H%M")
    satellite_filenames,output_file = get_filenames( radar_datetime )

    cnt = check_if_file_exists( radar_filename )
    for fh in satellite_filenames:
        cnt = cnt + check_if_file_exists( fh )

    if cnt != 11:
       print('missing input, skipping input')
       continue 
##
## Read in radar data 
##

    fid = nc.Dataset( radar_filename, 'r' )
    var = fid["precipitation"]
    radar_data = np.zeros((var.shape[0], var.shape[1]), dtype=np.float32)
    radar_data[:, :] = np.squeeze( var[...] * var.scale_factor )
    fid.close()

##
## Read in and re-project satellite data 
##

    satellite_data = np.zeros((2050, 2450, 10), dtype=np.float32)

    for n in range(len(satellite_filenames)):
        gdal_cmd = gdal_warp_string % (rainfields_proj4, satellite_filenames[n], temp_filepath)
        return_code = subprocess.call( [gdal_cmd], shell=True )

        fid = nc.Dataset(temp_filepath, 'r')
        temp_var = fid['Band1']
        satellite_data[ :,:,n ] = np.squeeze( temp_var[...] )
        fid.close()

        os.remove( temp_filepath )

    #print(satellite_data.shape)

##
## Output radar and projected satellite data into a new output datafile
##

    try:
       os.makedirs(os.path.dirname(output_file))
    except:
       pass

    fid = nc.Dataset(output_file, "w")

    fid.createDimension("channels", 10)
    fid.createDimension("y", 2050)
    fid.createDimension("x", 2450)

    radar_var = fid.createVariable( 'precipitation', 'f', ('y','x') )
    satellite_var = fid.createVariable( 'brightness', 'f', ('y','x','channels') )

    radar_var = fid['precipitation']
    radar_var[...] = radar_data[...]

    satellite_var = fid['brightness']
    satellite_var[...] = satellite_data[...]
    fid.close()

##
## Plot radar and satellite data 
##

fid = nc.Dataset( output_file, 'r' )
var = fid["precipitation"]
radar_data = np.zeros((var.shape[0], var.shape[1]), dtype=np.float32)
radar_data[:, :] = var[...]

var = fid["brightness"]
satellite_data = np.zeros((2050, 2450, 10), dtype=np.float32)
satellite_data[:,:,:] = var[...]
fid.close()

filename = 'input_image_check.png'

plt.subplot(1, 2, 1)
plt.imshow(radar_data)
plt.axis('off')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow( np.squeeze(satellite_data[:,:,8]) )
plt.axis('off')
plt.colorbar()

plt.tight_layout()
plt.savefig(filename)
plt.close('all')
