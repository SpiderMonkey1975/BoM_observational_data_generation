import sys
import glob
import numpy
import netCDF4
import subprocess
import os.path
import multiprocessing
import datetime


# Returns a sorted list of rainfields 3 files from input_dir
def get_rainfields_input_files(input_dir):
    return sorted(glob.glob("%s/*/*/*/*/310*.nc" % input_dir))



def datetime_to_path(obs_datetime):

    datetime_str = obs_datetime.strftime("%Y%m%d%H%M")
    result = "%s/%s/%s/%s" % (datetime_str[0:4],
                              datetime_str[4:6],
                              datetime_str[6:8],
                              datetime_str[8:12])
    return result



def reproj(ahi_file_list, output_file):

    rainfields_proj4 = "+proj=aea +lat_1=-18 +lat_2=-36 +lon_0=132.0 +lat_0=0.0 +a=6378137 +b=6356752.31414 +units=km"
    rainfields_te = (-2300, -5100, 2600, -1000)


    current_band = 7

    # For each AHI file
    for current_ahi_file in ahi_file_list:

        # Compute the temp filename
        temp_filepath = current_ahi_file[0:-3] + "_temp.nc"

        # Prepare the gdalwarp string
        gdal_warp_string = "gdalwarp -overwrite -r cubic -of NETCDF -co \"WRITE_BOTTOMUP=NO\" "
        gdal_warp_string += "-t_srs \"%s\" -te %f %f %f %f -tr 2 2 -srcnodata 1e+20 -dstnodata -1 %s %s"
        gdal_warp_string = gdal_warp_string % (rainfields_proj4,
                                               rainfields_te[0],
                                               rainfields_te[1],
                                               rainfields_te[2],
                                               rainfields_te[3],
                                               current_ahi_file,
                                               temp_filepath)

        return_code = subprocess.call([gdal_warp_string],
                                      shell=True)


        # If something when wrong, remove the temp file
        if return_code != 0:
            try:
                os.remove(temp_filepath)
            except:
                pass

            sys.stderr.write("Failed to reproject %s\n" % current_ahi_file)
            return


        # Open the temp file and get the reprojected variable
        temp_file = netCDF4.Dataset(temp_filepath, "r")
        temp_var = temp_file["Band1"]

        # Setup a variable for this band in the output file
        var_name = "channel_%.4d_brightness_temperature" % current_band       
        output_var = output_file.createVariable(var_name,
                                                temp_var.dtype,
                                                ("time", "y", "x"),
                                                zlib=True,
                                                complevel=6,
                                                shuffle=True,
                                                least_significant_digit=1,
                                                fill_value=-1.0)

        
        # Write the data from the temp var to the output var
        reproj_data = numpy.zeros((1, len(output_file.dimensions["y"]), len(output_file.dimensions["x"])),
                                  dtype=numpy.float32) 
        reproj_data[0, :, :] = temp_var[...]
        output_var[...] = reproj_data


        # Open the input file
        input_file = netCDF4.Dataset(current_ahi_file, "r")
        input_var = input_file[var_name]

        # Read in the useful attributes
        attrib_tuple = ("algorithm_name",
                        "central_wavelength",
                        "comment",
                        "long_name",
                        "units",
                        "valid_max",
                        "valid_min",
                        "coordinates")

        for attrib_name in attrib_tuple:
            output_var.setncattr(attrib_name,
                                 input_var.getncattr(attrib_name))


        input_file.close()
        temp_file.close()


        # Remove the temp file
        try:
            os.remove(temp_filepath)
        except:
            pass

        # Increment band
        current_band += 1
        reproj_data = None



    # Nothing went wrong
    return




def process(arg_tuple):


    rainfields_file = arg_tuple[0]
    ahi_base_directory = arg_tuple[1]
    output_base_directory = arg_tuple[2]

    # Get the obs time off the rainfields filename
    # 310_20190120_052000.prcp-c10.nc
    rainfields_filename = os.path.basename(rainfields_file)
    rainfields_datetime_str = "%s%s%s%s" % (rainfields_filename[4:8],
                                            rainfields_filename[8:10],
                                            rainfields_filename[10:12],
                                            rainfields_filename[13:17])

    rainfields_datetime = datetime.datetime.strptime(rainfields_datetime_str, "%Y%m%d%H%M")
    
    # Get the AHI datetime. The closest AHI obs time to the Rainfields datetime is actually 10 minutes before the
    # rainfields datetime
    ahi_datetime = rainfields_datetime - datetime.timedelta(minutes=10)

    # Check if the AHI datetime is a maintence time
    if (ahi_datetime.hour == 2 and ahi_datatime.minute == 40) or \
       (ahi_datetime.hour == 14 and ahi_datatime.minute == 40):

        # Just return without error, there will be no Himawari data to use
        return


    # Get the paths to the AHI files
    ahi_date_path = datetime_to_path(ahi_datetime)
    ahi_dir_path = "%s/%s" % (ahi_base_directory, ahi_date_path)
    ahi_file_list = sorted(glob.glob("%s/*.nc" % ahi_dir_path))
 

    # Ensure that there is 10 Himawari files
    if len(ahi_file_list) != 10:
        sys.stderr.write("Only %.2d/10 AHI files found for %s\n" % (len(ahi_file_list), ahi_datetime))
        return


    # Create the output file
    output_date_path = datetime_to_path(rainfields_datetime)
    output_filename = "%s_rf_ahi.nc" % rainfields_datetime_str
    output_filepath = "%s/%s/%s" % (output_base_directory,
                                    output_date_path,
                                    output_filename)

    try:
        os.makedirs(os.path.dirname(output_filepath))
    except:
        pass

    output_file = netCDF4.Dataset(output_filepath, "w")

    # Open the rainfields file
    rf_file = netCDF4.Dataset(rainfields_file, "r")

    
    # Setup the output file
    output_file.createDimension("time", 1)
    output_file.createDimension("y", len(rf_file.dimensions["y"]))
    output_file.createDimension("x", len(rf_file.dimensions["x"]))

    # Copy over valid_time, start_time, proj, y, x data and all their attributes
    var_list = ["valid_time", "start_time", "proj", "y", "x"]
    for current_var_name in var_list:

        input_var = rf_file[current_var_name]

        # Create this variable in the output_file
        output_var = output_file.createVariable(input_var.name,
                                                input_var.datatype,
                                                input_var.dimensions,
                                                zlib=True,
                                                complevel=6,
                                                shuffle=True)

        # Write the variables data
        output_var = output_file[current_var_name]
        output_var[...] = input_var[...]

        # Write the variables attributes
        for current_attribute_name in input_var.ncattrs():
            output_var.setncattr(current_attribute_name, input_var.getncattr(current_attribute_name))



    # Write the rainfields_precip to the output file, with the scale factor applied
    precip_var = rf_file["precipitation"]
    precip_data = numpy.zeros((1, precip_var.shape[0], precip_var.shape[1]),
                              dtype=numpy.float32)

    precip_data[0, :, :] = precip_var[...] * precip_var.scale_factor
    output_var = output_file.createVariable("precipitation",
                                            "f4",
                                            ("time", precip_var.dimensions[0], precip_var.dimensions[1]),
                                            zlib=True,
                                            complevel=6,
                                            shuffle=True,
                                            least_significant_digit=2,
                                            fill_value=-1.0)
    output_var[...] = precip_data

    # Write the precip attributes
    output_var.scale_factor_applied = precip_var.scale_factor
    for current_atrribute_name in ["grid_mapping", "add_offset", "long_name", "standard_name", "units"]:

        output_var.setncattr(current_atrribute_name,
                             precip_var.getncattr(current_atrribute_name))


    # Reproject each AHI band
    return_code = reproj(ahi_file_list, output_file)

    rf_file.close()
    output_file.close()



if __name__ == "__main__":


    # Check that the required args are being passed
    if len(sys.argv) != 4:
        print "Usage: python %s <input rainfields base directory> <input ahi base directory> <output base directory>" % sys.argv[0]
        sys.exit(1)


    # Get the input and output directories from the command line args
    rainfields_input_dir = sys.argv[1]
    ahi_input_dir = sys.argv[2]
    output_dir = sys.argv[3]

    
    rainfields_file_list = get_rainfields_input_files(rainfields_input_dir)

    # Put together an arg list
    arg_list = list()
    for f in rainfields_file_list:
        arg_list.append((f, ahi_input_dir, output_dir))


    pool = multiprocessing.Pool(processes=16, maxtasksperchild=10)
    pool.map(process, arg_list)

    

