from tensorflow.keras.optimizers import Adam
from datetime import datetime

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import glob, sys, argparse

sys.path.insert(0, '../neural_network_architecture/')
from basic_autoencoder import autoencoder
from unet import unet
from fc_densenet import Tiramisu

##
## Look for any user specified commandline arguments
##

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=1, help="set number of epochs used for training")
parser.add_argument('-g', '--num_gpu', type=int, default=1, help="set number of GPUs to be used for training")
parser.add_argument('-f', '--num_filter', type=int, default=32, help="set initial number of filters used in CNN layers for the neural networks")
parser.add_argument('-v', '--verbose', type=int, default=0, help="set to 1 if additional debugging info desired")
parser.add_argument('-b', '--batch_size', type=int, default=10, help="set the batch size used in training")
parser.add_argument('-n', '--neural_net', type=str, default='basic_autoencoder', help="set neural network design. Valid values are basic_autoencoder, unet and tiramisu")
parser.add_argument('-t', '--test_size', type=float, default=0.2, help="set fraction of input batches used for testing")
args = parser.parse_args()

if args.neural_net!='basic_autoencoder' and args.neural_net!='unet' and args.neural_net!='tiramisu':
   args.neural_net = 'basic_autoencoder'

##
## Form the neural network
##

if args.neural_net == 'tiramisu':
   model = Tiramisu( input_shape=(2050,2450,1),
                     n_filters_first_conv=args.num_filter,
                     n_pool = 2,
                     n_layers_per_block = [4,5,7,5,4] ) 

if args.neural_net == 'basic_autoencoder':
    model = autoencoder( args.num_filter, args.num_gpu )

if args.neural_net == 'unet':
    model = unet( args.num_filter, args.num_gpu )

model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001), metrics=['mae'])

if args.verbose != 0:
   model.summary()
   print( model.metrics_names )
   sys.exit()

##
## Get a list of input data files
##

input_file_list = []
cmd_str = '/data/combined_himawari_radar_data/2019/01/**/**/*.nc'
for fn in glob.iglob(cmd_str, recursive=True):
    input_file_list.append( fn )

input_file_list.sort()
input_file_list = list(dict.fromkeys(input_file_list))

if args.verbose != 0:
   print('# of input files located: ', len(input_file_list))

file_count = len( input_file_list )
if np.mod( file_count, args.batch_size ) > 0:
    file_count = file_count - np.mod( file_count, args.batch_size ) 

##
## Read in feature and target data for the specific day
##

def read_input_file( filename ):
    fid = nc.Dataset( filename, 'r' )
    x = np.array( fid['channel_0007_brightness_temperature'] )
    y = np.array( fid['precipitation'] )
    fid.close()
    return x, y

def model_fit( model, start, end, batch_size, input_file_list, train_flag ):
    losses = []
    for fn in range( start,end,batch_size ):
        x, y = read_input_file( input_file_list[fn] )
        for n in range( batch_size ):
            x2, y2 = read_input_file( input_file_list[fn+n] )
            x = np.concatenate((x,x2), axis=0)
            y = np.concatenate((y,y2), axis=0)

        x = x[ :,:,:,np.newaxis]
        y = y[ :,:,:,np.newaxis]

        if train_flag == 1:
           output = model.train_on_batch( x, y )
        else:
           output = model.test_on_batch( x, y )
        losses.append( output[1] )

    l = np.array( losses )
    return np.amax( l )

##
## Set the training-test split on the input data
##

#num_validation_batches = int(args.test_size * (file_count/args.batch_size))
num_validation_batches = 10 

##
## Perform model training
##

training_mse_losses = []
validation_mse_losses = []
tol = 10000.0

t1 = datetime.now()
for epoch in range( args.epochs ):
    t2 = datetime.now()
    train_loss = model_fit( model, num_validation_batches, file_count, args.batch_size, input_file_list, 1 )
    training_mse_losses.append( train_loss )

    valid_loss = model_fit( model, 0, num_validation_batches, args.batch_size, input_file_list, 0 )
    validation_mse_losses.append( valid_loss )
    epoch_time = (datetime.now()-t2 ).total_seconds()

    print("Epoch %2d: training MSE: %4.3f validation MSE: %4.3f" % (epoch,train_loss,valid_loss))
    if valid_loss < tol:
       tol = valid_loss
       weights_file = args.neural_net + '_model_weights.h5'
       model.save_weights( weights_file )

total_time = (datetime.now()-t1 ).total_seconds()

##
## Determine the Epoch at which the minimum MAE metric is observed during training 
## and validation
##

ind = np.argmin( np.array(training_mse_losses) )
ind2 = np.argmin( np.array(validation_mse_losses) )

print(" ")
print(" ")
print("=====================================================================================")
print("                          Rainfall Regression Network")
print("=====================================================================================")
print(" ")
print("   %s neural network design used with %3d initial filters" % (args.neural_net,args.num_filter))
print("   1 channel of satellite temperature data used")
print("   batch size of %2d images used" % args.batch_size)
print("   validation size set to %2d images" % (num_validation_batches*args.batch_size) )
print("   training lasted for %7.1f seconds" % total_time)
print(" ")
print("   Mean Absolute Error Metric")
print("       minimum observed during training was %4.3f at Epoch %2d" % (training_mse_losses[ind],ind))
print("       minimum observed during validation was %4.3f at Epoch %2d" % (validation_mse_losses[ind2],ind2))
print(" ")

##
## Output plot of training and validation errors
##

plt.plot( np.array(training_mse_losses),color='r' )
plt.plot( np.array(validation_mse_losses),color='b' )
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.title('Training and Validation Error')
plt.savefig( 'losses.png', transparent=True )

