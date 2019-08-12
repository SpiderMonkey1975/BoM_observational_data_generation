from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, History
from datetime import datetime

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import glob, sys, argparse

sys.path.insert(0, '/group/director2107/mcheeseman/BoM_observational_data_generation/neural_network_architecture/')
from basic_autoencoder import autoencoder
from unet import unet
from fc_densenet import Tiramisu

##
## Look for any user specified commandline arguments
##

parser = argparse.ArgumentParser()
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
   model, ref_model = Tiramisu( input_shape=(2050,2450,1),
                                n_filters_first_conv=args.num_filter,
                                n_pool = 2,
                                n_layers_per_block = [4,5,7,5,4] ) 

if args.neural_net == 'basic_autoencoder':
    model, ref_model = autoencoder( args.num_filter, args.num_gpu )

if args.neural_net == 'unet':
    model, ref_model = unet( args.num_filter, args.num_gpu )

model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001), metrics=['mae'])

if args.verbose != 0:
   model.summary()
   print( model.metrics_names )

##
## Get a list of input data files
##

input_file_list = []
cmd_str = '/group/director2107/mcheeseman/bom_data/2019/01/**/**/*.nc'
for fn in glob.iglob(cmd_str, recursive=True):
    input_file_list.append( fn )

input_file_list.sort()
input_file_list = list(dict.fromkeys(input_file_list))

input_file_list = input_file_list[ :400 ]

if args.verbose != 0:
   print('# of input files located: ', len(input_file_list))

##
## Read in feature and target data for the specific day
##

def read_input_file( filename ):
    fid = nc.Dataset( filename, 'r' )
    x = np.array( fid['channel_0007_brightness_temperature'] )
    y = np.array( fid['precipitation'] )
    fid.close()
    return x, y

x = np.zeros((len(input_file_list),2050,2450,1))
y = np.zeros((len(input_file_list),2050,2450,1))

for n in range( len(input_file_list) ):
    x[ n,:,:,0 ], y[ n,:,:,0 ] = read_input_file( input_file_list[n] )

##
## Set up the training of the model
##

filename = "model_weights_" + args.neural_net + "_" + str(args.num_filter) + "filters.h5"
if args.num_gpu == 1:
   checkpoint = ModelCheckpoint( filename, 
                                 monitor='val_mean_absolute_error', 
                                 save_best_only=True, 
                                 mode='min' )

earlystop = EarlyStopping( min_delta=0.5,
                           patience=2,
                           mode='min' )

history = History()

my_callbacks = [earlystop, history]
    
##
## Perform model training
##

t1 = datetime.now()
hist = model.fit( x, y, 
                  batch_size=args.batch_size,
                  epochs=500, 
                  verbose=2, 
                  validation_split=.25,
                  callbacks=my_callbacks, 
                  shuffle=True )
training_time = (datetime.now()-t1 ).total_seconds()

val_loss = hist.history['val_mean_absolute_error']
min_val_loss = 10000000.0
ind = -1
for n in range(len(val_loss)):
       if val_loss[n] < min_val_loss:
          min_val_loss = val_loss[n]
          ind = n+1

print(" ")
print(" ")
print("=====================================================================================")
print("                          Rainfall Regression Network")
print("=====================================================================================")
print(" ")
print("   %s neural network design used with %2d initial filters" % (args.neural_net,args.num_filter))
print("   1 channel of satellite temperature data used")
print("   batch size of %2d images used" % args.batch_size)
print("   training lasted for %7.1f seconds" % training_time)
print(" ")
print("   Mean Absolute Error Metric")
print("       minimum observed during validation was %4.3f at Epoch %2d" % (min_val_loss,ind))
print(" ")

##
## Output plot of training and validation errors
##

plt.plot( hist.history['mean_absolute_error'], color='r' )
plt.plot( hist.history['val_mean_absolute_error'], color='b' )
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.title('Model Error')
plt.legend(['Training','Validation'], loc='upper right')
plt.savefig( 'losses.png', transparent=True )

