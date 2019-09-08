from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, History
from datetime import datetime
from random import shuffle

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import glob, sys, argparse

sys.path.insert(0, '/group/director2107/mcheeseman/BoM_observational_data_generation/neural_network_architecture/')
from basic_autoencoder import autoencoder
from unet import unet

sys.path.insert(0, '/group/director2107/mcheeseman/BoM_observational_data_generation/plotting_routines')
from plotting_routines import plot_model_errors, plot_images 


##
## Look for any user specified commandline arguments
##

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--num_filter', type=int, default=16, help="set initial number of filters used in CNN layers for the neural networks")
parser.add_argument('-v', '--verbose', type=int, default=0, help="set to 1 if additional debugging info desired")
parser.add_argument('-b', '--batch_size', type=int, default=10, help="set the batch size used in training")
parser.add_argument('-n', '--neural_net', type=str, default='basic_autoencoder', help="set neural network design. Valid values are basic_autoencoder and unet")
parser.add_argument('-t', '--test_size', type=float, default=0.2, help="set fraction of input batches used for testing")
parser.add_argument('-l', '--learn_rate', type=float, default=0.0001, help="set learn rate for the optimizer")
parser.add_argument('-z', '--num_files', type=int, default=100, help="set number of input files to be read in per block of training")
parser.add_argument('-c', '--num_channels', type=int, default=10, help="set number of channels for each satellite input image")
args = parser.parse_args()

if args.neural_net!='basic_autoencoder' and args.neural_net!='unet' and args.neural_net!='tiramisu':
   args.neural_net = 'basic_autoencoder'

if args.neural_net!='unet':
   if args.batch_size > 5:
      args.batch_size = 5
   if args.num_filter < 16:
      args.num_filter = 16

if args.num_channels<1:
   args.num_channels=1
if args.num_channels>10:
   args.num_channels=10

##
## Form the neural network
##

image_dims = np.empty((3,),dtype=np.int)
image_dims[0] = 2050
image_dims[1] = 2450
image_dims[2] = args.num_channels 

if args.neural_net == 'basic_autoencoder':
    model, ref_model = autoencoder( image_dims, args.num_filter, 1 )

if args.neural_net == 'unet':
    model, ref_model = unet( image_dims, args.num_filter, 1 )

model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001), metrics=['mae'])

if args.verbose != 0:
   model.summary()
   print( model.metrics_names )

##
## Set up the training of the model
##

filename = "model_weights_" + args.neural_net + "_" + str(args.num_filter) + "filters.h5"
checkpoint = ModelCheckpoint( filename, 
                                 monitor='val_mean_absolute_error', 
                                 save_best_only=True, 
                                 mode='min' )

earlystop = EarlyStopping( min_delta=0.001,
                           monitor='val_mean_absolute_error', 
                           patience=5,
                           mode='min' )

history = History()

my_callbacks = [checkpoint, earlystop, history]

##
## Get a list of input data files
##

input_file_list = []
cmd_str = '/group/director2107/mcheeseman/bom_data/*.nc'
for fn in glob.iglob(cmd_str, recursive=True):
    input_file_list.append( fn )

input_file_list = list(dict.fromkeys(input_file_list))
shuffle( input_file_list )

if args.verbose != 0:
   print('# of input files located: ', len(input_file_list))

num_training_images = 400
training_input_file_list = input_file_list[ :num_training_images ]
test_input_file_list = input_file_list[ num_training_images:num_training_images+6 ]

##
## Perform the training
##

def read_input_file( filename ):
    fid = nc.Dataset( filename, 'r' )
    var = fid['brightness'] 
    x = var[ :,:,: ]

    var = fid['precipitation']
    y = var[ :,: ]

    x = x[ np.newaxis,:,:,: ]
    y = y[ np.newaxis,:,: ]

    fid.close()
    return x, y

def perform_training_block( model, input_files, batch_size ):

    x = np.empty((len(input_files),image_dims[0],image_dims[1],image_dims[2]))
    y = np.empty((len(input_files),image_dims[0],image_dims[1],1))

    t2 = datetime.now()
    n = 0
    for fid in input_files:
        x[ n,:,:,: ], y[ n,:,:,0 ] = read_input_file( fid )
        n = n + 1
    io_time = (datetime.now()-t2 ).total_seconds()
    
    t2 = datetime.now()
    hist = model.fit( x, y, 
                      batch_size=batch_size,
                      epochs=500, 
                      verbose=2, 
                      validation_split=.25,
                      callbacks=my_callbacks, 
                      shuffle=False )
    train_time = (datetime.now()-t2 ).total_seconds()

    print(' ')
    print(' time due to file reads was %7.1f seconds ' % io_time)
    print(' time to train file block was %7.1f seconds ' % train_time)

    return hist

training_errors = []
validation_errors = []

t1 = datetime.now()
for i in range( 0,len(training_input_file_list),args.num_files ):
    j = i + args.num_files
    if j > len(training_input_file_list):
       j = len(training_input_file_list)

    hist = perform_training_block( model, training_input_file_list[i:j], args.batch_size )
    print("   block of input files %4d - %4d trained" % (i, j))

    training_errors = training_errors + hist.history['mean_absolute_error']
    validation_errors = validation_errors + hist.history['val_mean_absolute_error']

training_time = (datetime.now()-t1 ).total_seconds()
print("   training took %7.1f seconds" % training_time)

plot_model_errors( args.neural_net, args.num_filter, training_errors, validation_errors )

##
## Compare radar data versus generated rainfall fields from trained neural net
##

satellite_input = np.zeros((5,image_dims[0],image_dims[1],image_dims[2]))
real_rainfall = np.zeros((5,image_dims[0],image_dims[1],1))

for n in range( 5 ):
    satellite_input[ n,:,:,: ], real_rainfall[ n,:,:,0 ] = read_input_file( test_input_file_list[n] )

plot_images( real_rainfall, 
             model.predict( satellite_input, batch_size=1, verbose=0 ), 
             args.neural_net, 
             args.num_filter )

