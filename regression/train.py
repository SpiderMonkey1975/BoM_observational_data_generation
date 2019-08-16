from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, History
from alt_model_checkpoint import AltModelCheckpoint
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
parser.add_argument('-g', '--num_gpu', type=int, default=1, help="set number of GPUs to be used for training")
parser.add_argument('-f', '--num_filter', type=int, default=32, help="set initial number of filters used in CNN layers for the neural networks")
parser.add_argument('-v', '--verbose', type=int, default=0, help="set to 1 if additional debugging info desired")
parser.add_argument('-b', '--batch_size', type=int, default=10, help="set the batch size used in training")
parser.add_argument('-n', '--neural_net', type=str, default='basic_autoencoder', help="set neural network design. Valid values are basic_autoencoder, unet and tiramisu")
parser.add_argument('-t', '--test_size', type=float, default=0.2, help="set fraction of input batches used for testing")
parser.add_argument('-z', '--num_files', type=int, default=390, help="set number of input files to be read in per block of training")
args = parser.parse_args()

if args.neural_net!='basic_autoencoder' and args.neural_net!='unet' and args.neural_net!='tiramisu':
   args.neural_net = 'basic_autoencoder'

if args.neural_net!='unet':
   if args.batch_size > 5:
      args.batch_size = 5
   if args.num_filter < 16:
      args.num_filter = 16

##
## Form the neural network
##

if args.neural_net == 'basic_autoencoder':
    model, ref_model = autoencoder( args.num_filter, args.num_gpu )

if args.neural_net == 'unet':
    model, ref_model = unet( args.num_filter, args.num_gpu )

model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001), metrics=['mae'])

if args.verbose != 0:
   model.summary()
   print( model.metrics_names )

##
## Set up the training of the model
##

filename = "model_weights_" + args.neural_net + "_" + str(args.num_filter) + "filters.h5"
if args.num_gpu == 1:
   checkpoint = ModelCheckpoint( filename, 
                                 monitor='val_mean_absolute_error', 
                                 save_best_only=True, 
                                 mode='min' )
else:
   checkpoint = AltModelCheckpoint( filename, ref_model )

earlystop = EarlyStopping( min_delta=0.001,
                           monitor='val_mean_absolute_error', 
                           patience=5,
                           mode='min' )

history = History()

my_callbacks = [earlystop, history]

##
## Get a list of input data files
##

input_file_list = []
cmd_str = '/group/director2107/mcheeseman/bom_data/2019/01/**/**/*.nc'
for fn in glob.iglob(cmd_str, recursive=True):
    input_file_list.append( fn )

input_file_list = list(dict.fromkeys(input_file_list))
shuffle( input_file_list )

input_file_list = input_file_list[ :20 ]

if args.verbose != 0:
   print('# of input files located: ', len(input_file_list))

##
## Perform the training
##

def read_input_file( filename ):
    fid = nc.Dataset( filename, 'r' )
    x = np.zeros((2050,2450,10))
    idx = 7
    for n in range(10):
        if idx<10:
           varname = 'channel_000' + str(idx) + '_brightness_temperature'
        else:
           varname = 'channel_00' + str(idx) + '_brightness_temperature'
        x[ :,:,n ] = np.array( fid[varname] )
        idx = idx + 1
    y = np.array( fid['precipitation'] )
    fid.close()
    return x, y

def perform_training_block( model, input_files, batch_size ):

    x = np.zeros((len(input_files),2050,2450,10))
    y = np.zeros((len(input_files),2050,2450,1))

    n = 0
    for fid in input_files:
        x[ n,:,:,: ], y[ n,:,:,0 ] = read_input_file( fid )
        n = n + 1
    
    hist = model.fit( x, y, 
                      batch_size=batch_size,
                      epochs=500, 
                      verbose=2, 
                      validation_split=.25,
                      callbacks=my_callbacks, 
                      shuffle=True )

    return hist

training_input_file_list = input_file_list[ :3000 ]
test_input_file_list = input_file_list[ 4000:4006 ]

training_errors = []
validation_errors = []

for i in range( 0,len(training_input_file_list),args.num_files ):
    j = i + args.num_files
    if j > len(input_file_list):
       j = len(input_file_list)

    t1 = datetime.now()
    hist = perform_training_block( model, training_input_file_list[i:j], args.batch_size )
    training_time = (datetime.now()-t1 ).total_seconds()
    print("   block of input files %4d - %4d trained for %7.1f seconds" % (i, j, training_time))

    print( len(hist.history['mean_absolute_error']) )
    training_errors.append( hist.history['mean_absolute_error'] )
    validation_errors.append( hist.history['val_mean_absolute_error'] )

print( len(training_errors) )
#print( training_errors.shape, validation_errors.shape )

plot_model_errors( args.neural_net, args.num_filter, training_errors, validation_errors )

##
## Compare radar data versus generated rainfall fields from trained neural net
##

#x = np.zeros((5,2050,2450,1))
#real_rainfall = np.zeros((5,2050,2450,1))

#for n in range( 5 ):
#    x[ n,:,:,0 ], real_rainfall[ n,:,:,0 ] = read_input_file( test_input_file_list[n] )

#fake_images = model.predict( x, batch_size=1, verbose=0 )

#plot_images( real_images, fake_images, args.neural_net, args.num_filter )

