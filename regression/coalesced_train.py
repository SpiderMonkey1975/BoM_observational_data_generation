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
from fully_connected import simple_net

sys.path.insert(0, '/group/director2107/mcheeseman/BoM_observational_data_generation/plotting_routines')
from plotting_routines import plot_fc_model_errors


##
## Look for any user specified commandline arguments
##

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--num_gpu', type=int, default=4, help="set number of GPUs to be used for training")
parser.add_argument('-b', '--batch_size', type=int, default=10, help="set the batch size used in training")
parser.add_argument('-t', '--tolerance', type=float, default=0.001, help="set the tolerance used for the early stopping callback")
args = parser.parse_args()

##
## Form the neural network
##

input_dims = np.empty((2,),dtype=np.int)
input_dims[0] = 724016 
input_dims[1] = 10

model, ref_model = simple_net( input_dims, args.num_gpu )
model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001), metrics=['mae'])

#model.summary()
#print( model.metrics_names )
#sys.exit()

##
## Set up the training of the model
##

history = History()

earlystop = EarlyStopping( min_delta=0.0001,
                           monitor='val_mean_absolute_error', 
                           patience=5,
                           mode='min' )

filename = "model_weights_fully_connected.h5"
if args.num_gpu == 1:
   checkpoint = ModelCheckpoint( filename, 
                                 monitor='val_mean_absolute_error', 
                                 save_best_only=True, 
                                 mode='min' )
   my_callbacks = [checkpoint, earlystop, history]
else:
   my_callbacks = [earlystop, history]

##
## Get a list of input data files
##

input_file_list = []
cmd_str = '/group/director2107/mcheeseman/bom_data/input_*.nc'
for fn in glob.iglob(cmd_str, recursive=True):
    input_file_list.append( fn )

input_file_list = list(dict.fromkeys(input_file_list))
shuffle( input_file_list )

num_training_images = len(input_file_list) - 5 
training_input_file_list = input_file_list[ :num_training_images ]
test_input_file_list = input_file_list[ num_training_images: ]

##
## Perform the training
##

def read_input_file( filename ):
    fid = nc.Dataset( filename, 'r' )
    x = np.array( fid['coalesced_brightness'] )
    y = np.array( fid['coalesced_precipitation'] )
  
    x = x[ np.newaxis,:,: ]
    y = y[ np.newaxis,: ]

    fid.close()
    return x, y


x = np.empty((num_training_images,input_dims[0],10))
y = np.empty((num_training_images,input_dims[0]))

t1 = datetime.now()
for n in range(num_training_images):
    x[ n,:,: ], y[ n,: ] = read_input_file( training_input_file_list[n] )
io_time = (datetime.now()-t1 ).total_seconds()
   
hist = model.fit( x, y, 
                  batch_size=args.batch_size,
                  epochs=500, 
                  verbose=2, 
                  validation_split=.125,
                  callbacks=my_callbacks, 
                  shuffle=False )
training_time = (datetime.now()-t1 ).total_seconds()

print("   training took %7.1f seconds" % training_time)
print("   I/O took %7.1f seconds" % io_time)

plot_fc_model_errors( 'simple_fully_connected', hist )

