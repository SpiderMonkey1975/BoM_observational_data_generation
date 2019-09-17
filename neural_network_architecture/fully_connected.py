import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten 
from tensorflow.keras.utils import multi_gpu_model

def simple_net( input_dims, num_layers, num_units ):
    ''' Python function that defines a simple encoder-decoder neural network design

        INPUT: image_dims -> 2D array containing the input data dimensions as so:
                                dim[0] => number of nonzero significant data points
                                dim[1] => number of channels in satellite input 
               num_layers -> # of Dense fully-connected layers in the neural network
               num_units  -> # of nodes in the first Dense layer.  Node count is double in each
                             successive Dense layer.
    '''

    input_layer = Input(shape = (input_dims[0], input_dims[1]))
    net = Flatten()( input_layer )

    for n in range(num_layers):
        net = Dense( units=num_units, activation='relu' )( net )
        net = Dropout(0.2)( net )
        num_units = num_units * 2

    net = Dense( units=input_dims[0], activation='relu' )( net )
    return Model( inputs=input_layer, outputs=net )

def create_simple_net( input_dims, num_gpus, num_layers, num_units ):
    ''' Python function that defines a simple encoder-decoder neural network design

        INPUT: image_dims  -> 2D array containing the input data dimensions as so:
                              dim[0] => number of nonzero significant data points
                              dim[1] => number of channels in satellite input
               num_gpus    -> number of GPUs to be used in the training
               num_layers -> # of Dense fully-connected layers in the neural network
               num_units  -> # of nodes in the first Dense layer.  Node count is double in each
                             successive Dense layer.
    '''
    if num_gpus == 1:
         model =simple_net( input_dims, num_layers, num_units )
    else:
         with tf.device("/cpu:0"):
              model = simple_net( input_dims, num_layers, num_units )
              model = multi_gpu_model( model, gpus=num_gpus )

    return model

