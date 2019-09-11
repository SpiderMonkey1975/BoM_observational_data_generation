import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten 
from tensorflow.keras.utils import multi_gpu_model

def simple_net( input_dims ):
    ''' Python function that defines a simple encoder-decoder neural network design

        INPUT: image_dims  -> 2D array containing the input data dimensions as so:
                              dim[0] => number of nonzero significant data points
                              dim[1] => number of channels in satellite input 
    '''

    input_layer = Input(shape = (input_dims[0], input_dims[1]))
    net = Flatten()( input_layer )

    net = Dense( units=16, activation='relu' )( net )
    net = Dropout(0.2)( net )
    net = Dense( units=32, activation='relu' )( net )
    net = Dropout(0.2)( net )
    net = Dense( units=64, activation='relu' )( net )
    net = Dropout(0.2)( net )
    net = Dense( units=input_dims[0], activation='sigmoid' )( net )

    return Model( inputs=input_layer, outputs=net )

def simple_net_multigpu( input_dims, num_gpus ):
    ''' Python function that defines a simple encoder-decoder neural network design

        INPUT: image_dims  -> 2D array containing the input data dimensions as so:
                              dim[0] => number of nonzero significant data points
                              dim[1] => number of channels in satellite input
               num_gpus    -> number of GPUs to be used in the training
    '''
    with tf.device("/cpu:0"):
         model = simple_net( input_dims )
         model = multi_gpu_model( model, gpus=num_gpus )

    return model
