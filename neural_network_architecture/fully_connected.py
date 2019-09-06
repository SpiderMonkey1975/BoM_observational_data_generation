import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten 
from tensorflow.keras.utils import multi_gpu_model

def construct_model( input_layer, output_layer, num_gpu ):
    if num_gpu>1:
       with tf.device("/cpu:0"):
            model = Model( inputs=input_layer, outputs=output_layer )
            parallel_model = multi_gpu_model( model, gpus=num_gpu )
    else:
       model = Model( inputs=input_layer, outputs=output_layer )
       parallel_model = model
    return parallel_model, model

def simple_net( input_dims, num_gpus ):
    ''' Python function that defines a simple encoder-decoder neural network design

        INPUT: image_dims  -> 2D array containing the input data dimensions as so:
                              dim[0] => number of nonzero significant data points
                              dim[1] => number of channels in satellite input 
               num_gpus    -> # of GPUs used in training the network
    '''

    input_layer = Input(shape = (input_dims[0], input_dims[1]))
    net = Flatten()( input_layer )

    net = Dense( units=32, activation='relu' )( net )
    net = Dropout(0.2)( net )
    net = Dense( units=64, activation='relu' )( net )
    net = Dropout(0.2)( net )
    net = Dense( units=724016, activation='sigmoid' )( net )

    return construct_model( input_layer, net, num_gpus ) 

