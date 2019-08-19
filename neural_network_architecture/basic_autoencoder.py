import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, Dense, Reshape, multiply, add, Dropout, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Input, BatchNormalization, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, UpSampling2D
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.regularizers import l2


def construct_model( input_layer, output_layer, num_gpu ):
    if num_gpu>1:
       with tf.device("/cpu:0"):
            model = Model( inputs=input_layer, outputs=output_layer )
            parallel_model = multi_gpu_model( model, gpus=num_gpu )
    else:
       model = Model( inputs=input_layer, outputs=output_layer )
       parallel_model = model
    return parallel_model, model

def autoencoder( image_dims, num_filters, num_gpus ):
    ''' Python function that defines a simple encoder-decoder neural network design

        INPUT: image_dims  -> 3D array containing the input image dimensions as so:
                              dim[0] => image width
                              dim[1] => image height
                              dim[2] => number of channels in image 
               num_filters -> # of filters in the first convolutional layer
               num_gpus    -> # of GPUs used in training the network
    '''

    #input_layer = Input(shape = (2050, 2450, 10))
    input_layer = Input(shape = (image_dims[0], image_dims[1], image_dims[2]))
    net = BatchNormalization(axis=3)( input_layer )

    # Encoding section
    net = Conv2D( num_filters, 5, strides=2, activation='relu', padding='valid')( net )
    net = BatchNormalization(axis=3)( net )
    net = Conv2D( num_filters*2, 3, strides=5, activation='relu', padding='valid')( net )
    net = BatchNormalization(axis=3)( net )
    net = Conv2D( num_filters*4, 5, strides=5, activation='relu', padding='valid')( net )
    net = BatchNormalization(axis=3)( net )

    # Decoding section
    net = Conv2DTranspose( num_filters*2, 5, strides=5, activation='relu', padding='valid')( net )
    net = BatchNormalization(axis=3)( net )
    net = Conv2DTranspose( num_filters, 3, strides=5, activation='relu', padding='valid')( net )
    net = BatchNormalization(axis=3)( net )
    net = Conv2DTranspose( 1, 5, strides=2, activation='relu', padding='valid')( net )
    net = BatchNormalization(axis=3)( net )

    net = Conv2D( 1, 4, strides=1, activation='relu', padding='valid')( net )
    return construct_model( input_layer, net, num_gpus ) 

