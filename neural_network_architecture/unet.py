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
            model = multi_gpu_model( model, gpus=num_gpu )
    else:
       model = Model( inputs=input_layer, outputs=output_layer )
    return model

##
##-------------  U-Net stuff ----------------------------------------------
##

def unet_encoder_block( net, num_filters ):
    net = MaxPooling2D(2)( net )
    net = BatchNormalization(axis=3)( net )
    net = Conv2D(num_filters, 3, strides=1, activation='relu', padding='same')(net)
    net = Conv2D(num_filters, 3, strides=1, activation='relu', padding='same')(net)
    return net

def unet_decoder_block( net, conv, num_filters ):
    net = UpSampling2D(2)(net)
    net = concatenate( [net,conv],axis=3 )
    net = Conv2D(num_filters, 3, strides=1, activation='relu', padding='same')(net)
    net = Conv2D(num_filters, 3, strides=1, activation='relu', padding='same')(net)
    net = BatchNormalization(axis=3)( net )
    return net

def unet( num_filters, num_gpus ):
    ''' Python function that defines U-Net architecture of an autoencoder neural network

        INPUT: num_filters -> # of filters in the first convolutional layer
               num_gpus    -> # of GPUs used in training the network
    '''
    input_layer = Input(shape = (2050, 2450, 1))

    conv_list = []

    net = Conv2D( 1, 3, strides=1, activation='relu', padding='same')(input_layer)
    net = Conv2D(num_filters, 3, strides=1, activation='relu', padding='same')(net)
    conv_list.append( Conv2D(num_filters, 3, strides=1, activation='relu', padding='same')(net) )

    for n in range(1,5):
        conv_list.append( unet_encoder_block( conv_list[n-1], num_filters*(2**n) ))

    net = conv_list[4]
    for n in range(3,-1,-1):
        net = unet_decoder_block( net, conv_list[n], num_filters*(2**n) )

    net = Conv2D(1, 3, strides=1, activation='relu', padding='same')(net)

    return construct_model( input_layer, net, num_gpus ) 

