import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, BatchNormalization, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, UpSampling2D
from tensorflow.keras.models import Model

def autoencoder( num_filters, num_layers, lat, lon ):
    '''Python function that creates an instance of a basic autoencoder neural network.  The network
       consists of (num_layers) of encoding/contracting layers followed by (num_layers) of decoding/expanding
       layers.  The first encoding layer possesses a convolution with (num_filters) filters. The number of
       filters double in each successive encoding layer.  The number of filters are halved in each 
       consecutive decoding layer.

       INPUT: num_filters -> # of filters in the first convolution layer
              num_layers  -> # of encoding (and decoding) layers in the neural network
              lat,lon     -> dimensions of the input images
    '''
    input_layer = Input(shape = (lat, lon, 1))
    net = BatchNormalization(axis=3)( input_layer )

    if num_layers>3:
       print("WARNING: a maximum 3-layer deep autoencoder can be used")
       print("         resetting number of layers to 3")
       num_layers = 3

    # Encoding section
    for n in range(num_layers):
        filter_cnt = num_filters * (2**n)
        if n == 0 or n == (num_layers-1):
           kernel_size = 5
        else:
           kernel_size = 3
        net = Conv2D( filter_cnt, kernel_size, strides=2, activation='relu', padding='same')( net )
        net = BatchNormalization(axis=3)( net )

    # Decoding section
    for n in range(num_layers):
        filter_cnt = num_filters * (2**(num_layers-1-n))
        if n == 0:
           kernel_size = 5
        elif n == (num_layers-1):
           kernel_size = 5
           filter_cnt = 1
        else:
           kernel_size = 3
        net = Conv2DTranspose( filter_cnt, kernel_size, strides=2, activation='relu', padding='same')( net )
        if n != (num_layers-1):
           net = BatchNormalization(axis=3)( net )

    return Model( inputs=input_layer, outputs=net ) 


