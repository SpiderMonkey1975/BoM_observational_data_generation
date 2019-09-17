import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, BatchNormalization, Conv2D, Conv2DTranspose
from tensorflow.keras.utils import multi_gpu_model

def autoencoder( image_dims, num_filters ):
    ''' Python function that defines a simple encoder-decoder neural network design

        INPUT: image_dims  -> 3D array containing the input image dimensions as so:
                              dim[0] => image width
                              dim[1] => image height
                              dim[2] => number of channels in image 
               num_filters -> # of filters in the first convolutional layer
    '''

    input_layer = Input(shape = (image_dims[0], image_dims[1], image_dims[2]))
    net = BatchNormalization(axis=3)( input_layer )

    # Encoding section
    net = Conv2D( num_filters, 5, strides=5, activation='relu', padding='same')( net )
#    net = BatchNormalization(axis=3)( net )
    net = Conv2D( num_filters*2, 5, strides=5, activation='relu', padding='same')( net )
    net = BatchNormalization(axis=3)( net )
    net = Conv2D( num_filters*4, 2, strides=2, activation='relu', padding='same')( net )
#    net = BatchNormalization(axis=3)( net )
    net = Conv2D( num_filters*8, 2, strides=1, activation='relu', padding='same')( net )
    net = Conv2D( num_filters*16, 2, strides=1, activation='relu', padding='same')( net )

    # Decoding section
    net = Conv2DTranspose( num_filters*8, 2, strides=1, activation='relu', padding='same')( net )
#    net = BatchNormalization(axis=3)( net )
    net = Conv2DTranspose( num_filters*4, 2, strides=2, activation='relu', padding='same')( net )
    net = BatchNormalization(axis=3)( net )
    net = Conv2DTranspose( num_filters*2, 5, strides=5, activation='relu', padding='same')( net )
#    net = BatchNormalization(axis=3)( net )
    net = Conv2DTranspose( 1, 5, strides=5, activation='relu', padding='same')( net )
    net = BatchNormalization(axis=3)( net )

    net = Conv2D( 1, 4, strides=1, activation='relu', padding='same')( net )
    return Model( inputs=input_layer, outputs=net )


def autoencoder_multigpu( input_dims, num_filters, num_gpus ):
    ''' Python function that defines a simple encoder-decoder neural network design

        INPUT: image_dims  -> 2D array containing the input data dimensions as so:
                              dim[0] => number of nonzero significant data points
                              dim[1] => number of channels in satellite input
               num_filters -> # of filters in the first convolutional layer
               num_gpus    -> number of GPUs to be used in the training
    '''
    with tf.device("/cpu:0"):
         model = autoencoder( input_dims, num_filters )
         model = multi_gpu_model( model, gpus=num_gpus )

    return model
