import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, BatchNormalization, Conv2D, concatenate, UpSampling2D
from tensorflow.keras.utils import multi_gpu_model


def unet( image_dims, num_filters ):
    ''' Python function that defines a modified U-Net autoencoder neural net architecture

        INPUT: image_dims  -> 3D array containing the input image dimensions as so:
                              dim[0] => image width
                              dim[1] => image height
                              dim[2] => number of channels in image 
               num_filters -> # of filters in the first convolutional layer
    '''

    input_layer = Input(shape = (images_dims[0], image_dims[1], image_dims[2]))

    net = Conv2D( 1, 3, strides=1, activation='relu', padding='same')(input_layer)
    net = Conv2D(num_filters, 3, strides=1, activation='relu', padding='same')(net)
    conv1 = Conv2D(num_filters, 3, strides=1, activation='relu', padding='same')(net)

    net = MaxPooling2D(2)( conv1 )
    net = BatchNormalization(axis=3)( net )
    net = Conv2D(num_filters*2, 3, strides=1, activation='relu', padding='same')(net)
    conv2 = Conv2D(num_filters*2, 3, strides=1, activation='relu', padding='same')(net)

    net = MaxPooling2D(5)( conv2 )
    net = BatchNormalization(axis=3)( net )
    net = Conv2D(num_filters*4, 3, strides=1, activation='relu', padding='same')(net)
    conv3 = Conv2D(num_filters*4, 3, strides=1, activation='relu', padding='same')(net)

    net = MaxPooling2D(5)( conv3 )
    net = BatchNormalization(axis=3)( net )
    net = Conv2D(num_filters*8, 3, strides=1, activation='relu', padding='same')(net)

    net = UpSampling2D(5)(net)
    net = concatenate( [net,conv3],axis=3 )
    net = Conv2D(num_filters*4, 3, strides=1, activation='relu', padding='same')(net)
    net = Conv2D(num_filters*4, 3, strides=1, activation='relu', padding='same')(net)
    net = BatchNormalization(axis=3)( net )

    net = UpSampling2D(5)(net)
    net = concatenate( [net,conv2],axis=3 )
    net = Conv2D(num_filters*2, 3, strides=1, activation='relu', padding='same')(net)
    net = Conv2D(num_filters*2, 3, strides=1, activation='relu', padding='same')(net)
    net = BatchNormalization(axis=3)( net )

    net = UpSampling2D(2)(net)
    net = concatenate( [net,conv1],axis=3 )
    net = Conv2D(num_filters, 3, strides=1, activation='relu', padding='same')(net)
    net = Conv2D(num_filters, 3, strides=1, activation='relu', padding='same')(net)
    net = BatchNormalization(axis=3)( net )

    net = Conv2D(1, 3, strides=1, activation='relu', padding='same')(net)

    return Model( inputs=input_layer, outputs=net )


def create_unet( input_dims, num_filters, num_gpus ):
    ''' Python function that defines a simple encoder-decoder neural network design

        INPUT: image_dims  -> 2D array containing the input data dimensions as so:
                              dim[0] => number of nonzero significant data points
                              dim[1] => number of channels in satellite input
               num_filters -> # of filters in the first convolutional layer
               num_gpus    -> number of GPUs to be used in the training
    '''
    if num_gpus > 1:
       with tf.device("/cpu:0"):
            model = unet( input_dims, num_filters )
            model = multi_gpu_model( model, gpus=num_gpus )
    else:
       model = unet( input_dims, num_filters )

    return model

