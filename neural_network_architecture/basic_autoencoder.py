from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, BatchNormalization, Conv2D, Conv2DTranspose

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
    return Model( inputs=input_layer, outputs=net )

