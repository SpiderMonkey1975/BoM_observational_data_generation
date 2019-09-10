from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten 

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

