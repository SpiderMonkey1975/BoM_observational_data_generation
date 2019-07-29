from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dropout, LeakyReLU, Dense, Flatten

def two_layer_classifier( num_filters ):
    """Create a simple CNN classification network.

       Parameters:
       num_filters(int): number of convolutional filters used in the first CNN layer of the network

    """
    model = Sequential()
    model.add( Conv2D(num_filters, 5, strides=2, padding='same', input_shape=(400,400,1)) )
    model.add( LeakyReLU(alpha=0.2) )
    model.add( Dropout(0.3) )

    model.add( Conv2D(2*num_filters, 5, strides=2, padding='same') )
    model.add( LeakyReLU(alpha=0.2) )
    model.add( Dropout(0.3) )

    model.add( Flatten() )
    model.add( Dense(units=1,activation='sigmoid') )
    return model
