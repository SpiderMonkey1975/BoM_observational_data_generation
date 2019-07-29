from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, ConvLSTM2D, BatchNormalization

def generate_model( num_filters, kernel, lat, lon ):
    model = Sequential()

    model.add( ConvLSTM2D(filters=num_filters, kernel_size=(kernel, kernel), input_shape=(None, lat, lon, 1), padding='same', return_sequences=True) )
    model.add( BatchNormalization() )

    for n in range(2):
        model.add( ConvLSTM2D(filters=num_filters, kernel_size=(kernel, kernel), padding='same', return_sequences=True) )
        model.add( BatchNormalization() )

    model.add( Conv3D(filters=1, kernel_size=(kernel, kernel, kernel), activation='sigmoid', padding='same', data_format='channels_last') )

    model.compile( loss='binary_crossentropy', optimizer='adadelta' )
    return model
