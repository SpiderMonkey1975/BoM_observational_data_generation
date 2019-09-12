from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, ConvLSTM2D, BatchNormalization

def convolutional_ltsm( num_filters, image_dims ):
   ''' Python function that creates a convolutional neural network in a LTSM framework.
       INPUT:  num_filters -> # of convolutions in the first layer of the LTSM
                image_dims -> 2D array containing the dimensions of the radar images
                              image_dims[0] => longitude dimension 
                              image_dims[1] => latitude dimension 
   '''
   kernel_dim = 2

   model = Sequential()
   model.add( ConvLSTM2D(filters=num_filters, 
                       kernel_size=kernel_dim, 
                       input_shape=(None, image_dims[1], image_dims[0], 1), 
                       padding='same', 
                       return_sequences=True) )
   model.add( BatchNormalization() )

   for n in range(3):
       num_filters = num_filters * 2
       model.add( ConvLSTM2D(filters=num_filters, kernel_size=kernel_dim, padding='same', return_sequences=True) )
       model.add( BatchNormalization() )

   model.add( Conv3D(filters=1, kernel_size=kernel_dim, activation='sigmoid', padding='same', data_format='channels_last') )

   model.compile( loss='binary_crossentropy', optimizer='adadelta' )
   return model 

