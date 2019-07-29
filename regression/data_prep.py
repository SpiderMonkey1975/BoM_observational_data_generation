import numpy as np

##
## Get the input satellite reflectance data at three bands [8, 10, 14]
##---------------------------------------------------------------------

b7 = np.load("/scratch/director2107/CRS_Data/b7.npy")
b7 = (b7 - b7.mean()) / b7.std()
b8 = np.load("/scratch/director2107/CRS_Data/b8.npy")
b8 = (b8 - b8.mean()) / b8.std()
b9 = np.load("/scratch/director2107/CRS_Data/b9.npy")
b9 = (b9 - b9.mean()) / b9.std()
b10 = np.load("/scratch/director2107/CRS_Data/b10.npy")
b10 = (b10 - b10.mean()) / b10.std()
b11 = np.load("/scratch/director2107/CRS_Data/b11.npy")
b11 = (b11 - b11.mean()) / b11.std()
b12 = np.load("/scratch/director2107/CRS_Data/b12.npy")
b12 = (b12 - b12.mean()) / b12.std()
b13 = np.load("/scratch/director2107/CRS_Data/b13.npy")
b13 = (b13 - b13.mean()) / b13.std()
b14 = np.load("/scratch/director2107/CRS_Data/b14.npy")
b14 = (b14 - b14.mean()) / b14.std()
b15 = np.load("/scratch/director2107/CRS_Data/b15.npy")
b15 = (b15 - b15.mean()) / b15.std()
b16 = np.load("/scratch/director2107/CRS_Data/b16.npy")
b16 = (b16 - b16.mean()) / b16.std()
print("Raw data read and normalized")

##
## Stack raw reflectance datasets in depth to create X 
##-----------------------------------------------------

#x = np.stack((b8, b10, b14), axis=3)
x = np.stack((b7, b8, b9, b10, b11, b12, b13, b14, b15, b16), axis=3)

# Verify dimensions of the data. 
print(x.shape)

##
## Output the stacked input dataset
##---------------------------------

np.save( "input_10layer.npy", x )
