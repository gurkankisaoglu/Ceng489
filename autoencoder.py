from __future__ import print_function
import keras
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import load_model

from keras.layers import Input, Dense
from keras.models import Model
from keras.models import load_model

# import matplotlib.pyplot as plt
data_augmentation = False

# The data, shuffled and split between train and test sets:
import numpy as np
import h5py as hp

traffic = pd.read_csv("sdn_datasets/train/train.400.csv")
testtraffic = pd.read_csv("sdn_datasets/validation/val.100.csv")
#testtraffic = pd.read_csv("sdn_datasets/test/test.10000.csv")

x_test = testtraffic.drop(["label"], axis=1)
x_train = traffic.drop(["label"], axis=1)
y_test = testtraffic['label']
y_train = traffic['label']



print('x_train shape:', x_train.shape)
print(x_train.shape, 'train samples')
print(x_test.shape, 'test samples')



# inp = keras.layers.Input(shape=(47*47*47,))
# inp = keras.layers.Input(shape=(22917,))
inp = keras.layers.Input(shape=(10,))

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= np.max(x_train)
x_test /= np.max(x_test)

##### Autoencoder starts #####
encoding_dim = 6
origin_dim = 10
# "encoded" is the encoded representation of the input
x = BatchNormalization()(inp)
print(x.shape)
# encoded = Dense(encoding_dim, activation='relu')(x)
x = Dense(10, activation='relu')(x)
print(x.shape)
x = Dense(8, activation='relu')(x)
print(x.shape)
encoded = Dense(encoding_dim, activation='relu')(x)
print(encoded.shape)

# "decoded" is the lossy reconstruction of the input
decoded = Dense(encoding_dim, activation='relu')(encoded)
print(decoded.shape)
decoded = Dense(8, activation='relu')(decoded)
print(decoded.shape)
decoded = Dense(10, activation='relu')(decoded)
print(decoded.shape)
decoded = Dense(origin_dim, activation='sigmoid')(decoded)
print(decoded.shape)
# this model maps an input to its reconstruction
autoencoder = Model(input=inp, output=decoded)
# this model maps an input to its encoded representation
encoder = Model(input=inp, output=encoded)


autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# fit our autoencoder!
history = autoencoder.fit(x_train, x_train,
                epochs=100,
                batch_size=16,
                shuffle=True,
                validation_data=(x_test, x_test))
##### Autoencoder ends	#####

import matplotlib.pyplot as plt
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#Save autoencoder model
encoder.save('trained_encoder_model.h5')




