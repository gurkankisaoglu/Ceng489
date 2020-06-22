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
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

batch_size = 32
num_classes = 6
epochs = 2
data_augmentation = False

# The data, shuffled and split between train and test sets:
import numpy as np

traffic = pd.read_csv("sdn_datasets/train/train.400.csv")
testtraffic = pd.read_csv("sdn_datasets/validation/val.100.csv")

x_test = testtraffic.drop(["label"], axis=1)
x_train = traffic.drop(["label"], axis=1)
y_test = testtraffic['label']
y_train = traffic['label']

y_train[y_train == "DDoS"] = 0
y_train[y_train == "Port_Scanning"] = 1
y_train[y_train == "OS_and_Service_Detection"] = 2
y_train[y_train == "Normal"] = 3
y_train[y_train == "Fuzzing"] = 4
y_train[y_train == "DoS"] = 5

y_test[y_test == "DDoS"] = 0
y_test[y_test == "Port_Scanning"] = 1
y_test[y_test == "OS_and_Service_Detection"] = 2
y_test[y_test == "Normal"] = 3
y_test[y_test == "Fuzzing"] = 4
y_test[y_test == "DoS"] = 5

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

y_train = y_train.reshape(2400)
y_test = y_test.reshape(600)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

inp = keras.layers.Input(shape=(10,))

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= np.max(x_train)
x_test /= np.max(x_test)

encoding_dim = 6
origin_dim = 10

# load autoencoder model
encoder = load_model('trained_encoder_model.h5')



tr_encoded_imgs = encoder.predict(x_train)
print(tr_encoded_imgs.shape)

te_encoded_imgs = encoder.predict(x_test)
print(te_encoded_imgs.shape)

inp2 = keras.layers.Input(shape=(encoding_dim,))

x = BatchNormalization()(inp2)

x = Dense(10, kernel_regularizer=keras.regularizers.l2(0.01))(x)
# x = BatchNormalization()(x)
x = Activation('relu')(x)
# x = Activation('tanh')(x)
#x = Activation('sigmoid')(x)

x = Dense(10, kernel_regularizer=keras.regularizers.l2(0.01))(x)
# x = BatchNormalization()(x)
x = Activation('relu')(x)
# x = Activation('tanh')(x)
#x = Activation('sigmoid')(x)

# x = Dense(8, kernel_regularizer=keras.regularizers.l2(0.01))(x)
##x = BatchNormalization()(x)
# x = Activation('relu')(x)

x = Dense(num_classes)(x)
y = Activation('softmax')(x)

model = keras.models.Model(inputs=inp2, outputs=y)

# model.add(Dropout(.4))

# model.add(Dense(num_classes))
# model.add(Activation('softmax'))

# initiate RMSprop optimizer
# opt = keras.optimizers.rmsprop(lr=0.000001, decay=1e-6)
opt = keras.optimizers.Adam(lr=.00001)
# opt = keras.optimizers.SGD(lr=.0000001,momentum=0.9,nesterov=True)

# model = load_model('my_model.h5')

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

history = model.fit(tr_encoded_imgs, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(te_encoded_imgs, y_test),  # validation_split=.3, #
                    shuffle=True)

y_pred = model.predict(te_encoded_imgs)
y_pred = np.argmax(y_pred,axis=1)
y_test = np.argmax(y_test,axis=1)

print(classification_report(y_test, y_pred, labels=[0, 1, 2, 3, 4, 5]))

print(history.history.keys())

# # summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# # summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
