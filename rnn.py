from keras.layers import Dense, Dropout, LSTM, Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder


def load_data(filename):
    print('Loading data...')
    df = pd.read_csv(filename)
    df = df.reindex(np.random.permutation(df.index))
    X = np.array([list(map(float, df['dur'].values)),
                  list(map(float, df['stddev'].values)),
                  list(map(float, df['min'].values)),
                  list(map(float, df['mean'].values)),
                  list(map(float, df['spkts'].values)),
                  list(map(float, df['dpkts'].values)),
                  list(map(float, df['sbytes'].values)),
                  list(map(float, df['dbytes'].values)),
                  list(map(float, df['max'].values)),
                  list(map(float, df['sum'].values))])
    X = np.transpose(X)
    print(X.shape)
    y = df['label']
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    print(X.shape)
    le = LabelEncoder()

    y = le.fit_transform(y)
    y = np.array(list(map(int, y)))
    return pad_sequences(X), y


def create_model(input_length):
    print('Creating model...')
    print(input_length)
    model = Sequential()
    # model.add(Embedding(input_dim=500000, output_dim=50, input_length=input_length))
    model.add(LSTM(256, input_shape=(1, 10), activation='sigmoid',
                   return_sequences=True, recurrent_activation='hard_sigmoid'))
    model.add(LSTM(256, activation='sigmoid', recurrent_activation='hard_sigmoid', dropout=0.2))
    model.add(Dense(6, activation='softmax'))

    print('Compiling...')
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


X_train, y_train = load_data('sdn_datasets/train/train.200.csv')
X_test, y_test = load_data('sdn_datasets/test/test.10000.csv')

model = create_model(len(X_train[0]))

print('Fitting model...')
hist = model.fit(X_train, y_train, batch_size=64, epochs=100, validation_split=0.3, verbose=1)

y_pred = model.predict_classes(X_test)
print(y_pred)
print(y_test)


print('Test accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, labels=[0, 1, 2, 3, 4, 5]))
