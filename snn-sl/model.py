import collections

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras.engine.training import Model


def build(batch_size,
          num_classes,
          optimizer='adam',
          loss='sparse_categorical_crossentropy',
          metrics=['accuracy']):
    """
    Build architecture
    """
    # Input shape:
    # (N, T, V, C)
    # (batch, timesteps, { joints, dimensions })
    # (87, 60, { 27, 3 })

    samples, timesteps, features = 87, 60, 54

    model = tf.keras.Sequential()

    # model.add(layers.BatchNormalization(axis=1))

    # TODO: try to skip invalid timesteps with Masking
    model.add(layers.Masking(mask_value=0., input_shape=(timesteps, features)))

    # model.add(layers.Embedding(input_dim=(60, 54), output_dim=16, mask_zero=True))

    # model.add(layers.Dense(24, input_dim=batch_size))

    # Recurrent layer
    # model.add(layers.LSTM(64, return_sequences=False,
    #                dropout=0.1, recurrent_dropout=0.1))
    model.add(layers.LSTM(32, return_sequences=True))
    model.add(layers.LSTM(128, return_sequences=True))
    model.add(layers.LSTM(256))

    # model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))

    # model.add(layers.RNN(nlstm.NestedLSTMCell(64, depth=2, input_shape=(60, 27, 3))))

    # Fully connected layer
    # model.add(layers.Dense(64, activation='relu'))

    # Regularization:
    # model.add(layers.Dropout(0.5))

    # Output layer:
    model.add(layers.Dense(num_classes))
    model.add(layers.Activation('softmax'))

    # ---------------------------------------------------------

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    # model.summary()
    # checkpointer = ModelCheckpoint(filepath=data_path + '/model-{epoch:02d}.hdf5', verbose=1)
    """
    model = tf.keras.Sequential()
    # Add an Embedding layer expecting input vocab of size 1000, and
    # output embedding dimension of size 64.
    model.add(layers.Embedding(input_dim=1000, output_dim=64))

    # Add a LSTM layer with 128 internal units.
    model.add(layers.LSTM(128))

    # Add a Dense layer with 10 units.
    model.add(layers.Dense(10))
    model.add(layers.Activation('softmax'))
    """
    """
    model = tf.keras.Sequential()
    # Add an Embedding layer expecting input vocab of size 1000, and
    # output embedding dimension of size 64.
    model.add(layers.Embedding(input_dim=1000, output_dim=64))

    # Add a LSTM layer with 128 internal units.
    model.add(layers.LSTM(128))

    # Add a Dense layer with 10 units.
    model.add(layers.Dense(10))

    model.summary()
    """

    return model


def train(model: Model, X_train, y_train, X_test, y_test, verbose=1):
    model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), epochs=10, verbose=verbose)

    # Evaluate:
    result = model.predict(X_train, batch_size=8, verbose=verbose)
    for value in result:
        print('%.1f' % np.argmax(value))


    # result = model.predict_classes(X_train, verbose=verbose)
    # for value in result:
    #     print('%.1f' % value)
