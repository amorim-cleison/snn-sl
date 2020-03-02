import collections

import numpy as np
import tensorflow as tf
from keras import losses, metrics, optimizers
from tensorflow.keras import layers
from tensorflow.python.keras.engine.training import Model


def build(batch_size,
          num_classes,
          optimizer='adam',
          loss='categorical_crossentropy',
          metrics=['accuracy']):
    """
    Build architecture
    """
    # Input shape:
    # (N, T, V, C)
    # (batch, timesteps, { joints, dimensions })
    # (87, 60, { 27, 3 })

    model = tf.keras.Sequential()

    # model.add(layers.Embedding(input_dim=num_classes, output_dim=2745))

    # model.add(layers.Dense(24, input_dim=batch_size))

    # Recurrent layer
    # model.add(layers.LSTM(64, return_sequences=False,
    #                dropout=0.1, recurrent_dropout=0.1))
    model.add(layers.LSTM(32, return_sequences=True))

    model.add(layers.LSTM(32, return_sequences=True))

    model.add(layers.LSTM(32))

    # model.add(layers.RNN(nlstm.NestedLSTMCell(64, depth=2, input_shape=(60, 27, 3))))

    # Fully connected layer
    # model.add(layers.Dense(64, activation='relu'))

    # Regularization:
    # model.add(layers.Dropout(0.5))

    # Output layer:
    model.add(layers.Dense(num_classes, activation='softmax'))
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


def train(model: Model, x, y, verbose=1):
    model.fit(x=x, y=y, epochs=10, batch_size=8, verbose=verbose)

    # Evaluate:
    result = model.predict(x, batch_size=8, verbose=verbose)
    for value in result:
        print('%.1f' % np.argmax(result[0]))
