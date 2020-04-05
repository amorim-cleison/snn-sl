import collections

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras.engine.training import Model


def build(num_classes,
          input_shape,
          optimizer='adam',
          loss='sparse_categorical_crossentropy',
          metrics=['accuracy']):
    """
    Build architecture
    """

    # LSTM:
    # model = __build_lstm(batch_size, num_classes)

    # Conv LSTM:
    # model = __build_conv_lstm(batch_size, num_classes)

    # AGC LSTM:
    model = __build_agc_lstm(input_shape, num_classes)

    # Compilation:
    # model.build(input_shape=input_shape)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.summary()
    # checkpointer = ModelCheckpoint(filepath=data_path + '/model-{epoch:02d}.hdf5', verbose=1)

    return model


def __build_lstm(batch_size, num_classes):
    """
    Build architecture
    """
    # Input shape:
    # (N, T, V, C)
    # (batch, timesteps, { joints, dimensions })
    # (87, 60, { 27, 3 })

    samples, timesteps, features = 87, 60, 54

    model = tf.keras.Sequential()

    # Skip invalid timesteps with Masking
    model.add(layers.Masking(mask_value=0.))

    model.add(layers.BatchNormalization(axis=1))

    # model.add(layers.Embedding(input_dim=(60, 54), output_dim=16, mask_zero=True))

    # model.add(layers.Dense(24, input_dim=batch_size))

    # Recurrent layer
    # model.add(layers.LSTM(64, return_sequences=False,
    #                dropout=0.1, recurrent_dropout=0.1))
    model.add(layers.LSTM(32, return_sequences=True))
    # model.add(layers.Conv1D(64, kernel_size=5))
    #     model.add(layers.Activation('relu'))
    #     model.add(layers.Dropout(0.5))

    model.add(layers.LSTM(128, return_sequences=True))

    model.add(layers.LSTM(256))

    # model.add(layers.RNN(nlstm.NestedLSTMCell(64, depth=2, input_shape=(60, 27, 3))))

    # Fully connected layer
    # model.add(layers.Dense(64, activation='relu'))

    # Regularization:
    # model.add(layers.Dropout(0.5))

    # Output layer:
    model.add(layers.Dense(num_classes))
    model.add(layers.Activation('softmax'))

    # ---------------------------------------------------------
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


def __build_agc_lstm(input_shape, num_classes):
    """
    Build architecture
    """
    model_layers = [
        layers.Input(shape=input_shape),
        # layers.TimeDistributed(layers.Flatten()),
        
        layers.Masking(mask_value=0.),

        # layers.Dense(64),
        # FA
        # layers.LSTM(128, return_sequences=True),

        # TAP:
        # layers.AveragePooling1D(pool_size=3, strides=3, data_format='channels_last'),

        # AGC LSTM 1:
        __acg_lstm_cell(40, return_sequences=True, name="agc_lstm_1"),

        # TAP:
        # layers.AveragePooling1D(pool_size=3, strides=3, data_format='channels_last'),
        __acg_lstm_cell(40, return_sequences=True, name="agc_lstm_2"),

        # TAP:
        # layers.AveragePooling1D(pool_size=3, strides=3, data_format='channels_last'),
        __acg_lstm_cell(40, return_sequences=True, name="agc_lstm_3"),


        layers.Conv3D(
                filters=1, 
                kernel_size=(3, 3, 3),
                activation='sigmoid',
                padding='same', 
                data_format='channels_last'),


        # layers.Convolution2D(
        #     filters=128,
        #     kernel_size=3,
        #     strides=1,
        #     padding='same',
        #     data_format="channels_last"),
        # layers.Flatten(data_format="channels_last", input_shape=input_shape),
        # Skip invalid timesteps with Masking
        # layers.BatchNormalization(axis=1),

        layers.TimeDistributed(layers.Flatten()),
        layers.Flatten(),

        # Output layer:
        layers.Dense(num_classes),
        layers.Activation('softmax')
    ]
    return tf.keras.Sequential(model_layers, "agc_lstm")

def __acg_lstm_cell(units, return_sequences, name=None) -> Model:
    agc_lstm_layers = [
        layers.ConvLSTM2D(
            filters=units,
            return_sequences=return_sequences,
            kernel_size=(3, 3),
            padding='same',
            data_format='channels_last'),
        layers.Dropout(0.5),
        layers.BatchNormalization()
    ]
    # Accuracy: ~ 0.55
    # agc_lstm_layers = [
    #     layers.Dense(64),
    #     layers.LSTM(units, return_sequences=return_sequences),
    # ]
    return tf.keras.Sequential(agc_lstm_layers, name)


def __build_conv_lstm(batch_size, num_classes):
    samples, timesteps, features = 87, 60, 54

    model = tf.keras.Sequential()

    model.add(
        layers.ConvLSTM2D(
            64,
            kernel_size=(3, 3),
            padding='valid',
            data_format='channels_first',
            input_shape=(timesteps, 1, 27, 3)))

    # Output layer:
    model.add(layers.Dense(num_classes))
    model.add(layers.Activation('softmax'))

    return model