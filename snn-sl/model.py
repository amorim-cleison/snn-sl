import collections

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers as l
from tensorflow.python.keras.engine.training import Model


def build(num_classes,
          input_shape,
          optimizer='adam',
          loss='sparse_categorical_crossentropy',
          metrics=['accuracy']):
    """
    Build architecture

    Input shape: (87, 3, 60, 27, 1)
    """
    model = agc_lstm(input_shape, num_classes)

    # Compilation:
    # model.build(input_shape=input_shape)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.summary()
    # checkpointer = ModelCheckpoint(filepath=data_path + '/model-{epoch:02d}.hdf5', verbose=1)

    return model


def agc_lstm(input_shape, num_classes):
    """
    Build architecture
    """
    layers = [
        l.Input(shape=input_shape),
        l.Permute((2, 4, 3, 1)),        # Input transformet to 'channels_last' (None, 60, 1, 27, 3)
        l.TimeDistributed(l.Flatten()), # Question: what is 'FC' layer here in paper?
        # FA                            # TODO: implement feature augmentation
        l.Masking(mask_value=0.),
        l.LSTM(40, return_sequences=True),
        
        l.AveragePooling1D(pool_size=3, strides=1, data_format='channels_last', name='avg_pool_1'),
        agc_lstm_cell(10, 'agc_lstm_1', True),

        l.AveragePooling1D(pool_size=3, strides=1, data_format='channels_last', name='avg_pool_2'),
        agc_lstm_cell(10, 'agc_lstm_2', True),

        l.AveragePooling1D(pool_size=3, strides=1, data_format='channels_last', name='avg_pool_3'),
        agc_lstm_cell(10, 'agc_lstm_3', False),

        l.Dense(num_classes),
        l.Activation('softmax')
    ]

    # for idx in range(0, num_hidden_layers):
    #     layers.insert(
    #         2 + idx,
    #         __acg_lstm_cell(
    #             40, return_sequences=True, name="agc_lstm_%0.0f" % (idx + 1)))

    return tf.keras.Sequential(layers, "agc_lstm")


def agc_lstm_cell(filters, name='agc_lstm_cell', return_sequences=True):
    layers = [
        l.Conv1D(int(filters * 0.5), kernel_size=3),        # TODO: implement GraphConvLSTM
        l.LSTM(filters, return_sequences=return_sequences)  # TODO: implement GraphConvLSTM
    ]
    return tf.keras.Sequential(layers, name)



def conv_lstm(input_shape, num_classes):
    """
    Implementation obtained from:
    https://medium.com/neuronio/an-introduction-to-convlstm-55c9025563a7

    Results:
    loss: 2.9957 - accuracy: 0.0862 - val_loss: 13.3078 - val_accuracy: 0.0455
    """
    trailer_input = l.Input(
        shape=input_shape, name='trailer_input')
    # shape = (60, 3, 1, 27)
    # permutation = l.Permute((1, 4, 2, 3))(trailer_input)
    first_ConvLSTM = l.ConvLSTM2D(
        filters=20,
        kernel_size=(3, 3),
        data_format='channels_last',
        recurrent_activation='hard_sigmoid',
        activation='tanh',
        padding='same',
        return_sequences=True)(trailer_input)
    first_BatchNormalization = l.BatchNormalization()(first_ConvLSTM)
    first_Pooling = l.MaxPooling3D(
        pool_size=(1, 2, 2), padding='same',
        data_format='channels_last')(first_BatchNormalization)

    second_ConvLSTM = l.ConvLSTM2D(
        filters=10,
        kernel_size=(3, 3),
        data_format='channels_last',
        padding='same',
        return_sequences=True)(first_Pooling)
    second_BatchNormalization = l.BatchNormalization()(second_ConvLSTM)
    second_Pooling = l.MaxPooling3D(
        pool_size=(1, 3, 3), padding='same',
        data_format='channels_last')(second_BatchNormalization)

    outputs = [conv_lstm_branch(second_Pooling, 'cat_{}'.format(c)) for c in range(num_classes)]

    merged = l.Concatenate()(outputs)

    seq = Model(inputs=trailer_input, outputs=merged, name='conv_lstm')

    return seq


def conv_lstm_branch(last_convlstm_layer, name):
    # branch_ConvLSTM = l.ConvLSTM2D(
    #     filters=5,
    #     kernel_size=(3, 3),
    #     data_format='channels_first',
    #     stateful=False,
    #     kernel_initializer='random_uniform',
    #     padding='same',
    #     return_sequences=True)(last_convlstm_layer)
    # branch_Pooling = l.MaxPooling3D(
    #     pool_size=(1, 2, 2), padding='same',
    #     data_format='channels_first')(branch_ConvLSTM)
    # flat_layer = l.TimeDistributed(l.Flatten())(branch_Pooling)

    # first_Dense = l.TimeDistributed(l.Dense(512, ))(flat_layer)
    # second_Dense = l.TimeDistributed(l.Dense(32, ))(first_Dense)

    # target = l.TimeDistributed(l.Dense(1), name=name)(second_Dense)

    branch_ConvLSTM = l.ConvLSTM2D(
        filters=5,
        kernel_size=(3, 3),
        data_format='channels_last',
        stateful=False,
        kernel_initializer='random_uniform',
        padding='same',
        return_sequences=False)(last_convlstm_layer)
    branch_Pooling = l.MaxPooling2D(
        pool_size=(2, 2), padding='same',
        data_format='channels_last')(branch_ConvLSTM)
    flat_layer = l.Flatten()(branch_Pooling)
    first_Dense = l.Dense(512, )(flat_layer)
    second_Dense = l.Dense(32, )(first_Dense)
    target = l.Dense(1, name=name)(second_Dense)
    return target


def convolutional_lstm(input_shape, num_classes, num_cells):
    """
    Implementation obtained from:
    https://keras.io/examples/conv_lstm/

    Results:
    loss: 0.7914 - accuracy: 0.8276 - val_loss: 3.1682 - val_accuracy: 0.0227

    Best accuracy:
    0.045977 (+/-0.130043) for 
    {'epochs': 10, 'input_shape': (60, 1, 27, 3), 'num_classes': 20, 'num_hidden_layers': 8}
    """
    layers = [
        l.Input(shape=input_shape),
        l.Masking(mask_value=0.),
        # --> 'convolutional_lstm_cells' <--
        l.Conv3D(
            filters=1,
            kernel_size=(3, 3, 3),
            activation='sigmoid',
            padding='same',
            data_format='channels_last'),
        l.Flatten(),
        l.Dense(num_classes),
        l.Activation('softmax')
    ]

    # Insert 'convolutional_lstm_cells'
    for cell in range(0, num_cells):
        layers.insert(
            2 + cell,
            convolutional_lstm_cell(
                40,
                return_sequences=True,
                name="convolutional_lstm_cell_%0.0f" % (cell + 1)))

    return tf.keras.Sequential(layers, "convolutional_lstm")


def convolutional_lstm_cell(units, return_sequences, name=None) -> Model:
    layers = [
        l.ConvLSTM2D(
            filters=units,
            return_sequences=return_sequences,
            kernel_size=(3, 3),
            padding='same',
            data_format='channels_last'),
        l.Dropout(0.5),
        l.BatchNormalization()
    ]
    return tf.keras.Sequential(layers, name)
