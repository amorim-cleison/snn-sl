import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers as l
from tensorflow.python.keras.engine.training import Model

from third_party.keras_dgl.layers import GraphConvLSTM
from models.base_model import BaseModel


class ConvLSTM(BaseModel):
    """
    ConvLSTM implementation obtained from:
    https://medium.com/neuronio/an-introduction-to-convlstm-55c9025563a7

    Results:
    loss: 2.9957 - accuracy: 0.0862 - val_loss: 13.3078 - val_accuracy: 0.0455
 
    """
    def __init__(self, input_shape, num_classes):
        super().__init__(input_shape, num_classes)

    def build(self):
        trailer_input = l.Input(shape=self.input_shape, name='trailer_input')
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

        outputs = [
            self.conv_lstm_branch(second_Pooling, 'cat_{}'.format(c))
            for c in range(self.num_classes)
        ]

        merged = l.Concatenate()(outputs)

        seq = Model(inputs=trailer_input, outputs=merged, name='conv_lstm')

        return seq

    def conv_lstm_branch(self, last_convlstm_layer, name):
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
