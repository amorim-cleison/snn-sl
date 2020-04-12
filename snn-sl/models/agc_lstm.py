import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers as l
from tensorflow.python.keras.engine.training import Model

from third_party.keras_dgl.layers import GraphConvLSTM
from models.base_model import BaseModel


class AttentionGraphConvLSTM(BaseModel):
    """
    AGC-LSTM implementation as presented in:
    'An Attention Enhanced Graph Convolutional LSTM Network for Skeleton-Based Action Recognition' 
    https://arxiv.org/abs/1902.09130

    Partial results:
    loss: 1.7926 - accuracy: 0.4483 - val_loss: 2.8913 - val_accuracy: 0.2273

    """

    def __init__(self, input_shape, num_classes):
        super().__init__(input_shape, num_classes)

    def build(self):
        input = l.Input(shape=self.input_shape)
        permute = l.Permute((2, 4, 3, 1))(
            input)  # Transformed to 'channels_last' (None, 60, 1, 27, 3)
        flatten = l.TimeDistributed(
            l.Flatten(), name='flatten')(
                permute)  # Question: what is 'FC' layer here in paper?

        fa = l.Lambda(self.agc_lstm_feature_augmentation, name='fa')(flatten)

        mask = l.Masking(mask_value=0.)(fa)

        normalize = l.BatchNormalization(axis=-1)(mask)

        lstm_1 = l.LSTM(40, return_sequences=True)(normalize)

        tap_1 = l.AveragePooling1D(
            pool_size=3, strides=1, data_format='channels_last',
            name='tap_1')(lstm_1)
        agc_lstm_1 = self.agc_lstm_cell(
            tap_1, filters=40, name='agc_lstm_1', return_sequences=True)

        tap_2 = l.AveragePooling1D(
            pool_size=3, strides=1, data_format='channels_last',
            name='tap_2')(agc_lstm_1)
        agc_lstm_2 = self.agc_lstm_cell(
            tap_2, filters=40, name='agc_lstm_2', return_sequences=True)

        tap_3 = l.AveragePooling1D(
            pool_size=3, strides=1, data_format='channels_last',
            name='tap_3')(agc_lstm_2)
        agc_lstm_3 = self.agc_lstm_cell(
            tap_3, filters=40, name='agc_lstm_3', return_sequences=False)

        fc = l.Dense(self.num_classes)(agc_lstm_3)
        activation = l.Activation('softmax')(fc)

        return Model(inputs=input, outputs=activation, name='agc_lstm')

    def agc_lstm_cell(self,
                      input,
                      filters,
                      name='agc_lstm_cell',
                      return_sequences=True):
        conv_1 = l.Conv1D(
            int(filters * 0.5), kernel_size=3, name='{}_conv_1'.format(name))(
                input)  # TODO: implement GraphConvLSTM
        lstm_1 = l.LSTM(
            filters,
            return_sequences=return_sequences,
            name='{}_lstm_1'.format(name))(
                conv_1)  # TODO: implement GraphConvLSTM
        return lstm_1

    @tf.function
    def agc_lstm_feature_augmentation(self, input):
        """
        'Concatenate spatial feature and feature difference between two consecutive frames 
        to compose an augmented feature.'
        (2019 - Si et al - An Attention Enhanced Graph Convolutional 
        LSTM Network for Skeleton-Based Action Recognition)
        """
        # tf.config.experimental_run_functions_eagerly(True)
        # assert tf.executing_eagerly()

        # Difference between consecutive frames:
        rolled = tf.roll(input, shift=-1, axis=1)
        diff = tf.subtract(rolled, input)

        # Concatenate spatial feature and feature difference:
        concat = tf.concat([input, diff], axis=-1)

        return concat
