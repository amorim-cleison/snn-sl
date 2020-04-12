import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers as l
from tensorflow.python.keras.engine.training import Model

from models.base_model import BaseModel


class ConvolutionalLSTM(BaseModel):
    """
    Convolutional LSTM implementation obtained from:
    https://keras.io/examples/conv_lstm/

    Results:
    loss: 0.7914 - accuracy: 0.8276 - val_loss: 3.1682 - val_accuracy: 0.0227

    Best accuracy:
    0.045977 (+/-0.130043) for 
    {'epochs': 10, 'input_shape': (60, 1, 27, 3), 'num_classes': 20, 'num_hidden_layers': 8}
    """
    def __init__(self, input_shape, num_classes):
        super().__init__(input_shape, num_classes)

    def build(self, num_cells=4):
        layers = [
            l.Input(shape=self.input_shape),
            l.Masking(mask_value=0.),
            # --> 'convolutional_lstm_cells' <--
            l.Conv3D(
                filters=1,
                kernel_size=(3, 3, 3),
                activation='sigmoid',
                padding='same',
                data_format='channels_last'),
            l.Flatten(),
            l.Dense(self.num_classes),
            l.Activation('softmax')
        ]

        # Insert 'convolutional_lstm_cells'
        for cell in range(0, num_cells):
            layers.insert(
                2 + cell,
                self.convolutional_lstm_cell(
                    40,
                    return_sequences=True,
                    name="convolutional_lstm_cell_%0.0f" % (cell + 1)))

        return tf.keras.Sequential(layers, "convolutional_lstm")

    def convolutional_lstm_cell(self, units, return_sequences,
                                name=None) -> Model:
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
