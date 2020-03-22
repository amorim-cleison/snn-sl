import collections

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers, models
from tensorflow.python.keras.engine.training import Model



import matplotlib
from matplotlib import pyplot as plt

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

    model.summary()
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



def visualize_intermediate_layers(model:Model, sample_input):
    layer_outputs = [layer.output for layer in model.layers] 
    
    # Creates a model that will return these outputs, given the model input
    activation_model = Model(inputs=model.input, outputs=layer_outputs)

    # Returns a list of five Numpy arrays: one array per layer activation 
    activations = activation_model.predict(np.asarray([sample_input])) 

    # Names of the layers, so you can have them as part of your plot
    layer_names = [layer.name for layer in model.layers] 

    layer_indexes = range(1, len(layer_names) + 1)

    for layer_name, layer_index, layer_activation in zip(layer_names, layer_indexes, activations):
        layer_activation = __normalize(layer_activation, 0, 255)
      
        if len(layer_activation.shape) == 3:
            display_grid = layer_activation[0,:,:]
        elif len(layer_activation.shape) == 2:
            display_grid = layer_activation

        scale = 2.
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        # plt.savefig("%03.0f_%s.png" % (layer_index, layer_name))

    plt.show()
    

# def visualize_intermediate_layers(model:Model, sample_input):
#     layer_outputs = [layer.output for layer in model.layers] 
    
#     # Extracts the outputs of the top 12 layers
#     activation_model = Model(inputs=model.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input
#     activations = activation_model.predict(np.asarray([sample_input])) # Returns a list of five Numpy arrays: one array per layer activation

#     layer_names = [layer.name for layer in model.layers] # Names of the layers, so you can have them as part of your plot
        
#     images_per_row = 16

#     for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
#         layer_activation = __normalize(layer_activation, 0, 255)
#         n_features = layer_activation.shape[-1] # Number of features in the feature map
#         size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
#         n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
#         display_grid = np.zeros((size * n_cols, images_per_row * size))

#         for col in range(n_cols): # Tiles each filter into a big horizontal grid
#             for row in range(images_per_row):
#                 if len(layer_activation.shape) != 3:
#                     continue

#                 channel_image = __make_activation_from_image_2(layer_activation, col, images_per_row, row)
                
#                 display_grid[col * size : (col + 1) * size, # Displays the grid
#                             row * size : (row + 1) * size] = channel_image
#         scale = 1. / size
#         plt.figure(figsize=(scale * display_grid.shape[1],
#                             scale * display_grid.shape[0]))
#         plt.title(layer_name)
#         plt.grid(False)
#         plt.imshow(display_grid, aspect='auto', cmap='viridis')
#         plt.show()


# def __make_activation_from_image(layer_activation, col, images_per_row, row):
#     channel_image = layer_activation[0,
#                                     # :, :,
#                                     :,
#                                     col * images_per_row + row]
#     channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
#     channel_image /= channel_image.std()
#     channel_image *= 64
#     channel_image += 128
#     channel_image = np.clip(channel_image, 0, 255).astype('uint8')
    
#     return channel_image


def __make_activation_from_image_2(layer_activation, col, images_per_row, row):
    channel_image = layer_activation[0,
                                    # :, :,
                                    :,
                                    col * images_per_row + row]
    return channel_image


def __normalize(x: np.ndarray, new_min: float, new_max: float):
    return ((x - x.min()) / (x.max() - x.min())) * (new_max - new_min) + new_min
