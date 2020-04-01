import collections

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.python.keras.engine.training import Model


def plot_intermediate_layers(model:Model, sample_input):
    __visualize_activations(model, sample_input, __plot_activation)
    

def print_intermediate_layers(model:Model, sample_input):
    __visualize_activations(model, sample_input, __print_activation)
    

def __visualize_activations(model:Model, sample_input, fn_visualization):
    layer_outputs = [layer.output for layer in model.layers] 
    
    # Creates a model that will return these outputs, given the model input
    activation_model = Model(inputs=model.input, outputs=layer_outputs)

    # Returns a list of five Numpy arrays: one array per layer activation 
    activations = activation_model.predict(np.asarray([sample_input])) 

    # Names of the layers, so you can have them as part of your plot
    layer_names = [layer.name for layer in model.layers] 

    layer_indexes = range(1, len(layer_names) + 1)

    for layer_name, layer_index, layer_activation in zip(layer_names, layer_indexes, activations):
        fn_visualization(layer_name, layer_index, layer_activation)


def __plot_activation(layer_name, layer_index, layer_activation):
    layer_activation = __normalize(layer_activation, 0, 255)
      
    if len(layer_activation.shape) == 3:
        display_grid = layer_activation[0,:,:]
    elif len(layer_activation.shape) == 2:
        display_grid = layer_activation
    else:
        display_grid = None

    if display_grid is not None:
        scale = 1.
        plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        # plt.savefig("%03.0f_%s.png" % (layer_index, layer_name))
        plt.show()


def __print_activation(layer_name, layer_index, layer_activation):
    print(80 * '=')
    print('Layer: ', layer_name)
    print('Activation shape: ', layer_activation.shape)
    print(80 * '_')
    print(layer_activation)
    print(80 * '_')



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
