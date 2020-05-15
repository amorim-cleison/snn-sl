import numpy as np
from matplotlib import pyplot as plt
from tensorflow.python.keras.engine.training import Model

from keras.utils import plot_model


class ModelVisualizer():
    def plot_intermediate_layers(self, model: Model, sample_input):
        self.__visualize_activations(model, sample_input,
                                     self.__plot_activation)

    def print_intermediate_layers(self, model: Model, sample_input):
        self.__visualize_activations(model, sample_input,
                                     self.__print_activation)

    def __visualize_activations(self, model: Model, sample_input,
                                fn_visualization):
        layer_outputs = [layer.output for layer in model.layers]

        # Creates a model that will return these outputs, given the model input
        activation_model = Model(inputs=model.input, outputs=layer_outputs)

        # Returns a list of five Numpy arrays: one array per layer activation
        activations = activation_model.predict(np.asarray([sample_input]))

        # Names of the layers, so you can have them as part of your plot
        layer_names = [layer.name for layer in model.layers]

        layer_indexes = range(1, len(layer_names) + 1)

        for layer_name, layer_index, layer_activation in zip(
                layer_names, layer_indexes, activations):
            fn_visualization(layer_name, layer_index, layer_activation)

    def __plot_activation(self, layer_name, layer_index, layer_activation):
        layer_activation = self.__normalize(layer_activation, 0, 255)

        if len(layer_activation.shape) == 3:
            display_grid = layer_activation[0, :, :]
        elif len(layer_activation.shape) == 2:
            display_grid = layer_activation
        else:
            display_grid = None

        if display_grid is not None:
            scale = 1.
            plt.figure(
                figsize=(scale * display_grid.shape[1],
                         scale * display_grid.shape[0]))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='viridis')
            # plt.savefig("%03.0f_%s.png" % (layer_index, layer_name))
            plt.plot()
            plt.show()

    def __print_activation(self, layer_name, layer_index, layer_activation):
        print(80 * '=')
        print('Layer: ', layer_name)
        print('Activation shape: ', layer_activation.shape)
        print(80 * '_')
        print(layer_activation)
        print(80 * '_')

    def plot_model_to_img(self, model: Model, file='model.png'):
        plot_model(model, expand_nested=True, to_file=file)

    def plot_training_history(self, history):
        """
        Plot training & validation accuracy values
        """
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

    def __make_activation_from_image_2(self, layer_activation, col,
                                       images_per_row, row):
        channel_image = layer_activation[0,
                                         # :, :,
                                         :, col * images_per_row + row]
        return channel_image

    def __normalize(self, x: np.ndarray, new_min: float, new_max: float):
        return ((x - x.min()) /
                (x.max() - x.min())) * (new_max - new_min) + new_min
