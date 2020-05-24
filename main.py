from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import random

import numpy as np

import data_loader as dl
from model_tuner import ModelTuner
from model_visualizer import ModelVisualizer
from models import AttentionGraphConvLSTM, Architecture

# Parameters:  ------------------------------
debug = True
data_folder = '../../data/asllvd-skeleton-20/normalized/'
# -------------------------------------------


def run():
    # Load data:
    X_train, y_train, X_test, y_test, num_classes = dl.load_data(data_folder)
    X_train, y_train, X_test, y_test = dl.prepare_data_and_label(
        X_train, y_train, X_test, y_test)

    input_shape = X_train.shape[1:]
    print("Input shape: ", input_shape)

    architecture = AttentionGraphConvLSTM(input_shape, num_classes)

    tune([architecture], X_train, y_train, X_test, y_test, num_classes,
         input_shape)

    # visualize(architecture, X_train)

    # train(architecture, X_train, y_train, X_test, y_test)


def tune(architectures, X_train, y_train, X_test, y_test, num_classes,
         input_shape):
    """
    Tune model hyperparameters
    """
    parameters = dict(
        architecture=architectures,
        epochs=[10],
        # batch_size=[8, 16],
        # optimizer=['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'],
        # loss=['sparse_categorical_crossentropy'],
        # metrics=[['accuracy']],
        # learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3],
        # momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9],
    )

    # Tune:
    print("Tuning...")
    tuner = ModelTuner()
    best_history = tuner.tune_hyperparameters(
        build_model, parameters, X_train, y_train, X_test, y_test, log=True)

    # Plot history:
    print("Plotting history...")
    visualizer = ModelVisualizer()
    visualizer.plot_training_history(best_history)
    
    # Finish
    print("Finished")


def visualize(architecture: Architecture, X_train):
    """
    Visualize model 
    """
    model = architecture.build()
    visualizer = ModelVisualizer()

    # Visualize model:
    visualizer.plot_model_to_img(model)

    # Visualize intermediate layers:
    visualizer.plot_intermediate_layers(model, random.choice(X_train))

    # Visualize intermediate layers:
    visualizer.print_intermediate_layers(model, random.choice(X_train))


def train(architecture: Architecture, X_train, y_train, X_test, y_test,
          verbose=1):
    """
    Train the model
    """
    # Losses:
    #   'binary_crossentropy', 'categorical_crossentropy', 'sparse_categorical_crossentropy'
    # Metrics:
    #   'categorical_accuracy', 'accuracy'
    model = build_model(
        architecture
    )

    # Train:
    print("Training...")
    history = model.fit(
        x=X_train,
        y=y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        verbose=verbose)

    # Plot history:
    print("Plotting history...")
    visualizer = ModelVisualizer()
    visualizer.plot_training_history(history)


def build_model(architecture: Architecture,
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']):
    """
    Build architecture
    """
    model = architecture.build()
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.summary()
    # checkpointer = ModelCheckpoint(
    #    filepath=data_path + '/model-{epoch:02d}.hdf5', verbose=1)
    return model


def configure():
    pass


def playground():
    A = np.asarray([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0]
    ])

    X = np.asarray([
        [2, 1,  5,  9],
        [4, 6, 10,  8],
        [6, 9, 15, 12]
    ])

    tmp = np.dot(A, X[0])

    print(tmp)
    

# K.eager(playground())
configure()
run()
