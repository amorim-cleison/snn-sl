from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import random

import numpy as np

import data_loader as dl
from model_tuner import ModelTuner
import visualizer as v
from model_builder import ModelBuilder
from models import *

# Configurations ----------------------------
debug = True
data_folder = '../../../data/asllvd-skeleton-20/normalized/'

# -------------------------------------------


def run():
    # Load data:
    X_train, y_train, X_test, y_test, num_classes = dl.load_data(data_folder)
    X_train, y_train, X_test, y_test = dl.prepare_data_and_label(
        X_train, y_train, X_test, y_test)

    input_shape = X_train.shape[1:]
    print("Input shape: ", input_shape)

    # Tune model:
    parameters = dict(
        architecture=[AttentionGraphConvLSTM(input_shape, num_classes)],
        epochs=[10],
        # batch_size=[8, 16],
        # optimizer=['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'],
        # loss=['sparse_categorical_crossentropy'],
        # metrics=[['accuracy']],
        # learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3],
        # momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9],
    )

    tuner = ModelTuner()
    tuner.tune_hyperparameters(
        build_model, parameters, X_train, y_train, X_test, y_test, log=True)

    # Visualize model:
    # v.plot_model_to_img(builder.build(num_classes, input_shape))

    # Visualize intermediate layers:
    # v.plot_intermediate_layers(builder.build(num_classes, input_shape), random.choice(X_train))

    # Visualize intermediate layers:
    # v.print_intermediate_layers(builder.build(num_classes, input_shape), random.choice(X_train))

    # Train model:
    # Losses:
    #   'binary_crossentropy',
    #   'categorical_crossentropy',
    #   'sparse_categorical_crossentropy'
    # Metrics:
    #   'categorical_accuracy'
    #   'accuracy'
    # model = m.build(
    #     loss='categorical_crossentropy',
    #     metrics=['accuracy'])
    # result = train(model, X_train, y_train)
    # print()


def build_model(architecture: BaseModel,
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']):
    """
    Build architecture
    """
    model = architecture.build()
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.summary()
    # checkpointer = ModelCheckpoint(filepath=data_path + '/model-{epoch:02d}.hdf5', verbose=1)
    return model


def train(model, X_train, y_train, X_test, y_test, verbose=1):
    history = model.fit(
        x=X_train,
        y=y_train,
        validation_data=(X_test, y_test),
        epochs=10,
        verbose=verbose)

    # Evaluate:
    result = model.predict(X_train, batch_size=8, verbose=verbose)
    for value in result:
        print('%.1f' % np.argmax(value))

    v.plot_training_history(history)

    # result = model.predict_classes(X_train, verbose=verbose)
    # for value in result:
    #     print('%.1f' % value)


def configure():
    pass


configure()
run()
