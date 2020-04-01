from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import data_loader as dl
import model as m
import tuner as t
import visualizer as v
import random

def run():
    # Load data:
    data_folder = '../../../data/asllvd-skeleton-20/normalized/'
    X_train, y_train, X_test, y_test, num_classes = dl.load_data(data_folder)
    X_train, y_train, X_test, y_test = dl.prepare_data(X_train, y_train, X_test, y_test)

    input_shape=X_train.shape[1:]
    print("Input shape: ", input_shape)

    # Tune model:
    parameters = dict(
        input_shape=[input_shape],
        # batch_size=[8, 16],
        epochs=[10],
        # optimizer=['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'],
        # optimizer=['adam'],
        # loss=['sparse_categorical_crossentropy'],
        # metrics=[['accuracy']],
        num_classes=[num_classes],
        # learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3],
        # momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
        )

    # Visualize intermediate layers:
    v.plot_intermediate_layers(m.build(num_classes, input_shape), random.choice(X_train))

    # Visualize intermediate layers:
    # v.print_intermediate_layers(m.build(num_classes, input_shape), random.choice(X_train))

    # Classification metrics can't handle a mix of multilabel-indicator and binary targets
    # t.tune_hyperparameters( m.build, parameters, 
    #                         X_train, y_train, 
    #                         X_test, y_test, 
    #                         log=True)

    # Train model:
    # Losses:
    #   'binary_crossentropy',
    #   'categorical_crossentropy',
    #   'sparse_categorical_crossentropy'
    # Metrics:
    #   'categorical_accuracy'
    #   'accuracy'
    # model = m.build(
    #     batch_size=batch_size,
    #     num_classes=num_classes,
    #     loss='categorical_crossentropy',
    #     metrics=['accuracy'])
    # result = train(model, X_train, y_train)
    # print()


def train(model, X_train, y_train, X_test, y_test, verbose=1):
    model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), epochs=10, verbose=verbose)

    # Evaluate:
    result = model.predict(X_train, batch_size=8, verbose=verbose)
    for value in result:
        print('%.1f' % np.argmax(value))

    # result = model.predict_classes(X_train, verbose=verbose)
    # for value in result:
    #     print('%.1f' % value)


run()
