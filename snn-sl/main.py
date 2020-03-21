from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import data_loader as dl
import model as m
import tuner as t


def run():
    # Load data:
    data_folder = '../../data/asllvd-skeleton-20/normalized/'
    X_train, y_train, X_test, y_test, num_classes = dl.load_data(data_folder)
    
    # Tune model:
    parameters = dict(
        batch_size=[8],
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

    # Exception has occurred: ValueError
    # Classification metrics can't handle a mix of multilabel-indicator and binary targets
    t.tune_hyperparameters(m.build, parameters, X_train, y_train, X_test, y_test, log=False)

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
    # result = m.train(model, X_train, y_train)
    # print()

run()
