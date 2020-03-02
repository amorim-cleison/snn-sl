from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import data_loader as dl
import model as model
import tuner as tn


def run():
    # Load data:
    data_folder = '../../data/asllvd-skeleton-20/normalized/'
    num_classes = 2745
    train_x, train_y, _, test_x, test_y, _ = dl.load_data(
        data_folder, num_classes)

    # Tune model:
    batch_size = [
        8
        # 8, 16
    ]
    epochs = [
        10
        # 10, 20
    ]
    optimizer = [
        'adam'
        # 'SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
    ]
    # learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
    # momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
    parameters = dict(
        batch_size=batch_size,
        epochs=epochs,
        optimizer=optimizer,
        num_classes=[num_classes])
    tn.tune_hyperparameters(model.build, parameters, train_x, train_y)

    # Train model:
    # Losses:
    #   'binary_crossentropy',
    #   'categorical_crossentropy',
    #   'sparse_categorical_crossentropy'
    # Metrics:
    #   'categorical_accuracy'
    #   'accuracy'
    # train_model(build_model(), train_x, train_y)
    # loss=losses.categorical_crossentropy,
    # metrics=[metrics.categorical_accuracy])


run()
