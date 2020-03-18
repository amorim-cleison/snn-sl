import sklearn.metrics as skmetrics
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


def tune_hyperparameters(build_fn,
                         parameters: dict,
                         X_train,
                         y_train,
                         X_test,
                         y_test,
                         cross_validation=3):
    """
    Run an exhaustive search over the parameters values for the model.

    Parameters
    ----------
    build_fn : function
        Function that builds the model

    parameters : dict
        Dictionary with parameters names (string) as keys and lists of
        parameter settings to try as values, or a list of such
        dictionaries, in which case the grids spanned by each dictionary
        in the list are explored. This enables searching over any sequence
        of parameter settings.

    X_train : array-like, shape = [n_samples, n_features]
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y_train : array-like, shape = [n_samples] or [n_samples, n_output], optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    cross_validation : integer, optional
        Determines the cross-validation splitting strategy.
    
    Returns
    ----------
    The results of the tuning search.
    """
    model = KerasClassifier(
        build_fn=build_fn, epochs=100, batch_size=10, verbose=2)

    search = GridSearchCV(
        estimator=model,
        param_grid=parameters,
        n_jobs=-1,
        cv=cross_validation,
        verbose=100,
        # scoring=score,
        return_train_score=False
        )

    # Fit search:
    search.fit(X_train, y_train, groups=None, validation_data=(X_test, y_test))

    # Print results:
    __print_results(search)


def __print_results(search, X_test=None, y_test=None):
    print()
    print("------------------------------------------------------")
    print("Best parameters set found on development set:")
    print()
    print(search.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = search.cv_results_['mean_test_score']
    stds = search.cv_results_['std_test_score']

    for mean, std, params in zip(means, stds, search.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()

    if (X_test is not None and y_test is not None):
        y_true, y_pred = y_test, search.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()
    
    print("------------------------------------------------------")