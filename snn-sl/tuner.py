import sklearn.metrics as skmetrics
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV


def tune_hyperparameters(build_fn,
                         parameters: dict,
                         x,
                         y,
                         verbose=2,
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

    x : array-like, shape = [n_samples, n_features]
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape = [n_samples] or [n_samples, n_output], optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    verbose : integer, optional
        Controls the verbosity: the higher, the more messages.

    cross_validation : integer, optional
        Determines the cross-validation splitting strategy.
    
    scoring : string, callable, list/tuple, dict or None, default: None
        A single string (see :ref:`scoring_parameter`) or a callable
        (see :ref:`scoring`) to evaluate the predictions on the test set.
    
    Returns
    ----------
    The results of the tuning search.
    """
    parameters['loss'] = ['categorical_crossentropy']
    parameters['metrics'] = [['accuracy']]

    model = KerasClassifier(
        build_fn=build_fn, epochs=100, batch_size=10, verbose=verbose)

    search = GridSearchCV(
        estimator=model,
        param_grid=parameters,
        n_jobs=-1,
        cv=cross_validation,
        verbose=verbose,
        scoring='accuracy')
    search_result = search.fit(x, y)

    # summarize results
    print("Best: %f using %s" % (search_result.best_score_,
                                 search_result.best_params_))

    means = search_result.cv_results_['mean_test_score']
    stds = search_result.cv_results_['std_test_score']
    params = search_result.cv_results_['params']

    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    return search_result
