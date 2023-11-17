import sys, os, re
sys.path.append('../')
import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.svm import SVR, NuSVR, LinearSVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA


def parameters_grid(regr_name, score_num, cv=3, n_comps=range(10, 250, 10)):
    '''

    '''

    scoring_parameters = ['neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_median_absolute_error', 'r2']

    SCORING_SEARCH = scoring_parameters[score_num]
    CV_SEARCH = cv
    ALPHA = np.logspace(-3, 3, 50)
    N_COMPONENTS = n_comps
    L1_RATIO = np.linspace(0.00001, 1, num=50)
    Cs = np.logspace(-3, 3, 50)
    NU = np.linspace(0.00001, 1, num=50)
    EPSILON = np.logspace(-2, 2, 50)
    TREES = [400, 425, 450, 475, 500, 525, 550, 575, 600]
    MAX_DEPTH = [2, 3, 4, 5]
    CRITERION = ['mse', 'mae']


    LOSS_LGD = ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']

    LOSS_ADA = ['linear', 'square', 'exponential']

    SOLVER = ['auto']#, 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag']

    # parameters passed to the model
    parameters = {
        'pca__n_components': N_COMPONENTS,
    }

    # hyper parameters passed outside the model
    hyperparameters = {
        'scoring': SCORING_SEARCH,
        'cv': CV_SEARCH,
    }


    if regr_name == 'AdaBoostRegressor':
        add_params = {'regr__n_estimators': TREES,
                      'regr__loss': LOSS_ADA}

    elif regr_name == 'ElasticNet':
        add_params = {'regr__alpha': ALPHA,
                      'regr__l1_ratio': L1_RATIO
                      }

    elif regr_name == 'GradientBoostingRegressor':
        add_params = {'regr__n_estimators': TREES,
                      'regr__max_depth': MAX_DEPTH}

    elif regr_name == 'Lasso':
        add_params = {'regr__alpha': ALPHA}

    elif regr_name == 'LinearSVR':
        add_params = {'regr__C': Cs}

    elif regr_name == 'NuSVR':
        add_params = {'regr__C': Cs,
                      'regr__nu': NU}

    elif regr_name in ['RandomForestRegressor', 'ExtraTreesRegressor']:
        add_params = {'regr__n_estimators': TREES,
                      'regr__criterion': CRITERION,
                      'regr__max_depth': MAX_DEPTH}

    elif regr_name == 'Ridge':
        add_params = {'regr__alpha': ALPHA,
                      'regr__solver': SOLVER}

    elif regr_name == 'SGDRegressor':
        add_params = {'regr__l1_ratio': L1_RATIO,
                      'regr__loss': LOSS_LGD}

    elif regr_name == 'SVR':
        add_params = {'regr__C': Cs,
                      'regr__epsilon': EPSILON}

    parameters.update(add_params)

    return parameters, hyperparameters

def pipe(number):
    '''
    Pipeline of transforms with a final estimator.
    '''


    models = [Ridge(normalize=True, random_state=74),
              ElasticNet(normalize=True, random_state=74),
              Lasso(normalize=True, random_state=74),
              SVR(kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, cache_size=200, verbose=False, max_iter=-1),
              NuSVR(kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, tol=0.001, cache_size=200, verbose=False, max_iter=-1),
              LinearSVR(epsilon=0.0, tol=0.0001, loss='epsilon_insensitive', fit_intercept=True, intercept_scaling=1.0, dual=True, verbose=0, random_state=74, max_iter=1000),
              SGDRegressor(penalty='elasticnet', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, n_iter=5, shuffle=True, verbose=0, epsilon=0.1, random_state=74, learning_rate='invscaling', eta0=0.01, power_t=0.25, warm_start=False, average=False),
              AdaBoostRegressor(base_estimator=None, learning_rate=1.0, loss='linear', random_state=74),
              RandomForestRegressor(n_estimators=10, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=True, oob_score=True, n_jobs=-1, random_state=74, verbose=0, warm_start=False),
              ExtraTreesRegressor(min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=True, oob_score=False, n_jobs=-1, random_state=74, verbose=0, warm_start=False),
              GradientBoostingRegressor(loss='ls', learning_rate=0.1, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_split=1e-07, init=None, random_state=74, max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')
              ]

    regressor = models[number]


    regr_name = type(regressor).__name__

    pipeline = Pipeline([
                        ('pca', PCA()),
                        ('regr', regressor),
    ])


    return pipeline, regr_name

if __name__ == '__main__':
    main()

'''

linear_model_ridge
linear_model_coordinate_descent

'''
