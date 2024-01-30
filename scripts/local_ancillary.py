#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 10:00:16 2018

@author: Harshvardhan
"""
import warnings
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm

def ignore_nans(X, y):
    """Removing rows containing NaN's in X and y"""

    if type(X) is pd.DataFrame:
        X_ = X.values.astype('float64')
    else:
        X_ = X

    if type(y) is pd.Series:
        y_ = y.values.astype('float64')
    else:
        y_ = y

    finite_x_idx = np.isfinite(X_).all(axis=1)
    finite_y_idx = np.isfinite(y_)

    finite_idx = finite_y_idx & finite_x_idx

    y_ = y_[finite_idx]
    X_ = X_[finite_idx, :]

    return X_, y_


def gather_local_stats(X, y, site):
    """Calculate local statistics"""
    y_labels = list(y.columns)

    biased_X = sm.add_constant(X)
    X_labels = list(biased_X.columns)

    local_params = []
    local_sse = []
    local_pvalues = []
    local_tvalues = []
    local_rsquared = []
    meanY_vector, lenY_vector = [], []

    for column in y.columns:
        curr_y = y[column]

        X_, y_ = ignore_nans(biased_X, curr_y)
        meanY_vector.append(np.mean(y_))
        lenY_vector.append(len(y_))

        # Printing local stats as well
        model = sm.OLS(y_, X_).fit()
        local_params.append(model.params)
        local_sse.append(model.ssr)
        local_pvalues.append(model.pvalues)
        local_tvalues.append(model.tvalues)
        local_rsquared.append(model.rsquared)

    keys = [
        "Coefficient", "Sum Square of Errors", "t Stat", "P-value",
        "R Squared", "covariate_labels"
    ]
    local_stats_list = []

    for index, _ in enumerate(y_labels):
        values = [
            local_params[index].tolist(), local_sse[index],
            local_tvalues[index].tolist(), local_pvalues[index].tolist(),
            local_rsquared[index], X_labels
        ]
        local_stats_dict = {key: value for key, value in zip(keys, values)}
        local_stats_list.append(local_stats_dict)

        beta_vector = [l.tolist() for l in local_params]

    return beta_vector, local_stats_list, meanY_vector, lenY_vector, site


def add_site_covariates(args, X):
    biased_X = sm.add_constant(X)
    site_covar_list = args["input"]["site_covar_list"]

    site_matrix = np.zeros(
        (np.array(X).shape[0], len(site_covar_list)), dtype=int)
    site_df = pd.DataFrame(site_matrix, columns=site_covar_list)

    select_cols = [
        col for col in site_df.columns
        if args["state"]["clientId"] in col[len('site_'):]
    ]

    site_df[select_cols] = 1

    biased_X.reset_index(drop=True, inplace=True)
    site_df.reset_index(drop=True, inplace=True)

    augmented_X = pd.concat([biased_X, site_df], axis=1)

    return augmented_X
