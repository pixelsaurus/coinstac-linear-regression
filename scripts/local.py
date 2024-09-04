#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
warnings.simplefilter("ignore")


import json
import numpy as np
import os
import pandas as pd
import sys

from scripts.regression import sum_squared_error, y_estimate
from scripts.utils import list_recursive, log
import scripts.local_ancillary as lc

def local_0(args):    
    input_list = args["input"]

    log(input_list, args['state'])

    lamb = input_list['lambda']

    cov_file = input_list['covariates'][0]
    cf = pd.read_csv(os.path.join(args["state"]["baseDirectory"], cov_file))

    data_file = input_list['data'][0]
    df = pd.read_csv(os.path.join(args["state"]["baseDirectory"], data_file))

    x_headers = input_list['x_headers']
    y_headers = input_list['y_headers']
    
    X = cf[x_headers]
    y = df[y_headers]

    columns_to_normalize = lc.check_cols_to_normalize(X)

    tol = input_list["tol"]
    eta = input_list["eta"]

    cache_dict = {
        "covariates": X.to_json(orient='records'),
        "dependents": y.to_json(orient='records'),
        "lambda": lamb,
    }

    computation_output_dict = {
        "output": {
            "computation_phase": "local_0",
            "x_headers": x_headers,
            "y_headers": y_headers,
            "tol": tol,
            "eta": eta,
            "columns_to_normalize": columns_to_normalize,
        },
        "cache": cache_dict
    }

    log(computation_output_dict, args['state'])

    return computation_output_dict

def local_1(args):
    input_list = args["input"]

    log(input_list, args['state'])
    """Read data from the local sites, perform local regressions and send
    local statistics to the remote site"""

    X = pd.read_json(args["cache"]["covariates"], orient='records')
    X = lc.normalize_columns(X, input_list["columns_to_normalize"])

    log(f'\n\nNormalizing the following column values to their z-scores: {input_list["columns_to_normalize"]} \n ', args['state'])

    y = pd.read_json(args["cache"]["dependents"], orient='records')
    y_labels = list(y.columns)

    site = args['state']['clientId']

    beta_vector, local_stats_list, meanY_vector, lenY_vector, site = lc.gather_local_stats(X, y, site)

    augmented_X = lc.add_site_covariates(args, X)

    beta_vec_size = augmented_X.shape[1]

    output_dict = {
        "beta_vector_local": beta_vector,
        "beta_vec_size": beta_vec_size,
        "augmented_X_labels": list(augmented_X.columns),
        "number_of_regressions": len(y_labels),
        "computation_phase": "local_1"
    }

    cache_dict = {
        "beta_vec_size": beta_vec_size,
        "number_of_regressions": len(y_labels),
        "covariates": augmented_X.to_json(orient='records'),
        "y_labels": y_labels,
        "mean_y_local": meanY_vector,
        "count_local": lenY_vector,
        "local_stats_list": local_stats_list,
    }

    computation_output = {
        "output": output_dict,
        "cache": cache_dict,
    }

    log(computation_output, args['state'])

    return computation_output


def local_2(args):
    input_list = args["input"]

    log(input_list, args['state'])

    X = pd.read_json(args["cache"]["covariates"], orient='records')
    y = pd.read_json(args["cache"]["dependents"], orient='records')

    X.to_csv(os.path.join(args['state']["outputDirectory"], f'{args["state"]["clientId"]}_X.csv'))
    y.to_csv(os.path.join(args['state']["outputDirectory"], f'{args["state"]["clientId"]}_y.csv'))

    beta_vec_size = args["cache"]["beta_vec_size"]
    number_of_regressions = args["cache"]["number_of_regressions"]

    mask_flag = args["input"].get("mask_flag",
                                  np.zeros(number_of_regressions, dtype=bool))

    biased_X = np.array(X)
    y = pd.DataFrame(y.values)

    w = args["input"]["remote_beta"]

    gradient = np.zeros((number_of_regressions, beta_vec_size))
    cost = np.zeros(number_of_regressions)

    for i in range(number_of_regressions):
        y_ = y[i]
        w_ = w[i]
        if not mask_flag[i]:
            gradient[i, :] = (
                1 / len(X)) * np.dot(biased_X.T, np.dot(biased_X, w_) - y_)
        cost[i] = lc.get_cost(y_actual=y[i], y_predicted=np.dot(biased_X, w_))

    output_dict = {
        "local_grad": gradient.tolist(),
        "local_cost": cost.tolist(),
        "computation_phase": "local_2"
    }

    cache_dict = {}

    computation_output = {
        "output": output_dict,
        "cache": cache_dict,
    }

    log(computation_output, args['state'])

    return computation_output


def local_3(args):
    input_list = args["input"]

    log(input_list, args['state'])

    output_dict = {
        "mean_y_local": args["cache"]["mean_y_local"],
        "count_local": args["cache"]["count_local"],
        "local_stats_list": args["cache"]["local_stats_list"],
        "y_labels": args["cache"]["y_labels"],
        "computation_phase": 'local_3'
    }

    cache_dict = {}

    computation_output = {
        "output": output_dict,
        "cache": cache_dict,
    }

    log(computation_output, args['state'])

    return computation_output


def local_4(args):
    input_list = args["input"]

    log(input_list, args['state'])

    """Computes the SSE_local, SST_local and varX_matrix_local

    Args:
        args (dictionary): {"input": {
                                "avg_beta_vector": ,
                                "mean_y_global": ,
                                "computation_phase":
                                },
                            "cache": {
                                "covariates": ,
                                "dependents": ,
                                "lambda": ,
                                "dof_local": ,
                                }
                            }

    Returns:
        computation_output (json): {"output": {
                                        "SSE_local": ,
                                        "SST_local": ,
                                        "varX_matrix_local": ,
                                        "computation_phase":
                                        }
                                    }

    Comments:
        After receiving  the mean_y_global, calculate the SSE_local,
        SST_local and varX_matrix_local

    """
    cache_list = args["cache"]
    input_list = args["input"]

    X = pd.read_json(cache_list["covariates"], orient='records')
    y = pd.read_json(cache_list["dependents"], orient='records')
    biased_X = np.array(X)

    avg_beta_vector = input_list["avg_beta_vector"]
    mean_y_global = input_list["mean_y_global"]

    SSE_local, SST_local = [], []
    for index, column in enumerate(y.columns):
        curr_y = y[column].values
        SSE_local.append(
            sum_squared_error( curr_y, y_estimate(biased_X, avg_beta_vector)[index]))
        SST_local.append(
            np.sum(
                np.square(np.subtract(curr_y, mean_y_global[index])),
                dtype=float))

    varX_matrix_local = np.dot(biased_X.T, biased_X)

    output_dict = {
        "SSE_local": SSE_local,
        "SST_local": SST_local,
        "varX_matrix_local": varX_matrix_local.tolist(),
        "computation_phase": "local_4"
    }

    cache_dict = {}

    computation_output = {
        "output": output_dict,
        "cache": cache_dict,
    }

    log(computation_output, args['state'])

    return computation_output



def start(PARAM_DICT):
    PHASE_KEY = list(list_recursive(PARAM_DICT, "computation_phase"))

    if not PHASE_KEY:
        return local_0(PARAM_DICT)
    elif 'remote_0' in PHASE_KEY:
        return local_1(PARAM_DICT)
    elif 'remote_1' in PHASE_KEY:
        return local_2(PARAM_DICT)
    elif 'remote_2a' in PHASE_KEY:
        return local_2(PARAM_DICT)
    elif 'remote_2b' in PHASE_KEY:
        return local_3(PARAM_DICT)
    elif 'remote_3' in PHASE_KEY:
        return local_4(PARAM_DICT)
    else:
        raise ValueError("Error occurred at Local")


