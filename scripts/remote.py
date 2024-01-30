#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script includes the remote computations for single-shot ridge
regression with decentralized statistic calculation
"""
import json
import sys
import scipy as sp
import numpy as np

from itertools import repeat

from scripts.regression import t_to_p
from scripts.utils import list_recursive, log
from scripts.remote_ancillary import get_stats_to_dict


def remote_0(args):
    #log(args, args['state'])

    input_list = args["input"]

    site_ids = list(input_list.keys())
    userID = list(input_list)[0]
    
    site_covar_list = [
        '{}_{}'.format('site', label) for index, label in enumerate(site_ids)
        if index
    ]

    X_labels = input_list[userID]["x_headers"]
    y_labels = input_list[userID]["y_headers"]

    tol = input_list[userID]["tol"]
    eta = input_list[userID]["eta"]

    output_dict = {
        "site_covar_list": site_covar_list,
        "computation_phase": "remote_0"
    }

    cache_dict = {
        "X_labels": X_labels,
        "y_labels": y_labels,
        "tol": tol,
        "eta": eta
    }

    computation_output_dict = {
        "output": output_dict,
        "cache": cache_dict,
    }

    #log(args, args['state'])

    return computation_output_dict


def remote_1(args):
    #log(args, args['state'])
    """Need this function for performing multi-shot regression"""
    input_list = args["input"]
    first_user_id = list(input_list.keys())[0]
    beta_vec_size = input_list[first_user_id]["beta_vec_size"]
    number_of_regressions = input_list[first_user_id]["number_of_regressions"]

    # Initial setup
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    count = 0

    tol = args["cache"]["tol"]
    eta = args["cache"]["eta"]

    wp, wc, mt, vt = [
        np.zeros((number_of_regressions, beta_vec_size), dtype=float)
        for _ in range(4)
    ]

    iter_flag = 1

    output_dict = {
        "remote_beta": wp.tolist(),
        "iter_flag": iter_flag,
        "computation_phase": "remote_1"
    }

    cache_dict = {
        "beta1": beta1,
        "beta2": beta2,
        "eps": eps,
        "tol": tol,
        "eta": eta,
        "count": count,
        "wp": wp.tolist(),
        "wc": wc.tolist(),
        "mt": mt.tolist(),
        "vt": vt.tolist(),
        "iter_flag": iter_flag,
        "number_of_regressions": number_of_regressions,
    }

    computation_output = {
        "output": output_dict,
        "cache": cache_dict,
    }

    #log(computation_output, args['state'])

    return computation_output


def remote_2(args):
    #log(args, args['state'])

    beta1 = args["cache"]["beta1"]
    beta2 = args["cache"]["beta2"]
    eps = args["cache"]["eps"]
    tol = args["cache"]["tol"]
    eta = args["cache"]["eta"]
    count = args["cache"]["count"]
    wp = np.array(args["cache"]["wp"], dtype=float)
    wc = np.array(args["cache"]["wc"], dtype=float)
    mt = args["cache"]["mt"]
    vt = args["cache"]["vt"]
    iter_flag = args["cache"]["iter_flag"]
    number_of_regressions = args["cache"]["number_of_regressions"]

    count = count + 1

    if not iter_flag:
        cache_dict = {"avg_beta_vector": wc.tolist(),
            "X_labels": args["cache"]["X_labels"]}

        output_dict = {
            "avg_beta_vector": wc.tolist(),
            "computation_phase": "remote_2b"
        }

        computation_output = {
            "output": output_dict,
            "cache": cache_dict,
        }
    else:
        input_list = args["input"]
        sorted_site_ids = sorted(list(input_list.keys()))

        if len(input_list) == 1:
            grad_remote = [
                np.array(args["input"][site]["local_grad"])
                for site in sorted_site_ids
            ]
            grad_remote = grad_remote[0]
        else:
            grad_remote = sum([
                np.array(args["input"][site]["local_grad"])
                for site in sorted_site_ids
            ])

        mt = beta1 * np.array(mt) + (1 - beta1) * grad_remote
        vt = beta2 * np.array(vt) + (1 - beta2) * (grad_remote**2)

        m = mt / (1 - beta1**count)
        v = vt / (1 - beta2**count)

        wc = wp - eta * m / (np.sqrt(v) + eps)

        mask_flag = np.linalg.norm(wc - wp, axis=1) <= tol

        if sum(mask_flag) == number_of_regressions:
            iter_flag = 0

        for i in range(mask_flag.shape[0]):
            if not mask_flag[i]:
                wp[i] = wc[i]

        output_dict = {
            "remote_beta": wc.tolist(),
            "mask_flag": mask_flag.astype(int).tolist(),
            "computation_phase": "remote_2a"
        }

        cache_dict = {
            "count": count,
            "wp": wp.tolist(),
            "wc": wc.tolist(),
            "mt": mt.tolist(),
            "vt": vt.tolist(),
            "iter_flag": iter_flag,
            "X_labels": args["cache"]["X_labels"]
        }

        computation_output = {
            "output": output_dict,
            "cache": cache_dict,
        }

        #log(computation_output, args['state'])

    return computation_output


def remote_3(args):
    #log(args, args['state'])
    """Computes the global beta vector, mean_y_global & dof_global

    Args:
        args (dictionary): {"input": {
                                "beta_vector_local": list/array,
                                "mean_y_local": list/array,
                                "count_local": int,
                                "computation_phase": string
                                },
                            "cache": {}
                            }

    Returns:
        computation_output (json) : {"output": {
                                        "avg_beta_vector": list,
                                        "mean_y_global": ,
                                        "computation_phase":
                                        },
                                    "cache": {
                                        "avg_beta_vector": ,
                                        "mean_y_global": ,
                                        "dof_global":
                                        },
                                    }

    """
    input_list = args["input"]
    userID = list(input_list)[0]
    sorted_site_ids = sorted(list(input_list.keys()))
    first_user_id = list(input_list)[0]

    avg_beta_vector = np.array(args["cache"]["avg_beta_vector"])

    all_local_stats_dicts = [{
        site: input_list[site]["local_stats_list"] for site in sorted_site_ids
    }]

    mean_y_local = [input_list[site]["mean_y_local"] for site in input_list]
    count_y_local = [np.array(input_list[site]["count_local"]) for site in input_list]
    mean_y_global = np.array(mean_y_local) * np.array(count_y_local)
    mean_y_global = np.average(mean_y_global, axis=0)

    dof_global = sum(count_y_local) - avg_beta_vector.shape[1]

    output_dict = {
        "avg_beta_vector": avg_beta_vector.tolist(),
        "mean_y_global": mean_y_global.tolist(),
        "computation_phase": "remote_3"
    }

    cache_dict = {
        "avg_beta_vector": avg_beta_vector.tolist(),
        "mean_y_global": mean_y_global.tolist(),
        "dof_global": dof_global.tolist(),
        "all_local_stats_dicts": all_local_stats_dicts
    }

    computation_output = {
        "output": output_dict,
        "cache": cache_dict,
    }

    #log(computation_output, args['state'])

    return computation_output


def remote_4(args):
    #log(args, args['state'])
    """
    Computes the global model fit statistics, r_2_global, ts_global, ps_global

    Args:
        args (dictionary): {"input": {
                                "SSE_local": ,
                                "SST_local": ,
                                "varX_matrix_local": ,
                                "computation_phase":
                                },
                            "cache":{},
                            }

    Returns:
        computation_output (json) : {"output": {
                                        "avg_beta_vector": ,
                                        "beta_vector_local": ,
                                        "r_2_global": ,
                                        "ts_global": ,
                                        "ps_global": ,
                                        "dof_global":
                                        },
                                    "success":
                                    }
    Comments:
        Generate the local fit statistics
            r^2 : goodness of fit/coefficient of determination
                    Given as 1 - (SSE/SST)
                    where   SSE = Sum Squared of Errors
                            SST = Total Sum of Squares
            t   : t-statistic is the coefficient divided by its standard error.
                    Given as beta/std.err(beta)
            p   : two-tailed p-value (The p-value is the probability of
                  seeing a result as extreme as the one you are
                  getting (a t value as large as yours)
                  in a collection of random data in which
                  the variable had no effect.)

    """
    input_list = args["input"]

    X_labels = args["cache"]["X_labels"]
    sorted_site_ids = sorted(list(input_list.keys()))
    y_labels = args["cache"]["y_labels"]
    all_local_stats_dicts = args["cache"]["all_local_stats_dicts"]

    cache_list = args["cache"]
    avg_beta_vector = cache_list["avg_beta_vector"]
    dof_global = cache_list["dof_global"]

    SSE_global = sum(
        [np.array(input_list[site]["SSE_local"]) for site in input_list])
    SST_global = sum(
        [np.array(input_list[site]["SST_local"]) for site in input_list])
    varX_matrix_global = sum([
        np.array(input_list[site]["varX_matrix_local"]) for site in input_list
    ])

    r_squared_global = 1 - (SSE_global / SST_global)
    MSE = SSE_global / np.array(dof_global)

    ts_global = []
    ps_global = []

    for i in range(len(MSE)):
        var_covar_beta_global = MSE[i] * sp.linalg.inv(varX_matrix_global)
        se_beta_global = np.sqrt(var_covar_beta_global.diagonal())
        ts = (avg_beta_vector[i] / se_beta_global).tolist()
        ps = t_to_p(ts, dof_global[i])
        ts_global.append(ts)
        ps_global.append(ps)

    # Block of code to print local stats as well
    sites = [site for site in input_list]

    keys = list(all_local_stats_dicts[0].keys())

    a_dict = []

    for i in range(len(all_local_stats_dicts[0][keys[0]])):
        obj = {}
        for v in all_local_stats_dicts[0].items():
            obj.update({v[0]:v[1][i]})
        a_dict.append(obj)

    # Block of code to print just global stats
    keys1 = [
        "Coefficient", "R Squared", "t Stat", "P-value", "Degrees of Freedom",
        "covariate_labels"
    ]

    X_labels.insert(0, 'const')

    sitesMinus = sites.pop()

    X_labels = X_labels + sites

    global_dict_list = get_stats_to_dict(keys1, avg_beta_vector,
                                         r_squared_global, ts_global,
                                         ps_global, dof_global,
                                         repeat(X_labels, len(y_labels)))

    # Print Everything
    keys2 = ["ROI", "global_stats", "local_stats"]
    dict_list = get_stats_to_dict(keys2, y_labels, global_dict_list, a_dict)

    output_dict = {"regressions": dict_list}

    computation_output = {"output": output_dict, "success": True}

    #raise Exception(computation_output)

    return computation_output


def start(PARAM_DICT):
    PHASE_KEY = list(list_recursive(PARAM_DICT, "computation_phase"))

    if "local_0" in PHASE_KEY:
        return remote_0(PARAM_DICT)
    elif "local_1" in PHASE_KEY:
        return remote_1(PARAM_DICT)
    elif "local_2" in PHASE_KEY:
        return remote_2(PARAM_DICT)
    elif "local_3" in PHASE_KEY:
        return remote_3(PARAM_DICT)
    elif "local_4" in PHASE_KEY:
        return remote_4(PARAM_DICT)
    else:
        raise ValueError("Error occurred at Remote")
