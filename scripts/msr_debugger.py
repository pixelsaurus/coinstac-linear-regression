

import numpy as np
import scipy as sp
import pandas as pd
import os
from tabulate import tabulate
#Prints floating points normally without scientific notation
np.set_printoptions(suppress=True)


beta_vec_size =None
number_of_regressions = None
X_labels = None
y_labels = None
W_RANDOM_INIT = []
rem_cache_dict = {
}
NUM_DEC_PLACES=3


def local_setup(number_of_sites, data_dir, norm_cols):
    global X_labels, y_labels
    site_data_dict={}
    for site in range(number_of_sites):
        X=pd.read_csv(os.path.join(data_dir,f'local{str(site)}_X.csv'))
        X_labels = X.columns
        if norm_cols != None:
            X= normalize_columns(X, norm_cols)

        X=np.array(X)

        y=pd.read_csv(os.path.join(data_dir, f'local{str(site)}_y.csv'))
        y_labels = y.columns
        y=pd.DataFrame(y.values)
        site_data_dict[f'local{site}'] = {
            'X' : X,
            'y' : y,
        }

    return site_data_dict

def get_pooled_data(number_of_sites, data_dir, norm_cols, save_data=False):
    site_data_dict = local_setup(number_of_sites, data_dir, norm_cols=None)

    X=site_data_dict['local0']['X']
    y=site_data_dict['local0']['y']

    #Combining all the data
    for site in range(1, number_of_sites):
        X = np.concatenate((X, site_data_dict[f'local{site}']['X']), axis=0)
        y = np.concatenate((y, np.array(site_data_dict[f'local{site}']['y'])), axis=0)

    X_df = pd.DataFrame(X, columns=X_labels)

    if norm_cols != None:
        X_df=normalize_columns(X_df, norm_cols)

    X=np.array(X_df)

    if save_data:
        suffix=""
        if norm_cols!=None:
            suffix = "_norm"+str(norm_cols)

        np.savetxt(data_dir + f'allPooled_X{suffix}.csv', X, delimiter=",", header=','.join(X_labels))
        np.savetxt(data_dir + f'allPooled_y{suffix}.csv', y, delimiter=",", header=','.join(y_labels))

    return X, y


def rem_setup(tol, eta):
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    count = 0
    #Initialize randomly
    global W_RANDOM_INIT
    if type(W_RANDOM_INIT) == list:
        W_RANDOM_INIT = np.random.random((number_of_regressions, beta_vec_size))
    wc = W_RANDOM_INIT.copy()
    print(f'wc: {wc}')
    wp, mt, vt = [
        np.zeros((number_of_regressions, beta_vec_size), dtype=float)
        for _ in range(3)
    ]
    prev_cost = [None]*number_of_regressions
    iter_flag=1

    global  rem_cache_dict
    rem_cache_dict.update( {
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
        "prev_cost":prev_cost,
    })

def local_gradient(X, y, w, mask_flag):
    gradient = np.zeros((number_of_regressions, beta_vec_size))
    cost = np.zeros(number_of_regressions)

    for i in range(number_of_regressions):
        y_ = y[i]
        w_ = w[i]
        if not mask_flag[i]:
            gradient[i, :] = (
                1 / len(X)) * np.dot(X.T, (np.dot(X, w_) - y_))

        cost[i] = get_cost(y_actual=y[i], y_predicted=np.dot(X, w_))

    return {'local_grad' : gradient, 'local_cost': cost}


def remote_gradient(input_dict):
    global  rem_cache_dict

    beta1 = rem_cache_dict["beta1"]
    beta2 = rem_cache_dict["beta2"]
    eps = rem_cache_dict["eps"]
    tol = rem_cache_dict["tol"]
    eta = rem_cache_dict["eta"]
    count = rem_cache_dict["count"]
    wp = np.array(rem_cache_dict["wp"], dtype=float)
    wc = np.array(rem_cache_dict["wc"], dtype=float)
    mt = rem_cache_dict["mt"]
    vt = rem_cache_dict["vt"]
    iter_flag = rem_cache_dict["iter_flag"]
    number_of_regressions = rem_cache_dict["number_of_regressions"]
    prev_cost = rem_cache_dict["prev_cost"]

    count = count + 1

    sorted_site_ids = sorted(list(input_dict.keys()))

    if len(input_dict) == 1:
        grad_remote = [
            np.array(input_dict[site]["local_grad"])
            for site in sorted_site_ids
        ]
        grad_remote = grad_remote[0]
    else:
        grad_remote = sum([
            np.array(input_dict[site]["local_grad"])
            for site in sorted_site_ids
        ])

    mt = beta1 * np.array(mt) + (1 - beta1) * grad_remote
    vt = beta2 * np.array(vt) + (1 - beta2) * (grad_remote**2)

    m = mt / (1 - beta1**count)
    v = vt / (1 - beta2**count)

    wc = wp - eta * m / (np.sqrt(v) + eps)

    mask_flag = np.linalg.norm(wc - wp, axis=1) <= tol

    #Compute curr_cost
    curr_cost = np.average(np.array([input_dict[site]["local_cost"] for site in sorted_site_ids]), axis=0)

    if None not in prev_cost:
        mask_flag = abs(np.array(prev_cost) - curr_cost) <= tol

    if sum(mask_flag) == number_of_regressions:
        iter_flag = 0

    for i in range(mask_flag.shape[0]):
        if not mask_flag[i]:
            wp[i] = wc[i]
            prev_cost[i] =curr_cost[i]

    rem_output_dict = {
        "remote_beta": wc.tolist(),
        "mask_flag": mask_flag.astype(int).tolist(),
        "iter_flag": iter_flag
        #"computation_phase": "remote_2a"
    }

    rem_cache_dict.update( {
        "count": count,
        "wp": wp.tolist(),
        "wc": wc.tolist(),
        "mt": mt.tolist(),
        "vt": vt.tolist(),
        "prev_cost": prev_cost,
        "iter_flag": iter_flag,
    })

    return rem_output_dict


def run_msr_fed(tol, eta, number_of_sites, data_dir, norm_cols=None):
    global beta_vec_size, number_of_regressions

    site_data_dict = local_setup(number_of_sites, data_dir, norm_cols=norm_cols)

    beta_vec_size= site_data_dict['local0']['X'].shape[1];
    number_of_regressions= len( site_data_dict['local0']['y'].columns);

    mask_flag = np.zeros(number_of_regressions, dtype=bool)
    rem_setup(tol=tol, eta=eta)

    print("\n\n###### Running MSR in federated manner ########")
    print(f'###### Number of regressions: {number_of_regressions} ########')

    run_iterations=rem_cache_dict['iter_flag']
    w_agg=rem_cache_dict['wc']
    while run_iterations:
        local_outputs_dict = {}
        for site in range(number_of_sites):
            local_outputs_dict[f'local{site}']={}
            X = site_data_dict[f'local{site}']['X']
            y = site_data_dict[f'local{site}']['y']
            local_outputs_dict[f'local{site}']= local_gradient(X, y, w_agg, mask_flag)

        remote_output=remote_gradient(local_outputs_dict)

        w_agg = remote_output['remote_beta']
        mask_flag = remote_output['mask_flag']
        run_iterations=remote_output['iter_flag']

    #print("Federated avg_beta_vector" + str(w_agg))
    print(f'Federated Solution converged in iterations: {rem_cache_dict["count"]}')
    results_df = metrics_fed(number_of_sites, site_data_dict, w_agg)


def run_msr_pooled(tol, eta, number_of_sites, data_dir, norm_cols=None):
    global beta_vec_size, number_of_regressions

    X, y = get_pooled_data(number_of_sites, data_dir, norm_cols, save_data=True)
    print("\n\n###### Running MSR on pooled data ########")

    beta_vec_size= X.shape[1];
    number_of_regressions= y.shape[1]#len(y.columns);

    print(f'###### Number of regressions: {number_of_regressions} ########')
    #MSR using all the pooled data
    y=pd.DataFrame(y)
    beta_vec_size= X.shape[1];
    number_of_regressions= len(y.columns);

    mask_flag = np.zeros(number_of_regressions, dtype=bool)
    rem_setup(tol=tol, eta=eta)

    run_iterations=rem_cache_dict['iter_flag']
    w_agg=rem_cache_dict['wc']
    while run_iterations:
        local_outputs_dict = {'combined':  local_gradient(X, y, w_agg, mask_flag)}
        remote_output=remote_gradient(local_outputs_dict)

        w_agg = remote_output['remote_beta']
        mask_flag = remote_output['mask_flag']
        run_iterations=remote_output['iter_flag']

    #print("Combined avg_beta_vector" + str(w_agg))
    print(f'Combined Solution converged in iterations: {rem_cache_dict["count"]}')
    results_df = metrics_pooled(X_all=X, y_actual_all=y, w_agg=w_agg)

def get_y_pred( X, w):
    y_pred=[]
    for i in range(number_of_regressions):
        w_=w[i]
        y_pred.append(np.dot(X, w_))
    return y_pred

def metrics_pooled( X_all, y_actual_all, w_agg ):
    mse_all =[]
    rmse_all=[]
    r2_score_all=[]

    dof = len(y_actual_all) - beta_vec_size
    y_pred_all = get_y_pred(X_all, w_agg)
    for i in range(number_of_regressions):
        y_pred=y_pred_all[i]
        y_actual=y_actual_all[i]

        sse = np.sum((y_pred - y_actual) ** 2)
        sst = np.sum((y_actual - np.mean(y_actual)) ** 2)
        r2_score_all.append(1 - (sse / sst))

        ss_reg =  np.sum((y_pred - np.mean(y_actual)) ** 2)

        mse = np.sum((y_pred - y_actual) ** 2)/dof
        mse_all.append(mse)
        rmse_all.append(np.sqrt(mse / len(y_actual_all)))

    varX_matrix = np.dot(X_all.T, X_all)
    ts_all, ps_all, se_beta_all = get_t_and_p_scores(mse_all, varX_matrix, w_agg, dof)
    ts_all_harsha, ps_all_harsha, se_beta_global_all_harsha = get_t_and_p_scores_harsha(mse_all, varX_matrix, w_agg, dof)

    '''
    results_dict = {}
    results_dict["MSE_pooled"] = mse_all
    results_dict["RMSE_pooled"] = rmse_all
    results_dict["r2_score_pooled"] = r2_score_all
    results_dict["ts_pooled"] = ts_all
    results_dict["ps_pooled"] = ps_all

    print(f'MSE : {str(mse_all)}')
    print(f'RMSE: {str(rmse_all)}')
    print(f'r2_score: {str(r2_score_all)}')
    print(f'ts_pooled: {str(ts_all)}')
    print(f'ps_pooled: {str(ps_all)}')
    print(f'ts_pooled: {str(ts_all_harsha)}')
    print(f'ps_pooled: {str(ps_all_harsha)}')

    results_df = pd.DataFrame(results_dict.items(), columns=y_labels)
    results_df = results_df.T
    print(results_df)

    tabulate(results_df, headers='keys', tablefmt='fancy_grid')
    '''

    print("**** Pooled Model Scores ****")
    results_dfs=[]

    for i in range(number_of_regressions):
        print(f'---- Results for : {y_labels[i]}')

        print(f'pooled MSE : {np.round(mse_all[i], NUM_DEC_PLACES)}')
        print(f'pooled RMSE : {np.round(rmse_all[i], NUM_DEC_PLACES)}')
        print(f'pooled r2_score: {np.round(r2_score_all[i], NUM_DEC_PLACES)}')

        temp_df=pd.DataFrame()
        temp_df["Coefficients"], temp_df["Standard Errors"], temp_df["t values"], temp_df["Probabilities"] = \
            [np.round(w_agg[i], NUM_DEC_PLACES), se_beta_all[i], ts_all[i], ps_all[i]]
        temp_df.index=X_labels
        print(temp_df)
        results_dfs.append(temp_df)

    return results_dfs



def metrics_fed(number_of_sites, site_data_dict, w_agg ):

    mean_y_global_all, dof_global_all = get_mean_y_global(number_of_sites, site_data_dict)
    sse_all_local=[]
    sst_all_local=[]
    rmse_all_local=[]
    varx_all_local=[]

    for site in range(number_of_sites):
        X = site_data_dict[f'local{site}']['X']
        y_actual_all = site_data_dict[f'local{site}']['y']
        m = len(y_actual_all)

        y_pred_all = get_y_pred(X, w_agg)
        sse_local = []
        sst_local = []
        rmse_local = []
        for i in range(number_of_regressions):
            y_pred = y_pred_all[i]
            y_actual = y_actual_all[i]
            mean_y_global = mean_y_global_all[i]

            sst_local.append(np.sum((y_actual - mean_y_global) ** 2))
            sse = np.sum((y_pred - y_actual) ** 2)
            sse_local.append(sse)
            rmse_local.append(np.sqrt(sse / m))

        sse_all_local.append(sse_local)
        sst_all_local.append(sst_local)
        rmse_all_local.append(rmse_local)
        varx_all_local.append(np.dot(X.T, X))

    sites_num_range =list(range(number_of_sites))
    SSE_global = sum([np.array(sse_all_local[site]) for site in sites_num_range])
    SST_global = sum([np.array(sst_all_local[site]) for site in sites_num_range])
    varX_matrix_global = sum([np.array(varx_all_local[site]) for site in sites_num_range])
    varX_matrix_global = varX_matrix_global.astype(float)

    r_squared_global = 1 - (SSE_global / SST_global)
    MSE_global = SSE_global / np.array(dof_global_all)
    RMSE_harsha = np.sqrt(MSE_global / number_of_sites)
    RMSE_sites_average = sum([np.array(rmse_all_local[site]) for site in sites_num_range])/number_of_sites

    ts_global, ps_global, se_beta_global_all= get_t_and_p_scores(MSE_global, varX_matrix_global, w_agg, dof_global_all)
    ts_global_harsha, ps_global_harsha, se_beta_global_all_harsha = get_t_and_p_scores_harsha(MSE_global, varX_matrix_global, w_agg, dof_global_all)


    '''
    
    print(f'MSE : {str(MSE_global)}')
    print(f'RMSE (harsha): {str(RMSE_harsha)}')
    print(f'RMSE_sites_average: {str(RMSE_sites_average)}')
    print(f'r2_score: {str(r_squared_global)}')

    print(f'ts_global: {str(ts_global)}')
    print(f'ps_global: {str(ps_global)}')
    print(f'ts_global_harsha: {str(ts_global_harsha)}')
    print(f'ps_global_harsha: {str(ps_global_harsha)}')


    results_dict = {}
    results_dict["MSE"] = MSE_global
    results_dict["RMSE_sites_average"] = RMSE_sites_average
    results_dict["RMSE (harsha)"] = RMSE_harsha
    results_dict["r2_score_global"] = r_squared_global
    results_dict["ts_global"] = ts_global
    results_dict["ps_global"] = ps_global
    results_df = pd.DataFrame(results_dict.items(), columns=y_labels)
    results_df = results_df.T

    #tabulate(results_df, headers='keys', tablefmt='fancy_grid')
    '''

    print("**** Federated Global Scores ****")
    results_dfs=[]

    for i in range(number_of_regressions):
        print(f'---- Results for : {y_labels[i]}')

        print(f'MSE : {np.round(MSE_global[i],NUM_DEC_PLACES)}')
        print(f'RMSE (harsha): {np.round(RMSE_harsha[i],NUM_DEC_PLACES)}')
        print(f'RMSE_sites_average: {np.round(RMSE_sites_average[i],NUM_DEC_PLACES)}')
        print(f'r2_score: {np.round(r_squared_global[i], NUM_DEC_PLACES)}')

        temp_df=pd.DataFrame()
        temp_df["Coefficients"], temp_df["Standard Errors"], temp_df["t values"], temp_df["Probabilities"] = \
            [np.round(w_agg[i], NUM_DEC_PLACES), se_beta_global_all[i], ts_global[i], ps_global[i]]
        temp_df.index=X_labels

        print(temp_df)
        results_dfs.append(temp_df)


    return results_dfs

def get_cost(y_actual, y_predicted):
    return np.average((y_actual-y_predicted)**2)

def get_mean_y_global(number_of_sites, site_data_dict):
    count_y_local=[]
    mean_y_local=[]
    mean_count_sums=0
    for site in range(number_of_sites):
        y_actual_all = site_data_dict[f'local{site}']['y']
        count_y_local.append([np.array(len(y_actual_all))])
        mean_y_local.append(y_actual_all.mean().to_list())

    count_y_local = np.array(count_y_local)
    mean_y_local = np.array(mean_y_local)
    mean_y_global = mean_y_local * count_y_local
    #TODO below is the BUG in the actual MSR code in remote_3()
    #mean_y_global = np.average(mean_y_global, axis=0)
    mean_y_global = mean_y_global.sum(axis=0) / np.sum(count_y_local)

    dof_global = sum(count_y_local) - beta_vec_size

    return mean_y_global, dof_global


def get_t_and_p_scores_harsha(MSE_global, varX_matrix_global, w_agg, dof_global_all):
    def t_to_p(ts_beta, dof):
        return [2 * sp.stats.t.sf(np.abs(t), dof) for t in ts_beta]

    ts_global = []
    ps_global = []

    for i in range(len(MSE_global)):
        var_covar_beta_global = MSE_global[i] * sp.linalg.inv(varX_matrix_global)
        se_beta_global = np.sqrt(var_covar_beta_global.diagonal())
        ts = (w_agg[i] / se_beta_global).tolist()
        ps = t_to_p(ts, dof_global_all)
        ts_global.append(ts)
        ps_global.append(ps)

    return ts_global, ps_global, se_beta_global

def get_t_and_p_scores(MSE_global, varX_matrix_global, w_agg, dof_global_all):
    def t_to_p(ts_beta, dof):
        return [2 * (1-sp.stats.t.cdf(np.abs(t), dof)) for t in ts_beta]

    ts_global = []
    ps_global = []
    se_beta_global=[]

    for i in range(len(MSE_global)):
        var_covar_beta_global = MSE_global[i] * sp.linalg.inv(varX_matrix_global)
        se_beta = np.sqrt(var_covar_beta_global.diagonal())
        ts = (w_agg[i] / se_beta).tolist()
        ps = t_to_p(ts, dof_global_all)
        ts_global.append(np.round(ts,NUM_DEC_PLACES))
        ps_global.append(np.round(ps,NUM_DEC_PLACES))
        se_beta_global.append((np.round(se_beta,NUM_DEC_PLACES)))

    return ts_global, ps_global, se_beta_global

def all_pooled_OLS(data_dir, file_suffix):
    import pandas as pd
    import numpy as np
    from sklearn import datasets, linear_model
    from sklearn.linear_model import LinearRegression
    import statsmodels.api as sm
    from scipy import stats
    global X_labels, Y_labels

    import pandas as pd
    all_X = pd.read_csv(data_dir + f'allPooled_X{file_suffix}.csv', sep=',')
    X_labels = all_X.columns
    all_y = pd.read_csv(data_dir + f'allPooled_y{file_suffix}.csv', sep=',')
    Y_labels = all_y.columns
    all_y=all_y.to_numpy()

    #for col in range(np.shape(all_y)[1]):
    #    est = sm.OLS(all_y[:,col], X)
    #    est2 = est.fit()
    #    print(est2.summary())
    for col in range(np.shape(all_y)[1]):
        print(f'\n#### Computing for : {Y_labels[col]}')

        X=all_X
        y=all_y[:, col]

        #### Method 1
        #est = sm.OLS(y, X)
        #est2 = est.fit()
        #print(est2.summary())


        #### Method 2
        X = all_X.loc[:, all_X.columns != '# const']
        lm = LinearRegression()
        lm.fit(X, y) #Removing constant as fit() adds it
        params = np.append(lm.intercept_, lm.coef_)
        predictions = lm.predict(X)
        newX = all_X
        MSE = (sum((y - predictions) ** 2)) / (len(newX) - len(newX.columns))

        var_b = MSE * (np.linalg.inv(np.dot(newX.T, newX)).diagonal())
        sd_b = np.sqrt(var_b)
        ts_b = params / sd_b

        p_values = [2 * (1 - stats.t.cdf(np.abs(i), (len(newX) - len(newX.columns)))) for i in ts_b]

        sd_b = np.round(sd_b, 3)
        ts_b = np.round(ts_b, 3)
        p_values = np.round(p_values, 3)
        params = np.round(params, 4)
        print(f'MSE: {MSE}')
        myDF3 = pd.DataFrame()
        myDF3["Coefficients"], myDF3["Standard Errors"], myDF3["t values"], myDF3["Probabilities"] = [params, sd_b, ts_b,
                                                                                                      p_values]
        myDF3.index=X_labels
        print(myDF3)

def normalize_columns(data_df, cols):
    for col in cols:
        data_df[col] = (data_df[col] - data_df[col].mean())/(data_df[col].std())
    return data_df


if __name__ == "__main__":
    tol=0.01#0.0001 #1000
    eta=0.01#0.01 #10000
    number_of_sites=8

    test_path= '../test/jess_input_debug/allFiles_withICV/'
    norm_cols=['ICV']

    print("\n\n\n###########################Running MSR CSV Debugger #################################\n\n")
    print(f'eta: {eta}, tol: {tol}')
    print(f'number_of_sites: {number_of_sites}')
    print(f'Normalizing columns (by removing mean value): {norm_cols}')
    #test_path= '../test/javier_jess_data/allFiles/'
    print(f'Data path: {test_path}')
    run_msr_fed(tol=tol, eta=eta, number_of_sites=number_of_sites, data_dir = test_path, norm_cols=norm_cols)
    run_msr_pooled(tol=tol, eta=eta, number_of_sites=number_of_sites, data_dir =test_path, norm_cols=norm_cols)
    print(f'X_labels: {X_labels}')
    print(f'y_labels: {y_labels}')

    all_pooled_OLS(data_dir=test_path, file_suffix="_norm['ICV']")

