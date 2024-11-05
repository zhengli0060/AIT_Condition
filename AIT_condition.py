import numpy as np
import pandas as pd
import rpy2.robjects as robjects
import os
from rpy2.robjects import pandas2ri
import statsmodels.api as sm
import indTest.HSIC2 as fasthsic

def AIT_test(data, Z, **params):
    alpha = params.get('alpha', 10 / data.shape[0])
    verbose = params.get('verbose', False)
    relation = params.get('relation', 'nonlinear')

    indexs = list(data.columns)
    if 'Treatment' not in indexs or 'Outcome' not in indexs:
        print('Please ensure the input is the variable of data!')
        exit(-1)

    if any(col.startswith('W') for col in data.columns):    # Determine whether covariates exist.
        if verbose: print("There are covariates.")
        if relation == 'linear':
            A, Z_data, W_data = linear_get_A_with_W(data, Z)
        else:
            A, Z_data, W_data = cf_with_W(data, Z)
    else:
        if verbose: print("There are no covariates.")
        if relation == 'linear':
            A = linear_get_A(data, Z)
            Z_data = data[Z].values.reshape(-1, 1)
        else:
            A, Z_data = cf_no_W(data, Z)

    pValue_Z = fasthsic.test(A, Z_data, alpha=alpha, verbose=verbose)
    if pValue_Z < alpha :
        valid_IV = 0
    else:
        valid_IV = 1
    return valid_IV, {'pValue_Z': pValue_Z}


def linear_get_A(df, Z):

    X_data = df['Treatment'].values.reshape(-1)
    Y_data = df['Outcome'].values.reshape(-1)
    Z_data = df[Z].values.reshape(-1)

    cov_YZ_given_W = np.cov(Y_data, Z_data)[0, 1]
    cov_XZ_given_W = np.cov(X_data, Z_data)[0, 1]

    if cov_XZ_given_W == 0:
        raise ValueError("Covariance of X and Z is zero, cannot divide by zero")
    f_hat = cov_YZ_given_W / cov_XZ_given_W
    A = Y_data - f_hat * X_data
    if len(A.shape) == 1: A = A.reshape(-1, 1)
    return A



def linear_get_A_with_W(df, Z):
    # Phase 1
    W_data = df.filter(like='W')
    first_stage = sm.OLS(df['Treatment'], sm.add_constant(pd.concat([W_data, df[Z]], axis=1))).fit()
    df['X_hat'] = first_stage.predict()
    # Phase 2
    second_stage = sm.OLS(df['Outcome'], sm.add_constant(pd.concat([df['X_hat'], W_data], axis=1))).fit()
    y_hat = second_stage.predict()

    A = df['Outcome'].values - y_hat
    Z_data = df[Z].values
    W_data = W_data.values
    return A.reshape(-1, 1), Z_data, W_data


def cf_with_W(data, Z):
    # path = os.path.join('control_IV/controlfunctionIV-main/R/using_cf.R')
    path = os.path.join('control_IV', 'controlfunctionIV-main', 'R', 'using_cf.R')
    robjects.r.source(path)
    pandas2ri.activate()

    r_dataframe = pandas2ri.py2rpy(data)
    result = robjects.r.using_R_cf_with_W(r_dataframe, Z)

    A = np.array(result).reshape(-1, 1)
    Z_data = data[Z].values.reshape(-1, 1)
    W_data = data['W'].values.reshape(-1, 1)
    return A, Z_data, W_data

def cf_no_W(data, Z):
    path = os.path.join('control_IV', 'controlfunctionIV-main', 'R', 'using_cf.R')
    robjects.r.source(path)
    pandas2ri.activate()

    r_dataframe = pandas2ri.py2rpy(data)
    result = robjects.r.using_R_cf_no_W(r_dataframe, Z)

    A = np.array(result).reshape(-1, 1)
    Z_data = data[Z].values.reshape(-1, 1)
    return A, Z_data