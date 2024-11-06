import numpy as np
import pandas as pd
import rpy2.robjects as robjects
import os
from rpy2.robjects import pandas2ri
import statsmodels.api as sm
import indTest.HSIC2 as fasthsic
from sklearn.ensemble import RandomForestRegressor
from rpy2.robjects import r
import rpy2.robjects.packages as rpackages
from sklearn.linear_model import LinearRegression

def AIT_test(data, Z, **params):
    alpha = params.get('alpha', 10 / data.shape[0])
    verbose = params.get('verbose', False)
    relation = params.get('relation', 'linear')

    indexs = list(data.columns)
    if 'Treatment' not in indexs or 'Outcome' not in indexs:
        print('Please ensure the input is the variable of data!')
        exit(-1)

    if any(col.startswith('W') for col in data.columns):    # Determine whether covariates exist.
        if verbose: print("There are covariates.")
        if relation == 'linear':
            A, Z_data = linear_get_A_with_W(data, Z)
        else:
            raise ValueError ('No such method')
    else:
        if verbose: print("There are no covariates.")
        if relation == 'linear':
            A = linear_get_A(data, Z)
            Z_data = data[Z].values.reshape(-1, 1)
        else:
            A, Z_data = cf_no_W(data, Z)

    pValue_Z = fasthsic.test(A, Z_data, alpha=alpha, verbose=verbose)
    if pValue_Z < alpha :
        valid_IV = False
    else:
        valid_IV = True
    return {'IV_validity':valid_IV,'pValue_Z': pValue_Z}


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
    if len(A.shape) == 1 : A = A.reshape(-1, 1)
    return A


def linear_get_A_with_W(df, Z):

    X_data = df['Treatment'].values.reshape(-1, 1)
    Y_data = df['Outcome'].values.reshape(-1, 1)
    Z_data = df[Z].values.reshape(-1, 1)
    W_data = df.filter(like='W').values

    # Linear regression
    model_YW = LinearRegression().fit(W_data, Y_data)
    residual_Y = Y_data - model_YW.predict(W_data)
    model_XW = LinearRegression().fit(W_data, X_data)
    residual_X = X_data - model_XW.predict(W_data)
    model_ZW = LinearRegression().fit(W_data, Z_data)
    residual_Z = Z_data - model_ZW.predict(W_data)

    if len(residual_Y.shape) == 2: residual_Y = residual_Y.reshape(-1)
    if len(residual_X.shape) == 2: residual_X = residual_X.reshape(-1)
    if len(residual_Z.shape) == 2: residual_Z = residual_Z.reshape(-1)

    cov_YZ_given_W = np.cov(residual_Y, residual_Z)[0, 1]
    cov_XZ_given_W = np.cov(residual_X, residual_Z)[0, 1]

    if cov_XZ_given_W == 0:
        raise ValueError("Covariance of X and Z is zero, cannot divide by zero")
    f_hat = cov_YZ_given_W / cov_XZ_given_W
    A = residual_Y - f_hat * residual_X
    if len(A.shape) == 1: A = A.reshape(-1, 1)


    return A, residual_Z.reshape(-1, 1)



def cf_no_W(data, Z):

    if not rpackages.isinstalled('readxl'):
        rpackages.importr('readxl')
    if not rpackages.isinstalled('Formula'):
        rpackages.importr('Formula')
    # path = os.path.join('control_IV/controlfunctionIV-main/R/using_cf.R')
    robjects.r.source('pretest.R')
    robjects.r.source('cf.R')
    path = os.path.join('using_cf.R')
    robjects.r.source(path)
    pandas2ri.activate()

    r_dataframe = pandas2ri.py2rpy(data)
    result = robjects.r.using_R_cf_no_W(r_dataframe, Z)

    A = np.array(result).reshape(-1, 1)
    Z_data = data[Z].values.reshape(-1, 1)
    return A, Z_data


# def cf_with_W(data, Z):
#
#     if not rpackages.isinstalled('readxl'):
#         rpackages.importr('readxl')
#     if not rpackages.isinstalled('Formula'):
#         rpackages.importr('Formula')
#     # path = os.path.join('control_IV/controlfunctionIV-main/R/using_cf.R')
#     robjects.r.source('pretest.R')
#     robjects.r.source('cf.R')
#     path = os.path.join('using_cf.R')
#     robjects.r.source(path)
#     pandas2ri.activate()
#     r_dataframe = pandas2ri.py2rpy(data)
#     result = robjects.r.using_R_cf_with_W(r_dataframe, Z)
#
#     A = np.array(result).reshape(-1, 1)
#     Z_data = data[Z].values.reshape(-1, 1)
#     W_data = data['W'].values.reshape(-1, 1)
#     return A, Z_data, W_data