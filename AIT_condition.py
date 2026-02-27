import random
import numpy as np
import pandas as pd
from utils.hsic_ind import hsic 
from utils.control_IV import fit_h_controlfunction as fit_h
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
np.random.seed(321)
random.seed(321)


def AIT_test(data: pd.DataFrame, outcome: str='Outcome', treatment: str='Treatment', instrument: str='IV1', covariates: list = [], rho: float=0.7, relation='constant', **params) -> dict:

    assert rho > 0 and rho < 1, "sample splitting ratio, rho, must be between 0 and 1"
    assert relation in ['constant', 'nonconstant'], "relation must be either 'constant' or 'nonconstant'"
    assert outcome in data.columns, "outcome must be in data.columns"
    assert treatment in data.columns, "treatment must be in data.columns"
    assert instrument in data.columns, "instrument must be in data.columns"
    assert all(col in data.columns for col in covariates), "all covariates must be in data.columns"

    n = data.shape[0]
    fith_size = int(n * rho)
    train_df = data.sample(n=fith_size, random_state=42) # for fitting h
    test_df = data.drop(train_df.index) # for testing A and Z independence
    test_size = len(test_df)
    alpha = params.get('alpha', None)
    if alpha is None:
        alpha = 0.01 if test_size >= 1000 else 0.05

    if relation == 'constant':
        if len(covariates) == 0:
            res = ait_test_constant_no_W(train_df, test_df, outcome, treatment, instrument)
        else:
            res = ait_test_constant_with_W(train_df, test_df, outcome, treatment, instrument, covariates)
    else:
        if len(covariates) == 0:
            res = ait_test_nonconstant_no_W(train_df, test_df, outcome, treatment, instrument)
        else:
            res = ait_test_nonconstant_with_W(train_df, test_df, outcome, treatment, instrument, covariates)

    
    pvalue = hsic(res['A_est'], res['Z_test'], **params)

    return {'IV_validity':pvalue > alpha,'pValue_Z': pvalue}


def ait_test_constant_no_W(train_df: pd.DataFrame, test_df: pd.DataFrame, outcome: str, treatment: str, instrument: str):

    cov_YZ = np.cov(train_df[outcome], train_df[instrument])[0, 1]
    cov_XZ = np.cov(train_df[treatment], train_df[instrument])[0, 1]
    if cov_XZ== 0:
        raise ValueError("Covariance of X and Z is zero, cannot divide by zero")
    beta = cov_YZ / cov_XZ

    
    A_est = test_df[outcome].values - beta * test_df[treatment].values
    
    return {'A_est': A_est, 'Z_test': test_df[instrument].values}


def ait_test_constant_with_W(train_df: pd.DataFrame, test_df: pd.DataFrame, outcome: str, treatment: str, instrument: str, covariates: list[str]):

    # Linear regression for Y,X,Z on W
    train_X = train_df[treatment].values.reshape(-1, 1)
    train_Y = train_df[outcome].values.reshape(-1, 1)
    train_Z = train_df[instrument].values.reshape(-1, 1)
    train_W = train_df[covariates].values

    model_YW = LinearRegression().fit(train_W, train_Y)
    train_residual_Y = train_Y - model_YW.predict(train_W)
    model_XW = LinearRegression().fit(train_W, train_X)
    train_residual_X = train_X - model_XW.predict(train_W)
    model_ZW = LinearRegression().fit(train_W, train_Z)
    train_residual_Z = train_Z - model_ZW.predict(train_W)

    cov_YZ_given_W = np.cov(train_residual_Y.reshape(-1), train_residual_Z.reshape(-1))[0, 1]
    cov_XZ_given_W = np.cov(train_residual_X.reshape(-1), train_residual_Z.reshape(-1))[0, 1]

    if cov_XZ_given_W == 0:
        raise ValueError("Covariance of X and Z is zero, cannot divide by zero")

    beta =  cov_YZ_given_W/cov_XZ_given_W

    test_X = test_df[treatment].values.reshape(-1, 1)
    test_Y = test_df[outcome].values.reshape(-1, 1)
    test_Z = test_df[instrument].values.reshape(-1, 1)
    test_W = test_df[covariates].values
    test_residual_Y = test_Y - model_YW.predict(test_W)
    test_residual_X = test_X - model_XW.predict(test_W)
    test_residual_Z = test_Z - model_ZW.predict(test_W)
    A_est = test_residual_Y - beta * test_residual_X
    
    return {'A_est': A_est, 'Z_test': test_residual_Z}


def ait_test_nonconstant_no_W(train_df: pd.DataFrame, test_df: pd.DataFrame, outcome: str, treatment: str, instrument: str):

    model = fit_h(train_df, outcome=outcome, treatment=treatment, instruments=[instrument])
    A_est = test_df[outcome].values - model(test_df[treatment])
    
    return {'A_est': A_est.values, 'Z_test': test_df[instrument].values}


def ait_test_nonconstant_with_W(train_df: pd.DataFrame, test_df: pd.DataFrame, outcome: str, treatment: str, instrument: str, covariates: list[str]):

    model = fit_h(train_df, outcome=outcome, treatment=treatment, instruments=[instrument], covariates=covariates)
    A_est = test_df[outcome].values - model(test_df[treatment], *[test_df[W] for W in covariates])

    model = RandomForestRegressor(n_estimators=100)
    model.fit(train_df[covariates].values, train_df[instrument].values.reshape(-1))

    test_residual_Z = test_df[instrument].values.reshape(-1) - model.predict(test_df[covariates].values)
    
    return {'A_est': A_est.values, 'Z_test': test_residual_Z.reshape(-1,1)}
