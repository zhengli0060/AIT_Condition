import numpy as np
import pandas as pd
import rpy2.robjects as robjects
import os
from rpy2.robjects import pandas2ri
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import IV2SLS
import indTest.HSIC2 as fasthsic

def AIT_test(data, Z, num_W_id, **params):
    alpha = params.get('alpha', 10 / data.shape[0])
    verbose = params.get('verbose', False)
    relation = params.get('relation', 'nonlinear')

    indexs = list(data.columns)
    if 'Treatment' not in indexs or 'Outcome' not in indexs:
        print('Please ensure the inpure is the variable of data!')
        exit(-1)

    if relation == 'linear':
        # A, Z_data, W_data, hat_va = linear_get_A_with_W(data, Z)
        # A, Z_data, W_data, hat_va = GIN_get_A(data, Z)
        A, Z_data, W_data, hat_va = linear_get_A_OLS(data, Z, num_W_id)
    else:
        A, Z_data, W_data = cf(data, Z)
        # A, Z_data, W_data,hat_va = KIV_get_A(data, Z, rate)


    valid_Z, pValue_Z = fasthsic.test(A, Z_data, alpha=alpha, verbose=verbose)

    # if pValue_Z < alpha or np.any(pValue_W < alpha):
    if pValue_Z < alpha :
        valid_IV = 0
    else:
        valid_IV = 1
    # valid_IV = 1
    # pValue_Z=1
    # pValue_W=1
    return valid_IV, {'pValue_Z': pValue_Z}, A




def linear_get_A_with_W(df, Z):
    # 直接从 DataFrame 中提取数据列
    X_data = df['Treatment'].values.reshape(-1, 1)
    Y_data = df['Outcome'].values.reshape(-1, 1)
    Z_data = df[Z].values.reshape(-1, 1)

    # 动态提取所有 W 列
    W_data = df.filter(like='W').values  # 获取所有以 'W' 开头的列

    # Linear regression for Y on W
    model_YW = LinearRegression().fit(W_data, Y_data)
    residual_Y = Y_data - model_YW.predict(W_data)

    # Linear regression for X on W
    model_XW = LinearRegression().fit(W_data, X_data)
    residual_X = X_data - model_XW.predict(W_data)

    # Linear regression for Z on W
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
    hat_value = {'beta': f_hat}

    return A, residual_Z.reshape(-1, 1), W_data, hat_value


# def linear_get_A_OLS(df, Z):
#     # 第一阶段：使用工具变量 Z 和控制变量 W 回归 X
#     first_stage = sm.OLS(df['X'], sm.add_constant(df[['W', Z]])).fit()
#     df['X_hat'] = first_stage.predict()
#
#     # 第二阶段：使用预测的 X_hat 回归 Y，控制变量 W
#     second_stage = sm.OLS(df['Y'], sm.add_constant(df[['X_hat', 'W']])).fit()
#     y_hat = second_stage.predict()
#     A = df['Y'].values - y_hat
#     Z_data = df[Z].values.reshape(-1, 1)
#     W_data = df['W'].values.reshape(-1, 1)
#     hat_value = None
#     return A.reshape(-1, 1), Z_data,W_data,hat_value

def linear_get_A_OLS(df, Z, num_W_id):
    # 第一阶段：使用工具变量 Z 和控制变量 W 回归 X
    # W = df.iloc[:, 1:num_W_id]  # 提取第2列到num_W_id-1列作为W
    W_data = df.filter(like='W')  # 获取所有以 'W' 开头的列
    first_stage = sm.OLS(df['Treatment'], sm.add_constant(pd.concat([W_data, df[Z]], axis=1))).fit()  # 将W和Z合并回归X
    df['X_hat'] = first_stage.predict()
    # 第二阶段：使用预测的 X_hat 回归 Y，控制变量 W
    second_stage = sm.OLS(df['Outcome'], sm.add_constant(pd.concat([df['X_hat'], W_data], axis=1))).fit()
    y_hat = second_stage.predict()
    # 计算 A
    A = df['Outcome'].values - y_hat
    # 提取 Z 和 W 数据
    Z_data = df[Z].values  # 这里是多列Z，保持原形状
    W_data = W_data.values  # 多列W，保持原形状
    hat_value = None  # 如果需要后续操作，可以在这里进一步处理 hat_value
    return A.reshape(-1, 1), Z_data, W_data, hat_value




def cf(data, Z):
    path = os.path.join('E:/testability_IV_JMLR/control_IV/controlfunctionIV-main/R/using_cf.R')
    robjects.r.source(path)
    # 激活pandas到R数据框的转换功能
    pandas2ri.activate()

    # 转换为R数据框
    r_dataframe = pandas2ri.py2rpy(data)
    result = robjects.r.using_R_cf(r_dataframe, Z)

    A = np.array(result).reshape(-1, 1)
    Z_data = data[Z].values.reshape(-1, 1)
    W_data = data['W'].values.reshape(-1, 1)
    return A, Z_data, W_data