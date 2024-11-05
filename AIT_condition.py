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



# def linear_get_A_with_W(df, Z):
#     # Phase 1
#     W_data = df.filter(like='W')
#     first_stage = sm.OLS(df['Treatment'], sm.add_constant(pd.concat([W_data, df[Z]], axis=1))).fit()
#     df['X_hat'] = first_stage.predict()
#     # Phase 2
#     second_stage = sm.OLS(df['Outcome'], sm.add_constant(pd.concat([df['X_hat'], W_data], axis=1))).fit()
#     y_hat = second_stage.predict()
#
#     A = df['Outcome'].values - y_hat
#     Z_data = df[Z].values
#     W_data = W_data.values
#     return A.reshape(-1, 1), Z_data, W_data


def linear_get_A_with_W(df, Z):
    # 第一阶段：使用工具变量 Z 和控制变量 W 回归 X
    # 确保 num_W_id 是整数类型
    # num_W = int(num_W_id.split('_')[-1])  # 从 'num_W_2' 中提取出数字部分
    # W = df.iloc[:, 1:num_W + 1]  # 根据提取出的整数值正确选择 W 列

    # W = df.iloc[:, 1:num_W_id]  # 提取第2列到num_W_id-1列作为W
    W = df.filter(like='W')
    # W_data = df.filter(like='W')  # 获取所有以 'W' 开头的列
    first_stage = sm.OLS(df['Y'], sm.add_constant(pd.concat([W, df[Z]], axis=1))).fit()  # 将W和Z合并回归X
    df['X_hat'] = first_stage.predict()
    # 第二阶段：使用预测的 X_hat 回归 Y，控制变量 W
    second_stage = sm.OLS(df['X'], sm.add_constant(pd.concat([df['X_hat'], W], axis=1))).fit()
    y_hat = second_stage.predict()
    # 计算 A
    A = df['Y'].values - y_hat
    # 提取 Z 和 W 数据
    Z_data = df[Z].values  # 这里是多列Z，保持原形状
    W_data = W.values  # 多列W，保持原形状
    hat_value = None  # 如果需要后续操作，可以在这里进一步处理 hat_value
    return A.reshape(-1, 1), Z_data, W_data

def cf_with_W(data, Z):

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
    result = robjects.r.using_R_cf_with_W(r_dataframe, Z)

    A = np.array(result).reshape(-1, 1)
    Z_data = data[Z].values.reshape(-1, 1)
    W_data = data['W'].values.reshape(-1, 1)
    return A, Z_data, W_data

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


def random_forest_residuals(dependent_var, independent_vars):
    # 确保因变量和自变量没有缺失值，并对齐行数
    combined_data = pd.concat([dependent_var, independent_vars], axis=1).dropna()
    dependent_var_clean = combined_data.iloc[:, 0]  # 第0列为dependent_var
    independent_vars_clean = combined_data.iloc[:, 1:]  # 后面的列为independent_vars

    model = RandomForestRegressor(n_estimators=100)
    model.fit(independent_vars_clean, dependent_var_clean.values.ravel())  # 使用 W 回归 Z
    residuals = dependent_var_clean.values.ravel() - model.predict(independent_vars_clean)
    return residuals