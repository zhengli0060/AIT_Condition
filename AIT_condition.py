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
    method = params.get('method', 'KI')
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

    if relation == 'linear':
        if method == 'KI':
            valid_Z, pValue_Z = KI(A, Z_data, alpha=alpha, verbose=verbose)
            # if pValue_Z > alpha:
            #     valid_W = []
            #     pValue_W = []
            #     for i in range(W_data.shape[1]):
            #         valid_w, pValue_w = KI(A, W_data[:, i], alpha=alpha, verbose=verbose)
            #         valid_W.append(valid_w)
            #         pValue_W.append(pValue_w)
            #     valid_W = np.array(valid_W)
            #     pValue_W = np.array(pValue_W)
            # else:
            #     valid_W = valid_Z
            #     pValue_W = pValue_Z
        elif method == 'fastHSIC':
            valid_Z, pValue_Z = fasthsic.test(A, Z_data, alpha=alpha, verbose=verbose)
            # if pValue_Z > alpha:
            #     valid_W = []
            #     pValue_W = []
            #     for i in range(W_data.shape[1]):
            #         valid_w, pValue_w = fasthsic.test(A, W_data[:, i], alpha=alpha, verbose=verbose)
            #         valid_W.append(valid_w)
            #         pValue_W.append(pValue_w)
            #     valid_W = np.array(valid_W)
            #     pValue_W = np.array(pValue_W)
            # else:
            #     valid_W = valid_Z
            #     pValue_W = pValue_Z
        else:
            raise ValueError('no such method')

    else:
        raise ValueError('no such method')

    # if pValue_Z < alpha or np.any(pValue_W < alpha):
    if pValue_Z < alpha :
        valid_IV = 0
    else:
        valid_IV = 1
    # valid_IV = 1
    # pValue_Z=1
    # pValue_W=1
    return valid_IV, {'pValue_Z': pValue_Z}, A


def getomega(df, X, Z):
    cov_m = np.cov(df, rowvar=False)
    col = list(df.columns)
    Xlist = []
    Zlist = []
    for i in X:
        t = col.index(i)
        Xlist.append(t)
    for i in Z:
        t = col.index(i)
        Zlist.append(t)
    B = cov_m[Xlist]
    B = B[:, Zlist]
    A = B.T
    u, s, v = np.linalg.svd(A)
    lens = len(X)
    omega = v.T[:, lens - 1]
    omegalen = len(omega)
    omega = omega.reshape(1, omegalen)

    re = omega
    re = re.reshape(-1)
    re[1] = re[1] / re[0] * (-1)
    re[2] = re[2] / re[0] * (-1)
    re[0] = re[0] / re[0]

    return re


def GIN_get_A(df, Z):
    omega = getomega(df, ['Outcome', 'Treatment', 'W'], [Z, 'W'])
    beta = omega[1]
    m = omega[2]

    # 直接从 DataFrame 中提取数据列
    X_data = df['Treatment'].values.reshape(-1, 1)
    Y_data = df['Outcome'].values.reshape(-1, 1)
    Z_data = df[Z].values.reshape(-1, 1)
    W_data = df['W'].values.reshape(-1, 1)

    # beta = 2
    # m = 2
    A = Y_data - beta * X_data - m * W_data

    if len(A.shape) == 1: A = A.reshape(-1, 1)
    hat_value = {'beta': beta, 'm': m}
    return A, Z_data, W_data, hat_value



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

def KI(X, Y, **params):
    alpha = params.get('alpha', 10 / X.shape[0])
    verbose = params.get('verbose', False)

    if len(X.shape) == 1:
        lens = len(X)
        X = X.reshape(lens, 1)
        Y = Y.reshape(lens, 1)

    data_CI = np.hstack((X, Y))
    if verbose: print(f'data_CI.shape:{data_CI.shape}')
    ki_obj = CIT(data_CI, "kci", est_width='manual', kwidthx=0.05, kwidthy=0.05)  # 创建 CIT 实例
    pValua = ki_obj(0, 1)
    if pValua < alpha:
        return False, pValua
    else:
        return True, pValua  # 0 表示A与Z不独立，Z为假，    1 表示A与Z独立，Z maybe为真


def KCI(X, Y, Z, **params):
    alpha = params.get('alpha', 10 / X.shape[0])
    verbose = params.get('verbose', False)

    if len(X.shape) == 1:
        lens = len(X)
        X = X.reshape(lens, 1)
        Y = Y.reshape(lens, 1)
        Z = Z.reshape(lens, 1)

    data_CI = np.hstack((X, Y, Z))
    if verbose: print(f'data_CI.shape:{data_CI.shape}')
    ki_obj = CIT(data_CI, "kci", est_width='manual', kwidthx=1, kwidthy=0.5, kwidthz=0.5)  # 创建 CIT 实例
    pValua = ki_obj(0, 1, [2])
    if pValua < alpha:
        return False, pValua
    else:
        return True, pValua  # 0 表示A与Z不独立，Z为假，    1 表示A与Z独立，Z maybe为真


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