import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri
import pandas as pd
import numpy as np





def controlfunctionIV_py(data: pd.DataFrame, outcome: str, treatment: str, instruments: list, covariates: list = None) -> np.ndarray:
    
    """
    controlfunctionIV::cf
    Parameters:
    ----------
    data : pd.DataFrame
    outcome : str
    treatment : str
    instruments : list
    covariates : list, optional
    Returns:
    -------
    A : np.ndarray, the shape is (n_samples, )
    """
    
    importr("controlfunctionIV")
    importr("stats") 
    # pandas -> R dataframe
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_data = ro.conversion.py2rpy(data)

    ro.globalenv['data'] = r_data
    ro.globalenv['outcome'] = outcome
    ro.globalenv['treatment'] = treatment
    ro.globalenv['instruments'] = ro.StrVector(instruments)
    if covariates is not None:
        ro.globalenv['covariates'] = ro.StrVector(covariates)
    else:
        ro.globalenv['covariates'] = ro.NULL

    r_code = """
    Y <- scale(data[,outcome])
    D <- scale(data[, treatment])
    Z <- as.matrix(data[, instruments, drop = FALSE])
    if (!is.null(covariates) && length(covariates) > 0) {
        W <- as.matrix(data[, covariates, drop = FALSE])
        formula_str <- "Y ~ D + I(D^2) + I(D^3) + I(D^4) + W | Z + I(Z^2) + I(Z^3) + W + I(W^2) + I(W^3)"
    } else {
        W <- NULL
        formula_str <- "Y ~ D + I(D^2) + I(D^3) + I(D^4) | Z + I(Z^2) + I(Z^3)"
    }
    cf_formula <- as.formula(formula_str)
    cf_model <- cf(cf_formula)
    
    coef <- cf_model$coefficients
    coef[is.na(coef)] <- 0

    if (!is.null(W)) {
        nW <- ncol(W)
        A <- Y - coef[2]*D - coef[3]*I(D^2) - coef[4]*I(D^3) - coef[5]*I(D^4) - W %*% coef[6:(5+nW)]
    } else {
        A <- Y - coef[2]*D - coef[3]*I(D^2) - coef[4]*I(D^3) - coef[5]*I(D^4)
    }
    """
    ro.r(r_code)

    
    with localconverter(ro.default_converter + pandas2ri.converter):
        A_py = ro.conversion.rpy2py(ro.globalenv['A'])

    return A_py.reshape(-1)



