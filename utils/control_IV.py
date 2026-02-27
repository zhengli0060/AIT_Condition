import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri
import pandas as pd
import numpy as np


def fit_h_controlfunction(data: pd.DataFrame, outcome: str, treatment: str, instruments: list, covariates: list = None, poly_powers: dict = None) -> callable:
    """
    Train control-function IV model on D1.
    Returns a model object that can later predict h(D,W).
    """

    importr("controlfunctionIV")
    importr("stats")
    # default powers for D, Z and W
    if poly_powers is None:
        poly_powers = {"D": [1, 2, 3, 4], "Z": [1, 2]}

    with localconverter(ro.default_converter + pandas2ri.converter):
        r_data = ro.conversion.py2rpy(data)

    ro.globalenv["data"] = r_data
    ro.globalenv["outcome"] = outcome
    ro.globalenv["treatment"] = treatment
    ro.globalenv["instruments"] = ro.StrVector(instruments)
    ro.globalenv["covariates"] = ro.StrVector(covariates) if covariates else ro.NULL

    # Build formula strings based on requested powers
    def powers_to_terms(var, powers):
        terms = []
        for p in powers:
            if p == 1:
                terms.append(f"{var}")
            else:
                terms.append(f"I({var}^{p})")
        return terms

    D_terms = powers_to_terms("D", poly_powers.get("D", [1, 2, 3, 4]))
    Z_terms = powers_to_terms("Z", poly_powers.get("Z", [1, 2]))

    # Convert Python lists to R code fragments
    D_left = " + ".join(D_terms)
    Z_right = " + ".join(Z_terms)

    if covariates:
        # include W (as covariates) on both sides
        r_formula = f'Y ~ {D_left} + W | {Z_right} + W + I(W^2) + I(W^3)'
    else:
        r_formula = f'Y ~ {D_left} | {Z_right}'

    r_code = f"""
    Y <- scale(data[, outcome])
    D <- scale(data[, treatment])
    Z <- as.matrix(data[, instruments, drop=FALSE])

    if (!is.null(covariates) && length(covariates) > 0) {{
        W <- as.matrix(data[, covariates, drop=FALSE])
    }} else {{
        W <- NULL
    }}

    cf_model <- cf(as.formula(\"{r_formula}\"))
    coef <- cf_model$coefficients
    coef[is.na(coef)] <- 0

    nW <- if (is.null(W)) 0 else ncol(W)
    names_coef <- names(coef)
    """

    ro.r(r_code)

    coef = np.array(ro.globalenv["coef"])
    nW = int(ro.globalenv["nW"][0])
    names_coef = list(ro.globalenv["names_coef"])
    # Build a sympy expression representing the fitted h(D,W)
    try:
        from sympy import symbols, lambdify
    except Exception:
        raise RuntimeError("sympy is required for constructing the predictor. Install sympy.")

    # Create symbols for D and covariates
    D_sym = symbols(treatment)
    W_syms = []
    if covariates:
        for w in covariates:
            W_syms.append(symbols(w))

    # Map coefficient names to sympy terms. The names from R will often be like 'D', 'I(D^2)', 'WW1'
    expr = 0
    for name, value in zip(names_coef, coef):
        
        if name == "D":
            term = D_sym
        elif name.startswith(f"I(D^"):
            power = int(name.split('^')[1].rstrip(')'))
            term = D_sym ** power
        elif covariates and name.startswith(f"W"):
            index = int(name[-1]) - 1  # W1, W2, ...
            term = W_syms[index]
        else:
            term = 0

        expr += float(value) * (term)

    # print("Fitted h(D,W) expression:", expr)
    # lambdify a predictor that takes pandas Series/arrays for treatment and covariates and returns hat_Y
    arg_symbols = [D_sym] + W_syms
    predictor_h = lambdify(arg_symbols, expr, modules=["numpy"]) if expr != 0 else None

    # print(f"Created predictor function: {predictor_h.__doc__}")

    """Predictor function for h(D,W)
    Used example:
    hat_Y = predictor_h(D_array, W1_array, W2_array, ...)
    where D_array is a numpy array or pandas Series of treatment values,
    and W1_array, W2_array, ... are arrays/Series for each covariate in the same order as provided during model fitting.
    """

    return predictor_h


