# ✨ Testability of Instrumental Variables in Additive Nonlinear, Non-Constant Effects Models
This is the code repository for the auxiliary-based independence test (AIT) condition, used to test whether a variable is a valid instrument.

## Examples
We provide several examples of running the AIT condition in `example.py`.

## 🤖 Main Function: `AIT_condition.py`
Auxiliary-based Independence Test (AIT) Condition

**Input:**
- **Data:** A set of observed variables, datatype: DataFrame. For example:

| Treatment | Outcome | IV1 | ... | IV_n | W_1 | ... | W_n |  
|-----------|---------|-----|-----|------|-----|-----|-----|  
| ***       | ***     | *** | *** | ***  | *** | *** | *** |  
| ***       | ***     | *** | *** | ***  | *** | *** | *** |

where `IVs` are candidate IVs and `Ws` are covariates.
- **Z:** The candidate IV being tested, datatype: str.
- **alpha:** Significance level, datatype: float.
- **relation:** The type of causal relationship from X to Y, datatype: str. It can be either `linear` or `nonlinear`.

**Output:**
- The result of the A and Z independence test, datatype: dict.

***Note:***
- If A and Z are independent, it implies that we cannot reject Z as a valid IV.
- If A and Z are dependent, it implies that Z is an invalid IV.
## 🛠️ Requirements
- Python 3.9.13  
- R 4.2.3  
- rpy2 3.5.16  
- kerpy 0.1.0  
- scikit-learn 1.5.2  
- numpy 1.22.4  
- pandas 2.2.3  

## 🗒️ Notes
> We adopt the control function IV estimator proposed by Guo and Small (2016) for Additive Non-Parametric IV Models, which is a two-stage approach.  
> - Guo Z, Small D. S. Control function instrumental variable estimation of nonlinear causal effect models[J]. *Journal of Machine Learning Research*, 2016, 17(100): 1-35.

> To check the statistical independence of `A` and `Z`, we employ the large-scale HSIC test proposed by Zhang et al. (2018).  
> - Zhang Q, Filippi S, Gretton A, et al. Large-scale kernel methods for independence testing[J]. *Statistics and Computing*, 2018, 28: 113-130.

## 📝 Citation
If you use this code, please cite the following paper:
    Testability of Instrumental Variables in Additive Nonlinear, Non-Constant Effects Models
