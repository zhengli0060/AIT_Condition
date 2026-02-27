import pandas as pd
import os
from AIT_condition import AIT_test


def example_constant():
    data_path = os.path.join(r'example_data/Example_data_constant.csv')
    data = pd.read_csv(data_path)

    for j in [1, 2]:
        Z = f'IV{j}'
        A_Z = AIT_test(data, instrument=Z, relation='constant') 
        if A_Z['IV_validity']: print(f"The candidate IV{j} is valid✔️ !!!")
        else: print(f"The candidate IV{j} is invalid ❌!")

def example_non_constant():
    data_path = os.path.join(r'example_data/Example_data_nonconstant.csv')
    data = pd.read_csv(data_path)

    for j in [1, 2]:
        Z = f'IV{j}'
        A_Z = AIT_test(data, instrument=Z, relation='nonconstant') 
        if A_Z['IV_validity']: print(f"The candidate IV{j} is valid✔️ !!!")
        else: print(f"The candidate IV{j} is invalid ❌!")


def example_with_covariates():

    data_path = os.path.join(r'example_data/Example_data_with_W.csv')
    data = pd.read_csv(data_path)
    for j in [1, 2]:
        Z = f'IV{j}'
        A_Z = AIT_test(data, instrument=Z, covariates=['W1','W2','W3','W4','W5'], relation='constant')
        if A_Z['IV_validity']: print(f"The candidate IV{j} is valid✔️ !!!")
        else: print(f"The candidate IV{j} is invalid ❌!")





if __name__ == '__main__':

    
    example_constant()

    example_non_constant()

    example_with_covariates()

