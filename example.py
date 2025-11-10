import pandas as pd
import os
from AIT_condition import AIT_test

def AIT_condition():

    data_path = os.path.join(r'example_data/linear_data_5000.csv')   # linear_data_5000.csv   # nonlinear_data_5000.csv  # Example_Data_compare_Ktest_5000.csv  # Example_Data_compare_PIM_5000.csv
    data = pd.read_csv(data_path)

    for j in [1, 2]:
        Z = f'IV{j}'
        A_Z = AIT_test(data, Z, relation='linear')   # relation = 'linear' or 'nonlinear'
        if A_Z['IV_validity']: print(f'The candidate IV{j} is valid !!!')
        else: print(f'The candidate IV{j} is invalid !')

if __name__ == '__main__':
    AIT_condition()

