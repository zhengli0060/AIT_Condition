import pandas as pd
import os
from AIT_condition import AIT_test
from rpy2.robjects import r
def AIT_condition():

    # data_path = os.path.join(r'example_data/rrbinom_simulation_3.csv')
    N=5000
    num_W = 5
    num_W_id = f'num_W_{num_W}'
    i = 0

    data_path = os.path.join(
        f'E:/testability_IV_JMLR/JMLR2023_Burauel2023instrument_validity_test/data_linear/sample_{N}_{num_W_id}/repeat_simulation/rrbinom_simulation_{i}.csv')
    data = pd.read_csv(data_path)
    # 更换特定列名（例如将 'A' 更换为 'X'，'C' 更换为 'Z'）
    data.rename(columns={'Y': 'Outcome', 'X': 'Treatment', 'Z1':'IV1', 'Z2':'IV2'}, inplace=True)
    r('setwd("control_IV/controlfunctionIV-main/R")')
    for j in [1, 2]:
        Z = f'IV{j}'
        A_Z = AIT_test(data, Z,relation='linear',verbose=True)
        print(A_Z)
        if A_Z['IV_validity']: print(f'The candidate IV{j} is valid !!!')
        else: print(f'The candidate IV{j} is invalid !')

if __name__ == '__main__':
    AIT_condition()

