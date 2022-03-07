import matplotlib.pyplot as plt

import pandas as pd

ACC = pd.read_csv('./datasets/CBS_DATA_ASD_ONLY/1206.01/1206.01_04_ACC_matched.csv')
BVP = pd.read_csv('./datasets/CBS_DATA_ASD_ONLY/1206.01/1206.01_04_BVP_matched.csv')
EDA = pd.read_csv('./datasets/CBS_DATA_ASD_ONLY/1206.01/1206.01_04_EDA_matched.csv')

x_acc = ACC['Timestamp']
acc_x_acc = ACC['ACC_X']
acc_y_acc = ACC['ACC_Y']
acc_z_acc = ACC['ACC_Z']
agg_acc = ACC['AGG']

x_bvp = BVP['Timestamp']
data_bvp = BVP['BVP']
agg_bvp = BVP['AGG']

x_eda = EDA['Timestamp']
data_eda = EDA['EDA']
agg_eda = EDA['AGG']

plt.plot(x_eda, EDA['EDA'])
plt.show()