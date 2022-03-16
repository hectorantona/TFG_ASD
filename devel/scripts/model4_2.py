import glob

import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential

import numpy as np

import pandas as pd

import random

from sklearn.preprocessing import StandardScaler

import tensorflow as tf

from typing import Tuple

# Randomize file order
# Gather input in 1 min gaps

# Temporal size in which you gather the gaps
# 120 = 30s, 240 = 1min, 720 = 3min
ONE_MIN = 40

def get_train_test_data(
    num_files: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Specify number of files you want to use for training and testing.
    80% goes to train, 20% goes to test
    '''
    all_files = glob.glob('./datasets/CBS_SESSIONS/*.csv')
    files_list = list(range(0, num_files))
    random.shuffle(files_list)
    threshold = round(num_files*0.8)
    train = pd.DataFrame()
    for it in range(0, threshold):
        train = pd.concat([train, pd.read_csv(all_files[files_list[it]], index_col = 0)])
    test = pd.DataFrame()
    for it in range(threshold, num_files):
        test = pd.concat([test, pd.read_csv(all_files[files_list[it]], index_col = 0)])
    print(files_list)
    print("Th", threshold)
    return train, test


def prepare_all_data(
    train: pd.DataFrame,
    test: pd.DataFrame,
): #-> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Separates both train and test into train_x, train_y and test_x, test_y respectively
    Gathers in 1 min gaps: 
        240 samples correspond to 1 min.
        x remains the same but reshaped in chuncks of 240
        y is resampled. If agg happened anytime in the gap, it corresponds to 1. 0 otherwise.
    Normalizes data (train_x and test_x)
    '''
    train_x = train[['ACC_X', 'ACC_Y', 'ACC_Z', 'BVP', 'EDA']]
    train_y = train[['Condition']]
    test_x = test[['ACC_X', 'ACC_Y', 'ACC_Z', 'BVP', 'EDA']]
    test_y = test[['Condition']]
    train_x = StandardScaler().fit_transform(train_x)
    test_x = StandardScaler().fit_transform(test_x)

    # Shave last rows so its multiple of 240 (everything converted to numpy) 
    n_rows_train = train_x.shape[0] - train_x.shape[0] % ONE_MIN
    train_x = train_x[:n_rows_train, :]
    train_y = train_y.to_numpy()
    train_y = train_y[:n_rows_train]
    
    n_rows_test = test_x.shape[0] - test_x.shape[0] % ONE_MIN
    test_x = test_x[:n_rows_test, :]
    test_y = test_y.to_numpy()
    test_y = test_y[:n_rows_test]

    # Reshape train_x and test_x into (X x 240 x 5) -+- X = og size / 240
    train_x = np.reshape(train_x, (int(n_rows_train / ONE_MIN), ONE_MIN, 5))
    test_x = np.reshape(test_x, (int(n_rows_test / ONE_MIN), ONE_MIN, 5))

    train_x = np.reshape(train_x, (int(n_rows_train / ONE_MIN), ONE_MIN*5))
    test_x = np.reshape(test_x, (int(n_rows_test / ONE_MIN), ONE_MIN*5))

    # Generate the shorter train_y and test_y into (X x 1) -+- X = og size / 240
    new_train_y = []
    new_test_y = []
    count_ones = 0
    count = 0
    for i in range(0, int(n_rows_train / ONE_MIN)):
        new_train_y.append(np.amax(train_y[ONE_MIN*i:(ONE_MIN*(i+1)-1)]))
        count_ones += np.amax(train_y[ONE_MIN*i:(ONE_MIN*(i+1)-1)])
        count += 1
    print('##########################')
    print(count_ones, count)
    print('##########################')
    for i in range(0, int(n_rows_test / ONE_MIN)):
        new_test_y.append(np.amax(test_y[ONE_MIN*i:(ONE_MIN*(i+1)-1)]))
    train_y = np.array(new_train_y)
    test_y = np.array(new_test_y)
    return train_x, train_y, test_x, test_y


def get_binary_model():
    model = Sequential()
    model.add(Dense(5, activation='relu', input_shape=(ONE_MIN*5,)))
    model.add(Dense(60, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(20, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(60, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model


train, test = get_train_test_data(250)
train_x, train_y, test_x, test_y = prepare_all_data(train, test)

print(type(train_x), type(test_x), type(train_y), type(test_y))
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)

class_weight = {0: 16., 1: 84.}

model = get_binary_model()
opt = keras.optimizers.SGD(learning_rate=1e-4) # Antes 1e-6
model.compile(
    loss= 'binary_crossentropy', # Try focal_loss as loss function (not found)
    metrics=['AUC'],
    optimizer=opt,
    
)
model.fit(
    train_x, train_y,
    validation_data=(test_x, test_y),
    batch_size=32, epochs=150, verbose=2,
    class_weight=class_weight
)

model.save('model4_v5.h5')

"""
Og - NN 60-40 (2 layers)
v1 - Reaches 0.6119, with 40 rows (10s) groups. 500 epochs. BS = 32. 
v2 - Reaches 0.6354, adding new layer (Autoencoder 60-20-60)
v3 - Reaches 0.6759, same as before but with weird start.
v4 - Reaches 0.6566, solid. LR to 1e-4 (before 1e-5) -> Less epochs (200), even less should work.
     Still (Autoencoder 60-20-60)
     File distribution: [24, 185, 182, 102, 244, 180, 239, 54, 129, 173, 212, 69, 62, 79, 241, 142, 219, 214, 162, 47, 145, 153, 115, 45, 116, 140, 59, 134, 209, 224, 38, 192, 131, 97, 72, 236, 167, 135, 9, 168, 204, 232, 1, 195, 40, 196, 216, 230, 63, 175, 163, 122, 80, 48, 30, 2, 111, 76, 70, 78, 73, 39, 77, 152, 95, 203, 92, 112, 0, 158, 18, 223, 132, 228, 19, 249, 247, 101, 246, 96, 98, 242, 193, 243, 83, 200, 233, 37, 218, 82, 49, 32, 7, 66, 248, 186, 105, 23, 146, 181, 124, 191, 4, 46, 26, 27, 215, 34, 199, 151, 113, 179, 53, 211, 231, 51, 207, 3, 81, 28, 85, 172, 206, 170, 171, 202, 149, 201, 213, 139, 238, 11, 184, 61, 29, 136, 15, 106, 160, 50, 222, 74, 100, 221, 164, 84, 56, 58, 99, 44, 52, 245, 235, 128, 93, 103, 67, 123, 22, 16, 12, 117, 229, 21, 43, 234, 31, 68, 169, 120, 107, 174, 188, 64, 148, 94, 33, 75, 183, 194, 178, 165, 35, 147, 126, 41, 5, 240, 225, 176, 125, 88, 197, 57, 87, 208, 133, 90, 17, 36, 109, 91, 166, 20, 14, 205, 10, 161, 8, 237, 127, 187, 55, 189, 144, 130, 108, 86, 25, 159, 110, 157, 60, 65, 119, 13, 226, 143, 104, 190, 138, 6, 220, 121, 114, 154, 227, 210, 150, 118, 198, 217, 141, 137, 155, 71, 89, 156, 177, 42]
"""

