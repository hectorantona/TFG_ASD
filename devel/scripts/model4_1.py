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
    model.add(Dense(1, activation='sigmoid'))
    return model


train, test = get_train_test_data(250)
train_x, train_y, test_x, test_y = prepare_all_data(train, test)

print(type(train_x), type(test_x), type(train_y), type(test_y))
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)

class_weight = {0: 16., 1: 84.}

model = get_binary_model()
opt = keras.optimizers.SGD(learning_rate=1e-5) # Antes 1e-6
model.compile(
    loss= 'binary_crossentropy', # Try focal_loss as loss function (not found)
    metrics=['AUC'],
    optimizer=opt,
    
)
model.fit(
    train_x, train_y,
    validation_data=(test_x, test_y),
    batch_size=32, epochs=700, verbose=2,
    class_weight=class_weight
)

model.save('model4_v2.h5')

"""
v1 - Reaches 0.6119, with 40 rows (10s) groups. 500 epochs. BS = 32.


"""

