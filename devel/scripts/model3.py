import glob

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

import pandas as pd

import random

from sklearn.preprocessing import StandardScaler

from typing import Tuple

# Randomize file order - does not work - Reaches 64%


def get_train_test_data(
    num_files: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_files = glob.glob('./datasets/CBS_SESSIONS_NORM/*.csv')
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


def get_binary_model():
    model = Sequential()
    model.add(Dense(5, activation='relu', input_shape=(5,))) # Change the input
    model.add(Dense(150, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='relu')) # Reduce layers and/or units
    model.add(Dropout(0.2))
    model.add(Dense(20, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model


train, test = get_train_test_data(60)
train_x = train[['ACC_X', 'ACC_Y', 'ACC_Z', 'BVP', 'EDA']]
train_y = train[['Condition']]
test_x = test[['ACC_X', 'ACC_Y', 'ACC_Z', 'BVP', 'EDA']]
test_y = test[['Condition']]
train_x = StandardScaler().fit_transform(train_x)
test_x = StandardScaler().fit_transform(test_x)

# print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)
class_weight = {0: 6., 1: 94.}

model = get_binary_model()
opt = keras.optimizers.SGD(learning_rate=1e-6)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['AUC'])
model.fit(
    train_x, train_y,
    validation_data=(test_x, test_y),
    batch_size=32, epochs=100, verbose=2,
    class_weight=class_weight
)

model.save('try4.h5')

all_files = glob.glob('./datasets/CBS_SESSIONS/*.csv')
pred_data = pd.read_csv(all_files[42], index_col = 0)
pred = pred_data[['ACC_X', 'ACC_Y', 'ACC_Z', 'BVP', 'EDA']]
pred = model.predict(pred)
pred = pd.DataFrame(pred)
# print(pred)
#pred.to_csv('dnd.csv')

