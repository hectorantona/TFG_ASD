import glob

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

import pandas as pd

from sklearn.preprocessing import StandardScaler

from typing import Tuple

# Reaches 77.5% - Not reliable

def get_train_test_data(
    train_files: int,
    test_files: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_files = glob.glob('./datasets/CBS_SESSIONS/*.csv')
    train = pd.DataFrame()
    it = 0
    for it in range(0, train_files - 1):
        train = pd.concat([train, pd.read_csv(all_files[it], index_col = 0)])
    test = pd.DataFrame()
    for it in range(train_files, train_files + test_files - 1):
        test = pd.concat([test, pd.read_csv(all_files[it], index_col = 0)])
    return train, test


def get_binary_model():
    model = Sequential()
    model.add(Dense(5, activation='relu', input_shape=(5,)))
    model.add(Dense(150, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(20, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model


train, test = get_train_test_data(40, 20)
train_x = train[['ACC_X', 'ACC_Y', 'ACC_Z', 'BVP', 'EDA']]
train_y = train[['Condition']]
test_x = test[['ACC_X', 'ACC_Y', 'ACC_Z', 'BVP', 'EDA']]
test_y = test[['Condition']]
train_x = StandardScaler().fit_transform(train_x)
test_x = StandardScaler().fit_transform(test_x)

# print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)

model = get_binary_model()
opt = keras.optimizers.Adam(learning_rate=1e-6)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['AUC'])
model.fit(train_x, train_y, validation_data=(test_x, test_y), batch_size=100, epochs=50, verbose=2)

model.save('model2.h5')

"""
all_files = glob.glob('./datasets/CBS_SESSIONS/*.csv')
pred_data = pd.read_csv(all_files[42], index_col = 0)
pred = pred_data[['ACC_X', 'ACC_Y', 'ACC_Z', 'BVP', 'EDA']]
pred = model.predict(pred)
pred = pd.DataFrame(pred)
print(pred)
pred.to_csv('dnd.csv')
"""

