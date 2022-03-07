import glob

from keras import Sequential
from keras.models import load_model

import pandas as pd

model = load_model('try2.h5')

all_files = glob.glob('./datasets/CBS_SESSIONS/*.csv')

for i in range(0, 100):
    file_to_test = pd.read_csv(all_files[i])
    test_x = file_to_test[['ACC_X', 'ACC_Y', 'ACC_Z', 'BVP', 'EDA']]
    test_y = file_to_test[['Condition']]
    score = model.evaluate(test_x, test_y, verbose = 0)
    print(score)

"""
pred = model.predict(test_x)
pred = pd.DataFrame(pred)
pred.to_csv('42.csv')"""