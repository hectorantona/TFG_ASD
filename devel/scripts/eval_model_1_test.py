import glob

from keras import Sequential
from keras.models import load_model

import pandas as pd

model = load_model('model3_200ep.h5')

all_files = glob.glob('./datasets/CBS_SESSIONS/*.csv')

check = pd.DataFrame()

files = [70, 71, 74, 77, 79, 89, 90, 99]
for i in files: #range(75, 100):
    check = pd.concat([check, pd.read_csv(all_files[i], index_col = 0)])

print(check.shape)

test_x = check[['ACC_X', 'ACC_Y', 'ACC_Z', 'BVP', 'EDA']]
test_y = check[['Condition']]    
score = model.evaluate(test_x, test_y, verbose = 0)
print(score)

"""
pred = model.predict(test_x)
pred = pd.DataFrame(pred)
pred.to_csv('42.csv')"""