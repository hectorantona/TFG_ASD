import glob

import pandas as pd

# Script to see which files have agg and how many.

all_files = glob.glob('./datasets/CBS_SESSIONS/*.csv')

for file in all_files:

    file_to_check = pd.read_csv(file, index_col = 0)

    """
    for index, row in file_to_check.iterrows():
        if row['Condition'] > 0.1:
            # i = 0
            print(index)
    """

    file_to_check = file_to_check['Condition']
    print(file_to_check.sum(), file)


