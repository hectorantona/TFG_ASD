import glob

import pandas as pd

# Shows all agg and al rows to compute de rate (imbalanced dataset)

all_files = glob.glob('./datasets/CBS_SESSIONS/*.csv')
total_ones = 0
total_rows = 0
for file in all_files:
    file_to_check = pd.read_csv(file, index_col = 0)
    total_ones += file_to_check['Condition'].sum()
    print(file_to_check.shape[0], type(file_to_check.shape[0]))
    total_rows += file_to_check.shape[0]

print(total_ones, total_rows)
