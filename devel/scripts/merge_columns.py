import glob

import pandas as pd

from sklearn.preprocessing import StandardScaler

def get_patient_id(path: str) -> str:
    """
    Read patient_id from:
    ./datasets/CBS_DATA_ASD_ONLY\4365.01\4365.01_01_ACC_matched.csv
    And obtain 1175 as str
    """
    patient_id = path.split('\\')[1]
    patient_id = patient_id.split('.')[0]
    return patient_id

FOLDER = '1206.01'
PATH = './datasets/Raw_Dataset/CBS_DATA_ASD_ONLY/'+ '**' +'/'
NAME = ''

all_files = glob.glob(PATH + "/*.csv")
it = 1215
patient_id = 0
session = 1
final_name = ''
while it < len(all_files):
    # get the patient id for the final name
    new_patient_id = get_patient_id(all_files[it])
    ACC_filename = all_files[it]
    BVP_filename = all_files[it + 1]
    EDA_filename = all_files[it + 2]
    acc = pd.read_csv(ACC_filename, index_col = 0)[['ACC_X', 'ACC_Y', 'ACC_Z']]
    bvp = pd.read_csv(BVP_filename, index_col = 0)[['BVP']]
    eda = pd.read_csv(EDA_filename, index_col = 0)[['EDA']]
    cond = pd.read_csv(EDA_filename, index_col = 0)[['Condition']] # Chunks of 3min and flatten the input
    complete = acc.merge(bvp, on='Timestamp')
    complete = complete.merge(eda, on='Timestamp')
    complete = StandardScaler().fit_transform(complete)
    complete = pd.DataFrame(
        complete,
        columns = ['ACC_X', 'ACC_Y', 'ACC_Z', 'BVP', 'EDA'],
        index=cond.index
    )
    complete['Condition'] = cond

    if patient_id == new_patient_id:
        session += 1
    else:
        session = 1
    patient_id = new_patient_id
    if session < 10:
        final_name = f"./datasets/CBS_SESSIONS_NORM/{new_patient_id}_0{session}.csv"
    else:
        final_name = f"./datasets/CBS_SESSIONS_NORM/{new_patient_id}_{session}.csv"

    complete.to_csv(final_name)
    print(final_name)
    it += 3


