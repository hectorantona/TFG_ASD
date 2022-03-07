
import numpy as np
import pickle
import os
import pandas as pd
import glob


def get_user_ids_dictionary_from_directory_root(dir_path, pathStyle='/', idNDigits=4):
    return get_user_ids_dictionary_from_directory_list(list_folders_in_dir(dir_path), pathStyle, idNDigits)


def list_folders_in_dir(dir):
    # list all folders including the current path
    folders = [x[0] for x in os.walk(dir)]
    # delete current path from list
    # del folders[0]
    return folders[1:]


def get_user_ids_dictionary_from_directory_list(dirList, pathStyle='/', idNDigits=4):
    # ids = [x.replace(path + pathStyle, '') for x in folders]
    uid = [x.split(pathStyle)[-1][0:idNDigits] for x in dirList]
    new_id = np.unique(uid, return_inverse=True)[1]
    return dict(zip(new_id + 1, uid))


def select_feat_from_feat_code(feat_code):
    selected_feat = []

    if feat_code == -1:
        selected_feat = ['TimePastAggression']

    elif feat_code == 0:
        selected_feat = ['AGGObserved', 'TimePastAggression']

    elif feat_code == 1:
        selected_feat = ['Xfirst', 'Xlast', 'Xmax', 'Xmin', 'Xmean', 'Xmedian', 'Xnunique',
                                       'Xstd', 'Xsum', 'Xvar', 'Yfirst', 'Ylast', 'Ymax', 'Ymin', 'Ymean',
                                       'Ymedian', 'Ynunique', 'Ystd', 'Ysum', 'Yvar', 'Zfirst', 'Zlast',
                                       'Zmax', 'Zmin', 'Zmean', 'Zmedian', 'Znunique', 'Zstd', 'Zsum', 'Zvar']

    elif feat_code == 2:
        selected_feat = ['EDAfirst', 'EDAlast', 'EDAmax', 'EDAmin', 'EDAmean', 'EDAmedian',
                                       'EDAnunique', 'EDAstd', 'EDAsum', 'EDAvar']

    elif feat_code == 3:
        selected_feat = ['BVPfirst', 'BVPlast', 'BVPmax', 'BVPmin', 'BVPmean', 'BVPmedian',
                                       'BVPnunique', 'BVPstd', 'BVPsum', 'BVPvar']

    elif feat_code == 4:
        selected_feat = ['IBIfirst', 'IBIlast', 'IBImax', 'IBImin', 'IBImean', 'IBImedian', 'IBInunique', 'IBIstd',
                         'IBIsum', 'IBIvar']

    elif feat_code == 5:
        selected_feat = ['BVPfirst', 'BVPlast', 'BVPmax', 'BVPmin', 'BVPmean', 'BVPmedian',
                                       'BVPnunique', 'BVPstd', 'BVPsum', 'BVPvar', 'EDAfirst', 'EDAlast',
                                       'EDAmax', 'EDAmin', 'EDAmean', 'EDAmedian', 'EDAnunique', 'EDAstd',
                                       'EDAsum', 'EDAvar', 'IBIfirst', 'IBIlast', 'IBImax', 'IBImin',
                                       'IBImean', 'IBImedian', 'IBInunique', 'IBIstd', 'IBIsum', 'IBIvar']
    elif feat_code == 6:
        selected_feat = ['Xfirst', 'Xlast', 'Xmax', 'Xmin', 'Xmean', 'Xmedian', 'Xnunique',
                                       'Xstd', 'Xsum', 'Xvar', 'Yfirst', 'Ylast', 'Ymax', 'Ymin', 'Ymean',
                                       'Ymedian', 'Ynunique', 'Ystd', 'Ysum', 'Yvar', 'Zfirst', 'Zlast',
                                       'Zmax', 'Zmin', 'Zmean', 'Zmedian', 'Znunique', 'Zstd', 'Zsum', 'Zvar',
                                       'BVPfirst', 'BVPlast', 'BVPmax', 'BVPmin', 'BVPmean', 'BVPmedian',
                                       'BVPnunique', 'BVPstd', 'BVPsum', 'BVPvar', 'EDAfirst', 'EDAlast',
                                       'EDAmax', 'EDAmin', 'EDAmean', 'EDAmedian', 'EDAnunique', 'EDAstd',
                                       'EDAsum', 'EDAvar', 'IBIfirst', 'IBIlast', 'IBImax', 'IBImin',
                                       'IBImean', 'IBImedian', 'IBInunique', 'IBIstd', 'IBIsum', 'IBIvar']

    elif feat_code == 7:
        selected_feat = ['Xfirst', 'Xlast', 'Xmax', 'Xmin', 'Xmean', 'Xmedian', 'Xnunique',
                         'Xstd', 'Xsum', 'Xvar', 'Yfirst', 'Ylast', 'Ymax', 'Ymin', 'Ymean',
                         'Ymedian', 'Ynunique', 'Ystd', 'Ysum', 'Yvar', 'Zfirst', 'Zlast',
                         'Zmax', 'Zmin', 'Zmean', 'Zmedian', 'Znunique', 'Zstd', 'Zsum', 'Zvar',
                         'BVPfirst', 'BVPlast', 'BVPmax', 'BVPmin', 'BVPmean', 'BVPmedian',
                         'BVPnunique', 'BVPstd', 'BVPsum', 'BVPvar', 'EDAfirst', 'EDAlast',
                         'EDAmax', 'EDAmin', 'EDAmean', 'EDAmedian', 'EDAnunique', 'EDAstd',
                         'EDAsum', 'EDAvar', 'IBIfirst', 'IBIlast', 'IBImax', 'IBImin',
                         'IBImean', 'IBImedian', 'IBInunique', 'IBIstd', 'IBIsum', 'IBIvar', 'AGGObserved',
                         'TimePastAggression']
    else:       # Including and beyond useFeaturesCode == 7
        print('Not implemented yet!')
    return selected_feat


def feature_generator(input_dict, resampling_period, agg_category, non_agg_category):
    list_all_instances = input_dict['dataAll']
    bin_feat_per_session = []
    bin_label_per_session = []

    for ins in range(len(list_all_instances)):
        list_per_instance = list_all_instances[ins]
        bin_feat = pd.DataFrame()
        bin_label = pd.DataFrame()
        acc_data_received = False
        for data_source in range(len(list_per_instance)):
            df = list_per_instance[data_source]
            df['Condition'].fillna(0, inplace=True)  # fill NaN with 0
            df['Condition'][df['Condition'].isin(agg_category)] = 1  # 1 to Agg state
            df['Condition'][df['Condition'].isin(non_agg_category)] = 0  # 0 to nonAgg state

            if df['Condition'].nunique() > 2:
                print(df['Condition'].nunique())
                assert ('Have unknown labels!')

            evidence = list(df.columns.values)
            evidence.remove('Condition')
            if len(evidence) > 1:
                acc_data_received = True
            else:
                pass
            for e in range(len(evidence)):
                bin_feat[evidence[e] + 'first'] = pd.to_numeric(df[evidence[e]].dropna()).resample(resampling_period).first()
                bin_feat[evidence[e] + 'last'] = pd.to_numeric(df[evidence[e]].dropna()).resample(resampling_period).last()
                bin_feat[evidence[e] + 'max'] = pd.to_numeric(df[evidence[e]].dropna()).resample(resampling_period).max()
                bin_feat[evidence[e] + 'min'] = pd.to_numeric(df[evidence[e]].dropna()).resample(resampling_period).min()
                bin_feat[evidence[e] + 'mean'] = pd.to_numeric(df[evidence[e]].dropna()).resample(resampling_period).mean()
                bin_feat[evidence[e] + 'median'] = pd.to_numeric(df[evidence[e]].dropna()).resample(resampling_period).median()
                bin_feat[evidence[e] + 'nunique'] = pd.to_numeric(df[evidence[e]].dropna()).resample(resampling_period).nunique()
                bin_feat[evidence[e] + 'std'] = pd.to_numeric(df[evidence[e]].dropna()).resample(resampling_period).std()
                bin_feat[evidence[e] + 'sum'] = pd.to_numeric(df[evidence[e]].dropna()).resample(resampling_period).sum()
                bin_feat[evidence[e] + 'var'] = pd.to_numeric(df[evidence[e]].dropna()).resample(resampling_period).var()
                bin_label[evidence[e] + 'Label'] = df.Condition.resample(resampling_period).max()

        bin_label = bin_label.apply(lambda x: x.dropna().max(), axis=1)

        # Add elapsed time to/till aggression features
        if np.sum(bin_label) == 0:
            # no Aggression obeserved!
            bin_feat['AGGObserved'] = 0
            bin_feat['TimePastAggression'] = 50000000
        else:
            agg_observed_feat = bin_label.copy()
            time_past_agg_feat = bin_label.copy()
            agg_ind = np.where(bin_label != 0)[0]  # index of agression Labels
            min_agg = np.min(agg_ind)  # get index of first Agg episode
            counter = 0
            for t in range(0, len(bin_label)):
                if t >= min_agg:
                    agg_observed_feat[t] = 1
                    if t in agg_ind:
                        counter = 0
                    else:
                        pass
                    time_past_agg_feat[t] = counter
                    counter = counter + 1
                else:
                    agg_observed_feat[t] = 0
                    time_past_agg_feat[t] = 50000000

            # adding new time since last agg features
            bin_feat['AGGObserved'] = agg_observed_feat
            bin_feat['TimePastAggression'] = time_past_agg_feat

        if not acc_data_received:
            for ax in ['X', 'Y', 'Z']:
                bin_feat[ax + 'first'] = 0
                bin_feat[ax + 'last'] = 0
                bin_feat[ax + 'max'] = 0
                bin_feat[ax + 'min'] = 0
                bin_feat[ax + 'mean'] = 0
                bin_feat[ax + 'median'] = 0
                bin_feat[ax + 'nunique'] = 0
                bin_feat[ax + 'std'] = 0
                bin_feat[ax + 'sum'] = 0
                bin_feat[ax + 'var'] = 0
        else:
            proper_order_of_feats = bin_feat.columns

        bin_feat_per_session.append(bin_feat[proper_order_of_feats])
        bin_label_per_session.append(bin_label)

    output_dict = {'features': bin_feat_per_session, 'labels': bin_label_per_session}
    return output_dict


def feat_extraction_from_user_csv_files(uid, root_dir, resampling_period, agg_category, non_agg_category,
                                        path_style='/'):
    folders_for_uid = glob.glob(root_dir + path_style + '*' + uid + '*')
    data_list_for_uid = []
    #    print(folders_for_uid)
    for folder_count in range(len(folders_for_uid)):
        # load all csv files in folder
        csv_files = sorted(glob.glob(folders_for_uid[folder_count] + path_style + '*.csv'), reverse=True,
                           key=os.path.getsize)

        list_per_instance = []
        for file in csv_files:
            # loading each csv file
            df = pd.read_csv(file, index_col=None, header=0, dtype=object)

            # using Timestamp as index
            df_index_time = df.set_index('Timestamp')

            # the command bellow was done so that pandas could recognize df_index_time as a time index
            df_index_time = df_index_time.set_index(pd.to_datetime(df_index_time.index))
            # df_index_time = df_index_time.set_index(pd.to_datetime(df_index_time.index, unit='ms'))

            # appending all different data in one userList in which each element
            # is a different instance (e.g., EDA, IBI, etc)
            list_per_instance.append(df_index_time)

        data_list_for_uid.append(list_per_instance)
    user_dict = {"dataAll": data_list_for_uid}
    output_dict = feature_generator(user_dict, resampling_period, agg_category, non_agg_category)
    return output_dict


# old function used to process raw data from csv files and generated concatenated feature vectors (instances)
def data_processing(data_path, resampling_period, agg_category, non_agg_category, num_observation_frames,
                    num_prediction_frames, feat_code, feature_file_name, instances_file_name):

    # get user ids
    sbj_id_dict = get_user_ids_dictionary_from_directory_root(data_path)

    # extract features from csv files
    if not os.path.isfile(feature_file_name):
        data_dict = {}

        # extract features for each user
        for user_id in sbj_id_dict.values():
            user_data_dict = feat_extraction_from_user_csv_files(user_id, data_path, resampling_period, agg_category,
                                                                 non_agg_category)
            data_dict[user_id] = user_data_dict

        pickle_out = open(feature_file_name, 'wb')
        pickle.dump(data_dict, pickle_out)
        # pickle.close()

    else:
        print('loading data...')
        pickle_in = open(feature_file_name, 'rb')
        data_dict = pickle.load(pickle_in)

    # building the training and test data:
    if not os.path.isfile(instances_file_name):
        # get dictionary with instances (e.g. 12 concatenated 15 feature frames) and labels for all subjects
        dict_of_instances_arrays, dict_of_labels_arrays = gen_instances_from_raw_feat_dictionary(data_dict,
                                                                                                 num_observation_frames,
                                                                                                 num_prediction_frames,
                                                                                                 feat_code)

        temp_data_list = [dict_of_instances_arrays, dict_of_labels_arrays]
        with open(instances_file_name, 'wb') as f:
            # The protocol version used is detected automatically, so we do not
            # have to specify it.
            pickle.dump(temp_data_list, f)

    else:
        print('loading instance data...')
        with open(instances_file_name, 'rb') as f:
            # The protocol version used is detected automatically, so we do not
            # have to specify it.
            [dict_of_instances_arrays, dict_of_labels_arrays] = pickle.load(f)

    return dict_of_instances_arrays, dict_of_labels_arrays, sbj_id_dict


# main csv file processing function
def extract_features_from_raw_csv_files(data_path, resampling_period, agg_category, non_agg_category,
                                        feature_file_name):
    """
    :param data_path: path to the data-set root directory
    :param resampling_period: string with new pandas resampling time (e.g., '15S')
    :param agg_category: list of aggression labels
    :param non_agg_category: list of non aggression labels
    :param feature_file_name: name of the file to write and/or read extracted features.
    :return: raw_features_data_dictionary
    """
    # get user ids
    sbj_id_dict = get_user_ids_dictionary_from_directory_root(data_path)

    # extract features from csv files
    if not os.path.isfile(feature_file_name):
        raw_features_data_dictionary = {}

        # extract features for each user
        for user_id in sbj_id_dict.values():
            user_data_dict = feat_extraction_from_user_csv_files(user_id, data_path, resampling_period, agg_category,
                                                                 non_agg_category)
            raw_features_data_dictionary[user_id] = user_data_dict

        pickle_out = open(feature_file_name, 'wb')
        pickle.dump(raw_features_data_dictionary, pickle_out)
        # pickle.close()

    else:
        print('loading data...')
        pickle_in = open(feature_file_name, 'rb')
        raw_features_data_dictionary = pickle.load(pickle_in)

    return raw_features_data_dictionary


#
def gen_classifier_instances_from_session_data_frame(feat_data_frame_per_session, label_data_frame_per_session,
                                                     num_observation_frames, num_prediction_frames, selected_feat):

    # using only the selected features
    feat_data_frame_per_session = feat_data_frame_per_session[selected_feat]
    n_different_features = len(selected_feat)

    # removing NaN from extracted features
    feat_data_frame_per_session.fillna(0, inplace=True)

    total_num_of_frames = len(feat_data_frame_per_session)
    n_instances = total_num_of_frames - num_prediction_frames - num_observation_frames + 1
    # idxs = feat_data_frame_per_session.index

    print('n_instances ', n_instances)
    instances_array = np.zeros((n_instances, len(selected_feat)*num_observation_frames))
    labels_array = np.zeros((n_instances, 1))

    for i in range(0, n_instances):
#        temp_np_array = np.array(feat_data_frame_per_session.ix[i: i + num_observation_frames])
        temp_np_array = np.array(feat_data_frame_per_session.iloc[i: i + num_observation_frames])
        instances_array[i] = np.reshape(temp_np_array, [1, -1], order='C')

#        labels_array[i] = label_data_frame_per_session.ix[i + num_observation_frames: i + num_observation_frames + num_prediction_frames].sum()
        labels_array[i] = label_data_frame_per_session.iloc[i + num_observation_frames: i + num_observation_frames + num_prediction_frames].sum()
        # if labels_array[i] > 0:
        #     labels_array[i] = 1
        # else:
        #     labels_array[i] = -1

    # add std of each feature
    # each feature repeats every len(selected_feat)
    # feat_idx = [0  n_different_features n_different_features*2,...]
    feat_std_array = np.zeros([n_instances, n_different_features])
    for feat_count in range(n_different_features):
        idx = [feat_count + n_different_features*cc for cc in range(0, num_observation_frames)]
        feat_std_array[:, feat_count] = np.std(instances_array[:, idx], axis=1)
    instances_array = np.concatenate((instances_array, feat_std_array), axis=1)

    return instances_array, labels_array


# main instance generator function
def gen_instances_from_raw_feat_dictionary(feat_dict, num_observation_frames, num_prediction_frames, feat_code):
    """
    :param feat_dict:
    :param num_observation_frames:
    :param num_prediction_frames:
    :param feat_code:
    :return:  dict_of_instances_arrays, dict_of_labels_arrays
    """

    # select features from feat_code
    selected_feat = select_feat_from_feat_code(feat_code)

    # print(len(selected_feat))
    dict_of_instances_arrays = {}
    dict_of_labels_arrays = {}

    # loops over each subject
    for subjID in feat_dict:
        print('subjID', subjID)
        instances_array_per_subject = np.array([]).reshape(0, num_observation_frames * len(selected_feat)
                                                           + len(selected_feat))

        labels_array_per_subject = np.array([]).reshape(0, 1)

        feat_dic_per_subj = feat_dict[subjID]

        # print(len(feat_dic_per_subj['features']), len(feat_dic_per_subj['labels']))

        # getting list of features and labels from each the feature dictionary from subject.
        feat_list_per_subj = feat_dic_per_subj['features']
        label_list_per_subj = feat_dic_per_subj['labels']

        # loop over sessions for a given subject
        for i in range(len(label_list_per_subj)):
            # getting pandas data frame for each session within a subject
            feat_data_frame_per_session = feat_list_per_subj[i]
            label_data_frame_per_session = label_list_per_subj[i]

            # print(feat_data_frame_per_session)
            # gen classifier instances #
            session_instances_array, session_labels_array = gen_classifier_instances_from_session_data_frame(
                feat_data_frame_per_session, label_data_frame_per_session, num_observation_frames,
                num_prediction_frames, selected_feat)

            # concatenate instances in a numpy array
            instances_array_per_subject = np.concatenate((instances_array_per_subject, session_instances_array), axis=0)
            labels_array_per_subject = np.concatenate((labels_array_per_subject, session_labels_array), axis=0)

        dict_of_instances_arrays[subjID] = instances_array_per_subject
        dict_of_labels_arrays[subjID] = labels_array_per_subject
    return dict_of_instances_arrays, dict_of_labels_arrays


def gen_instances_from_raw_feat_dictionary_multiscale(feat_dict, num_observation_frames, min_num_prediction_frames,
                                                      max_num_prediction_frames, step=1, feat_code=6):
    """

    :param feat_dict:
    :param num_observation_frames: number of observed frames in the past (e.g., 12)
    :param min_num_prediction_frames: number of observed frames in the future (e.g., 4)
    :param max_num_prediction_frames: maximum number of observed frames in the future (e.g., 4+4)
    :param step: controls the labels we want to keep. Meaning if min_num_prediction_frames=4 and
    max_num_prediction_frames=8 and step=2 this function will return y's for 4, 6 and 8 frames in the future (3 scales).
    :param feat_code: feature code to collect from previously extracted features.
    :return: x (np array of features) and y (list with np arrays of labels for multiples scales).
    """

    y = []
    x, yy = gen_instances_from_raw_feat_dictionary(feat_dict, num_observation_frames, min_num_prediction_frames,
                                                   feat_code)
    y.append(yy)
    for num_prediction_frames in range(min_num_prediction_frames + step, max_num_prediction_frames + 1, step):
        _, yy = gen_instances_from_raw_feat_dictionary(feat_dict, num_observation_frames, num_prediction_frames,
                                                       feat_code)
        y.append(yy)

    dict_of_instances_arrays = x
    list_dict_of_labels_arrays = y

    return dict_of_instances_arrays, list_dict_of_labels_arrays


def test_function():
    data_path = '/home/tales/DataBases/test_new_data'
    path_style = '/'
    subject_id_coding = 4  # Number of digits in subject ID coding
    uid = '4356'
    folders_for_uid = glob.glob(data_path + path_style + '*' + uid + '*')
    data_list_for_uid = []
    #    print(folders_for_uid)
    for folder_count in range(len(folders_for_uid)):
        # load all csv files in folder
        csv_files = sorted(glob.glob(folders_for_uid[folder_count] + path_style + '*.csv'), reverse=True,
                           key=os.path.getsize)

        list_per_instance = []
        for file in csv_files:
            # loading each csv file
            df = pd.read_csv(file, index_col=None, header=0, dtype=object)

            # using Timestamp as index
            df_index_time = df.set_index('Timestamp')

            # the command bellow was done so that pandas could recognize df_index_time as a time index
            df_index_time = df_index_time.set_index(pd.to_datetime(df_index_time.index, unit='ms'))

            # appending all different data in one userList in which each element
            # is a different instance (e.g., EDA, IBI, etc)
            list_per_instance.append(df_index_time)


def get_normalization_constants(data_array, axis=0):
    means = np.mean(data_array, axis=axis)
    stds = np.std(data_array, axis=axis)
    stds[stds == 0] = 1
    # return np.linalg.norm(data_array, axis=axis)
    return [means, stds]


def normalize_data(data_array, normalization_constants, axis=0):
    # removing zeros and avoiding a division by zero.
    # normalization_constants[normalization_constants == 0] = 1
    # if axis == 1:
    #     return (data_array.transpose() / normalization_constants).transpose()
    #
    # return data_array / normalization_constants

    means = normalization_constants[0]
    stds = normalization_constants[1]
    if axis == 1:
        return ((data_array.transpose() - means) / stds).transpose()

    return (data_array - means)/(stds)




if __name__ == '__main__':
    test_function()
