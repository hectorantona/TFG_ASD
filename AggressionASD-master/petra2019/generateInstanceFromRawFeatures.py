import numpy as np
import fileManipulation as fm
import pickle
import os
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import pandas as pd
# def gen_instances_from_raw_feat_dictionary(outputDict, prevTimeRange, futureTimeRange, useFeaturesCode):


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

    elif feat_code == 8:
        selected_feat = ['ACCNormmean']

    else:       # Including and beyond useFeaturesCode == 7
        print('Not implemented yet!')
    return selected_feat


def gen_classifier_instances_from_session_data_frame(feat_data_frame_per_session, label_data_frame_per_session,
                                                     num_observation_frames, num_prediction_frames, selected_feat,
                                                     agg_intensity_clf_per_sbj=None):

    # using only the selected features
    feat_data_frame_per_session_sfeat = feat_data_frame_per_session[selected_feat]
    n_different_features = len(selected_feat)

    # removing NaN from extracted features
    feat_data_frame_per_session_sfeat.fillna(0, inplace=True)

    total_num_of_frames = len(feat_data_frame_per_session_sfeat)
    n_instances = total_num_of_frames - num_prediction_frames - num_observation_frames + 1
    # idxs = feat_data_frame_per_session.index

    # for loop over all possible frames
        # instance [i] = concatenation of num_observation_frames
        # label [i] = 0 or 1 by looking at the future num_prediction_frames
    print('n_instances ', n_instances)
    instances_array = np.zeros((n_instances, len(selected_feat)*num_observation_frames))
    labels_array = np.zeros((n_instances, 1))

    if agg_intensity_clf_per_sbj is not None:
        # acc_feats = select_feat_from_feat_code(1)
        # acc_feats = ['Xmean', 'Ymean', 'Zmean']
        acc_feats = ['ACCNormmean']
        X = feat_data_frame_per_session[acc_feats]
        X = np.array(X)
        # X = np.linalg.norm(X, axis=1)

        y = label_data_frame_per_session.to_numpy()
        # agg_idx = y > 0
        agg_idx = np.where(y > 0)[0]
        if agg_idx.shape[0] is not 0:
            print('here')
            X = X[agg_idx]
            km_pred = agg_intensity_clf_per_sbj.predict(X.reshape(-1, 1))
            for cc in range(agg_intensity_clf_per_sbj.n_clusters):
                y[agg_idx[km_pred == cc]] = cc + 1
            label_data_frame_per_session = pd.DataFrame(y)

    for i in range(0, n_instances):
        temp_np_array = np.array(feat_data_frame_per_session_sfeat.ix[i: i + num_observation_frames])
        # temp_np_array = np.array(feat_data_frame_per_session.loc[i: i + num_observation_frames])
        instances_array[i] = np.reshape(temp_np_array, [1, -1], order='C')

        # labels_array[i] = label_data_frame_per_session.ix[i + num_observation_frames: i + num_observation_frames + num_prediction_frames].sum()
        # Set the label as the maximum aggression intensity:
        labels_array[i] = label_data_frame_per_session.ix[
                          i + num_observation_frames: i + num_observation_frames + num_prediction_frames].max()
        # labels_array[i] = label_data_frame_per_session.loc[
        #                   i + num_observation_frames: i + num_observation_frames + num_prediction_frames].max()
        if agg_intensity_clf_per_sbj is None:
            if labels_array[i] > 0:
                labels_array[i] = 1
            else:
                labels_array[i] = -1
        else:
            if labels_array[i] == 0:
                labels_array[i] = -1

    # add std of each feature
    # each feature repeats every len(selected_feat)
    # feat_idx = [0  n_different_features n_different_features*2,...]
    feat_std_array = np.zeros([n_instances, n_different_features])
    for feat_count in range(n_different_features):
        idx = [feat_count + n_different_features*cc for cc in range(0, num_observation_frames)]
        feat_std_array[:, feat_count] = np.std(instances_array[:, idx], axis=1)
    instances_array = np.concatenate((instances_array, feat_std_array), axis=1)

    return instances_array, labels_array


def gen_instances_from_raw_feat_dictionary(feat_dict, num_observation_frames, num_prediction_frames, feat_code, agg_intensity_clf=None):

    # select features from feat_code
    selected_feat = select_feat_from_feat_code(feat_code)

    # print(len(selected_feat))
    dict_of_instances_arrays = {}
    dict_of_labels_arrays = {}

    id_blacklist = []
    # loops over each subject
    for subjID in feat_dict:
        try:
            if agg_intensity_clf is not None:
                try:
                    agg_intensity_clf_per_sbj = agg_intensity_clf[subjID]
                except:
                    agg_intensity_clf_per_sbj = None
            else:
                agg_intensity_clf_per_sbj = None

            print('subjID', subjID)
            instances_array_per_subject = np.array([]).reshape(0, num_observation_frames * len(selected_feat) + len(selected_feat))
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
                session_instances_array, session_labels_array = gen_classifier_instances_from_session_data_frame(
                    feat_data_frame_per_session, label_data_frame_per_session, num_observation_frames,
                    num_prediction_frames, selected_feat, agg_intensity_clf_per_sbj)

                instances_array_per_subject = np.concatenate((instances_array_per_subject, session_instances_array), axis=0)
                labels_array_per_subject = np.concatenate((labels_array_per_subject, session_labels_array), axis=0)

            dict_of_instances_arrays[subjID] = instances_array_per_subject
            dict_of_labels_arrays[subjID] = labels_array_per_subject
        except:
            id_blacklist.append(subjID)
            print('Not possible to construct data for sbj ' + subjID)
    return dict_of_instances_arrays, dict_of_labels_arrays, id_blacklist


#
def data_processing(dataPath, binSize, aggCategory, nonAggCategory, num_observation_frames, num_prediction_frames,
                    feat_code, featureFileName, instances_file_name, agg_intensity_clf=False, n_intensity_cat=3,
                    from_scratch=True):

    sbjIDDict = fm.getUserIdsDictionaryFromDirectoryRoot(dataPath, binSize, aggCategory, nonAggCategory)

    if not os.path.isfile(featureFileName) or from_scratch:
        dataDict = {}
        for userID in sbjIDDict.values():
            userDataDict = fm.featExtractFromRawDataForAGivenUser(userID, dataPath, binSize, aggCategory,
                                                                  nonAggCategory)
            dataDict[userID] = userDataDict

        pickleOut = open(featureFileName, 'wb')
        pickle.dump(dataDict, pickleOut)
        # pickle.close()

    else:
        print('loading data...')
        pickleIn = open(featureFileName, 'rb')
        dataDict = pickle.load(pickleIn)

    # building the training and test data:
    if not os.path.isfile(instances_file_name) or agg_intensity_clf or from_scratch:
        if agg_intensity_clf:
            print('here')
            agg_intensity_clf = get_kmeans_for_accelerometer_features_of_agg_episodes(featureFileName,
                                                                                o_plot_histograms=False,
                                                                                n_clusters=n_intensity_cat,
                                                                                      return_data=False)

        # get dictionary with instances (e.g. 12 concatenated 15 feature frames) and labels for all subjects
        dict_of_instances_arrays, dict_of_labels_arrays, id_blacklist = \
            gen_instances_from_raw_feat_dictionary(dataDict, num_observation_frames,
                                                                                   num_prediction_frames,
                                                                                   feat_code, agg_intensity_clf)

        # remove blacklisted IDs from IDsdict
        if len(id_blacklist) != 0:
            for v in id_blacklist:
                for k in sbjIDDict.keys():
                    if sbjIDDict[k] == v:
                        blacklisted_key = k
                del sbjIDDict[blacklisted_key]

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

    return dict_of_instances_arrays, dict_of_labels_arrays, sbjIDDict


# def get_accelerometer_features_and_num_of_aggression_episodes(featureFileName, o_plot_histograms=False, n_clusters=3):
def get_kmeans_for_accelerometer_features_of_agg_episodes(featureFileName, o_plot_histograms=False, n_clusters=3,
                                                          return_data=False):

    # get only norm of accelerometer data
    feat_code = 8

    feats = select_feat_from_feat_code(feat_code)

    print('loading data...')
    pickleIn = open(featureFileName, 'rb')

    # dictionary of data frames
    dataDict = pickle.load(pickleIn)

    count = 1
    kmeans_dict = {}
    for key in dataDict.keys():

        sbj_data_dict = dataDict[key]
        sbj_feat_list = sbj_data_dict['features']
        sbj_label_list = sbj_data_dict['labels']

        sbj_feat_array = []
        sbj_label_array = []
        for i in range(len(sbj_feat_list)):
            # tmp_array = np.linalg.norm(np.array(sbj_feat_list[i][['Xmean', 'Ymean', 'Zmean']]), axis=1)
            tmp_array = np.array(sbj_feat_list[i]['ACCNormmean'])
            sbj_feat_array = np.concatenate((sbj_feat_array, tmp_array), axis=0)
            sbj_label_array = np.concatenate((sbj_label_array, sbj_label_list[i]), axis=0)

        try:
            X = sbj_feat_array[sbj_label_array >= 1].reshape(-1, 1)
            # kmeans = KMeans(init= np.array([min(X), max(X)]), n_clusters=2, random_state=0).fit(X)
            if n_clusters == 3:
                # kmeans = KMeans(init=np.array([min(X), (min(X) + max(X)) / 2, max(X)]), n_clusters=n_clusters, random_state=0).fit(X)
                kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
                means = kmeans.cluster_centers_.copy()
                for i in range(n_clusters):
                    idx = np.argmax(means)
                    kmeans.labels_[idx] = n_clusters - i -1
                    means[idx] = -10000

            elif n_clusters == 2:
                # kmeans = KMeans(init=np.array([min(X), max(X)]), n_clusters=n_clusters,
                #                 random_state=0).fit(X)
                kmeans = KMeans(n_clusters=n_clusters,
                                random_state=0).fit(X)

                means = kmeans.cluster_centers_
                if means[0] > means[1]:
                    print('KMeans Centroids not in crescent order. Inverting kmmean labels!')
                    kmeans.labels_ = abs(kmeans.labels_ -1)

            else:
                kmeans = KMeans(n_clusters=n_clusters,
                                random_state=0).fit(X)

            kmeans_dict[key] = kmeans

            if o_plot_histograms:
                km_pred = kmeans.predict(X)
                # plt.figure()
                # plt.hist(X[km_pred == 0], bins=20)
                # plt.hist(X[km_pred == 1], bins=20)
                # plt.hist(X[km_pred == 2], bins=20)
                plt.figure()
                plt.hist(X[km_pred == 0])
                plt.hist(X[km_pred == 1])
                plt.hist(X[km_pred == 2])
                # if count == 8:
                #     print(kmeans)
                #     print(X.T)
                #     print(km_pred)
                #     break
                # count += 1

                print(kmeans.cluster_centers_)
        except:
            print('Could not process ' + str(key))
            print(X.shape[0])
            print(sbj_label_array)

    if return_data:
        return kmeans_dict, dataDict
    return kmeans_dict
