#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 11:52:49 2019

@author: tales
"""

from fileManipulation import dictval2str
import generateInstanceFromRawFeatures as ginst
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn import preprocessing
import numpy as np
import matplotlib
from sklearn.cluster import KMeans

# matplotlib.use('Agg')
from matplotlib import pyplot as plt
# import matplotlib.pyplot as plt

import itertools
import classifier_cv

from joblib import Parallel, delayed
import joblib

#####################
# Configuration
#####################

dataPath = '/home/tales/DataBases/smallASDData/Data'
pathStyle = '/'
subjectIDCoding = 4  # Number of digits in subject ID coding
# num_observation_frames = 12
# num_prediction_frames = 4
num_observation_frames = 12
num_prediction_frames = 8


# feat_code = 7                       # use all features
feat_code = 6  # use all features but 'AGGObserved' and 'TimePastAggression'

o_normalize_data = True  # normalize data?
o_perform_pca = True  # perform PCA?
n_pcs = 10  # number of principal components

# classifier config
cv_folds = 4
cv_reps = 6

# cv_folds = 2
# cv_reps = 1

n_intensity_cat = 3

# classifier_type = 'LR'
classifier_type = 'SVM'

o_perform_parameters_search = False

# lr_par = {'C': 100, 'penalty': 'l2', 'tol': 0.01, 'solver': 'lbfgs', 'max_iter': 500}
lr_par = {'C': 1000000, 'penalty': 'l2', 'tol': 0.01, 'solver': 'saga', 'max_iter': 500}
svm_par = {'gamma': 0.1, 'C': 100, 'kernel': 'rbf', 'probability': True}

# figure config
labels_fontsize = 16

# are csv files preprocessed?
# file name for the pre-processed data
featureFileName = dataPath + pathStyle + 'rawFeatureDataAllSubjects.b'
instances_file_name = dataPath + pathStyle + 'instancesFileName_pb_' + str(num_observation_frames) + '_fb_' \
                      + str(num_prediction_frames) + '_' + str(feat_code) + '.b'

aggCategory = ['aggression', 'agg', 'aggression (?)', 'property destruction', 'noncompliance',
               'SIB', 'ED', 'kicked table', 'aggression/SIB', 'elopement/drop on the floor',
               'aggression/SIB/property destruction', 'dropped on floor', 'sib', 'elopement',
               'AGG', 'Agg', 'Agg/SIB']
nonAggCategory = ['sleeping', 'mild ED/shoved by peer']

binSize = '15S'  # Frame size in seconds

#####################
# Feature Extraction
#####################

dict_of_instances_arrays, dict_of_labels_arrays, sbjIDDict = ginst.data_processing(dataPath, binSize, aggCategory,
                                                                                   nonAggCategory,
                                                                                   num_observation_frames,
                                                                                   num_prediction_frames, feat_code,
                                                                                   featureFileName, instances_file_name,
                                                                                   agg_intensity_clf=True, n_intensity_cat=n_intensity_cat)

#####################################
#  Data Analysis and Classification #
#####################################

# # normalizing data
# if o_normalize_data:
#     for sbj in sbjIDDict.keys():
#         dict_of_instances_arrays[sbjIDDict[sbj]] = preprocessing.normalize(dict_of_instances_arrays[sbjIDDict[sbj]],
#                                                                            norm='l2', axis=0)
norm_constants_dict = {}
if o_normalize_data:
    # train one PCA for each subj using data from other subjs.
    # get data from all subjects but the current
    for sbj in sbjIDDict.keys():
        blacklist = [sbjIDDict[sbj]]
        instances_from_other_sbj = \
            np.concatenate([tmp_inst_array for key, tmp_inst_array in dict_of_instances_arrays.items() if key not in blacklist], axis=0)

        norm_const = classifier_cv.get_normalization_constants(instances_from_other_sbj, axis=0)
        norm_constants_dict[sbjIDDict[sbj]] = norm_const
        # print( np.linalg.norm(classifier_cv.normalize_data(instances_from_other_sbj, norm_const) - preprocessing.normalize(instances_from_other_sbj, axis=0) ))




# getting all data
all_subjects_instances_array = \
    np.concatenate([tmp_inst_array for tmp_inst_array in dict_of_instances_arrays.values()], axis=0)
all_subjects_labels = np.concatenate([tmp_label_array for tmp_label_array in dict_of_labels_arrays.values()], axis=0)

pcas_dict = {}
if o_perform_pca:
    # train one PCA for each subj using data from other subjs.
    # get data from all subjects but the current
    for sbj in sbjIDDict.keys():
        blacklist = [sbjIDDict[sbj]]
        instances_from_other_sbj = \
            np.concatenate([tmp_inst_array for key, tmp_inst_array in dict_of_instances_arrays.items() if key not in blacklist], axis=0)

        pca = PCA(n_components=n_pcs)
        pca.fit(instances_from_other_sbj)
        pcas_dict[sbjIDDict[sbj]] = pca
        # pcas.append(pca)


#########################################################
#  Classification with Person-dependent Models          #
#########################################################


if classifier_type == 'LR':
    clf_function = LogisticRegression
    if o_perform_parameters_search:
        Cs = [.1, 1, 100, 500, 1e3, 1e5, 1e6, 1e8]
        # Cs = [.1, 100]
        reg = ['l2', 'l1']
    else:
        Cs = [lr_par['C']]
        reg = [lr_par['penalty']]

    par_list = list(itertools.product(Cs, reg))
    clf_par = lr_par
    par_order = ['C', 'penalty']

elif classifier_type == 'SVM':
    clf_par = svm_par
    clf_function = svm.SVC

    if o_perform_parameters_search:
        # Cs = [.1, 1, 100, 1e3, 1e5]
        # gammas = [1e-3, 1e-1, 1, 2, 5, 10]
        Cs = [100, 1e3]
        gammas = [0.5, 1, 2, 3]
    else:
        Cs = [svm_par['C']]
        gammas = [svm_par['gamma']]

    par_list = list(itertools.product(Cs, gammas))
    par_order = ['C', 'gamma']
else:
    pass

# plot colors
colors = np.random.uniform(0.3, 0.9, (22, 3))

# for loop over all possible parammeters
for pars in par_list:

    clf_par[par_order[0]] = pars[0]
    clf_par[par_order[1]] = pars[1]
    print(clf_par)

    classifier = clf_function(**clf_par)

    color_count = 0
    # plt.figure()
    # plt.figure()

    # get number of cores in the cpu
    n_jobs = min(joblib.cpu_count(), sbjIDDict.keys().__len__())

    for sbj in sbjIDDict.keys():
        yy = dict_of_labels_arrays[sbjIDDict[sbj]]
        if len(np.unique(yy)) < n_intensity_cat + 1:
            print('Escaping SBJ (at least one class is not present)' + sbjIDDict[sbj])
            continue
        o_enough_data = True
        for i in range(n_intensity_cat):
            if len(yy[yy == i + 1]) < 4:
                o_enough_data = False
        if o_enough_data is False:
            print('Escaping SBJ (not enough data for one class)' + sbjIDDict[sbj])
            continue

        # if sbjIDDict[sbj] != '1101':
        #     print('Breaking ' + sbjIDDict[sbj])
        #     continue

        res = classifier_cv.classify_instance_cv_with_agg_intensity(
            classifier, dict_of_instances_arrays[sbjIDDict[sbj]],
            dict_of_labels_arrays[sbjIDDict[sbj]], cv_folds,
            cv_reps, o_perform_pca, pcas_dict,
            o_normalize_data,
            norm_constants_dict, sbjIDDict[sbj],
            o_get_pd_for_diff_agg_intensity=True)

        # plt.figure()
        nc = n_intensity_cat
        tprs, mean_fpr, aucs, sbjID = res
        tprs_array = np.zeros((len(mean_fpr), nc))
        auc_array = np.zeros(nc)
        # tprs is a list (one for cv run) of dictionaries
        for l in range(len(tprs)):
            tpr_d = tprs[l]
            auc_d = aucs[l]
            for key in tpr_d.keys():
                tprs_array[:, key] += tprs[l][key]
                auc_array[key] += auc_d[key]

        tprs_array = tprs_array / len(tprs)
        auc_array = auc_array / len(tprs)

        plt.figure()

        plt.plot(mean_fpr, tprs_array[:, 0], '-b', mean_fpr, tprs_array[:, 1], '--g', mean_fpr, tprs_array[:, 2], '-.r')
        plt.legend(['Low', 'Moderate', 'High'])
        plt.title(sbjID)
        print(str(auc_array) + ' ' + sbjID)
