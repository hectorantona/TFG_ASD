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
import csv
import pickle

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
num_prediction_frames = 4


# feat_code = 7                       # use all features
feat_code = 6  # use all features but 'AGGObserved' and 'TimePastAggression'

o_normalize_data = True  # normalize data?
o_perform_pca = True  # perform PCA?
n_pcs = 10  # number of principal components

# classifier config
cv_folds = 4
cv_reps = 6

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
                                                                                   featureFileName, instances_file_name)

#####################################
#  Data Analysis and Classification #
#####################################

# normalizing data #
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

# get all data
all_subjects_instances_array = \
    np.concatenate([tmp_inst_array for tmp_inst_array in dict_of_instances_arrays.values()], axis=0)
all_subjects_labels = np.concatenate([tmp_label_array for tmp_label_array in dict_of_labels_arrays.values()], axis=0)


# PCA #
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
    plt.figure()


    # get number of cores in the cpu
    n_jobs = min(joblib.cpu_count(), sbjIDDict.keys().__len__())

    # train and test classifier using parallel for
    results_list = Parallel(n_jobs=n_jobs)(
        delayed(classifier_cv.classify_instance_cv)(classifier, dict_of_instances_arrays[sbjIDDict[sbj]],
                                                    dict_of_labels_arrays[sbjIDDict[sbj]], cv_folds,
                                                    cv_reps, o_perform_pca, pcas_dict,
                                                    o_normalize_data,
                                                    norm_constants_dict, sbjIDDict[sbj]) for sbj in sbjIDDict.keys())

    # plot ROCs and compute average AUCs
    average_auc_all_subjects = []
    std_auc_all_subjects = []

    for res in results_list:
        mean_fpr, mean_tpr, mean_auc, std_auc, tprs, id = res
        average_auc_all_subjects.append(mean_auc)
        std_auc_all_subjects.append(std_auc)

        plt.plot(mean_fpr, mean_tpr, color=colors[color_count, :], linestyle='--', linewidth=1.5)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=colors[color_count, :], alpha=.1)

        color_count += 1

    plt.grid(color=[.9, .9, .9], linestyle='--')
    plt.xlim([-0.005, 1.005])
    plt.ylim([-0.005, 1.005])
    plt.xlabel('False Positive Rate (1-Specificity)', fontweight='bold', fontsize=labels_fontsize)
    plt.ylabel('True Positive Rate (Sensitivity)', fontweight='bold', fontsize=labels_fontsize)
    # plt.xlabel('False Positive Rate', fontweight='bold')
    # plt.ylabel('True Positive Rate', fontweight='bold')
    # plt.title('Receiver operating characteristic example')
    plt.tick_params(labelsize=14)
    # plt.legend(loc="lower right")
    plt.show()

    print('E[AUC_sbj]', np.mean(average_auc_all_subjects))
    print('STD[AUC_sbj]', np.mean(std_auc_all_subjects))

    # #########################################################
    # #  Classification with Population Model                     #
    # #########################################################
    print('Building global model')


    mean_fpr, mean_tpr, mean_auc, std_auc, tprs = classifier_cv.classify_instance_parallel_cv(classifier,
                                                                                              all_subjects_instances_array,
                                                                                              all_subjects_labels,
                                                                                              cv_folds,
                                                                                              cv_reps, n_jobs,
                                                                                              o_perform_pca,
                                                                                              n_pcs, o_normalize_data)

    plt.plot(mean_fpr, mean_tpr, color='b', linestyle='-', linewidth=1.5)
    print('E[AUC_cv]_global', mean_auc)
    print('STD[AUC_global]', std_auc)

    # #########################################################
    #    Save Simulation Data in a binary file                #
    # #########################################################

    population_model_list = [mean_fpr, mean_tpr, mean_auc, std_auc]
    results = [results_list, population_model_list]
    file_name = './Results/results' + classifier_type + '.data'
    # save data in a csv file:
    with open(file_name, 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(results, filehandle)

    # #########################################################
    # #  Save Figures                                         #
    # #########################################################

    if o_perform_pca:
        roc_fig_file = './Figs/ROCs_localModels_' + classifier_type + '_Par' + dictval2str(clf_par, 3) + '_PCA_' + str(
            n_pcs) + '_ftCode' \
                       + str(feat_code) + '.pdf'
    else:
        roc_fig_file = './Figs/ROCs_localModels_' + classifier_type + '_Par' + dictval2str(clf_par, 3) + '_ftCode' + str(
            feat_code) + '.pdf'

    plt.tick_params(labelsize=(labels_fontsize - 2))
    plt.savefig(roc_fig_file, dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format='pdf',
                transparent=False, bbox_inches='tight', pad_inches=0.1,
                frameon=None, metadata=None)

    plt.close()
