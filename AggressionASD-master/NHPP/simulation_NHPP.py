#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 11:52:49 2019

@author: tales
"""

import featureextraction as fext
# from poissonprocess import NonHomogeneousPoissonProcess as nhpp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from poissonprocess import agg_pred_cv, agg_pred_leave_one_subj_out, agg_pred_data_split
from plottools import plot_roc_with_std, roc_fig_labels_and_style

#####################
# Configuration
#####################

# File read config:
data_path = '/scratch/talesim/smallASDData/Data'
# data_path = '/home/tales/DataBases/smallASDData/Data'
# data_path = '/home/tales/DataBases/smallASDData/Data'
# data_path = '/home/tales/DataBases/test_new_data'
path_style = '/'
subject_id_coding = 4  # Number of digits in subject ID coding


# Raw Feature extraction config:

# num_observation_frames = 12
# num_prediction_frames = 4
num_observation_frames = 12
num_prediction_frames = 4

# feat_code = 7                       # use all features
# feat_code = 6  # use all features but 'AGGObserved' and 'TimePastAggression'
feat_code = 1  # use only acceleration XYZ features.


# feature vector every feature_sampling_period in seconds
# binSize = '15S'  # Frame size in seconds
# feature_sampling_period_str = '15S'
feature_sampling_period_in_seconds = 15
feature_sampling_period_str = str(feature_sampling_period_in_seconds) + 'S'

# observation window slide
observation_window_slide = num_observation_frames   # generates independent poisson Y_k random variable
# observation_window_slide = 1    # a lot of superposition


# High level data processing:

o_normalize_data = False  # normalize data?
# o_use_my_normalization = True # False uses sklearn scale method.
# o_perform_pca = False  # perform PCA?
o_perform_pca = False    # perform PCA?
n_pcs = 10  # number of principal components
# n_pcs = 100  # number of principal components
o_whiten = True  # whiten PCs?

## NHPP config
# estimation_method = 'LS'
# estimation_method = 'WLS'
# intensity_model = 'linear'
# intensity_model = 'explin'
# estimation_method = 'MLE'
intensity_model = 'svr'
# intensity_model = 'dnn'
estimation_method = ''

# model parameters
if intensity_model == 'linear':
    model_parameters = None
elif intensity_model == 'explin':
    # model_parameters = regularization_parameter
    model_parameters = 0.1
elif intensity_model == 'svr':
    model_parameters = {'gamma': 0.5, 'C': 1000, 'kernel': 'rbf'}
    # model_parameters = {'gamma': 0.5, 'C': 1, 'kernel': 'rbf'}
    # model_parameters = {'gamma': 0.5, 'C': 10, 'kernel': 'rbf'}
else:
    model_parameters = {}

# Classifier config:
cv_folds = 4
cv_reps = 6
# cv_folds = 2
# cv_reps = 1


# figure config
labels_font_size = 16
# plot colors
colors = np.random.uniform(0.3, 0.9, (22, 3))
color_count = 0
# plt.figure()


# are csv files preprocessed?
# file name for the pre-processed data
feature_file_name = data_path + path_style + 'rawFeatureDataAllSubjects.b'
instances_file_name = data_path + path_style + 'instancesFileName_' + str(feat_code) + '.b'

agg_category = ['aggression', 'agg', 'aggression (?)', 'property destruction', 'noncompliance',
               'SIB', 'ED', 'kicked table', 'aggression/SIB', 'elopement/drop on the floor',
               'aggression/SIB/property destruction', 'dropped on floor', 'sib', 'elopement',
               'AGG', 'Agg', 'Agg/SIB']

non_agg_category = ['sleeping', 'mild ED/shoved by peer']



###############################
# Front-End: Feature Extraction
###############################

# extract features from raw csv files:

feature_dict = fext.extract_features_from_raw_csv_files(data_path, feature_sampling_period_str, agg_category,
                                                        non_agg_category, feature_file_name)

# build processing feature vectors (instances) from raw features
#TODO: add a supperposition variable which control the supperposition of the prediction window. Or, create a new function
# function for the Poisson process case.
dict_of_instances_arrays, dict_of_labels_arrays = fext.gen_instances_from_raw_feat_dictionary(feature_dict,
                                                                                              num_observation_frames,
                                                                                              num_prediction_frames,
                                                                                              feat_code)

############################################
# Front-End 2: High level feature processing
############################################

#TODO: Add PCA/feature selection/other high-level feature processing here.

if o_normalize_data or o_perform_pca:
    norm_constants_dict = {}
    scaler = {}
    pcas_dict = {}

    for sbj in dict_of_instances_arrays.keys():
        blacklist = [sbj]
        instances_from_other_sbj = \
            np.concatenate(
                [tmp_inst_array for key, tmp_inst_array in dict_of_instances_arrays.items() if key not in blacklist],
                axis=0)
    # all_subjects_instances_array = \
    #     np.concatenate([tmp_inst_array for tmp_inst_array in dict_of_instances_arrays.values()], axis=0)
        X_temp = instances_from_other_sbj.copy()

        if o_normalize_data:
            # norm_const_list = fext.get_normalization_constants(all_subjects_instances_array)
            # norm_const = fext.get_normalization_constants(instances_from_other_sbj, axis=0)
            # norm_const = fext.get_normalization_constants(X_temp, axis=0)
            # norm_constants_dict[sbj] = norm_const
            scaler[sbj] = preprocessing.StandardScaler().fit(X_temp)
            X_temp = scaler[sbj].transform(X_temp)

        if o_perform_pca:
            # pca = PCA(n_components=n_pcs)
            # pca.fit(all_subjects_instances_array)
            pca = PCA(n_components=n_pcs, whiten=o_whiten)
            pca.fit(X_temp)
            X_temp = pca.transform(X_temp)
            pcas_dict[sbj] = pca



#################################################################################
# Back-End: Signal processing (classification, regression, Poisson modeling, etc.)
#################################################################################

#QUESTIONS:
#   1- What is T (the period for which the linear model works)?
#   2- T_f = feature_sampling_period_in_seconds * num_prediction_frames  => T = T_f * n, n = length(y)
#   3- N = T/Tf = n
#   4- a = alpha * N/T,  b = N/T * beta



keys = []

auc_means = []
auc_stds = []

fig_count = 2
for key in dict_of_instances_arrays.keys():
    keys.append(key)
    # print(keys)

    # data for 'key' subj
    X = dict_of_instances_arrays[key]

    if o_normalize_data:
        print('Normalizing data.')
        # X = fext.normalize_data(X, norm_constants_dict[key])
        X = scaler[key].transform(X)

    if o_perform_pca:
        print('Perform PCA.')
        X = pcas_dict[key].transform(X)

    n, d = X.shape
    y = np.reshape(dict_of_labels_arrays[key], (n,))
    prediction_duration = feature_sampling_period_in_seconds * num_prediction_frames

#   Run Cross Validation
    random_state = 1
    # print('aqui')
    # print(X.shape, y.shape)
    mean_fpr, mean_tpr, mean_auc, std_auc, tprs = agg_pred_cv(X, y, cv_folds, cv_reps, random_state, prediction_duration,
                                                              estimation_method, intensity_model, model_parameters)

    color_count = plot_roc_with_std(mean_fpr, mean_tpr, tprs, colors, color_count, fig_num=1)
    auc_means.append(mean_auc)
    auc_stds.append(std_auc)

    agg_pred_data_split(X, y, random_state, prediction_duration, estimation_method, intensity_model, model_parameters,
                        o_perform_normalization=True, o_perform_PCA=o_perform_pca, n_pcs=10, o_whiten=True, data_split=0.5,
                        fig_num=fig_count)
    fig_count += 1
    # break

roc_fig_labels_and_style(fig_num=1)


print('E[AUC_sbj]', np.mean(auc_means))
print('STD[AUC_sbj]', np.mean(std_auc))

# #TODO: process all data using subject global models.
#
# # global leave-one-out model
# global_cv_fig_handle = plt.figure()
# color_count = 0
# rocs_fig_handle = plt.figure()
# roc_aucs = []
#
# for sbj in dict_of_instances_arrays.keys():
#     all_instances = \
#         np.concatenate(
#             [tmp_inst_array for key, tmp_inst_array in dict_of_instances_arrays.items()], axis=0)
#     all_labels = np.concatenate(
#             [tmp_inst_array for key, tmp_inst_array in dict_of_labels_arrays.items()], axis=0)
#
#
# # global model cv
# random_state = 1
# X = all_instances
# y = all_labels
# prediction_duration = feature_sampling_period_in_seconds * num_prediction_frames
# mean_fpr, mean_tpr, mean_auc, std_auc, tprs = agg_pred_cv(X, y, cv_folds, cv_reps, random_state, prediction_duration,
#                                                           estimation_method, intensity_model, model_parameters,
#                                                           o_perform_normalization=True, o_perform_PCA=True, n_pcs=n_pcs,
#                                                           o_whiten=True)
#
# plot_roc_with_std(mean_fpr, mean_tpr, tprs, colors, color_count, fig_num=global_cv_fig_handle.number)
# roc_fig_labels_and_style()
# print('E[AUC_glob]', mean_auc, ' +- ', std_auc)

#
# # for sbj in dict_of_instances_arrays.keys():
# for sbj in {'1161'}:
#     blacklist = [sbj]
#     instances_from_other_sbj = \
#         np.concatenate(
#             [tmp_inst_array for key, tmp_inst_array in dict_of_instances_arrays.items() if key not in blacklist],
#             axis=0)
#     labels_from_other_sbj = np.concatenate(
#             [tmp_inst_array for key, tmp_inst_array in dict_of_labels_arrays.items() if key not in blacklist],
#             axis=0)
#
#     X_train = instances_from_other_sbj
#     y_train = labels_from_other_sbj
#     X_test = dict_of_instances_arrays[sbj]
#     y_test = dict_of_labels_arrays[sbj]
#
#     print('Train data shape: ', X_train.shape)
#     print('Test data shape: ', X_test.shape)
#     print('All data shape: ', all_instances.shape)
#
#     if o_normalize_data:
#         print('Normalizing data.')
#         norm_const = fext.get_normalization_constants(X_train, axis=0)
#         # norm_const = fext.get_normalization_constants(np.concatenate((X_train, X_test)), axis=0)
#         X_train = fext.normalize_data(X_train, norm_const)
#         X_test = fext.normalize_data(X_test, norm_const)
#
#         # scaler = preprocessing.StandardScaler().fit(np.concatenate((X_train, X_test)))
#         # X_train = scaler.transform(X_train)
#         # X_test = scaler.transform(X_train)
#         # print(np.mean(X_train, axis=0))
#         # print(np.std(X_train, axis=0))
#         # print(np.mean(X_test, axis=0))
#         # print(np.std(X_test, axis=0))
#
#     if o_perform_pca:
#         print('Perform PCA.')
#         pca = PCA(n_components=n_pcs, whiten=o_whiten)
#         pca = pca.fit(X_train)
#         # pca = pca.fit(np.concatenate((X_train, X_test)))
#         X_train = pca.transform(X_train)
#         X_test = pca.transform(X_test)
#         print(np.mean(X_train, axis=0))
#         print(np.std(X_train, axis=0))
#         print(np.mean(X_test, axis=0))
#         print(np.std(X_test, axis=0))
#
#     # prediction_duration = 1
#     prediction_duration = feature_sampling_period_in_seconds * num_prediction_frames
#     fpr_points, tpr_points, roc_auc = agg_pred_leave_one_subj_out(X_train, y_train, X_test, y_test, prediction_duration,
#                                                                   estimation_method, intensity_model,
#                                                                   model_param=model_parameters, o_plot_results=True)
#
#     # fpr_points, tpr_points, roc_auc = agg_pred_leave_one_subj_out(X_test, y_test, X_test, y_test, prediction_duration,
#     #                                                               estimation_method, intensity_model,
#     #                                                               model_param=model_parameters, o_plot_results=True)
#
#     roc_aucs.append(roc_auc)
#     plt.figure(rocs_fig_handle.number)
#     plt.plot(fpr_points, tpr_points, color=colors[color_count, :], linestyle='--', linewidth=1.5)
#     color_count += 1
#
#     print('AUC_sbj', roc_auc)
#
#
# roc_fig_labels_and_style()
#
#
# roc_fig_labels_and_style()
# print('E[AUC_sbj]', np.mean(roc_aucs), ' +- ', np.std(roc_aucs))

