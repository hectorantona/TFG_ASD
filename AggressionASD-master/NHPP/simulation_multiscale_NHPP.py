#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 11:52:49 2019

@author: tales
"""

import featureextraction as fext
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
from poissonprocess import agg_pred_cv, agg_pred_leave_one_subj_out, agg_pred_data_split, multiscale_cv
from plottools import plot_roc_with_std, roc_fig_labels_and_style

import pickle
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

#####################
# Configuration
#####################

# File read config:
data_path = '/scratch/talesim/smallASDData/Data'
# data_path = '/home/tales/DataBases/s_smallASDData/Data'
# data_path = '/home/tales/DataBases/smallASDData/Data'
path_style = '/'
subject_id_coding = 4  # Number of digits in subject ID coding


# Raw Feature extraction config:

num_observation_frames = 12
num_prediction_frames = 4
# num_observation_frames = 1
# num_prediction_frames = 1

# feat_code = 7                       # use all features
# feat_code = 6  # use all features but 'AGGObserved' and 'TimePastAggression'
feat_code = 6  # use only acceleration XYZ features.

# feature vector every feature_sampling_period in seconds
# binSize = '15S'  # Frame size in seconds
feature_sampling_period = '15S'
feature_sampling_period_in_seconds = 15

# observation window slide
observation_window_slide = num_observation_frames   # generates independent poisson Y_k random variable
# observation_window_slide = 1    # a lot of superposition


# High level data processing:

o_normalize_data = True  # normalize data?
o_perform_pca = True  # perform PCA?
n_pcs = 10  # number of principal components
o_whiten = False  # whiten PCs?

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
labels_fontsize = 16

# are csv files preprocessed?
# file name for the pre-processed data
feature_file_name = data_path + path_style + 'rawFeatureDataAllSubjects_' + str(feat_code) + '_nhpp.b'
instances_file_name = data_path + path_style + 'instancesFileName_' + str(feat_code) + '_nhpp.b'

agg_category = ['aggression', 'agg', 'aggression (?)', 'property destruction', 'noncompliance',
               'SIB', 'ED', 'kicked table', 'aggression/SIB', 'elopement/drop on the floor',
               'aggression/SIB/property destruction', 'dropped on floor', 'sib', 'elopement',
               'AGG', 'Agg', 'Agg/SIB']

non_agg_category = ['sleeping', 'mild ED/shoved by peer']



###############################
# Front-End: Feature Extraction
###############################

# extract features from raw csv files:

feature_dict = fext.extract_features_from_raw_csv_files(data_path, feature_sampling_period, agg_category,
                                                        non_agg_category, feature_file_name)

# get data from raw features:
max_num_prediction_frames = num_prediction_frames + 4
# max_num_prediction_frames = num_prediction_frames + 0
step = 1
x, y = fext.gen_instances_from_raw_feat_dictionary_multiscale(feature_dict, num_observation_frames, num_prediction_frames,
                                                      max_num_prediction_frames, step, feat_code)



############################################
# Front-End 2: High level feature processing
############################################


dict_of_instances_arrays = x
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
# Back-End: Signal processing (classification, regression, Poisson modeling, etc.
#################################################################################

keys = []

auc_means = []
auc_stds = []

fig_count = 2
random_state = 1

prediction_duration = list(np.arange(num_prediction_frames, max_num_prediction_frames + 1, step))
# prediction_duration = [4]
# person dependent models
# for each subject
results_dict = dict()
for key in x:
    y_list = []

    # add all label arrays to y_list
    for ll in y:
        y_list.append(ll[key])

    # call cv function here.
    [mean_fpr, mean_tpr, mean_auc, std_auc, tprs] = \
        multiscale_cv(x[key], y_list, cv_folds, cv_reps, random_state, prediction_duration, estimation_method,
                      intensity_model, model_param=model_parameters, o_perform_normalization=o_normalize_data,
                      o_perform_PCA=o_perform_pca, n_pcs=10, o_whiten=o_whiten, pca_model=pcas_dict[key],
                      scaler_model=scaler[key])

    results_dict.update({key: [mean_fpr, mean_tpr, mean_auc, std_auc, tprs]})


#    for i in range(len(mean_tpr)):
#        plt.plot(mean_fpr, mean_tpr[i], label="pred time = " + str(15 * int(prediction_duration[i])))
#    plt.legend()
#    plt.show()


#
## Save simulation
# results = [mean_fpr, mean_tpr, mean_auc, std_auc, tprs]
file_name = 'results_msnhpp_' + intensity_model + '_repX.data'
# save data in a csv file:
with open(file_name, 'wb') as filehandle:
    # store the data as binary data stream
    # pickle.dump(results, filehandle)
    pickle.dump(results_dict, filehandle)

plt.figure()
for key in results_dict:
    [mean_fpr, mean_tpr, mean_auc, std_auc, tprs] = results_dict[key]
    for i in range(len(mean_tpr)):
        plt.plot(mean_fpr, mean_tpr[i], label="pred time = " + str(15*int(prediction_duration[i])) + ", SID=" + key)
plt.legend()
fig_name = 'ms_nhpp' + str(intensity_model)
plt.savefig(fig_name + '_2.png')
plt.savefig(fig_name + '_2.pdf')
plt.close()
#
# plt.show()
