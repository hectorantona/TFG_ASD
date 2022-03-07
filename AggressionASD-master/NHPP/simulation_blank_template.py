#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 11:52:49 2019

@author: tales
"""

import featureextraction as fext

#####################
# Configuration
#####################

# File read config:
data_path = '/home/tales/DataBases/smallASDData/Data'
path_style = '/'
subject_id_coding = 4  # Number of digits in subject ID coding


# Raw Feature extraction config:

# num_observation_frames = 12
# num_prediction_frames = 4
num_observation_frames = 1
num_prediction_frames = 1

# feat_code = 7                       # use all features
# feat_code = 6  # use all features but 'AGGObserved' and 'TimePastAggression'
feat_code = 6  # use only acceleration XYZ features.

# feature vector every feature_sampling_period in seconds
# binSize = '15S'  # Frame size in seconds
feature_sampling_period = '15S'

# observation window slide
observation_window_slide = num_observation_frames   # generates independent poisson Y_k random variable
# observation_window_slide = 1    # a lot of superposition


# High level data processing:

o_normalize_data = True  # normalize data?
o_perform_pca = True  # perform PCA?
n_pcs = 10  # number of principal components


# Classifier config:
cv_folds = 4
cv_reps = 6

# classifier_type = 'LR'
classifier_type = 'SVM'

o_perform_parameters_search = 0
# lr_par = {'C': 100, 'penalty': 'l2', 'tol': 0.01, 'solver': 'lbfgs', 'max_iter': 500}
lr_par = {'C': 1000000, 'penalty': 'l2', 'tol': 0.01, 'solver': 'saga', 'max_iter': 500}
svm_par = {'gamma': 1, 'C': 1000, 'kernel': 'rbf', 'probability': True}

# figure config
labels_fontsize = 16

# are csv files preprocessed?
# file name for the pre-processed data
feature_file_name = data_path + path_style + 'rawFeatureDataAllSubjects_' + str(feat_code) + '_.b'
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

feature_dict = fext.extract_features_from_raw_csv_files(data_path, feature_sampling_period, agg_category,
                                                        non_agg_category, feature_file_name)

# build processing feature vectors (instances) from raw features
dict_of_instances_arrays, dict_of_labels_arrays = fext.gen_instances_from_raw_feat_dictionary(feature_dict,
                                                                                              num_observation_frames,
                                                                                              num_prediction_frames,
                                                                                              feat_code)

############################################
# Front-End 2: High level feature processing
############################################




#################################################################################
# Back-End: Signal processing (classification, regression, Poisson modeling, etc.
#################################################################################


