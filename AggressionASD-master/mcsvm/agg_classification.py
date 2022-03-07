
import featGen
import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot as plt
import math
from sklearn.model_selection import KFold
import numpy as np
from numpy import interp
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import argparse
from sklearn import svm
from math import trunc
from classifier_cv import GMMClassifier
import classifier_cv
import pickle
import os
from joblib import Parallel, delayed
import joblib
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score
import pandas as pd


def data_preprocess(data_path, path_style='/', num_observation_frames=12, num_prediction_frames=4, feat_code=6,
                    o_return_list_of_sessions=False, o_is_new_dataset=True, bin_size='15S', o_multiclass=True,
                    o_run_from_scratch=False, o_agg_intensity_classifier=False):
    # Load data
    agg_cat = ['aggression', 'agg', 'aggression (?)', 'property destruction', 'noncompliance',
                   'SIB', 'ED', 'kicked table', 'aggression/SIB', 'elopement/drop on the floor',
                   'aggression/SIB/property destruction', 'dropped on floor', 'sib', 'elopement',
                   'AGG', 'Agg', 'Agg/SIB']
    non_agg_cat = ['sleeping', 'mild ED/shoved by peer']


    # data_dict, uid_dict = featGen.feature_extraction(data_path, bin_size, agg_cat, non_agg_cat, feat_code, path_style=path_style,
    #                                    o_is_new_dataset=o_is_new_dataset, o_run_from_scracth=o_run_from_scratch,
    #                                                  o_multiclass=o_multiclass)
    data_dict, uid_dict = featGen.feature_extraction(data_path, bin_size, agg_cat, non_agg_cat, path_style=path_style,
                                                     o_is_new_dataset=o_is_new_dataset,
                                                     o_run_from_scracth=o_run_from_scratch,
                                                     o_multiclass=o_multiclass)
    # print(data_dict)
    # print("!!!!!!!!!!HHHHEEEEERRRREEEEEE!!!!!!!!!!!!!")
    agg_intensity_clf = None
    if o_agg_intensity_classifier is True and o_multiclass is False:
        n_clusters = 3
        agg_intensity_clf = featGen.kmeans_accnorm(data_dict, o_plot_histograms=False, n_clusters=n_clusters, return_data=False)

    o_run_from_scratch = True
    dict_of_instances_arrays, dict_of_labels_arrays, id_blacklist, dict_of_superposition_lists = \
        featGen.gen_instances_from_raw_feat_dictionary(data_dict, num_observation_frames, num_prediction_frames, feat_code,
                                                   agg_intensity_clf=agg_intensity_clf, o_is_new_dataset=o_is_new_dataset,
                                                   o_multiclass=o_multiclass,
                                                   o_return_list_of_sessions=o_return_list_of_sessions,
                                                   outdir=data_path, o_run_from_scracth=o_run_from_scratch, bin_size=bin_size)

    # remove blacklisted IDs from IDsdict
    if len(id_blacklist) != 0:
        for v in id_blacklist:
            for k in uid_dict.keys():
                if uid_dict[k] == v:
                    blacklisted_key = k
            del uid_dict[blacklisted_key]

    return dict_of_instances_arrays, dict_of_labels_arrays, id_blacklist, uid_dict, dict_of_superposition_lists

def run_mc_individual_classification_lso(X, y, cv_folds=10, cv_reps=5, o_normalize_data=False, o_perform_pca=False, n_pcs=10,
                                    clf_obj=svm.SVC, clf_par=None, output_file_name='out.data'):
    """
    :param X: Dictionary of lists of feature vectors. X[id][i] is a (n_samples, n_features) numpy array, for subject
    'id' and session 'i'.
    :param y: Dictionary of labels. y[id][i] is a (n_samples,) numpy array of labels for subject 'id' and session 'i'.
    y[id][i] is a 1D array of labels corresponding to the i-th session.
    :param cv_folds:
    :param cv_reps:
    :param o_normalize_data:
    :param o_perform_pca:
    :param n_pcs:
    :param clf_obj:
    :param clf_par:
    :param output_file_name:
    """

    classes = [0, 1, 2, 3]
    n_classes = 4
    # preprocess
    print("Pre-processing: Normalizing/PCA with data from other subjects...")
    if o_normalize_data or o_perform_pca:
        # train one PCA for each subj using data from other subjs.
        # get data from all subjects but the current
        if o_normalize_data:
            norm_constants_dict = dict()
        else:
            norm_constants_dict = None
        if o_perform_pca:
            pcas_dict = dict()
        else:
            pcas_dict = None
        # get norm constants and pca obj for each id
        for ko in X.keys():
            blacklist = ko
            instances_from_other_sbj = \
                np.concatenate([xlist[i] for key, xlist in X.items() if key not in blacklist for i in range(len(xlist))], axis=0)
            # instances_from_other_sbj = \
            #     np.concatenate([tmp_inst_array for key, tmp_inst_array in X.items() if
            #                     key not in blacklist], axis=0)

            if o_normalize_data:
                norm_const = classifier_cv.get_normalization_constants(instances_from_other_sbj, axis=0)
                norm_constants_dict[ko] = norm_const
                norm_data = classifier_cv.normalize_data(instances_from_other_sbj, norm_const)
            else:
                norm_data = instances_from_other_sbj
            if o_perform_pca:
                pca = PCA(n_components=n_pcs)
                pca.fit(norm_data)
                pcas_dict[ko] = pca

    mean_fpr = np.linspace(0, 1, 300)
    tprs = {}
    aucs = {}
    results = {}
    for id in X.keys():
        # gen data set

        # check min number of sessions
        if len(X[id]) < 2*cv_folds:
            print(f"Warning: ID {id} has less than {cv_folds} (cv_folds) sessions and will not be used.")
            continue

        # aucs[id] = []
        # tprs[id] = []

        Xlist = X[id]
        dummyX = list(range(len(Xlist)))

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i] = []
            tpr[i] = []
            roc_auc[i] = []

        fpr["micro"] = []
        tpr["micro"] = []
        roc_auc["micro"] = []

        fpr["macro"] = []
        tpr["macro"] = []
        roc_auc["macro"] = []

        mean_fpr = np.linspace(0, 1, 300)
        # tprs = []
        # aucs = []
        count = 0
        for rep in range(cv_reps):
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=rep)
            for train_idx, test_idx in kf.split(dummyX):
            # for fold in range(cv_folds):
                # create cv splits
                # n_sessions_per_fold = len(dummyX) // cv_folds
                # n_training_sessions = (cv_folds - 1) * n_sessions_per_fold
                # train_idx = list(np.random.choice(dummyX, n_training_sessions))
                # test_idx = [dummyX[i] for i in range(len(dummyX)) if i not in train_idx]
                X_train = np.concatenate([X[id][i] for i in train_idx], axis=0)
                y_train = np.concatenate([y[id][i] for i in train_idx], axis=0).ravel()
                X_test = np.concatenate([X[id][i] for i in test_idx], axis=0)
                y_test = np.concatenate([y[id][i] for i in test_idx], axis=0).ravel()


                if len(np.unique(y_train)) < 4 or len(np.unique(y_test)) < 2:
                    print("Fold has only one class for training: aborting CV fold!")
                    continue

            y_train = label_binarize(y_train, classes=classes)
            y_test = label_binarize(y_test, classes=classes)

            classifier = OneVsRestClassifier(clf_obj(**clf_par)).fit(X_train, y_train)

            # test classifiers
            y_pred = classifier.decision_function(X_test)

            # save performance metrics
            # fpr, tpr, thresholds = roc_curve(y_test, y_pred[:, 1])
            #
            # tprs.append(interp(mean_fpr, fpr, tpr))
            # tprs[-1][0] = 0.0
            # roc_auc = auc(fpr, tpr)
            # if math.isnan(roc_auc):
            #     print('!!!!!!NAN!!!!!!!!', roc_auc)
            #
            # aucs.append(roc_auc)

            for i in range(n_classes):
                # fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
                # roc_auc[i] = auc(fpr[i], tpr[i])
                fpr_, tpr_, _ = roc_curve(y_test[:, i], y_pred[:, i])
                roc_auc_ = auc(fpr_, tpr_)
                fpr[i].append(fpr_)
                tpr[i].append(interp(mean_fpr, fpr_, tpr_))
                roc_auc[i].append(roc_auc_)

            fpr_, tpr_, _ = roc_curve(y_test.ravel(), y_pred.ravel())
            roc_auc_ = auc(fpr_, tpr_)
            fpr["micro"].append(fpr_)
            tpr["micro"].append(interp(mean_fpr, fpr_, tpr_))
            roc_auc["micro"].append(auc(fpr_, tpr_))

            # First aggregate all false positive rates
            all_fpr = np.unique(np.concatenate([fpr[i][count] for i in range(n_classes)]))

            # Then interpolate all ROC curves at this points
            mean_tpr = np.zeros_like(mean_fpr)
            for i in range(n_classes):
                # mean_tpr += interp(all_fpr, fpr[i][rep], tpr[i][rep])
                mean_tpr += tpr[i][count]

            # Finally average it and compute AUC
            mean_tpr /= n_classes

            fpr["macro"].append(all_fpr)
            tpr["macro"].append(mean_tpr)
            roc_auc["macro"].append(auc(mean_fpr, tpr["macro"][count]))
            count += 1

            print("AVERAGE MACRO AUC = {0:0.4f})".format(np.mean(roc_auc["macro"])))
            print("STD AUC = {0:0.4f})".format(np.std(roc_auc["macro"])))
            # results = [mean_fpr, tprs, aucs]
            results[id] = [mean_fpr, tpr, roc_auc, num_observation_frames, num_prediction_frames]
            # with open(output_file_name, 'wb') as filehandle:
            #     # store the data as binary data stream
            #     pickle.dump(results, filehandle)
            #
            # return results

    with open(output_file_name, 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(results, filehandle)

    return results


def run_mc_population_classification_lso(X, y, cv_folds=5, cv_reps=5, o_normalize_data=False, o_perform_pca=False, n_pcs=10,
                                    clf_obj=svm.SVC, clf_par=None, output_file_name='out.data'):
    """
    Do cross validation for population models with leave some subjects out.
    :param X: Dictionary of feature vectors. X[id] is a (n_samples, n_features) numpy array, for subject 'id'.
    :param y: Dictionary of labels. y[id] is a (n_samples,) numpy array of labels for subject 'id'.
    :param cv_folds: number of cross-validation (cv) folds.
    :param cv_reps: number of cross-validation (cv) repetitions.
    :param o_normalize_data: boolean (default: False) zero-mean one standard deviation (whitening).
    :param o_perform_pca: boolean (default: False).
    :param n_pcs: int number of principal components.
    :param clf_obj: classifier constructor
    :param clf_par: dict with classifier's parameters.
    :param output_file_name: str with 'path+filename'
    :return: list containing [mean_fpr, tprs, aucs]
    """

    # get number of cores in the cpu
    # n_jobs = min(joblib.cpu_count(), sbjIDDict.keys().__len__())
    # n_jobs = min(joblib.cpu_count(), len(X))
    # n_jobs = min(joblib.cpu_count(), cv_reps)

    train_per = (1.0 - 1.0/cv_folds)
    ids_list = np.array(list(X.keys()))
    n_train_ids = round(train_per * len(X))
    # n_test_ids = len(X) - n_train_ids

    n_classes = 4
    classes = [0, 1, 2, 3]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i] = []
        tpr[i] = []
        roc_auc[i] = []

    fpr["micro"] = []
    tpr["micro"] = []
    roc_auc["micro"] = []

    fpr["macro"] = []
    tpr["macro"] = []
    roc_auc["macro"] = []

    mean_fpr = np.linspace(0, 1, 300)
    # tprs = []
    # aucs = []
    count = 0
    for rep in range(cv_reps):
        print("Running CV repetition #", rep)
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=rep)
        # kf.get_n_splits(ids_list)
        # for cv in range(cv_folds):
        cv_count = 0
        for train_idx, test_idx in kf.split(ids_list):
            print("Running CV Fold #", cv_count)
            cv_count += 1
            # train_ids = list(np.random.choice(ids_list, n_train_ids, replace=False))
            # test_ids = [ii for ii in ids_list if ii not in train_ids]
            train_ids = ids_list[train_idx]
            test_ids = ids_list[test_idx]
            X_train = np.concatenate([X[k] for k in train_ids], axis=0)
            y_train = np.concatenate([y[k] for k in train_ids], axis=0).ravel()

            X_test = np.concatenate([X[k] for k in test_ids], axis=0)
            y_test = np.concatenate([y[k] for k in test_ids], axis=0).ravel()

            if o_normalize_data:
                print("Normalizing data...")
                # normalize data
                norm_const = classifier_cv.get_normalization_constants(X_train, axis=0)
                X_train = classifier_cv.normalize_data(X_train, norm_const)
                X_test = classifier_cv.normalize_data(X_test, norm_const)

            if o_perform_pca:
                print("Performing PCA...")
                # perform pca
                pca = PCA(n_components=n_pcs)
                pca.fit(X_train)
                X_train = pca.transform(X_train)
                X_test = pca.transform(X_test)

            # # train classifier
            # classifier = clf_obj(**clf_par)
            # classifier.fit(X_train, y_train)
            # # test classifier
            # y_pred = classifier.predict_proba(X_test)

            y_train = label_binarize(y_train, classes=classes)
            y_test = label_binarize(y_test, classes=classes)

            classifier = OneVsRestClassifier(clf_obj(**clf_par)).fit(X_train, y_train)

            # test classifiers
            y_pred = classifier.decision_function(X_test)

            # save performance metrics
            # fpr, tpr, thresholds = roc_curve(y_test, y_pred[:, 1])
            #
            # tprs.append(interp(mean_fpr, fpr, tpr))
            # tprs[-1][0] = 0.0
            # roc_auc = auc(fpr, tpr)
            # if math.isnan(roc_auc):
            #     print('!!!!!!NAN!!!!!!!!', roc_auc)
            #
            # aucs.append(roc_auc)

            for i in range(n_classes):
                # fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
                # roc_auc[i] = auc(fpr[i], tpr[i])
                fpr_, tpr_, _ = roc_curve(y_test[:, i], y_pred[:, i])
                roc_auc_ = auc(fpr_, tpr_)
                fpr[i].append(fpr_)
                tpr[i].append(interp(mean_fpr, fpr_, tpr_))
                roc_auc[i].append(roc_auc_)

            fpr_, tpr_, _ = roc_curve(y_test.ravel(), y_pred.ravel())
            roc_auc_ = auc(fpr_, tpr_)
            fpr["micro"].append(fpr_)
            tpr["micro"].append(interp(mean_fpr, fpr_, tpr_))
            roc_auc["micro"].append(auc(fpr_, tpr_))

            # First aggregate all false positive rates
            all_fpr = np.unique(np.concatenate([fpr[i][count] for i in range(n_classes)]))

            # Then interpolate all ROC curves at this points
            mean_tpr = np.zeros_like(mean_fpr)
            for i in range(n_classes):
                # mean_tpr += interp(all_fpr, fpr[i][rep], tpr[i][rep])
                mean_tpr += tpr[i][count]

            # Finally average it and compute AUC
            mean_tpr /= n_classes

            fpr["macro"].append(all_fpr)
            tpr["macro"].append(mean_tpr)
            roc_auc["macro"].append(auc(mean_fpr, tpr["macro"][count]))
            count += 1

    print("AVERAGE MACRO AUC = {0:0.4f})".format(np.mean(roc_auc["macro"])))
    print("STD AUC = {0:0.4f})".format(np.std(roc_auc["macro"])))
    # results = [mean_fpr, tprs, aucs]
    results = [mean_fpr, tpr, roc_auc, num_observation_frames, num_prediction_frames]
    with open(output_file_name, 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(results, filehandle)

    return results


def run_individual_classification_lso(X, y, cv_folds=10, cv_reps=5, o_normalize_data=False, o_perform_pca=False, n_pcs=10,
                                    clf_obj=svm.SVC, clf_par=None, output_file_name='out.data'):
    """

    :param X: Dictionary of lists of feature vectors. X[id][i] is a (n_samples, n_features) numpy array, for subject
    'id' and session 'i'.
    :param y: Dictionary of labels. y[id][i] is a (n_samples,) numpy array of labels for subject 'id' and session 'i'.
    y[id][i] is a 1D array of labels corresponding to the i-th session.
    :param cv_folds:
    :param cv_reps:
    :param o_normalize_data:
    :param o_perform_pca:
    :param n_pcs:
    :param clf_obj:
    :param clf_par:
    :param output_file_name:
    """

    # preprocess
    print("Pre-processing: Normalizing/PCA with data from other subjects...")
    if o_normalize_data or o_perform_pca:
        # train one PCA for each subj using data from other subjs.
        # get data from all subjects but the current
        if o_normalize_data:
            norm_constants_dict = dict()
        else:
            norm_constants_dict = None
        if o_perform_pca:
            pcas_dict = dict()
        else:
            pcas_dict = None
        # get norm constants and pca obj for each id
        for ko in X.keys():
            blacklist = ko
            instances_from_other_sbj = \
                np.concatenate([xlist[i] for key, xlist in X.items() if key not in blacklist for i in range(len(xlist))], axis=0)
            # instances_from_other_sbj = \
            #     np.concatenate([tmp_inst_array for key, tmp_inst_array in X.items() if
            #                     key not in blacklist], axis=0)

            if o_normalize_data:
                norm_const = classifier_cv.get_normalization_constants(instances_from_other_sbj, axis=0)
                norm_constants_dict[ko] = norm_const
                norm_data = classifier_cv.normalize_data(instances_from_other_sbj, norm_const)
            else:
                norm_data = instances_from_other_sbj
            if o_perform_pca:
                pca = PCA(n_components=n_pcs)
                pca.fit(norm_data)
                pcas_dict[ko] = pca

    mean_fpr = np.linspace(0, 1, 300)
    tprs = {}
    aucs = {}

    for id in X.keys():
        # gen data set

        # check min number of sessions
        if len(X[id]) < cv_folds:
            print(f"Warning: ID {id} has less than {cv_folds} (cv_folds) sessions and will not be used.")
            continue

        aucs[id] = []
        tprs[id] = []

        Xlist = X[id]
        dummyX = list(range(len(Xlist)))

        for rep in range(cv_reps):
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=rep)
            for train_idx, test_idx in kf.split(dummyX):
            # for fold in range(cv_folds):
                # create cv splits
                # n_sessions_per_fold = len(dummyX) // cv_folds
                # n_training_sessions = (cv_folds - 1) * n_sessions_per_fold
                # train_idx = list(np.random.choice(dummyX, n_training_sessions))
                # test_idx = [dummyX[i] for i in range(len(dummyX)) if i not in train_idx]
                X_train = np.concatenate([X[id][i] for i in train_idx], axis=0)
                y_train = np.concatenate([y[id][i] for i in train_idx], axis=0).ravel()
                X_test = np.concatenate([X[id][i] for i in test_idx], axis=0)
                y_test = np.concatenate([y[id][i] for i in test_idx], axis=0).ravel()

                y_train[y_train >= 1] = 1
                y_test[y_test >= 1] = 1

                if len(np.unique(y_train)) == 1:
                    print("Fold has only one class for training: aborting CV fold!")
                    continue

                # train and test classifier
                classifier = clf_obj(**clf_par)
                classifier.fit(X_train, y_train)
                y_pred = classifier.predict_proba(X_test)

                # save performance metrics
                fpr, tpr, thresholds = roc_curve(y_test, y_pred[:, 1])
                roc_auc = auc(fpr, tpr)
                if math.isnan(roc_auc):
                    print('!!!!!!NAN!!!!!!!!', roc_auc)
                    continue

                tprs[id].append(interp(mean_fpr, fpr, tpr))
                tprs[id][-1][0] = 0.0

                aucs[id].append(roc_auc)
                # save results

    # return results
    results = [mean_fpr, tprs, aucs]
    with open(output_file_name, 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(results, filehandle)

    return results


def train_population_model(X, y, o_normalize_data=False, o_perform_pca=False, n_pcs=10,
                                    clf_obj=svm.SVC, clf_par=None, output_file_name='out.data'):
    """
    Do cross validation for population models with leave some subjects out.
    :param X: Dictionary of feature vectors. X[id] is a (n_samples, n_features) numpy array, for subject 'id'.
    :param y: Dictionary of labels. y[id] is a (n_samples,) numpy array of labels for subject 'id'.
    :param cv_folds: number of cross-validation (cv) folds.
    :param cv_reps: number of cross-validation (cv) repetitions.
    :param o_normalize_data: boolean (default: False) zero-mean one standard deviation (whitening).
    :param o_perform_pca: boolean (default: False).
    :param n_pcs: int number of principal components.
    :param clf_obj: classifier constructor
    :param clf_par: dict with classifier's parameters.
    :param output_file_name: str with 'path+filename'
    :return: list containing [mean_fpr, tprs, aucs]
    """

    X_train = np.concatenate([X[k] for k in X.keys()], axis=0)
    y_train = np.concatenate([y[k] for k in y.keys()], axis=0).ravel()

    y_train[y_train >= 1] = 1

    if o_normalize_data:
        print("Normalizing data...")
        # normalize data
        norm_const = classifier_cv.get_normalization_constants(X_train, axis=0)
        X_train = classifier_cv.normalize_data(X_train, norm_const)

    if o_perform_pca:
        print("Performing PCA...")
        # perform pca
        pca = PCA(n_components=n_pcs)
        pca.fit(X_train)
        X_train = pca.transform(X_train)

    # train classifier
    classifier = clf_obj(**clf_par)
    classifier.fit(X_train, y_train)

    with open(output_file_name, 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(classifier, filehandle)

    return classifier


def run_population_classification_lso_tps(X, y, cv_folds=5, cv_reps=5, o_normalize_data=False, o_perform_pca=False, n_pcs=10,
                                    clf_obj=svm.SVC, clf_par=None, output_file_name='out.data', min_session=4):
    """
    Do cross validation for population models with leave some subjects out.
    :param X: Dictionary of feature vectors. X[id] is a (n_samples, n_features) numpy array, for subject 'id'.
    :param y: Dictionary of labels. y[id] is a (n_samples,) numpy array of labels for subject 'id'.
    :param cv_folds: number of cross-validation (cv) folds.
    :param cv_reps: number of cross-validation (cv) repetitions.
    :param o_normalize_data: boolean (default: False) zero-mean one standard deviation (whitening).
    :param o_perform_pca: boolean (default: False).
    :param n_pcs: int number of principal components.
    :param clf_obj: classifier constructor
    :param clf_par: dict with classifier's parameters.
    :param output_file_name: str with 'path+filename'
    :param min_session: int minimum number of sessions for the individual to be considered.
    :return: list containing [cv][test_sbj][session][y_session, y_pred]
    """

    # get number of cores in the cpu
    # n_jobs = min(joblib.cpu_count(), sbjIDDict.keys().__len__())
    # n_jobs = min(joblib.cpu_count(), len(X))
    # n_jobs = min(joblib.cpu_count(), cv_reps)

    train_per = (1.0 - 1.0/cv_folds)
    ids_list = np.array(list(X.keys()))
    n_train_ids = round(train_per * len(X))
    # n_test_ids = len(X) - n_train_ids

    n_session_group = 5
    mean_fpr = np.linspace(0, 1, 300)
    tprs = []
    aucs = []
    aucs_per_session = []

    save_ys_per_id_and_session = []
    cross_cv_count = 0
    for rep in range(cv_reps):
        print("Running CV repetition #", rep)
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=rep)
        # kf.get_n_splits(ids_list)
        # for cv in range(cv_folds):
        cv_count = 0
        # cv split for ids
        for train_idx, test_idx in kf.split(ids_list):
            print("Running CV Fold #", cv_count)
            cv_count += 1

            save_ys_per_id_and_session.append([])
            # train_ids = list(np.random.choice(ids_list, n_train_ids, replace=False))
            # test_ids = [ii for ii in ids_list if ii not in train_ids]
            train_ids = ids_list[train_idx]
            test_ids = ids_list[test_idx]

            # Training data
            X_train = np.concatenate([X[k][i] for k in train_ids for i in range(len(X[k]))], axis=0)
            y_train = np.concatenate([y[k][i] for k in train_ids for i in range(len(y[k]))], axis=0).ravel()

            # X_test = np.concatenate([X[k] for k in test_ids], axis=0)
            # y_test = np.concatenate([y[k] for k in test_ids], axis=0).ravel()
            X_test_list = [X[k] for k in test_ids]
            y_test_list = [y[k] for k in test_ids]

            # making it two classes (0: non-aggression, 1: aggression)
            y_train[y_train >= 1] = 1
            # y_test[y_test >= 1] = 1

            if o_normalize_data:
                print("Normalizing data...")
                # normalize data
                norm_const = classifier_cv.get_normalization_constants(X_train, axis=0)
                X_train = classifier_cv.normalize_data(X_train, norm_const)
                # X_test = classifier_cv.normalize_data(X_test, norm_const)

            if o_perform_pca:
                print("Performing PCA...")
                # perform pca
                pca = PCA(n_components=n_pcs)
                pca.fit(X_train)
                X_train = pca.transform(X_train)
                # X_test = pca.transform(X_test)

            # train classifier
            classifier = clf_obj(**clf_par)
            classifier.fit(X_train, y_train)

            # # test classifier
            # y_pred = classifier.predict_proba(X_test)

            # test classifer per session:
            test_user_count = 0
            for X_id, y_id in zip(X_test_list, y_test_list):
                # if len(X_id) < min_session:
                #     print("Aborting: Subject has less than minimum required sessions.")
                #     continue
                aucs_per_session.append([[]])
                save_ys_per_id_and_session[cross_cv_count].append([])
                session_count = 0
                for X_session, y_session in zip(X_id, y_id):

                    # y_session[y_session >= 1] = 1
                    # if sum(y_session) < 10:
                    #     continue

                    save_ys_per_id_and_session[cross_cv_count][test_user_count].append([])
                    if o_normalize_data:
                        X_session = classifier_cv.normalize_data(X_session, norm_const)
                    if o_perform_pca:
                        X_session = pca.transform(X_session)

                    y_pred = classifier.predict_proba(X_session)
                    save_ys_per_id_and_session[cross_cv_count][test_user_count][session_count] = [y_session, y_pred[:,1]]
                    session_count += 1
                    # save performance metrics
                    # fpr, tpr, thresholds = roc_curve(y_session, y_pred[:, 1])
                    #
                    # tprs.append(interp(mean_fpr, fpr, tpr))
                    # tprs[-1][0] = 0.0
                    # roc_auc = auc(fpr, tpr)
                    # if math.isnan(roc_auc):
                    #     print('!!!!!!NAN!!!!!!!!', roc_auc)
                    #
                    # aucs_per_session[test_user_count].append(roc_auc)
                    # aucs.append(roc_auc)
                test_user_count += 1
            cross_cv_count += 1
    print("MEAN AUC = {0:0.4f})".format(np.mean(aucs)))
    print("STD AUC = {0:0.4f})".format(np.std(aucs)))
    # results = [mean_fpr, tprs, aucs, aucs_per_session]
    results = save_ys_per_id_and_session
    with open(output_file_name, 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(results, filehandle)

    return results


def run_population_classification_lso(X, y, cv_folds=5, cv_reps=5, o_normalize_data=False, o_perform_pca=False, n_pcs=10,
                                    clf_obj=svm.SVC, clf_par=None, output_file_name='out.data'):
    """
    Do cross validation for population models with leave some subjects out and return test results per session.
    :param X: Dictionary of lists of feature vectors. X[id][i] is a (n_samples, n_features) numpy array, for subject
    'id' and session 'i'.
    :param y: Dictionary of labels. y[id][i] is a (n_samples,) numpy array of labels for subject 'id' and session 'i'.
    y[id][i] is a 1D array of labels corresponding to the i-th session.
    :param cv_folds: number of cross-validation (cv) folds.
    :param cv_reps: number of cross-validation (cv) repetitions.
    :param o_normalize_data: boolean (default: False) zero-mean one standard deviation (whitening).
    :param o_perform_pca: boolean (default: False).
    :param n_pcs: int number of principal components.
    :param clf_obj: classifier constructor
    :param clf_par: dict with classifier's parameters.
    :param output_file_name: str with 'path+filename'
    :return: list containing [mean_fpr, tprs, aucs]
    """

    # get number of cores in the cpu
    # n_jobs = min(joblib.cpu_count(), sbjIDDict.keys().__len__())
    # n_jobs = min(joblib.cpu_count(), len(X))
    # n_jobs = min(joblib.cpu_count(), cv_reps)

    train_per = (1.0 - 1.0/cv_folds)
    ids_list = np.array(list(X.keys()))
    n_train_ids = round(train_per * len(X))
    # n_test_ids = len(X) - n_train_ids

    mean_fpr = np.linspace(0, 1, 300)
    tprs = []
    aucs = []
    for rep in range(cv_reps):
        print("Running CV repetition #", rep)
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=rep)
        # kf.get_n_splits(ids_list)
        # for cv in range(cv_folds):
        cv_count = 0
        for train_idx, test_idx in kf.split(ids_list):
            print("Running CV Fold #", cv_count)
            cv_count += 1
            # train_ids = list(np.random.choice(ids_list, n_train_ids, replace=False))
            # test_ids = [ii for ii in ids_list if ii not in train_ids]
            train_ids = ids_list[train_idx]
            test_ids = ids_list[test_idx]
            X_train = np.concatenate([X[k] for k in train_ids], axis=0)
            y_train = np.concatenate([y[k] for k in train_ids], axis=0).ravel()

            X_test = np.concatenate([X[k] for k in test_ids], axis=0)
            y_test = np.concatenate([y[k] for k in test_ids], axis=0).ravel()

            # making it two classes (0: non-aggression, 1: aggression)
            y_train[y_train >= 1] = 1
            y_test[y_test >= 1] = 1

            if o_normalize_data:
                print("Normalizing data...")
                # normalize data
                norm_const = classifier_cv.get_normalization_constants(X_train, axis=0)
                X_train = classifier_cv.normalize_data(X_train, norm_const)
                X_test = classifier_cv.normalize_data(X_test, norm_const)

            if o_perform_pca:
                print("Performing PCA...")
                # perform pca
                pca = PCA(n_components=n_pcs)
                pca.fit(X_train)
                X_train = pca.transform(X_train)
                X_test = pca.transform(X_test)

            # train classifier
            classifier = clf_obj(**clf_par)
            classifier.fit(X_train, y_train)
            # test classifier
            y_pred = classifier.predict_proba(X_test)

            # save performance metrics
            fpr, tpr, thresholds = roc_curve(y_test, y_pred[:, 1])

            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            if math.isnan(roc_auc):
                print('!!!!!!NAN!!!!!!!!', roc_auc)

            aucs.append(roc_auc)
    print("MEAN AUC = {0:0.4f})".format(np.mean(aucs)))
    print("STD AUC = {0:0.4f})".format(np.std(aucs)))
    results = [mean_fpr, tprs, aucs]
    with open(output_file_name, 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(results, filehandle)

    return results


def run_individual_classification_sessionsplits(X, y, test_prop=0.2, o_normalize_data=False, o_perform_pca=False, n_pcs=10,
                                    clf_obj=svm.SVC, clf_par=None, output_file_name='out.data',
                                     dict_of_superposition_lists=None, min_sessions=5):
    """

    :param X: Dictionary of lists of feature vectors. X[id][i] is a (n_samples, n_features) numpy array, for subject
    'id' and session 'i'.
    :param y: Dictionary of labels. y[id][i] is a (n_samples,) numpy array of labels for subject 'id' and session 'i'.
    y[id][i] is a 1D array of labels corresponding to the i-th session.
    :param test_prop: proportion of each session to be used for testing.
    :param o_normalize_data:
    :param o_perform_pca:
    :param n_pcs:
    :param clf_obj:
    :param clf_par:
    :param output_file_name:
    :param dict_of_superposition_lists: dictionary of overlapping feature vectors. dict_of_superposition_lists[id] is a
    list with len = n_samples in X[id] whose each element dict_of_superposition_lists[id][i] is a list [prev, post]
    indicating the index offset of feature vectors overlapping X[id][i].
    :return:
    """


    # # split each session in 2
    n_features = X[next(iter(X))][0].shape[1]
    tprs = []
    aucs = []
    fig_ax = None
    for key in X.keys():
        Xtrain = np.zeros((0, n_features))
        ytrain = np.zeros((0,))
        Xtest = np.zeros((0, n_features))
        ytest = np.zeros((0,))

        n_sessions = len(X[key])
        if n_sessions < min_sessions:
            print("Aborting subject " + str(key) + ". Number of sessions is smaller than " + str(min_sessions))
            continue

        for i in range(len(X[key])):
            X_session = X[key][i]
            y_session = y[key][i].ravel()
            n_samples = len(X_session)
            first_test_sample_idx = trunc(n_samples*(1-test_prop))

            n_overlapping_samples = dict_of_superposition_lists[key][i][first_test_sample_idx][0]
            last_training_sample_idx = first_test_sample_idx - n_overlapping_samples
            Xtrain = np.concatenate([Xtrain, X_session[0:last_training_sample_idx]])
            Xtest = np.concatenate([Xtest, X_session[first_test_sample_idx:len(X_session)]])
            ytrain = np.concatenate([ytrain, y_session[0:last_training_sample_idx]])
            ytest = np.concatenate([ytest, y_session[first_test_sample_idx:len(y_session)]])

        if o_normalize_data:
            print("Normalizing data...")
            # normalize data
            norm_const = classifier_cv.get_normalization_constants(Xtrain, axis=0)
            Xtrain = classifier_cv.normalize_data(Xtrain, norm_const)
            Xtest = classifier_cv.normalize_data(Xtest, norm_const)

        if o_perform_pca:
            print("Performing PCA...")
            # perform pca
            pca = PCA(n_components=n_pcs)
            pca.fit(Xtrain)
            Xtrain = pca.transform(Xtrain)
            Xtest = pca.transform(Xtest)

        ytrain[ytrain > 0.0] = 1
        ytest[ytest > 0.0] = 1

        print("Training Classifier with " + str(len(Xtrain)) + " samples.")

        classifier = clf_obj(**clf_par)
        classifier.fit(Xtrain, ytrain)

        print("Testing Classifier with " + str(len(Xtest)) + " samples.")
        ypred = classifier.predict_proba(Xtest)

        mean_fpr = np.linspace(0, 1, 300)
        fpr, tpr, thresholds = roc_curve(ytest, ypred[:, 1])
        tpr = interp(mean_fpr, fpr, tpr)
        # plt.plot(mean_fpr, tpr, color=[0.9, 0.9, 0.9])
        # fig_ax, _ = plot_roc(mean_fpr, tpr, fig_ax=fig_ax, color=[0.9, 0.9, 0.9])
        auc_ = auc(mean_fpr, tpr)
        aucs.append(auc_)
        tprs.append(tpr)
        print("AUC = " + str(auc_))
    print("MEAN AUC = " + str(np.mean(aucs)))
    # plt.plot(mean_fpr, np.mean(tprs, axis=0))
    # plot_roc(mean_fpr, np.mean(tprs, axis=0), fig_ax=fig_ax, label='Average ROC (AUC = {0:0.2f})'
    #                        ''.format(np.mean(aucs)))
    # plt.show()

    # plot_individual_models_rocs(mean_fpr, tprs, aucs, fig_name_with_path='./fig')

    # save models and data

    results = [mean_fpr, tprs, aucs]
    with open(output_file_name, 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(results, filehandle)

    return mean_fpr, tprs, aucs


def run_population_classification_sessionsplits(X, y, test_prop=0.2, o_normalize_data=False, o_perform_pca=False, n_pcs=10,
                                    clf_obj=svm.SVC, clf_par=None, output_file_name='out.data',
                                     dict_of_superposition_lists=None):
    """

    :param X: Dictionary of lists of feature vectors. X[id][i] is a (n_samples, n_features) numpy array, for subject
    'id' and session 'i'.
    :param y: Dictionary of labels. y[id][i] is a (n_samples,) numpy array of labels for subject 'id' and session 'i'.
    y[id][i] is a 1D array of labels corresponding to the i-th session.
    :param test_prop: proportion of each session to be used for testing.
    :param o_normalize_data:
    :param o_perform_pca:
    :param n_pcs:
    :param clf_obj:
    :param clf_par:
    :param output_file_name:
    :param dict_of_superposition_lists: dictionary of overlapping feature vectors. dict_of_superposition_lists[id] is a
    list with len = n_samples in X[id] whose each element dict_of_superposition_lists[id][i] is a list [prev, post]
    indicating the index offset of feature vectors overlapping X[id][i].
    :return:
    """


    # # split each session in 2
    n_features = X[next(iter(X))][0].shape[1]
    Xtrain = np.zeros((0, n_features))
    ytrain = np.zeros((0,))
    Xtest = np.zeros((0, n_features))
    ytest = np.zeros((0,))
    for key in X.keys():
        for i in range(len(X[key])):
            X_session = X[key][i]
            y_session = y[key][i].ravel()
            n_samples = len(X_session)
            first_test_sample_idx = trunc(n_samples*(1-test_prop))

            n_overlapping_samples = dict_of_superposition_lists[key][i][first_test_sample_idx][0]
            last_training_sample_idx = first_test_sample_idx - n_overlapping_samples
            Xtrain = np.concatenate([Xtrain, X_session[0:last_training_sample_idx]])
            Xtest = np.concatenate([Xtest, X_session[first_test_sample_idx:len(X_session)]])
            ytrain = np.concatenate([ytrain, y_session[0:last_training_sample_idx]])
            ytest = np.concatenate([ytest, y_session[first_test_sample_idx:len(y_session)]])

    if o_normalize_data:
        print("Normalizing data...")
        # normalize data
        norm_const = classifier_cv.get_normalization_constants(Xtrain, axis=0)
        Xtrain = classifier_cv.normalize_data(Xtrain, norm_const)
        Xtest = classifier_cv.normalize_data(Xtest, norm_const)

    if o_perform_pca:
        print("Performing PCA...")
        # perform pca
        pca = PCA(n_components=n_pcs)
        pca.fit(Xtrain)
        Xtrain = pca.transform(Xtrain)
        Xtest = pca.transform(Xtest)

    ytrain[ytrain > 0.0] = 1
    ytest[ytest > 0.0] = 1

    print("Training Classifier with " + str(len(Xtrain)) + " samples.")

    classifier = clf_obj(**clf_par)
    classifier.fit(Xtrain, ytrain)

    print("Testing Classifier with " + str(len(Xtest)) + " samples.")
    ypred = classifier.predict_proba(Xtest)

    # mean_fpr = np.linspace(0, 1, 300)
    fpr, tpr, thresholds = roc_curve(ytest, ypred[:, 1])
    auc_ = auc(fpr, tpr)
    # tpr = interp(mean_fpr, fpr, tpr)
    # plt.plot(fpr, tpr)
    # plt.show()
    # print("AUC = " + str(auc_))

    results = [fpr, tpr, auc_]
    with open(output_file_name, 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(results, filehandle)

    return fpr, tpr, auc_


def run_individual_classification_sc(X, y, cv_folds=10, cv_reps=5, o_normalize_data=False, o_perform_pca=False, n_pcs=10,
                                    clf_obj=svm.SVC, clf_par=None, output_file_name='out.data',
                                     dict_of_superposition_lists=None):
    """

    :param X: Dictionary of feature vectors. X[id] is a (n_samples, n_features) numpy array, for subject 'id'.
    :param y: Dictionary of labels. y[id] is a (n_samples,) numpy array of labels for subject 'id'.
    :param cv_folds:
    :param cv_reps:
    :param o_normalize_data:
    :param o_perform_pca:
    :param n_pcs:
    :param clf_obj:
    :param clf_par:
    :param output_file_name:
    :param dict_of_superposition_lists: dictionary of overlapping feature vectors. dict_of_superposition_lists[id] is a
    list with len = n_samples in X[id] whose each element dict_of_superposition_lists[id][i] is a list [prev, post]
    indicating the index offset of feature vectors overlapping X[id][i].
    """
    print("Pre-processing: Normalizing/PCA with data from other subjects...")
    if o_normalize_data or o_perform_pca:
        # train one PCA for each subj using data from other subjs.
        # get data from all subjects but the current
        if o_normalize_data:
            norm_constants_dict = dict()
        else:
            norm_constants_dict = None
        if o_perform_pca:
            pcas_dict = dict()
        else:
            pcas_dict = None
        for ko in X.keys():
            blacklist = ko
            instances_from_other_sbj = \
                np.concatenate([tmp_inst_array for key, tmp_inst_array in X.items() if
                                key not in blacklist], axis=0)

            if o_normalize_data:
                norm_const = classifier_cv.get_normalization_constants(instances_from_other_sbj, axis=0)
                norm_constants_dict[ko] = norm_const
                norm_data = classifier_cv.normalize_data(instances_from_other_sbj, norm_const)
            else:
                norm_data = instances_from_other_sbj
            if o_perform_pca:
                pca = PCA(n_components=n_pcs)
                pca.fit(norm_data)
                pcas_dict[ko] = pca

    print("Ensuring binary class labels")
    # making sure it has one positive class:
    del_keys = []
    for key in y.keys():
        # y['3174'][y['3174'] >= 1] = 1
        # if key == '4356':
        #     print(key)
        y[key][y[key] >= 1] = 1
        if np.sum(y[key]) <= cv_folds:
            print("Subject " + str(key) + " has too few behavioral episodes and will not be used!")
            del_keys.append(key)
    # delete del_keys from data
    del_keys.append('1253')
    for key in del_keys:
        del X[key]
        del y[key]
        del dict_of_superposition_lists[key]

    print("Initializing parallel model training and classification for each subject...")

    # get number of cores in the cpu
    n_jobs = min(joblib.cpu_count(), len(X))
    print("NJOBS = " + str(n_jobs))

    # train and test classifier using parallel for
    # results_list = Parallel(n_jobs=n_jobs)(
    #     delayed(classifier_cv.classify_instance_cv)(clf_obj, clf_par, X[key],
    #                                                 y[key], cv_folds,
    #                                                 cv_reps, o_perform_pca, pcas_dict,
    #                                                 o_normalize_data,
    #                                                 norm_constants_dict, key,
    #                                                 dict_of_superposition_lists=dict_of_superposition_lists[key])
    #     for key in X.keys())
    results_list = []
    for key in X.keys():
        if key is '1235':
            print(key)
        res = classifier_cv.classify_instance_cv(clf_obj, clf_par, X[key], y[key], cv_folds, cv_reps, o_perform_pca,
                                                  pcas_dict, o_normalize_data, norm_constants_dict, key,
                                                    dict_of_superposition_lists=dict_of_superposition_lists[key])
        results_list.append(res)

    results = results_list
    with open(output_file_name, 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(results, filehandle)

    return results


def run_population_classification_sc(X, y, cv_folds=5, cv_reps=5, o_normalize_data=False, o_perform_pca=False, n_pcs=10,
                                    clf_obj=svm.SVC, clf_par=None, output_file_name='out.data',
                                     dict_of_superposition_lists=None):
    """

    :param X: Dictionary of feature vectors. X[id] is a (n_samples, n_features) numpy array, for subject 'id'.
    :param y: Dictionary of labels. y[id] is a (n_samples,) numpy array of labels for subject 'id'.
    :param cv_folds:
    :param cv_reps:
    :param o_normalize_data:
    :param o_perform_pca:
    :param n_pcs:
    :param clf_obj:
    :param clf_par:
    :param output_file_name:
    :param dict_of_superposition_lists: dictionary of overlapping feature vectors. dict_of_superposition_lists[id] is a
    list with len = n_samples in X[id] whose each element dict_of_superposition_lists[id][i] is a list [prev, post]
    indicating the index offset of feature vectors overlapping X[id][i].
    :return:
    """

    # get number of cores in the cpu
    # n_jobs = min(joblib.cpu_count(), sbjIDDict.keys().__len__())
    # n_jobs = min(joblib.cpu_count(), len(X))
    n_jobs = min(joblib.cpu_count(), cv_reps)

    classifier = clf_obj(**clf_par)

    all_subjects_instances_array = \
        np.concatenate([tmp_inst_list for tmp_inst_list in X.values()], axis=0)
    all_subjects_labels = np.concatenate([tmp_label_list for tmp_label_list in y.values()],
                                         axis=0)

    X = all_subjects_instances_array
    y = all_subjects_labels.ravel()
    # making it single class
    y[y >= 1] = 1

    color_count = 0
    plt.figure()

    mean_fpr, mean_tpr, mean_auc, std_auc, tprs = classifier_cv.classify_instance_parallel_cv(classifier, X, y, cv_folds,
                                                                                              cv_reps, n_jobs, o_perform_pca,
                                                                                              n_pcs, o_normalize_data,
                                                                                              dict_of_superposition_lists=dict_of_superposition_lists)
    results = [mean_fpr, mean_tpr, mean_auc, std_auc, tprs]
    with open(output_file_name, 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(results, filehandle)

    return results


def plot_roc(fpr, tpr, fig_ax=None, color='C4', label=None, labels_fontsize=14):
    # def plot_roc(fpr, tpr, fig_ax=None, color=[0.2, 0.3, 0.9], label=None, labels_fontsize=14):
    # fig_handler = None
    if fig_ax is None:
        fig_handler = plt.figure()
        fig_ax = fig_handler.add_subplot(1, 1, 1)

    fig_ax.plot(fpr, tpr, color=color, label=label)
    if label is not None:
        fig_ax.legend()
    fig_ax.set_xlabel('False Positive Rate (1-Specificity)', fontweight='bold', fontsize=labels_fontsize)
    fig_ax.set_ylabel('True Positive Rate (Sensitivity)', fontweight='bold', fontsize=labels_fontsize)
    # plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.grid(color=[.9, .9, .9], linestyle='--')
    # return fig_ax, fig_handler


def plot_individual_models_rocs(mean_fpr, tprs, aucs, fig_name_with_path='fig', o_show_fig=True):
    n_curves = len(tprs)
    fig_handler = plt.figure()
    fig_ax = fig_handler.add_subplot(1, 1, 1)
    # plot individual ROCs
    for i in range(n_curves - 1):
        plot_roc(mean_fpr, tprs[i], fig_ax=fig_ax, color=[0.9, 0.9, 0.9])
    # Add legend to last individual ROC
    plot_roc(mean_fpr, tprs[i+1], fig_ax=fig_ax, color=[0.9, 0.9, 0.9], label='Individual ROCs')

    # plot average ROC:
    label = 'Average ROC (AUC = {0:0.2f})'.format(np.mean(aucs))
    plot_roc(mean_fpr, np.mean(tprs, axis=0), fig_ax=fig_ax, label=label)

    pdf_fig = fig_name_with_path + '.pdf'
    fig_handler.savefig(pdf_fig)
    png_fig = fig_name_with_path + '.png'
    fig_handler.savefig(png_fig)

    if o_show_fig:
        plt.show()


def plot_average_rocs(mean_fpr, tprs, aucs, fig_name_with_path='fig', o_show_fig=True):
    n_curves = len(tprs)
    fig_handler = plt.figure()
    fig_ax = fig_handler.add_subplot(1, 1, 1)

    # plot average ROC:
    label = rf'Average ROC, AUC={np.mean(aucs):.2f} $\pm$ {np.std(aucs):.2f}'
    m = np.mean(tprs, axis=0)
    s = np.std(tprs, axis=0)
    plot_roc(mean_fpr, m, fig_ax=fig_ax, label=label)
    plt.fill_between(mean_fpr, m - s, m + s, alpha=0.1, color='b')
    # print(f"MEAN AUC = {average_auc:.2f}")
    plt.legend()
    plt.grid(color=[.9, .9, .9], linestyle='--')

    pdf_fig = fig_name_with_path + '.pdf'
    fig_handler.savefig(pdf_fig)
    png_fig = fig_name_with_path + '.png'
    fig_handler.savefig(png_fig)

    if o_show_fig:
        plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--outputdir", type=str, default="Results", help="Output directory (default: ./Results/)")
    parser.add_argument("-op", type=bool, default=False, help="Only plot figures without running sym.")
    parser.add_argument("-a", "--algorithm", type=str, default="SVM", help="classifier name default (SVM). Use 'LR' for"
                                                                           " Logistic Regression")
    parser.add_argument("-g", "--gamma", type=float, default=0.1, help="Inverse of kernel bandwidth"
                                                                       " (Only required for nonlinear SVMs)")
    parser.add_argument("-r", "--regtype", type=str, default='l2', help="Regularization type (Only required for LR)")
    parser.add_argument("-C", type=float, default=100, help="classifier regularization parameter")
    parser.add_argument("-nc", "--n_components", type=int, default=20, help="Number of Gaussians (default is 20)")
    parser.add_argument("-k", "--kernel", type=str, default="rbf", help="Kernel name (only for SVM: e.g., rbf, poly, etc)")
    parser.add_argument("-tp", type=int, default=12, help="number of bins in the past (default is 12 = 3 min)")
    parser.add_argument("-tf", type=int, default=4, help="Number of bins in the future (default is 4 = 1 min)")
    parser.add_argument("-pca", type=int, default=0, help="Perform PCA (default is 1 (True))")
    parser.add_argument("-n", "--normalize", type=int, default=1,
                        help="Perform zero-mean one-std normalization (default is 1 (True))")
    parser.add_argument("-m", "--model", type=int, default=0,
                        help="model type (default is 0: population model. Use 1 for individual models, 2 for "
                             "leave-one-subject-out population models, "
                             "and 3 for individual models with leave-one-session out)")
    parser.add_argument("-s", "--min_sessions", type=int, default=2,
                        help="minimum number of sessions per subject (default is 2, only needed with -m 3)")
    parser.add_argument("-n_pcs", type=int, default=10, help="Number of Principal Components (default is 10)")
    parser.add_argument("-cv_folds", type=int, default=10, help="Number of cv folds (default is 10)")
    parser.add_argument("-cv_reps", type=int, default=1, help="Number of cv repetitions (default is 5)")
    parser.add_argument("-fc", "--feature_code", type=int, default=6, help="Feature Code: 6 (default) for all features "
                                                                           "but aggression info (aggression observed "
                                                                           "and time since aggression), 7 to include "
                                                                           "agg information.")
    parser.add_argument("-bs", "--bin_size", type=int, default='15', help="Bin duration in seconds (default=15)")

    args = parser.parse_args()
    print(args)

    o_is_new_dataset = True
    # data_path = '/scratch/talesim/new_CBS_data'
    # data_path = '/scratch/talesim/new_CBS_data_full'
    # data_path = '/scratch/talesim/tes/'

    # data_path = '/home/tales/DataBases/test_t4'
    # data_path = '/home/tales/DataBases/new_CBS_data'
    # dataPath = '/home/tales/DataBases/new_data_t1'
    # data_path = '/home/tales/DataBases/newT3'
    # data_path = '/home/tales/DataBases/Tranche 5'
    # data_path = '/home/tales/DataBases/Test_T3'
    # data_path = '/home/tales/DataBases/t3_small'

    # data_path = '/scratch/talesim/new_CBS_data_small'
    # data_path = '/home/tales/DataBases/new_CBS_data_small'
    # data_path = '/home/tales/DataBases/new_CBS_data_full'
    data_path = './CBS_DATA'
    # data_path = '/home/tales/DataBases/CBS_Test'

    path_style = '/'

    """
    About model_options:
        POP_SC: Population model single class with KFold or Stratified KFold cross-validation 
        IND_SC: Individual model single class with KFold or Stratified KFold cross-validation
        POP_SS: Population model with session split 
        IND_SS: Individual models with session splits    
        
        POP_LSO_TPS: POP_LSO with testing per session.
    """
    model_options = {'POP_SC': 0, 'IND_SC': 1, 'POP_SS': 2, 'IND_SS': 3, 'POP_LSO': 4, 'IND_LSO': 5, 'POP_MC_LSO': 6,
                     'IND_MC_LSO': 7, 'POP_CLUST': 8, 'TRAIN_POP': 9, 'POP_LSO_TPS': 10}
    # o_return_list_of_sessions should be True only if LOSO is selected.
    o_return_list_of_sessions = False

    # subjectIDCoding = 4  # Number of digits in subject ID coding
    # num_observation_frames = 12
    # num_prediction_frames = 4
    num_observation_frames = args.tp
    num_prediction_frames = args.tf

    bin_size = str(args.bin_size) + 'S'

    # feat_code = 7                       # use all features
    # feat_code = 6  # use all features but 'AGGObserved' and 'TimePastAggression'
    # feat_code = 1  # use only ACC data
    feat_code = args.feature_code

    o_normalize_data = args.normalize  # normalize data?
    o_perform_pca = args.pca  # perform PCA?
    n_pcs = args.n_pcs  # number of principal components

    fig_file_name = args.outputdir + '/sim_fc{}_tp{}_tf{}_bs{}'.format(feat_code, args.tp,
                                                                        args.tf, args.bin_size)

    if args.algorithm == 'SVM':
        clf_par = {'gamma': args.gamma, 'C': args.C, 'kernel': args.kernel, 'probability': True}
        clf_alg = svm.SVC
        fig_file_name +='_clf{}_par{}_{}_model'.format(args.algorithm, args.kernel + str(args.C) + "_"
                                                       + str(args.gamma), list(model_options.keys())[args.model])

    elif args.algorithm == 'LR':
        clf_par = {'C': args.C, 'penalty': args.regtype, 'tol': 0.01, 'solver': 'saga', 'max_iter': 500}
        # clf_par = {'gamma': args.gamma, 'C': args.C, 'kernel': args.kernel, 'probability': True}
        clf_alg = LogisticRegression
        fig_file_name += '_clf{}_par{}_{}_model'.format(args.algorithm, str(args.C) + "_",
                                                        list(model_options.keys())[args.model])

    elif args.algorithm == 'GMM':
        clf_par = {'n_components': args.n_components}
        clf_alg = GMMClassifier
        fig_file_name += '_clf{}_par{}_{}_model'.format(args.algorithm, args.n_components,
                                                        list(model_options.keys())[args.model])

    else:
        exit()

    data_file_name = fig_file_name + '.data'
    o_multiclass = True
    o_agg_intensity_classifier = False
    # os.makedirs(args.outputdir)
    # if args.model == model_options['POP_SC']:
    #     args.outputdir += '/POP_SC'
    #     if not os.path.isdir(args.outputdir):
    #         os.makedirs(args.outputdir)
    # elif args.model == model_options['IND_SC']:
    #     args.outputdir += '/IND_SC'
    #     if not os.path.isdir(args.outputdir):
    #         os.makedirs(args.outputdir)
    # el
    if (args.model == model_options['POP_SS']) or (args.model == model_options['POP_LSO_TPS']):
        # args.outputdir += '/POP_SS'
        # if not os.path.isdir(args.outputdir):
        #     os.makedirs(args.outputdir)
        o_return_list_of_sessions = True
    elif args.model == model_options['IND_SS'] or args.model == model_options['IND_LSO'] or args.model == model_options['IND_MC_LSO']:
        # args.outputdir += '/IND_SS'
        # if not os.path.isdir(args.outputdir):
        #     os.makedirs(args.outputdir)
        o_return_list_of_sessions = True
    elif args.model == model_options['POP_CLUST']:
        o_multiclass = False
        # o_run_from_scratch = True
        o_agg_intensity_classifier = True
    else:
        # print("Invalid model option!")
        pass

    if not args.op:
        dict_of_instances_arrays, dict_of_labels_arrays, id_blacklist, uid_dict, dict_of_superposition_lists\
            = data_preprocess(data_path, path_style='/', num_observation_frames=num_observation_frames,
                              num_prediction_frames=num_prediction_frames, feat_code=feat_code,
                              o_return_list_of_sessions=o_return_list_of_sessions, bin_size=bin_size,
                              o_multiclass=o_multiclass, o_agg_intensity_classifier=o_agg_intensity_classifier)

        if args.model == model_options['POP_SC']:
            # single class populations models

            results = run_population_classification_sc(dict_of_instances_arrays, dict_of_labels_arrays,
                                                       cv_folds=args.cv_folds, cv_reps=args.cv_reps,
                                                       o_normalize_data=o_normalize_data, o_perform_pca=o_perform_pca,
                                                       n_pcs=n_pcs, clf_obj=clf_alg, clf_par=clf_par,
                                                       output_file_name=data_file_name,
                                                       dict_of_superposition_lists=dict_of_superposition_lists)

        elif args.model == model_options['IND_SC']:
            # loo populations models

            results = run_individual_classification_sc(dict_of_instances_arrays, dict_of_labels_arrays,
                                                       cv_folds=args.cv_folds, cv_reps=args.cv_reps,
                                                       o_normalize_data=o_normalize_data, o_perform_pca=o_perform_pca,
                                                       n_pcs=n_pcs, clf_obj=clf_alg, clf_par=clf_par,
                                                       output_file_name=data_file_name,
                                                       dict_of_superposition_lists=dict_of_superposition_lists)
        elif args.model == model_options['POP_SS']:
            # single class populations models

            results = run_population_classification_sessionsplits(dict_of_instances_arrays, dict_of_labels_arrays,
                                                       test_prop=0.2,
                                                       o_normalize_data=o_normalize_data, o_perform_pca=o_perform_pca,
                                                       n_pcs=n_pcs, clf_obj=clf_alg, clf_par=clf_par,
                                                       output_file_name=data_file_name,
                                                       dict_of_superposition_lists=dict_of_superposition_lists)
        elif args.model == model_options['IND_SS']:

            results = run_individual_classification_sessionsplits(dict_of_instances_arrays, dict_of_labels_arrays,
                                                       test_prop=0.2,
                                                       o_normalize_data=o_normalize_data, o_perform_pca=o_perform_pca,
                                                       n_pcs=n_pcs, clf_obj=clf_alg, clf_par=clf_par,
                                                       output_file_name=data_file_name, min_sessions=args.min_sessions,
                                                       dict_of_superposition_lists=dict_of_superposition_lists)
        elif args.model == model_options['POP_LSO']:
            results = run_population_classification_lso(dict_of_instances_arrays, dict_of_labels_arrays,
                                                       cv_folds=args.cv_folds, cv_reps=args.cv_reps,
                                                       o_normalize_data=o_normalize_data, o_perform_pca=o_perform_pca,
                                                       n_pcs=n_pcs, clf_obj=clf_alg, clf_par=clf_par,
                                                       output_file_name=data_file_name)

        elif args.model == model_options['POP_MC_LSO'] or args.model == model_options['POP_CLUST']:
            results = run_mc_population_classification_lso(dict_of_instances_arrays, dict_of_labels_arrays,
                                                           cv_folds=args.cv_folds, cv_reps=args.cv_reps,
                                                           o_normalize_data=o_normalize_data,
                                                           o_perform_pca=o_perform_pca,
                                                           n_pcs=n_pcs, clf_obj=clf_alg, clf_par=clf_par,
                                                           output_file_name=data_file_name)
        elif args.model == model_options['IND_LSO']:
            results = run_individual_classification_lso(dict_of_instances_arrays, dict_of_labels_arrays,
                                                        cv_folds=args.cv_folds, cv_reps=args.cv_reps,
                                                        o_normalize_data=o_normalize_data, o_perform_pca=o_perform_pca,
                                                        n_pcs=n_pcs, clf_obj=clf_alg, clf_par=clf_par,
                                                        output_file_name=data_file_name)

        elif args.model == model_options['IND_MC_LSO']:
            results = run_mc_individual_classification_lso(dict_of_instances_arrays, dict_of_labels_arrays,
                                                        cv_folds=args.cv_folds, cv_reps=args.cv_reps,
                                                        o_normalize_data=o_normalize_data, o_perform_pca=o_perform_pca,
                                                        n_pcs=n_pcs, clf_obj=clf_alg, clf_par=clf_par,
                                                        output_file_name=data_file_name)

        elif args.model == model_options['TRAIN_POP']:
            classifier = train_population_model(dict_of_instances_arrays, dict_of_labels_arrays,
                                                        o_normalize_data=o_normalize_data, o_perform_pca=o_perform_pca,
                                                        n_pcs=n_pcs, clf_obj=clf_alg, clf_par=clf_par,
                                                        output_file_name=data_file_name)

        elif args.model == model_options['POP_LSO_TPS']:
            results = run_population_classification_lso_tps(dict_of_instances_arrays, dict_of_labels_arrays,
                                                        cv_folds=args.cv_folds, cv_reps=args.cv_reps,
                                                        o_normalize_data=o_normalize_data, o_perform_pca=o_perform_pca,
                                                        n_pcs=n_pcs, clf_obj=clf_alg, clf_par=clf_par,
                                                        output_file_name=data_file_name)

        else:
            pass

    # else:
    #     fig_file_name = args.outputdir + '/sim_tp_{}_tf_{}_clf_{}_{}_model.data'.format(args.tp, args.tf,
    #                                                                                     args.algorithm,
    #                                                                                     list(model_options.keys())[
    #                                                                                         args.model])
    #     data_file_name = fig_file_name + '.data'
    #

    with open(data_file_name, 'rb') as filehandle:
        # read the data as binary data stream
        results = pickle.load(filehandle)

    if args.model == model_options['POP_SC']:
        print("AUC = " + str(results[2]))
        # plt.plot(results[0], results[1])
        label = 'Population ROC (AUC = {0:0.2f})'.format(results[2])
        plot_roc(results[0], results[1], label=label)
        plt.savefig(fig_file_name + '.png')
        plt.savefig(fig_file_name + '.pdf')
        plt.show()

    elif args.model == model_options['IND_SC']:
        fig_handler = plt.figure()
        fig_ax = fig_handler.add_subplot(1, 1, 1)
        av_auc = 0
        av_tpr = np.zeros_like(results[0][0])
        for ll in results:
            # plt.plot(ll[0], ll[1], color=[0.9, 0.9, 0.9])
            plot_roc(ll[0], ll[1], fig_ax=fig_ax, color=[0.9, 0.9, 0.9])
            av_auc += ll[2]
            av_tpr += ll[1]
        av_auc /= len(results)
        av_tpr /= len(results)
        # plt.plot(ll[0], av_tpr)
        label = 'Average ROC (AUC = {0:0.2f})'.format(av_auc)
        plot_roc(ll[0], av_tpr, fig_ax=fig_ax, label=label)
        print(av_auc)
        # if True:
        plt.show()

    elif args.model == model_options['POP_SS']:
        print("AUC = " + str(results[2]))
        # plt.plot(results[0], results[1])
        label = 'Population ROC (AUC = {0:0.2f})'.format(results[2])
        plot_roc(results[0], results[1], label=label)
        plt.savefig(fig_file_name + '.png')
        plt.savefig(fig_file_name + '.pdf')
        plt.show()

    elif args.model == model_options['IND_SS']:
        mean_fpr, tprs, aucs = results
        plot_individual_models_rocs(mean_fpr, tprs, aucs, fig_name_with_path=fig_file_name)

    elif args.model == model_options['POP_LSO']:
        print("AUC = " + str(results[2]))
        # plt.plot(results[0], results[1])
        mean_auc = np.mean(results[2])
        mean_tpr = np.mean(results[1], axis=0)
        label = 'Population ROC (AUC = {0:0.2f})'.format(mean_auc)
        plot_roc(results[0], mean_tpr, label=label)
        plt.savefig(fig_file_name + '.png')
        plt.savefig(fig_file_name + '.pdf')
        plt.show()

    elif args.model == model_options['IND_LSO']:
        # print(results)
        mean_fpr = results[0]
        tprs_dict = results[1]
        aucs_dict = results[2]
        plt.figure()
        tprs = []
        average_auc = 0
        count = 0
        for id in tprs_dict:
            mean_tpr = np.mean(tprs_dict[id], axis=0)
            mean_id_auc = np.mean(aucs_dict[id])
            std_id_auc = np.std(aucs_dict[id])
            average_auc += np.sum(aucs_dict[id])
            count += len(aucs_dict[id])
            print(f"ID = {id}, AUC = {mean_id_auc} +- {std_id_auc}")
            if math.isnan(mean_id_auc):
                continue
            # plt.plot(mean_fpr, mean_tpr, color=[0.7, 0.7, 0.7], label=f'{id}, AUC={mean_id_auc:.2f}')
            plt.plot(mean_fpr, mean_tpr, color=[0.8, 0.8, 0.8])
            tprs.append(mean_tpr)
        average_tprs = np.mean(tprs, axis=0)
        average_auc = average_auc/count
        print(f"MEAN AUC = {average_auc:.2f}")
        plt.plot(mean_fpr, average_tprs, label=f'Average, AUC={average_auc:.2f}')
        plt.legend()
        plt.grid(color=[.9, .9, .9], linestyle='--')
        plt.savefig(fig_file_name + '.png')
        plt.savefig(fig_file_name + '.pdf')
        plt.show()

    elif args.model == model_options['POP_MC_LSO'] or args.model == model_options['POP_CLUST']:
        from multiclass_svm import plot_rocs
        [mean_fpr, tpr, roc_auc, num_observation_frames, num_prediction_frames] = results
        # remove not a number from tpr and roc_aucs

        class_dict = {0: "No", 1: "ED", 2: "SIB", 3: "AGG", 'micro': 'Micro', 'macro': 'Average'}
        if args.model == model_options['POP_CLUST']:
            class_dict = {1: "LOW", 2: "MID", 3: "HIGH"}
        plt_colors = ['black', 'red', 'crimson', 'darkgreen', 'navy', 'gold']
        plt_ls = [':', '-.', '--', '-', '-']
        for key in roc_auc.keys():
            rlist = roc_auc[key]
            roc_auc[key] = [x for x in rlist if (np.isnan(x) == 0)]
            tlist = tpr[key]
            tpr[key] = [x for x in tlist if (sum(np.isnan(x) == 0))]

        count = 0
        for key in tpr.keys():
            if str(key) == 'micro' or str(key) == 'macro':
                continue
            if len(tpr[key]) != 0:
                # plt.plot(mean_fpr, sum(tpr[key])/len(tpr[key]), label=class_dict[key] + f', AUC={np.mean(roc_auc[key]):.2f}')
                mean_tpr = np.mean(tpr[key], axis=0)
                conf_interval = 1.98*np.std(tpr[key], axis=0)
                plt.fill_between(mean_fpr, mean_tpr - conf_interval, mean_tpr + conf_interval, alpha=0.1, color=plt_colors[count])
                count += 1
        count = 0
        for key in tpr.keys():
            if str(key) == 'micro' or str(key) == 'macro':
                continue
            if args.model == model_options['POP_CLUST'] and key != 0:
                plot_label = class_dict[key]
                if plot_label == 'No':
                    plot_label = 'Combined'

            if len(tpr[key]) != 0:
                # plt.plot(mean_fpr, sum(tpr[key])/len(tpr[key]), label=class_dict[key] + f', AUC={np.mean(roc_auc[key]):.2f}')
                mean_tpr = np.mean(tpr[key], axis=0)
                conf_interval = 1.98*np.std(tpr[key], axis=0)
                # plt.plot(mean_fpr, mean_tpr, label=class_dict[key] + f', AUC={np.mean(roc_auc[key]):.2f}', color=plt_colors[count],  linestyle=plt_ls[count])
                plt.plot(mean_fpr, mean_tpr, label=plot_label + f', AUC={np.mean(roc_auc[key]):.2f}',
                         color=plt_colors[count], linestyle=plt_ls[count])
                print("AUC[" + str(class_dict[key]) + f"] = {np.mean(roc_auc[key]):.4f} ")
                # print(key)
                # plt.fill_between(mean_fpr, mean_tpr - conf_interval, mean_tpr + conf_interval, alpha=0.1, color=plt_colors[count])
                count += 1
        plt.ylim([-0.05, 1.05])
        plt.legend(loc='lower right')
        plt.grid(color=[.9, .9, .9], linestyle='--')
        labels_fontsize = 14
        plt.xlabel(r'False Positive Rate (1-Specificity)', fontweight='bold', fontsize=labels_fontsize)
        plt.ylabel('True Positive Rate (Sensitivity)', fontweight='bold', fontsize=labels_fontsize)
        plt.savefig(fig_file_name + '.png', dpi=300)
        plt.savefig(fig_file_name + '.pdf')
        plt.show()

    elif args.model == model_options['IND_MC_LSO']:
        # mean_fpr =
        # mean_tpr_dict
        tpr_dict = {}
        for id in results.keys():
            tpr_dict[id] = []
        mean_fpr = results[id][0]
        for id in results.keys():
            plt.figure()
            dict_of_tprs = results[id][1]
            for key in dict_of_tprs.keys():
                if not np.isnan(np.sum(tpr)):
                    tpr_dict[id].append(tpr)
            print(np.mean(tpr_dict[id], axis=0))
            plt.plot(mean_fpr, np.mean(tpr_dict[id], axis=0).ravel())
        plt.show()

##
##
    elif args.model == model_options['POP_LSO_TPS']:
        # -a LR -m 10 -C 0.1 -k rbf -s 12 -fc 7 -pca 0 -n_pcs 20 -bs 15 -tp 12 -tf 4 -o ./CBS_ASD_ONLY/ALR.M10.PCA0.FC7.BS15.Tp12.Tf4 -cv_folds 5 -cv_reps 2

        """
        results[cv_itt][sbj][session][data]
        """
        n_session_grouping = 1
        # n_session_grouping = 3

        labels_fontsize = 14
        max_n_sessions = 27
        # results is a list with [cv][test_sbj][session][y_session, y_pred]



        # idx_list = [5, 10, 15, 20, 26]
        idx_list = [i for i in range(n_session_grouping, max_n_sessions, n_session_grouping)]

        idx_list[-1] += max_n_sessions - idx_list[-1] - 1

        auc_cv_vs_session_g = []
        n_cvs_itt = len(results)
        cv_count = 0
        for cv_results in results:
            auc_cv_vs_session_g.append([])
            session_g_idx = 0
            init_idx = 0
            for end_idx in idx_list:
                y_s = np.empty((0,))
                y_p = np.empty((0,))
                auc_cv_vs_session_g[cv_count].append([])
                for sbj_l in cv_results:
                    for i in range(init_idx, min(len(sbj_l), end_idx)):
                        y_s = np.concatenate([y_s, sbj_l[i][0].ravel()])
                        y_p = np.concatenate([y_p, sbj_l[i][1]])
                y_s[y_s > 0] = 1
                if y_s.shape[0] == 0:
                    roc_auc = np.nan
                else:
                    fpr, tpr, thresholds = roc_curve(y_s, y_p)
                    roc_auc = auc(fpr, tpr)
                auc_cv_vs_session_g[cv_count][session_g_idx] = roc_auc
                init_idx = idx_list[session_g_idx]
                session_g_idx += 1
            cv_count += 1
        df = pd.DataFrame(auc_cv_vs_session_g)
        df.boxplot(grid=False)
        plt.grid(color=[.9, .9, .9], linestyle='--')
        plt.ylabel("AUC", fontsize=labels_fontsize, fontweight='bold')
        # tick
        if n_session_grouping > 1:
            a = [1] + [tt + 1 for tt in idx_list]
            tk = []
            for i in range(len(a)-1):
                tk.append(str(a[i]) + "-" + str(a[i + 1]-1))
            plt.xticks(fontsize=labels_fontsize-2)
            plt.xticks(ticks=np.arange(len(tk)) + 1, labels=tk)
            plt.xlabel("Session Group", fontsize=labels_fontsize, fontweight='bold')
        else:
            # plt.xticks(fontsize=labels_fontsize - 3)
            plt.xticks(ticks=np.arange(max_n_sessions) + 1, labels=np.arange(max_n_sessions) + 1)
            plt.xlabel("Session", fontsize=labels_fontsize, fontweight='bold')

        plt.show()
        # mean_fpr, tprs, aucs, aucs_per_session = results
        # aucs = np.array(aucs)
        # non_nan_idxs = ~np.isnan(aucs)
        # aucs = aucs[non_nan_idxs]
        # # for i in range(len(tprs)):
        # #     if ~non_nan_idxs[i]:
        # #         del tprs[i]
        #
        # df = pd.DataFrame(aucs_per_session)
        # conc_aucs = []
        # step = 5
        # count = 0
        # while True:
        #     conc_aucs.append(np.concatenate(aucs_per_session[count: max(count + step, len(df.columns))]).tolist())
        #     count += step
        #     if count > len(df.columns):
        #         break
        # plt.figure()
        #
        # df.boxplot(grid=False)
        # plt.grid(color=[.9, .9, .9], linestyle='--')
        # plt.xlabel('Session', fontsize=labels_fontsize)
        # plt.ylabel('AUC', fontsize=labels_fontsize)
        # plt.show()
        #
        # clean_tprs = []
        # for tpr in tprs:
        #     if np.isnan(np.sum(tpr)):
        #         continue
        #     clean_tprs.append(tpr)
        #
        # plot_average_rocs(mean_fpr, clean_tprs, aucs, fig_name_with_path=fig_file_name, o_show_fig=True)
        #
        # # plotting betas
        # from scipy.stats import beta
        # a, b, loc, scale = beta.fit(aucs)
        # rv = beta(a, b, loc=loc, scale=scale)
        # t = np.linspace(0, 1, 100)
        # plt.figure()
        # plt.plot(t, rv.pdf(t))
        # plt.ylabel(r'Beta PDF', fontsize=labels_fontsize)
        # plt.xlabel(r'AUC', fontsize=labels_fontsize)
        # plt.grid(color=[.9, .9, .9], linestyle='--')

    elif args.model == model_options['TRAIN_POP']:

        # python agg_classification.py  -a LR -cv_folds 5 -cv_reps 1 -m 9 -C 0.1 -k rbf -s 12 -fc 7 -pca 0 -n_pcs 20 -bs 15 -tp 12 -tf 4 -o ./tttALR.M9PCA0.FC6.BS15.Tp12.Tf4 -op 1


        classifier = results
        n_coefs = len(classifier.coef_.ravel())
        # n_largest_coefs = 132  # 50% of the coefficient mass
        # n_largest_coefs = 238 # 70% of the coefficient mass
        n_largest_coefs = n_coefs  # 100% of the coefficient mass
        # n_largest_coefs = 132  #asd

        if args.op == 1:
            o_return_list_of_sessions = False
            dict_of_instances_arrays, dict_of_labels_arrays, id_blacklist, uid_dict, dict_of_superposition_lists \
                = data_preprocess(data_path, path_style='/', num_observation_frames=num_observation_frames,
                                  num_prediction_frames=num_prediction_frames, feat_code=feat_code,
                                  o_return_list_of_sessions=o_return_list_of_sessions, bin_size=bin_size,
                                  o_multiclass=o_multiclass, o_agg_intensity_classifier=o_agg_intensity_classifier)

        # plot abs Classifier parameters
        labels_fontsize = 14
        plt.bar(np.linspace(0, n_coefs, n_coefs), np.abs(classifier.coef_.ravel()))
        plt.xlabel(r'Coefficient Index', fontsize=labels_fontsize)
        plt.ylabel(r'$\|w\|$', fontsize=labels_fontsize)
        plt.grid(color=[.9, .9, .9], linestyle='--')
        plt.show()

        selected_feat = featGen.select_feat_from_feat_code(feat_code, o_is_new_dataset=o_is_new_dataset)
        std_feat = ['STD_' + selected_feat[i] for i in range(len(selected_feat))]
        features = selected_feat*num_observation_frames + std_feat
        # get largest coef.

        w = classifier.coef_.ravel().copy()
        max_coef_index = np.argmax(w)
        w[max_coef_index] = 0
        second_max_coef_index = np.argmax(w)

        idx = np.argsort(np.abs(w))
        sw = w[idx]
        sw = sw[-n_largest_coefs:]
        npfeat = np.array(features)
        sfeat = npfeat[idx[-n_largest_coefs:]]

        print("Max coef = " + features[max_coef_index])
        print("2nd Max coef = " + features[second_max_coef_index])
        X = np.concatenate([dict_of_instances_arrays[k] for k in dict_of_instances_arrays.keys()], axis=0)

        if o_normalize_data:
            print("Normalizing data...")
            # normalize data
            norm_const = classifier_cv.get_normalization_constants(X, axis=0)
            X = classifier_cv.normalize_data(X, norm_const)

        y = np.concatenate([dict_of_labels_arrays[k] for k in dict_of_labels_arrays.keys()], axis=0)
        print("Here!")
        y[y>=1] = 1
        n_plus = int(np.sum(y))
        n_minus = len(y) - n_plus
        X_minus = np.empty((n_minus, X.shape[1]))
        X_plus = np.empty((n_plus, X.shape[1]))

        c_minus = 0
        c_plus = 0
        for i in range(len(X)):
            if y[i] == 0:
                X_minus[c_minus] = X[i]
                c_minus += 1
                    # X[i, max_coef_index], X[i, second_max_coef_index], '.b')
            else:
                X_plus[c_plus] = X[i]
                c_plus += 1
                # X_plus = np.concatenate([X_plus, X[i].reshape(1, -1)], axis=0)
                # X_plus.append(X[i])
                # plt.plot(X[i, max_coef_index], X[i, second_max_coef_index], '.r')
        # plt.plot(X_minus[:, max_coef_index], X_minus[:, second_max_coef_index], '.b')
        # plt.plot(X_plus[:, max_coef_index], X_plus[:, second_max_coef_index], '.r')
        sX_minus = X_minus[:, idx[-n_largest_coefs:]]
        sX_plus = X_plus[:, idx[-n_largest_coefs:]]
        sw = sw[-n_largest_coefs:]
        print("Plotting!")
        # plt.figure()
        # plt.plot(sX_minus[:, ['ACC' in feat for feat in sfeat]].dot(sw[['ACC' in feat for feat in sfeat]]),
        #          sX_minus[:, ['EDA' in feat for feat in sfeat]].dot(sw[['EDA' in feat for feat in sfeat]]), '.b', alpha=0.1)
        # plt.plot(sX_plus[:, ['ACC' in feat for feat in sfeat]].dot(sw[['ACC' in feat for feat in sfeat]]),
        #          sX_plus[:, ['EDA' in feat for feat in sfeat]].dot(sw[['EDA' in feat for feat in sfeat]]), '.r', alpha=0.1)
        # plt.xlabel('ACC Top Features')
        # plt.ylabel('EDA Top Features')
        # plt.grid(color=[.9, .9, .9], linestyle='--')
        # # plt.xlim(0, 10)
        # plt.show()
        #
        # plt.figure()
        # plt.plot(sX_minus[:, ['ACC' in feat for feat in sfeat]].dot(sw[['ACC' in feat for feat in sfeat]]),
        #          sX_minus[:, ['EDA' in feat for feat in sfeat]].dot(sw[['EDA' in feat for feat in sfeat]]), '.b',
        #          alpha=0.1)
        # plt.plot(sX_plus[:, ['ACC' in feat for feat in sfeat]].dot(sw[['ACC' in feat for feat in sfeat]]),
        #          sX_plus[:, ['EDA' in feat for feat in sfeat]].dot(sw[['EDA' in feat for feat in sfeat]]), '.r',
        #          alpha=0.1)
        # plt.xlabel('ACC Top Features')
        # plt.ylabel('EDA Top Features')
        # plt.grid(color=[.9, .9, .9], linestyle='--')
        # plt.xlim(-1, 1)
        # plt.ylim(-1, 1)
        # plt.show()
        #
        # plt.figure()
        # plt.plot(sX_minus[:, 1:n_largest_coefs//2].dot(sw[1:n_largest_coefs//2]),
        #          sX_minus[:, n_largest_coefs//2:].dot(sw[n_largest_coefs//2:]), '.b',
        #          alpha=0.1)
        # plt.plot(sX_plus[:, 1:n_largest_coefs//2].dot(sw[1:n_largest_coefs//2]),
        #          sX_plus[:,  n_largest_coefs//2:].dot(sw[ n_largest_coefs//2:]), '.r',
        #          alpha=0.1)
        # plt.xlabel('first half of Top Features')
        # plt.ylabel('second half of Top Features')
        # plt.grid(color=[.9, .9, .9], linestyle='--')
        # # plt.xlim(-1, 1)
        # # plt.ylim(-1, 1)
        # plt.show()

        # 3d scatter plot
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x_m_acc = sX_minus[:, ['ACC' in feat for feat in sfeat]].dot(sw[['ACC' in feat for feat in sfeat]])
        x_m_bvp = sX_minus[:, ['BVP' in feat for feat in sfeat]].dot(sw[['BVP' in feat for feat in sfeat]])
        x_m_eda = sX_minus[:, ['EDA' in feat for feat in sfeat]].dot(sw[['EDA' in feat for feat in sfeat]])

        x_m_bvp_eda = sX_minus[:, [('BVP' in feat) or ('EDA' in feat) for feat in sfeat]].dot(
            sw[[('BVP' in feat) or ('EDA' in feat) for feat in sfeat]])

        x_p_acc = sX_plus[:, ['ACC' in feat for feat in sfeat]].dot(sw[['ACC' in feat for feat in sfeat]])
        x_p_bvp = sX_plus[:, ['BVP' in feat for feat in sfeat]].dot(sw[['BVP' in feat for feat in sfeat]])
        x_p_eda = sX_plus[:, ['EDA' in feat for feat in sfeat]].dot(sw[['EDA' in feat for feat in sfeat]])

        x_p_bvp_eda = sX_plus[:, [('BVP' in feat) or ('EDA' in feat) for feat in sfeat]].dot(
            sw[[('BVP' in feat) or ('EDA' in feat) for feat in sfeat]])

        ax.scatter(x_m_acc, x_m_bvp, x_m_eda, 'b')
        ax.scatter(x_p_acc, x_p_bvp, x_p_eda, 'r')
        ax.set_xlabel('ACC Combined Feature')
        ax.set_ylabel('BVP Combined Feature')
        ax.set_zlabel('EDA Combined Feature')
        plt.show()


        def plot_mgaussians(Xm, Xp, xlabel, ylabel):
            mm = np.mean(Xm, axis=0)
            covm = np.cov(Xm.T)
            # stdm = np.std(Xm, axis=0)
            mp = np.mean(Xp, axis=0)
            covp = np.cov(Xp.T)
            # stdp = np.std(Xp, axis=0)
            # plt.figure()
            N = 40
            # if len(mp) < 30:
            #     return
            # N = min([40, len(mp)])

            XX, YY = np.meshgrid(np.linspace(-2, 2, N), np.linspace(-2, 2, N))
            from scipy.stats import multivariate_normal
            rv_m = multivariate_normal(mm, covm)
            rv_p = multivariate_normal(mp, covp)
            pos = np.empty(XX.shape + (2,))
            pos[:, :, 0] = XX
            pos[:, :, 1] = YY
            pd_m = rv_m.pdf(pos)
            pd_p = rv_p.pdf(pos)
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            # ax.plot_surface(XX, YY, 0.12*pd_m, cmap='viridis', linewidth=0, alpha=0.3, label='No Behavior')
            # ax.plot_surface(XX, YY, 0.87*pd_p, linewidth=0, alpha=0.3, label='Aggregated Behavior')
            ax.plot_surface(XX, YY, pd_m, cmap='viridis', linewidth=0, alpha=0.3, label='No Behavior')
            ax.plot_surface(XX, YY, pd_p, linewidth=0, alpha=0.3, label='Aggregated Behavior')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_zlabel('Probability Density')
            # plt.title("Multivariate Normal Distribution")
            # plt.legend()
            plt.show()


        # pair-wise plots:
        # ACC vs BVP
        Xm = np.concatenate([x_m_acc.reshape(-1, 1), x_m_bvp.reshape(-1, 1)], axis=1)
        Xp = np.concatenate([x_p_acc.reshape(-1, 1), x_p_bvp.reshape(-1, 1)], axis=1)
        plt.figure()
        plt.plot(Xm[:, 0], Xm[:, 1], '.', label='No Behavior')
        plt.plot(Xp[:, 0], Xp[:, 1], '.', label='Aggregated Behavior')
        plt.legend()
        plt.xlabel('ACC Combined Feature')
        plt.ylabel('BVP Combined feature')
        plt.grid(color=[.9, .9, .9], linestyle='--')
        plt.show()

        plot_mgaussians(Xm, Xp, 'ACC Combined Feature', 'BVP Combined feature')


        # ACC vs EDA

        Xm = np.concatenate([x_m_acc.reshape(-1, 1), x_m_eda.reshape(-1, 1)], axis=1)
        Xp = np.concatenate([x_p_acc.reshape(-1, 1), x_p_eda.reshape(-1, 1)], axis=1)
        plt.figure()
        plt.plot(Xm[:, 0], Xm[:, 1], '.', label='No Behavior')
        plt.plot(Xp[:, 0], Xp[:, 1], '.', label='Aggregated Behavior')
        plt.legend()
        plt.xlabel('ACC Combined Feature')
        plt.ylabel('EDA Combined feature')
        plt.grid(color=[.9, .9, .9], linestyle='--')
        plt.show()

        plot_mgaussians(Xm, Xp, 'ACC Combined Feature', 'EDA Combined feature')


        # BVP vs EDA

        Xm = np.concatenate([x_m_bvp.reshape(-1, 1), x_m_eda.reshape(-1, 1)], axis=1)
        Xp = np.concatenate([x_p_bvp.reshape(-1, 1), x_p_eda.reshape(-1, 1)], axis=1)
        plt.figure()
        plt.plot(Xm[:, 0], Xm[:, 1], '.', label='No Behavior')
        plt.plot(Xp[:, 0], Xp[:, 1], '.', label='Aggregated Behavior')
        plt.legend()
        plt.xlabel('BVP Combined Feature')
        plt.ylabel('EDA Combined feature')
        plt.grid(color=[.9, .9, .9], linestyle='--')
        plt.show()

        plot_mgaussians(Xm, Xp, 'BVP Combined Feature', 'EDA Combined feature')

        # ACC VS (BVP + EDA)
        Xm = np.concatenate([x_m_acc.reshape(-1, 1), x_m_bvp_eda.reshape(-1, 1)], axis=1)
        Xp = np.concatenate([x_p_acc.reshape(-1, 1), x_p_bvp_eda.reshape(-1, 1)], axis=1)
        plt.figure()
        plt.plot(Xm[:, 0], Xm[:, 1], '.', label='No Behavior')
        plt.plot(Xp[:, 0], Xp[:, 1], '.', label='Aggregated Behavior')
        plt.legend()
        plt.xlabel('ACC Combined Feature')
        plt.ylabel('BVP + EDA Combined feature')
        plt.grid(color=[.9, .9, .9], linestyle='--')
        plt.show()

        plot_mgaussians(Xm, Xp, 'ACC Combined Feature', 'BVP + EDA Combined feature')

        if args.feature_code == 7:
            # augmented features vs rest
            x_m_aug = sX_minus[:, [('AGG' in feat) or ('Time' in feat) for feat in sfeat]].dot(
                sw[[('AGG' in feat) or ('Time' in feat) for feat in sfeat]])
            x_p_aug = sX_plus[:, [('AGG' in feat) or ('Time' in feat) for feat in sfeat]].dot(
                sw[[('AGG' in feat) or ('Time' in feat) for feat in sfeat]])

            x_m_acc_bvp_eda = sX_minus[:, [('ACC' in feat) or ('BVP' in feat) or ('EDA' in feat) for feat in sfeat]].dot(
                sw[[('ACC' in feat) or ('BVP' in feat) or ('EDA' in feat) for feat in sfeat]])
            x_p_acc_bvp_eda = sX_plus[:, [('ACC' in feat) or ('BVP' in feat) or ('EDA' in feat) for feat in sfeat]].dot(
                sw[[('ACC' in feat) or ('BVP' in feat) or ('EDA' in feat) for feat in sfeat]])

            alpha = 0.1
            plt.figure()
            plt.plot(x_m_aug, x_m_acc, '.', label='No Behavior', alpha=alpha)
            plt.plot(x_p_aug, x_p_acc, '.', label='Aggregated Behavior', alpha=alpha)
            plt.legend()
            plt.xlabel('AGG Combined Feature')
            plt.ylabel('ACC Combined feature')
            plt.grid(color=[.9, .9, .9], linestyle='--')
            plt.show()

            plt.figure()
            plt.plot(x_m_aug, x_m_bvp_eda, '.', label='No Behavior', alpha=alpha)
            plt.plot(x_p_aug, x_p_bvp_eda, '.', label='Aggregated Behavior', alpha=alpha)
            plt.legend()
            plt.xlabel('AGG Combined Feature')
            plt.ylabel('BVP+EDA Combined feature')
            plt.grid(color=[.9, .9, .9], linestyle='--')
            plt.show()

            plt.figure()
            plt.plot(x_m_aug, x_m_eda, '.', label='No Behavior', alpha=alpha)
            plt.plot(x_p_aug, x_p_eda, '.', label='Aggregated Behavior', alpha=alpha)
            plt.legend()
            plt.xlabel('AGG Combined Feature')
            plt.ylabel('EDA Combined feature')
            plt.grid(color=[.9, .9, .9], linestyle='--')
            plt.show()

            plt.figure()
            plt.plot(x_m_aug, x_m_acc_bvp_eda, '.', label='No Behavior', alpha=alpha)
            plt.plot(x_p_aug, x_p_acc_bvp_eda, '.', label='Aggregated Behavior', alpha=alpha)
            plt.legend()
            plt.xlabel('AGG Combined Feature')
            plt.ylabel('ACC+BVP+EDA Combined feature')
            plt.grid(color=[.9, .9, .9], linestyle='--')
            plt.show()


        # # gaussian fit plots:
        # # Xm = np.concatenate([x_m_acc.reshape(-1, 1), x_m_bvp.reshape(-1, 1), x_m_eda.reshape(-1, 1)], axis=1)
        # # Xp = np.concatenate([x_p_acc.reshape(-1, 1), x_p_bvp.reshape(-1, 1), x_p_eda.reshape(-1, 1)], axis=1)
        # Xm = np.concatenate([x_m_acc.reshape(-1, 1), x_m_eda.reshape(-1, 1)], axis=1)
        # Xp = np.concatenate([x_p_acc.reshape(-1, 1), x_p_eda.reshape(-1, 1)], axis=1)
        # # get marginals:
        # mm = np.mean(Xm, axis=0)
        # covm = np.cov(Xm.T)
        # # stdm = np.std(Xm, axis=0)
        # mp = np.mean(Xp, axis=0)
        # covp = np.cov(Xp.T)
        # # stdp = np.std(Xp, axis=0)
        # plt.figure()
        # N = 40
        # XX, YY = np.meshgrid(np.linspace(-2, 2, N), np.linspace(-2, 2, N))
        # from scipy.stats import multivariate_normal
        # rv_m = multivariate_normal(mm, covm)
        # rv_p = multivariate_normal(mp, covp)
        # pos = np.empty(XX.shape + (2,))
        # pos[:, :, 0] = XX
        # pos[:, :, 1] = YY
        # pd_m = rv_m.pdf(pos)
        # pd_p = rv_p.pdf(pos)
        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # ax.plot_surface(XX, YY, pd_m, cmap='viridis', linewidth=0, alpha=0.3, label='No Behavior')
        # ax.plot_surface(XX, YY, pd_p, linewidth=0, alpha=0.3, label='Aggregated Behavior')
        # ax.set_xlabel('ACC feats')
        # ax.set_ylabel('EDA feats')
        # ax.set_zlabel('Probability Density')
        # # plt.title("Multivariate Normal Distribution")
        # # plt.legend()
        # plt.show()


        ## Combined feature evolution

        if args.op == 1:
            o_return_list_of_sessions = True
            dict_of_instances_arrays, dict_of_labels_arrays, id_blacklist, uid_dict, dict_of_superposition_lists \
                = data_preprocess(data_path, path_style='/', num_observation_frames=num_observation_frames,
                                  num_prediction_frames=num_prediction_frames, feat_code=feat_code,
                                  o_return_list_of_sessions=o_return_list_of_sessions, bin_size=bin_size,
                                  o_multiclass=o_multiclass, o_agg_intensity_classifier=o_agg_intensity_classifier)

        bins_before_episodes = 40
        comb_feat_evol_mean = np.zeros((4, bins_before_episodes))
        comb_feat_evol_var = np.zeros((4, bins_before_episodes))
        baseline_comb_feat_evol_mean = np.zeros((4, bins_before_episodes))
        baseline_comb_feat_evol_var = np.zeros((4, bins_before_episodes))
        n = 0
        bln = 0
        c = 0
        c2 = 0
        for k in dict_of_instances_arrays.keys():
            c2 += len(dict_of_instances_arrays[k])
            for j in range(len(dict_of_instances_arrays[k])):
                c+=1
                Xsession = dict_of_instances_arrays[k][j]
                Ysession = dict_of_labels_arrays[k][j]
                Ysession[Ysession > 0] = 1
                if np.sum(Ysession) == 0:
                    if len(Ysession) < bins_before_episodes + 10:
                        continue
                    # create baseline here.
                    # normalize data
                    Xsession = classifier_cv.normalize_data(Xsession, norm_const)
                    # get data to analyze
                    Xsorted = Xsession[5: bins_before_episodes + 5]
                    Xsorted = Xsorted[:, idx[-n_largest_coefs:]]
                    x_acc = Xsorted[:, ['ACC' in feat for feat in sfeat]].dot(sw[['ACC' in feat for feat in sfeat]])
                    x_bvp = Xsorted[:, ['BVP' in feat for feat in sfeat]].dot(sw[['BVP' in feat for feat in sfeat]])
                    x_eda = Xsorted[:, ['EDA' in feat for feat in sfeat]].dot(sw[['EDA' in feat for feat in sfeat]])

                    x_bvp_eda = Xsorted[:, [('BVP' in feat) or ('EDA' in feat) for feat in sfeat]].dot(
                        sw[[('BVP' in feat) or ('EDA' in feat) for feat in sfeat]])

                    # bl_old_mean_square = np.copy(comb_feat_evol_mean[0, :] ** 2)
                    # comb_feat_evol_mean[0, :] += x_acc + 1/(n+1) * (x_acc - comb_feat_evol_mean[0, :])
                    # comb_feat_evol_var[0, :] += old_mean_square - comb_feat_evol_mean[0, :]**2 + 1/(n+1) *(x_acc**2 - comb_feat_evol_var[0, :] - old_mean_square)

                    bln += 1
                    baseline_comb_feat_evol_mean[0, :] += x_acc
                    baseline_comb_feat_evol_var[0, :] += ((bln * x_acc - baseline_comb_feat_evol_mean[0, :]) ** 2) / (bln * (bln + 1))

                    baseline_comb_feat_evol_mean[1, :] += x_bvp
                    baseline_comb_feat_evol_var[1, :] += ((bln * x_bvp - baseline_comb_feat_evol_mean[1, :]) ** 2) / (bln * (bln + 1))

                    baseline_comb_feat_evol_mean[2, :] += x_eda
                    baseline_comb_feat_evol_var[2, :] += ((bln * x_eda - baseline_comb_feat_evol_mean[2, :]) ** 2) / (bln * (bln + 1))

                    baseline_comb_feat_evol_mean[3, :] += x_bvp_eda
                    baseline_comb_feat_evol_var[3, :] += ((bln * x_bvp_eda - baseline_comb_feat_evol_mean[3, :]) ** 2) / (bln * (bln + 1))

                    continue

                first_agg_index = np.nonzero(Ysession)[0][0]
                if first_agg_index < bins_before_episodes:
                    continue

                # normalize data
                Xsession = classifier_cv.normalize_data(Xsession, norm_const)
                # get data to analyze
                Xsorted = Xsession[first_agg_index - bins_before_episodes + 1: first_agg_index + 1]
                Xsorted = Xsorted[:, idx[-n_largest_coefs:]]
                x_acc = Xsorted[:, ['ACC' in feat for feat in sfeat]].dot(sw[['ACC' in feat for feat in sfeat]])
                x_bvp = Xsorted[:, ['BVP' in feat for feat in sfeat]].dot(sw[['BVP' in feat for feat in sfeat]])
                x_eda = Xsorted[:, ['EDA' in feat for feat in sfeat]].dot(sw[['EDA' in feat for feat in sfeat]])

                x_bvp_eda = Xsorted[:, [('BVP' in feat) or ('EDA' in feat) for feat in sfeat]].dot(
                    sw[[('BVP' in feat) or ('EDA' in feat) for feat in sfeat]])

                # old_mean_square = np.copy(comb_feat_evol_mean[0, :]**2)
                # comb_feat_evol_mean[0, :] += x_acc + 1/(n+1) * (x_acc - comb_feat_evol_mean[0, :])
                # comb_feat_evol_var[0, :] += old_mean_square - comb_feat_evol_mean[0, :]**2 + 1/(n+1) *(x_acc**2 - comb_feat_evol_var[0, :] - old_mean_square)

                n += 1
                comb_feat_evol_mean[0, :] += x_acc
                comb_feat_evol_var[0, :] += ((n*x_acc - comb_feat_evol_mean[0, :])**2)/(n*(n+1))

                comb_feat_evol_mean[1, :] += x_bvp
                comb_feat_evol_var[1, :] += ((n * x_bvp - comb_feat_evol_mean[1, :]) ** 2) / (n * (n + 1))

                comb_feat_evol_mean[2, :] += x_eda
                comb_feat_evol_var[2, :] += ((n * x_eda - comb_feat_evol_mean[2, :]) ** 2) / (n * (n + 1))

                comb_feat_evol_mean[3, :] += x_bvp_eda
                comb_feat_evol_var[3, :] += ((n * x_bvp_eda - comb_feat_evol_mean[3, :]) ** 2) / (n * (n + 1))


        fontsize = 14
        xlabel = '# of bins to episode'
        legends = ['ACC', 'BVP', 'EDA', 'BVP+EDA']
        for i in [0, 1, 2, 3]:
            m = comb_feat_evol_mean[i, :]/n
            m_bl = baseline_comb_feat_evol_mean[i,:]/bln
            s = np.sqrt(comb_feat_evol_var[i, :]/(n+1))
            plt.figure()
            xticks = -bins_before_episodes + np.linspace(1, bins_before_episodes, bins_before_episodes)
            plt.plot(xticks, m, label=legends[i])
            plt.plot(xticks, m_bl, '--', label=legends[i])
            # plt.fill_between(xticks, m - 1.98 * s, m + 1.98 * s, alpha=0.1, color='b')
            plt.fill_between(xticks, m - s, m + s, alpha=0.1, color='b')
            plt.legend(fontsize=fontsize, loc='lower left')
            # plt.xticks(fontsize=ticksize)
            # plt.yticks(fontsize=ticksize)
            plt.xlabel(xlabel, fontsize=fontsize)
            plt.show()

    else:
       pass
