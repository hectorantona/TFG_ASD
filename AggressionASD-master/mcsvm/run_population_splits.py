import featGen
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import numpy as np
from numpy import interp
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA
import argparse
from sklearn import svm
from sklearn.model_selection import train_test_split, KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
import classifier_cv
import pickle
import os
import glob
from multiclass_svm import data_preprocess, plot_rocs


def run_population_split(X_dict, y_dict, n_classes=4, test_size=0.2, o_normalize_data=False, o_perform_pca=False, n_pcs=10,
                                  clf_obj=svm.SVC, clf_par=None, output_file_name='out.data'):

    X_list = list(X_dict.values())
    y_list = list(y_dict.values())

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    mean_fpr = np.linspace(0, 1, 300)

    # create dummy variable to obtain split idxs
    temp_cv_x = np.arange(len(X_dict))

    itt_count = 0
    while True:

        train_idx, test_idx = train_test_split(temp_cv_x, test_size=test_size)

        X_train = np.concatenate([X_list[i] for i in train_idx])
        X_test = np.concatenate([X_list[i] for i in test_idx])
        y_train = np.concatenate([y_list[i] for i in train_idx])
        y_test = np.concatenate([y_list[i] for i in test_idx])

        y_train = label_binarize(y_train, classes=np.arange(n_classes).astype(float))
        y_test = label_binarize(y_test, classes=np.arange(n_classes).astype(float))

        if sum(sum(y_train) == 0) or sum(sum(y_test) == 0):
            # print("Split has train or test without one of the classes!")
            # print(sum(y_train) == 0)
            # print(sum(y_test) == 0)
            itt_count += 1
            if itt_count == 20:
                print("No split found in 20 tries!")
                return -1
        else:
            print("Split found!")
            print(train_idx, test_idx)
            break

    if o_normalize_data:
        norm_constants = classifier_cv.get_normalization_constants(X_train)
        X_train = classifier_cv.normalize_data(X_train, norm_constants).copy()
        X_test = classifier_cv.normalize_data(X_test, norm_constants).copy()

    if o_perform_pca:
        pca = PCA(n_components=n_pcs)
        pca.fit(X_train)
        X_train = pca.transform(X_train).copy()
        X_test = pca.transform(X_test).copy()

    # create and train classifiers
    # clf = OneVsRestClassifier(svm.SVC(**svm_par)).fit(X_train, y_train)
    clf = OneVsRestClassifier(clf_obj(**clf_par)).fit(X_train, y_train)

    # test classifiers
    y_score = clf.decision_function(X_test)
    # fpr, tpr, thresholds = metrics.roc_curve(binary_labels[test], probas_[:, 1])

    for i in range(n_classes):
        fpr_, tpr_, _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc_ = auc(fpr_, tpr_)
        fpr[i] = fpr_
        tpr[i] = interp(mean_fpr, fpr_, tpr_)
        # roc_auc_2 = auc(mean_fpr, tpr[i])
        roc_auc[i] = roc_auc_

    print("roc_aucs: " + str(roc_auc))
    output_file_name += "_" + str(np.random.randint(10000)) + ".bin"
    print("Saving split results in " + output_file_name)
    results = [mean_fpr, tpr, roc_auc, clf]

    with open(output_file_name, 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(results, filehandle)


def merge_pop_split_results_and_plots(dir, num_observation_frames=12, num_prediction_frames=4, n_classes=4):
    files = os.listdir(dir)
    print(files)
    all_tpr = dict()
    all_auc = dict()
    for i in range(n_classes):
        all_tpr[i] = []
        all_auc[i] = []
    all_tpr["macro"] = []
    all_auc["macro"] = []

    # for file in files:
    for file in glob.glob(dir + '*.bin'):
        # file_with_path = dir + file
        with open(file, 'rb') as filehandle:
            results = pickle.load(filehandle)
            print(results)
            [fpr, tpr_dict, roc_auc_dict, clf] = results
        temp_tpr = np.zeros_like(fpr)
        temp_auc = 0
        n_classes = len(tpr_dict)
        for i in range(n_classes):
            all_tpr[i].append(tpr_dict[i])
            all_auc[i].append(roc_auc_dict[i])
            temp_tpr += tpr_dict[i]
            temp_auc += roc_auc_dict[i]
        all_tpr["macro"].append(temp_tpr/n_classes)
        all_auc["macro"].append(temp_auc/n_classes)
    plot_rocs(fpr, all_tpr, all_auc, num_observation_frames=num_observation_frames,
              num_prediction_frames=num_prediction_frames, outputdir=dir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--outputdir", type=str, default="Results",
                        help="Output directory (default: ./Results/)")
    parser.add_argument("-op", type=bool, default=False, help="Only plot figures without running sym.")
    parser.add_argument("-g", "--gamma", type=float, default=0.1, help="Inverse of kernel bandwidth")
    parser.add_argument("-C", type=float, default=100, help="SVM regularization parameter")
    parser.add_argument("-k", "--kernel", type=str, default="rbf", help="Kernel name (e.g., rbf, poly, etc)")
    parser.add_argument("-tp", type=int, default=12, help="number of bins in the past (default is 12 = 3 min)")
    parser.add_argument("-tf", type=int, default=4, help="Number of bins in the future (default is 4 = 1 min)")
    parser.add_argument("-pca", type=bool, default=True, help="Perform PCA (default is True)")
    parser.add_argument("-n", "--normalize", type=bool, default=True,
                        help="Perform zero-mean one-std normalization (default is True)")
    parser.add_argument("-ts", "--test_size", type=bool, default=0.2,
                        help="test size proportion between 0 and 1 (default is 0.2)")
    # # parser.add_argument("-m", "--model", type=int, default=0,
    # #                     help="model type (default is 0: population model. Use 1 for individual models, 2 for "
    # #                          "leave-one-subject-out population models, "
    # #                          "and 3 for individual models with leave-one-session out)")
    # parser.add_argument("-s", "--min_sessions", type=int, default=2,
    #                     help="minimum number of sessions per subject (default is 2, only needed with -m 3)")
    parser.add_argument("-n_pcs", type=int, default=10, help="Number of Principal Components (default is 10)")
    parser.add_argument("-cv_reps", type=int, default=10, help="Number of data splits and runs(default is 10)")

    args = parser.parse_args()

    o_is_new_dataset = True
    # data_path = '/scratch/talesim/new_CBS_data'
    # data_path = '/scratch/talesim/new_CBS_data_small'
    # data_path = '/scratch/talesim/new_CBS_data_full'
    # data_path = '/scratch/talesim/tes/'
    # data_path = '/home/tales/DataBases/new_CBS_data_small'
    # data_path = '/home/tales/DataBases/test_t4'
    # data_path = '/home/tales/DataBases/new_CBS_data'
    data_path = '/home/tales/DataBases/new_CBS_data_full'
    # dataPath = '/home/tales/DataBases/new_data_t1'
    # data_path = '/home/tales/DataBases/newT3'
    # data_path = '/home/tales/DataBases/Tranche 5'
    # data_path = '/home/tales/DataBases/Test_T3'
    # data_path = '/home/tales/DataBases/t3_small'
    path_style = '/'

    """
    About model_options:
    POP: Population models
    IND: Individual models
    LOO: Population models with leave one individual out cv
    LOSO: Individual models with leave one session out cv
    """
    model_options = {'POP': 0, 'IND': 1, 'LOO': 2, 'LOSO': 3}
    # o_return_list_of_sessions should be True only if LOSO is selected.
    o_return_list_of_sessions = False

    # subjectIDCoding = 4  # Number of digits in subject ID coding
    # num_observation_frames = 12
    # num_prediction_frames = 4
    num_observation_frames = args.tp
    num_prediction_frames = args.tf

    # feat_code = 7                       # use all features
    feat_code = 6  # use all features but 'AGGObserved' and 'TimePastAggression'
    # feat_code = 1  # use only ACC data

    o_normalize_data = args.normalize  # normalize data?
    o_perform_pca = args.pca  # perform PCA?
    n_pcs = args.n_pcs  # number of principal components

    svm_par = {'gamma': args.gamma, 'C': args.C, 'kernel': args.kernel, 'probability': True}

    outputdir = args.outputdir + '/PopSplits'
    if not os.path.isdir(args.outputdir):
        os.makedirs(args.outputdir)

    if not args.op:
        dict_of_instances_arrays, dict_of_labels_arrays, id_blacklist, uid_dict \
            = data_preprocess(data_path, path_style='/', num_observation_frames=num_observation_frames,
                              num_prediction_frames=num_prediction_frames, feat_code=feat_code,
                              o_return_list_of_sessions=o_return_list_of_sessions)

        # loo populations models
        data_file_name = outputdir + '/sim_tp_{}_tf_{}_svm_PSS_model'.format(args.tp, args.tf)
        run_population_split(dict_of_instances_arrays, dict_of_labels_arrays, test_size=args.test_size,
                                          o_normalize_data=o_normalize_data,
                                          o_perform_pca=o_perform_pca,
                                          n_pcs=n_pcs, clf_obj=svm.SVC, clf_par=svm_par,
                                          output_file_name=data_file_name)

    else:
        merge_pop_split_results_and_plots(outputdir, num_observation_frames=num_observation_frames,
                                          num_prediction_frames=num_prediction_frames)