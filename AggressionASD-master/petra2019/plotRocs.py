
from matplotlib import pyplot as plt
import csv
import pickle
import numpy as np


def plot_rocs_from_data_file(data_file, labels_fontsize=16):
    # plot colors
    np.random.seed(0)
    colors = np.random.uniform(0.3, 0.9, (22, 3))
    color_count = 0

    # load data from binary data:
    with open(data_file, 'rb') as filehandle:
        # read the data as binary data stream
        [results_list, [mean_fpr_pop, mean_tpr_pop, mean_auc_pop, std_auc_pop]] = pickle.load(filehandle)


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


    print('E[AUC_sbj]', np.mean(average_auc_all_subjects))
    print('STD[AUC_sbj]', np.mean(std_auc_all_subjects))

    plt.plot(mean_fpr_pop, mean_tpr_pop, color='b', linestyle='-', linewidth=1.5)
    print('E[AUC_cv]_global', mean_auc_pop)
    print('STD[AUC_global]', std_auc_pop)

    plt.show()

if __name__ == '__main__':
    labels_fontsize = 16
    classifier_type = 'LR'
    file_name = './Results/results' + classifier_type + '.data'
    plot_rocs_from_data_file(file_name, labels_fontsize)


    classifier_type = 'SVM'
    file_name = './Results/results' + classifier_type + '.data'
    plot_rocs_from_data_file(file_name, labels_fontsize)