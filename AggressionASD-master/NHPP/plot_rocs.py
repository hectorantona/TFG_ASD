
from matplotlib import pyplot as plt
import pickle
import numpy as np

labels_fontsize = 16

# file_name = 'results_svr_2.data'
file_name = 'results_msnhpp_svr_repX.data'
with open(file_name, 'rb') as filehandle:
    # read the data as binary data stream
    results_dict = pickle.load(filehandle)

prediction_duration = [4, 5, 6, 7, 8]
colorstr = '-k'
mean_tpr = []
mean_auc = []
std_auc = []
for tf_idx in range(len(prediction_duration)):
    mean_tpr.append([])
    mean_auc.append([])
    std_auc.append([])


for tf_idx in range(len(prediction_duration)):
    for key in results_dict:
        [mean_fpr, mean_tpr_k, mean_auc_k, std_auc_k, tprs_k] = results_dict[key]
        mean_tpr[tf_idx].append(mean_tpr_k[tf_idx])
        mean_auc[tf_idx].append(mean_auc_k[tf_idx])
        std_auc[tf_idx].append(std_auc_k[tf_idx])

plt.figure()
for tf_idx in range(len(prediction_duration)):
    mm_tpr = np.mean(mean_tpr[tf_idx], axis=0)
    plt.plot(mean_fpr, mm_tpr, label='Tf = ' + str(15*int(prediction_duration[tf_idx])) + ', AUC =' + \
                                     '{: {prec}}'.format(np.mean(mean_auc[tf_idx]), prec='.2') + ' +- ' +
                                                                                   '{: {prec}}'.format(np.mean(std_auc[tf_idx]), prec='.2'))
    std_tpr = np.std(mean_tpr[tf_idx], axis=0)
    tprs_upper = np.minimum(mm_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mm_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr   , tprs_lower, tprs_upper, alpha=.1)

plt.grid(color=[.9, .9, .9], linestyle='--')
plt.legend()
plt.xlabel('False Positive Rate (1-Specificity)', fontweight='bold', fontsize=labels_fontsize)
plt.ylabel('True Positive Rate (Sensitivity)', fontweight='bold', fontsize=labels_fontsize)
# plt.xlabel('False Positive Rate', fontweight='bold')
# plt.ylabel('True Positive Rate', fontweight='bold')
# plt.title('Receiver operating characteristic example')
plt.tick_params(labelsize=14)
# plt.legend(loc="lower right")
plt.show()



plt.figure()
for tf_idx in range(len(prediction_duration)):
    mean_tpr = []
    mean_auc = []
    std_auc = []
    file_name = 'results_nhpp_svr' + '_tf_' + str(prediction_duration[tf_idx]) + '.data'
    with open(file_name, 'rb') as filehandle:
        # read the data as binary data stream
        results_dict = pickle.load(filehandle)

    for key in results_dict:
        m_fpr, m_tpr, m_auc, s_auc, tprs = results_dict[key]
        mean_tpr.append(m_tpr)
        mean_auc.append(m_auc)
        std_auc.append(s_auc)

    mm_tpr = np.mean(mean_tpr, axis=0)
    plt.plot(m_fpr, mm_tpr, label='Tf = ' + str(15*int(prediction_duration[tf_idx])) + ', AUC =' + \
                                    '{: {prec}}'.format(np.mean(mean_auc), prec='.2') + ' +- ' +
                                                                                  '{: {prec}}'.format(np.mean(std_auc), prec='.2'))
    std_tpr = np.std(mean_tpr, axis=0)
    tprs_upper = np.minimum(mm_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mm_tpr - std_tpr, 0)
    plt.fill_between(m_fpr, tprs_lower, tprs_upper, alpha=.1)

plt.grid(color=[.9, .9, .9], linestyle='--')
plt.legend()
plt.xlabel('False Positive Rate (1-Specificity)', fontweight='bold', fontsize=labels_fontsize)
plt.ylabel('True Positive Rate (Sensitivity)', fontweight='bold', fontsize=labels_fontsize)
# plt.xlabel('False Positive Rate', fontweight='bold')
# plt.ylabel('True Positive Rate', fontweight='bold')
# plt.title('Receiver operating characteristic example')
plt.tick_params(labelsize=14)
# plt.legend(loc="lower right")
plt.show()


# for key in results_dict:
#     [mean_fpr, mean_tpr_k, mean_auc_k, std_auc_k, tprs_k] = results_dict[key]
#     for tf_idx in range(len(mean_tpr_k)):
#         mean_tpr[tf_idx].append(mean_tpr_k[tf_idx])
#         mean_auc[tf_idx].append(mean_auc_k[])
#         std_auc[tf_idx].append(std_auc_k)

# for tf_idx in range(len(mean_tpr_k)):
#     plt.plot(mean_tpr, mean(mean_tpr[tf_idx]))
#
#     for i in range(len(mean_tpr)):
#         if i == 0:
#             colorstr = '-k'
#         else:
#             colorstr = '--r'
#
#         plt.plot(mean_fpr, mean_tpr[i], colorstr, label="pred time = " + str(15*int(prediction_duration[i])) + ", key=" + key)
# plt.legend()
# plt.show()