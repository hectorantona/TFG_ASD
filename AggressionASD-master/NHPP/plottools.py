import matplotlib.pyplot as plt
import numpy as np


def plot_roc_with_std(mean_fpr, mean_tpr, tprs, colors, color_count, fig_num=None):
    if fig_num:
        plt.figure(fig_num)

    plt.plot(mean_fpr, mean_tpr, color=colors[color_count, :], linestyle='--', linewidth=1.5)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=colors[color_count, :], alpha=.1)
    color_count += 1

    return color_count


def roc_fig_labels_and_style(labels_font_size=16, fig_num=None):
    if fig_num:
        plt.figure(fig_num)
    plt.grid(color=[.9, .9, .9], linestyle='--')
    plt.xlim([-0.005, 1.005])
    plt.ylim([-0.005, 1.005])
    plt.xlabel('False Positive Rate (1-Specificity)', fontweight='bold', fontsize=labels_font_size)
    plt.ylabel('True Positive Rate (Sensitivity)', fontweight='bold', fontsize=labels_font_size)
    # plt.xlabel('False Positive Rate', fontweight='bold')
    # plt.ylabel('True Positive Rate', fontweight='bold')
    # plt.title('Receiver operating characteristic example')
    plt.tick_params(labelsize=14)
    # plt.legend(loc="lower right")
    plt.show()


def plot_arrival_and_prediction_results(y, p, expected_number_of_arrivals, labels_font_size=16, fig_num=None):
    # plot results
    if fig_num:
        plt.figure(fig_num)
    else:
        plt.figure()

    plt.rc('text', usetex=True)
    # plt.rc('text.latex', preamble=r'\usepackage{amsmath},\usepackage{amssymb}')
    plt.rc('font', family='serif')

    plt.subplot(311)
    plt.plot(y, label=r'$N_k$')
    plt.legend()
    plt.grid(linestyle='-', color=(0.9, 0.9, 0.9))

    plt.subplot(312)
    plt.plot(expected_number_of_arrivals, label=r'$\Lambda_k = \mathbb{E}(N_k)$')
    plt.legend()
    plt.grid(linestyle='-', color=(0.9, 0.9, 0.9))

    plt.subplot(313)
    plt.plot(p, label=r'$P(N_k > 0)$')
    plt.legend()
    plt.ylim(-0.1, 1.1)
    plt.grid(linestyle='-', color=(0.9, 0.9, 0.9))

    plt.xlabel(r'\textbf{Frames}')
    # plt.ylabel(r'\textit{voltage} (mV)', fontsize=labels_font_size)


def plot_prediction_and_feat_norm_results(y, feat_norm, expected_number_of_arrivals, labels_font_size=16, fig_num=None):
    # plot results
    if fig_num:
        plt.figure(fig_num)
    else:
        plt.figure()

    plt.rc('text', usetex=True)
    # plt.rc('text.latex', preamble=r'\usepackage{amsmath},\usepackage{amssymb}')
    plt.rc('font', family='serif')

    plt.subplot(311)
    plt.plot(y, label=r'$N_k$')
    plt.legend()
    plt.grid(linestyle='-', color=(0.9, 0.9, 0.9))

    plt.subplot(312)
    plt.plot(expected_number_of_arrivals, label=r'$\Lambda_k = \mathbb{E}(N_k)$')
    plt.legend()
    plt.grid(linestyle='-', color=(0.9, 0.9, 0.9))

    plt.subplot(313)
    # plt.plot(p, label=r'$P(N_k > 0)$')
    plt.plot(feat_norm, label=r'$\|\mathbf{x}\|$')
    plt.legend()
    plt.grid(linestyle='-', color=(0.9, 0.9, 0.9))

    plt.xlabel(r'\textbf{Frames}')
    # plt.ylabel(r'\textit{voltage} (mV)', fontsize=labels_font_size)
