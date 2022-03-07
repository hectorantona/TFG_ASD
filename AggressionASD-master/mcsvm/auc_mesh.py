from matplotlib import pyplot as plt
import pickle
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib


def plot_auc_mesh(mean_fpr, tpr, roc_auc, num_observation_frames, num_prediction_frames, labels_fontsize=16, show=False):
    # Plot all ROC curves
    class_dict = {0: "No", 1: "ED", 2: "SIB", 3: "Agg", 'macro': 'Average'}
    n_classes = 4


if __name__ == '__main__':

    params = {
        'text.latex.preamble': ['\\usepackage{gensymb}'],
        'image.origin': 'lower',
        'image.interpolation': 'nearest',
        'image.cmap': 'coolwarm',
        'axes.grid': False,
        'savefig.dpi': 150,  # to adjust notebook inline plot size
        'axes.labelsize': 8,  # fontsize for x and y labels (was 10)
        'axes.titlesize': 8,
        'font.size': 8,  # was 10
        'legend.fontsize': 6,  # was 10
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'text.usetex': True,
        'figure.figsize': [7, 5],
        'font.family': 'serif',
    }
    matplotlib.rcParams.update(params)


    num_observation_frames = np.array([4, 8, 12])
    num_prediction_frames = np.array([4, 8, 12])
    tp_in_min = np.array([1, 2, 3])
    tf_in_min = np.array([1, 2, 3])
    bin_time_length = 15

    labels_fontsize = 12

    class_dict = {0: "No", 1: "ED AUC", 2: "SIB AUC", 3: "Agg AUC", 'macro': 'Average AUC'}
    n_classes = 4

    plots = [1, 2, 3, 'macro']
    count = 1
    # fig = plt.figure(figsize=(8, 6))

    fig = plt.figure()
    for p in plots:
        roc_aucs = []
        for npf in num_prediction_frames:
            for nof in num_observation_frames:
                data_file_name = './Results/sim_tp_{}_tf_{}_svm.data'.format(nof, npf)
                with open(data_file_name, 'rb') as filehandle:
                    # read the data as binary data stream
                    [mean_fpr, tpr, roc_auc, _, _] = pickle.load(filehandle)

                roc_aucs.append(np.mean(roc_auc[p]))

        roc_aucs = np.array(roc_aucs)
        _xx, _yy = np.meshgrid(tp_in_min, tf_in_min)
        x, y = _xx.ravel(), _yy.ravel()
        bottom = np.zeros_like(roc_aucs)
        width = depth = 1

        # setup the figure and axes

        ax1 = fig.add_subplot(2, 2, count, projection='3d')
        # ax1.bar3d(x, y, bottom, width, depth, roc_aucs, shade=True)
        # surf = Axes3D.plot_surface(ax1, x.reshape(3, 3), y.reshape(3, 3), roc_aucs.reshape(3, 3), cmap=cm.coolwarm,
        #                    linewidth=0, antialiased=False)
        surf = Axes3D.plot_surface(ax1, _xx, _yy, roc_aucs.reshape(3, 3),  cmap=cm.coolwarm,
                                   linewidth=0, antialiased=False, vmin=0.7, vmax=1)
        # ax1.set_title('Shaded')
        # ax1.text2D(0.05, 0.95, "2D Text", transform=ax1.transAxes)
        # ax1.view_init(elev=5., azim=-30)
        ax1.view_init(elev=15., azim=-120)
        # ax1.xaxis.set_major_locator(plt.MaxNLocator(4))
        # ax1.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax1.set_zlim(0.70, 1)

        # if count == 1:
        #     ax1.xaxis.set_major_locator(plt.MultipleLocator(1))
        #     ax1.set_xlabel(r'$\tau_p$ (minutes)', fontsize=labels_fontsize)
        # elif count == 2:
        #     ax1.zaxis.set_major_locator(plt.MultipleLocator(0.05))
        #     ax1.set_zlabel(r'AUC', fontsize=labels_fontsize)
        # elif count == 3:
        #     ax1.yaxis.set_major_locator(plt.MultipleLocator(1))
        #     ax1.set_ylabel(r'$\tau_f$ (minutes)', fontsize=labels_fontsize)
        # elif count == 4:
        #     ax1.yaxis.set_major_locator(plt.MultipleLocator(1))
        #     ax1.zaxis.set_major_locator(plt.MultipleLocator(0.05))
        #     ax1.set_xlabel(r'$\tau_p$ (minutes)', fontsize=labels_fontsize)
        #     ax1.set_zlabel(r'AUC', fontsize=labels_fontsize)

        # ax1.zaxis.set_major_locator(plt.MultipleLocator(0.1))
        ax1.zaxis.set_major_formatter(plt.NullFormatter())
        ax1.xaxis.set_major_locator(plt.MultipleLocator(1))
        ax1.yaxis.set_major_locator(plt.MultipleLocator(1))

        # ax1.set_xlabel(r'$\tau_p$ (min)', fontsize=labels_fontsize)
        # ax1.set_ylabel(r'$\tau_f$ (min)', fontsize=labels_fontsize)
        ax1.set_xlabel(r'$\tau_p$', fontsize=labels_fontsize)
        ax1.set_ylabel(r'$\tau_f$', fontsize=labels_fontsize)

        # fig.colorbar(surf, shrink=0.5, aspect=5)

        # if count == 2 or count == 4:
        #     fig.colorbar(surf, shrink=0.5, aspect=5)
            # ax1.set_zlabel(r'AUC', fontsize=labels_fontsize)
            # ax1.colorbar()
        ax1.set_title(r''+class_dict[p])
        count = count + 1
        fig.subplots_adjust(right=0.9)
    # plt.tight_layout()
    # fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(surf, cax=cbar_ax, fraction=0.1)
    # fig.colorbar(surf, fraction=0.2)
    plt.savefig('./Results/AUC_mesh.pdf')
    plt.savefig('./Results/AUC_mesh.png')
    plt.show()