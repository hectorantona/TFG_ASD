import numpy as np
from cvxopt import solvers, matrix, spdiag, log, mul
from cvxopt import exp as cvx_exp
# from cvxopt import div as cvx_div
from math import factorial, exp
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn import metrics
from scipy import interp
import math
from sklearn.svm import SVR
from plottools import plot_arrival_and_prediction_results, plot_prediction_and_feat_norm_results
import featureextraction as fext
from sklearn.decomposition import PCA
from scipy import sparse
from sklearn import preprocessing
import matplotlib.pyplot as plt


class NonHomogeneousPoissonProcess:

    def __init__(self, X, y, prediction_duration, estimation_method='LS', model='linear', model_param=0.0):
        """
        :param X: (n x d) data (numpy) array
        :param y: (n,) arrivals count (numpy) array
        :param prediction_duration: the duration of the counting process window T/N
        :param estimation_method: 'LS' for Least-Squares, 'WLS' for Weighted LS, 'MLE' for Maximum Likelihood Estimation
        :param model: 'linear' for linear rate functions,...
        :param model_param: Model parameters. For MLE is a regularization parameter, for SVR is a dictionary with the SVR
         parameters.
        """
        self.n, self.d = X.shape
        self.y = y
        self.estimation_method = estimation_method
        self.model = model

        if self.model is 'linear' or 'explin':
            self.X = np.hstack((X, np.ones((self.n, 1))))
            self.d = self.d + 1
        else:
            self.X = X

        self.prediction_duration = prediction_duration
        self.w_opt = []
        self.model_param = model_param

    def fit_model_over_multiple_scales(self, n_scales, ar_ord):
        pass

    def fit_model(self):
        if self.model == 'linear':
            if self.estimation_method == 'LS':
                w = np.linalg.lstsq(self.X, self.y, rcond=None)[0]

                # transform w into rate parameters
                # convert w = [beta^T, alpha]^T to w = T_f * [beta^T, alpha]^T = [b^T, a]^T
                # self.w_opt = w/self.prediction_duration
                self.w_opt = w
                # print(self.X.dot(w))

            elif self.estimation_method == 'WLS':
                # implement weighted LS
                max_iteration = 20
                min_norm_diff = 1e-10

                # xi is weight vector of the WLS:
                self.xi = np.ones(self.y.shape)

                XX = np.zeros(self.X.shape)

                xi_sqr_diff_norm = math.inf
                for count in range(max_iteration):
                    if xi_sqr_diff_norm < min_norm_diff:
                        break
                    XX = np.dot(np.sqrt(np.diag(self.xi)), self.X)
                    # for i in range(self.n):
                    #     XX[i, :] = np.sqrt(self.xi[i]) * self.X[i, :]
                    yy = np.sqrt(self.xi)*self.y

                    # update coefficients
                    self.w_opt = np.linalg.lstsq(XX, yy, rcond=None)[0]

                    # update weights
                    xi_old = self.xi
                    # residual_coeffs = np.exp(np.linalg.lstsq(self.X, np.log((self.y - np.dot(self.X, self.w_opt))**2), rcond=None)[0])
                    # self.xi = 1/np.maximum(np.sqrt(np.exp(np.dot(self.X, residual_coeffs))), 1e-3)

                    lambdas = np.maximum(np.dot(self.X, self.w_opt), 1e-3)
                    # # lambdas = np.array([max(np.dot(self.w_opt, self.X[i, :]), 1e-3) for i in range(self.n)])
                    num = self.n / lambdas
                    sum_inv_lambdas = sum(1/lambdas)
                    self.xi = num/sum_inv_lambdas
                    # # self.xi = np.maximum(1/np.diag(np.outer(self.y - lambdas, self.y - lambdas)), 3)
                    # self.xi = np.maximum(1 / ((self.y - np.dot(self.X, self.w_opt))**2), 5)
                    xi_sqr_diff_norm = np.linalg.norm((xi_old - self.xi)**2)
                    print('Iteration ' + str(count) + ': normdiff = ' + str(xi_sqr_diff_norm))

                pass
            elif self.estimation_method == 'MLE':
                # MLE for linear models.
                pass
        elif self.model == 'explin':
            # MLE is the only option
            print('Solving by Maximum Likelihood Estimation (MLE)')
            sol_dict = solvers.cp(self.cvx_format_explin_cost)
            self.w_opt = np.array(sol_dict['x']).reshape((self.d,))

        elif self.model == 'svr':
            # MLE is the only option
            print('Training SVM')
            # clf = SVR(gamma=0.5, C=1000.0, kernel='rbf')
            svr = SVR(**self.model_param)
            svr.fit(self.X, self.y.ravel())
            # print(clf)
            self.w_opt = svr

        elif self.model == 'dnn':
            # implement a deep neural network
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers

            def build_model():
                model = keras.Sequential([
                    # layers.Dense(200, activation=tf.nn.sigmoid, input_shape=[self.X.shape[1]]),
                    # layers.Dense(200, activation=tf.nn.sigmoid, input_shape=[self.X.shape[1]]),
                    # layers.Dense(200, activation=tf.nn.relu, input_shape=[self.X.shape[1]]),
                    layers.Dense(20, activation=tf.nn.relu, input_shape=[self.X.shape[1]]),
                    layers.Dense(20, activation=tf.nn.relu),
                    # layers.Dense(64, activation=tf.nn.relu, input_shape=[self.X.shape[1]]),
                    # layers.Dense(128, activation=tf.nn.relu),
                    layers.Dense(1)
                ])

            # def build_model():
            #     model = keras.Sequential([
            #         layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
            #         layers.Dense(64, activation=tf.nn.relu),
            #         layers.Dense(1)
            #     ])

                # optimizer = tf.keras.optimizers.RMSprop(0.001)
                optimizer = tf.keras.optimizers.Adam()
                # optimizer = tf.keras.optimizers.Adadelta()

                model.compile(loss='mean_squared_error',
                              optimizer=optimizer,
                              metrics=['mean_absolute_error', 'mean_squared_error'])
                return model

            # model = build_model()
            #
            # model.summary()
            EPOCHS = 1000

            # history = model.fit(self.X, self.y, epochs=EPOCHS, validation_split=0.2, verbose=0, callbacks=[PrintDot()])
            # history = model.fit(self.X, self.y, epochs=EPOCHS, validation_split=0.2, verbose=0)

            import pandas as pd
            # hist = pd.DataFrame(history.history)
            # hist['epoch'] = history.epoch
            # hist.tail()

            import matplotlib.pyplot as plt

            def plot_history(history):
              hist = pd.DataFrame(history.history)
              hist['epoch'] = history.epoch

              plt.figure()
              plt.xlabel('Epoch')
              plt.ylabel('Mean Abs Error [MPG]')
              plt.plot(hist['epoch'], hist['mean_absolute_error'],
                       label='Train Error')
              plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
                       label='Val Error')
              plt.ylim([0, 5])
              plt.legend()

              plt.figure()
              plt.xlabel('Epoch')
              plt.ylabel('Mean Square Error [$MPG^2$]')
              plt.plot(hist['epoch'], hist['mean_squared_error'],
                       label='Train Error')
              plt.plot(hist['epoch'], hist['val_mean_squared_error'],
                       label='Val Error')
              plt.ylim([0, 20])
              plt.legend()
              plt.show()

            # plot_history(history)

            model = build_model()
            model.summary()

            # The patience parameter is the amount of epochs to check for improvement
            early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)

            history = model.fit(self.X, self.y, epochs=EPOCHS, validation_split=0.2, verbose=0, callbacks=[early_stop])
            self.w_opt = model
            plot_history(history)

        else:
            # should implement other nonlinear-models
            print('Selected model not implemented!')
            pass

    def cvx_format_explin_cost(self, w=None, z=None):
        if w is None:
            return 0, matrix(0.0, (self.d, 1))

        y = matrix(self.y)
        X = matrix(self.X)
        Xw = X*w
        exp_Xw = cvx_exp(Xw)
        # print(Xw)
        one_vec_n = matrix(1.0, (self.n, 1))
        # f = -y.T * Xw + exp_Xw.T * one_vec_n
        # f = self.reg_const * (-y.T * Xw + exp_Xw.T*one_vec_n) + 0.5*(w.T*w)
        f = (-y.T * Xw + exp_Xw.T * one_vec_n) + 0.5 * self.model_param * (w.T * w)
        # print(cvx_exp(Xw).T)

        # compute gradient:  -sum_k y_k x_k + sum_k x_k exp(Xw_k)/Xw_k = sum_k gamma_k x_k
        # gamma_k = exp(Xw_k) - y_k
        gamma_vec = exp_Xw - y

        # Df = sum_k gamma_k x_k = (diag(gamma_vec) * X)^T * 1_n
        # Df = (spdiag(gamma_vec) * X).T * one_vec_n
        Df = (spdiag(gamma_vec) * X).T * one_vec_n + self.model_param * w
        # Df = self.reg_const * (spdiag(gamma_vec) * X).T * one_vec_n + w
        # Df = matrix(0.0, (self.d, 1))
        # for k in range(self.n):
        #     Df += gamma_vec[k] * X[k, :].T

        # cvx wants the transpose of the gradient
        Df = Df.T
        if z is None:
            return f, Df

        # Hessian:
        H = matrix(0.0, (self.d, self.d))
        # Compute Hessian  => H = sum_k exp_XW_k * (x_k * x_k^T)
        for k in range(self.n):
            H += exp_Xw[k] * (X[k, :].T * X[k, :])

        # H = z[0] * H
        H = z[0] * H + self.model_param * spdiag(matrix(1.0, (self.d, 1)))
        # H = self.reg_const*z[0]*H + spdiag(matrix(1.0, (self.d, 1)))
        return f, Df, H

    def compute_intensity(self, x_bar):
        if self.model == 'linear':
            # return x_bar.dot(self.w_opt)
            # hard clipping to avoid negative intensity measures (avoid negative probabilities)
            return max(x_bar.dot(self.w_opt), 0)

        elif self.model == 'explin':
            return cvx_exp(x_bar.dot(self.w_opt))

        elif self.model == 'svr':
            x = x_bar.reshape(-1, 1)
            x = x.transpose()
            return max(self.w_opt.predict(x), 0)

        elif self.model == 'dnn':
            x = x_bar.reshape(-1, 1)
            x = x.transpose()
            # test_predictions = self.w_opt.predict(x).flatten()
            # return max(self.w_opt.predict(x).flatten(), 0)
            return max(self.w_opt.predict(x), 0)
        else:
            pass

    def compute_arrival_probability(self, X, num_of_arrivals=-1):
        """
        :param X: (n x d) data (numpy) array
        :param num_of_arrivals: number of arrivals (or aggression episodes) if num_of_arrivals=-1 (default) then
               1 - P(Y = 0) is computed.
        :return: P(Y = num_of_arrivals) (or 1 - P(Y = 0))
        """

        dim = X.ndim
        X1 = np.copy(X)

        if dim == 1:
            n = 1
            X1 = np.hstack((X1, np.ones(1))).reshape(1, -1)
        else:
            n, d = X.shape
            if d == 1:
                temp = d
                d=n
                n=temp
                X1 = X1.T
            X1 = np.hstack((X1, np.ones((n, 1))))
            #X1 = np.vstack((X1, np.ones((1, n))))

        # define the average x_bar vector (if n=1, then x_bar = X1)
        intensity_measure = self.compute_intensity(X1[0])
        if n > 1:
            # prediction for more than one time step
            for i in range(1, n):
                intensity_measure += self.compute_intensity(X1[i])
            # x_bar = np.mean(X1, axis=0)

        # compute the intensity measure: Lambda(t_1, t_2) = w^T * x_bar

        # intensity_measure = x_bar.dot(self.w_opt)
        # intensity_measure = self.compute_intensity(x_bar)
        # print(intensity_measure)

        # compute probabilities
        if num_of_arrivals == -1:
            temp_num_of_arrivals = 0
            probability = 1 - ((intensity_measure ** temp_num_of_arrivals) / factorial(temp_num_of_arrivals)) * exp(
                -intensity_measure)
        else:
            probability = ((intensity_measure**num_of_arrivals)/factorial(num_of_arrivals)) * exp(-intensity_measure)

        return probability, intensity_measure

    def perform_cross_validation(self, X, y, cv_folds, cv_reps):
        pass

    def __str__(self):
        #TODO: make a meaningful implementation of this function.
        return "NHPP: model = %s, estimation method is %s" % (self.model, self.estimation_method)


def agg_pred_leave_one_subj_out(X_train, y_train, X_test, y_test, prediction_duration, estimation_method,
                                model, model_param=None, o_plot_results=False):

    nhpp = NonHomogeneousPoissonProcess(X_train, y_train, prediction_duration, estimation_method, model,
                                        model_param)
    nhpp.fit_model()
    n_test = y_test.shape[0]
    p = np.zeros((n_test, 1))
    expected_agg_episodes = np.zeros((n_test, 1))

    fpr_points = np.linspace(0, 1, 300)
    for j in range(n_test):
        # p0, intensity_measure = nhpp.compute_arrival_probability(X_test[j, :], 0)
        # p[j] = 1-p0
        p[j], expected_agg_episodes[j] = nhpp.compute_arrival_probability(X_test[j, :], -1)

    if o_plot_results:
        plot_arrival_and_prediction_results(y_test, p, expected_agg_episodes)

    lab = np.copy(y_test)
    lab[lab > 0] = 1

    fpr, tpr, thresholds = metrics.roc_curve(lab, p)
    # plt.figure()
    # plt.plot(fpr, tpr)

    tpr_points = interp(fpr_points, fpr, tpr)
    # tpr_points[-1][0] = 0.0
    roc_auc = metrics.auc(fpr, tpr)
    if math.isnan(roc_auc):
        print('!!!!!!NAN!!!!!!!!', roc_auc)

    return fpr_points, tpr_points, roc_auc


def agg_pred_cv(X, y, cv_folds, cv_reps, random_state, prediction_duration, estimation_method, model, model_param=None,
                o_perform_normalization=False, o_perform_PCA=False, n_pcs=10, o_whiten=True):

    tprs = []
    aucs = []
    # coefs = []
    mean_fpr = np.linspace(0, 1, 300)
    # labels = labels_array.ravel()
    lab = np.copy(y)
    print(y.shape, X.shape, lab.shape)
    lab[lab > 0] = 1
    print(y.shape, X.shape, lab.shape)
    for i in range(cv_reps):
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        # skf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        # skf.get_n_splits(X, y)

        for train_index, test_index in skf.split(X, lab):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            if o_perform_normalization:
                print('Scaling data')
                norm_const = fext.get_normalization_constants(X_train, axis=0)
                # norm_const = fext.get_normalization_constants(np.concatenate((X_train, X_test)), axis=0)
                X_train = fext.normalize_data(X_train, norm_const)
                X_test = fext.normalize_data(X_test, norm_const)

            if o_perform_PCA:
                print('Performing PCA')
                pca = PCA(n_components=n_pcs, whiten=o_whiten)
                pca = pca.fit(X_train)
                # pca = pca.fit(np.concatenate((X_train, X_test)))
                X_train = pca.transform(X_train)
                X_test = pca.transform(X_test)

            nhpp = NonHomogeneousPoissonProcess(X_train, y_train, prediction_duration, estimation_method, model,
                                                model_param)
            nhpp.fit_model()
            n_test = y_test.shape[0]
            p = np.zeros((n_test, 1))
            expected_agg_episodes = np.zeros((n_test, 1))

            for j in range(n_test):
                # p0, intensity_measure = nhpp.compute_arrival_probability(X_test[j, :], 0)
                # p[j] = 1-p0
                p[j], intensity_measure = nhpp.compute_arrival_probability(X_test[j, :], -1)
                expected_agg_episodes[j] = intensity_measure

            lab2 = np.copy(y_test)
            lab2[lab2 > 0] = 1

            fpr, tpr, thresholds = metrics.roc_curve(lab2, p)
            # plt.figure()
            # plt.plot(fpr, tpr)

            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = metrics.auc(fpr, tpr)
            if math.isnan(roc_auc):
                print('!!!!!!NAN!!!!!!!!', roc_auc)

            # cv_roc_aucs.append(roc_auc)
            aucs.append(roc_auc)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    # return mean_fpr, mean_tpr, mean_auc, std_auc, tprs
    return [mean_fpr, mean_tpr, mean_auc, std_auc, tprs]


def agg_pred_data_split(X, y, random_state, prediction_duration, estimation_method, model, model_param=None,
                o_perform_normalization=False, o_perform_PCA=False, n_pcs=10, o_whiten=True, data_split=0.5, fig_num=None):

    n, d = X.shape
    n_train = math.floor(n/2)
    n_test = n - n_train
    X_train = X[0:n_train, :]
    y_train = y[0:n_train]
    X_test = X[n_train+1:n, :]
    y_test = y[n_train+1:n]

    if o_perform_normalization:
        print('Scaling data')
        norm_const = fext.get_normalization_constants(X_train, axis=0)
        # norm_const = fext.get_normalization_constants(np.concatenate((X_train, X_test)), axis=0)
        X_train = fext.normalize_data(X_train, norm_const)
        X_test = fext.normalize_data(X_test, norm_const)

    if o_perform_PCA:
        print('Performing PCA')
        pca = PCA(n_components=n_pcs, whiten=o_whiten)
        pca = pca.fit(X_train)
        # pca = pca.fit(np.concatenate((X_train, X_test)))
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)

    nhpp = NonHomogeneousPoissonProcess(X_train, y_train, prediction_duration, estimation_method, model, model_param)

    nhpp.fit_model()
    n_test = y_test.shape[0]
    p = np.zeros((n_test, 1))
    expected_agg_episodes = np.zeros((n_test, 1))

    for j in range(n_test):
        # p0, intensity_measure = nhpp.compute_arrival_probability(X_test[j, :], 0)
        # p[j] = 1-p0
        p[j], intensity_measure = nhpp.compute_arrival_probability(X_test[j, :], -1)
        expected_agg_episodes[j] = intensity_measure

    lab2 = np.copy(y_test)
    lab2[lab2 > 0] = 1

    # plt.figure()
    # plt.plot(np.linalg.norm(X_test, axis=1))
    xx = np.linalg.norm(X_test, axis=1)
    plot_prediction_and_feat_norm_results(y_test, xx, expected_agg_episodes, fig_num=fig_num)
    # plot_arrival_and_prediction_results(y_test, np.linalg.norm(X_test, axis=1), expected_agg_episodes)
    # plot_arrival_and_prediction_results(y_test, p, expected_agg_episodes)
    # fpr, tpr, thresholds = metrics.roc_curve(lab2, p)

    return np.linalg.norm(X_test, axis=1)


def multiscale_cv(X, y_list, cv_folds, cv_reps, random_state, prediction_duration, estimation_method, model,
                  model_param=None, o_perform_normalization=False, o_perform_PCA=False, n_pcs=10, o_whiten=True,
                  pca_model=None, scaler_model=None):
    """

    :param X: (n,d) feature matrix
    :param y_list: list of label arrays for different scales. Each label array is (n_i,) with n_i <= n
    :param cv_folds: # of cv folds
    :param cv_reps: # of repetitions
    :param random_state: random state
    :param prediction_duration: list with prediction duration for each label
    :param estimation_method: 'LS' for Least-Squares, 'WLS' for Weighted LS, 'MLE' for Maximum Likelihood Estimation
    :param model:  'linear' for linear rate functions,...
    :param model_param: Model parameters. For MLE is a regularization parameter, for SVR is a dictionary with the SVR
         parameters.
    :param o_perform_normalization: perform normalization of everything
    :param o_perform_PCA: boolean True or False
    :param n_pcs: # of principal components
    :param o_whiten: whiten before pca with only training samples.
    :return:
    """

    ar_ord = None
    n_bins = 12
    # n_future_inst = 4

    n_scales = len(y_list)

    # find minimum label length
    # min_len = len(y_list[n_scales - 1])
    min_len = len(y_list[-1])
    max_idx = min_len - 1
    X = X[0:max_idx]

    # y_smallest = y_list[-1]
    y_poisson_train = y_list[0]

    tprs = [None]*n_scales
    aucs = [None]*n_scales
    mean_tpr = [None]*n_scales
    mean_auc = [None]*n_scales
    std_auc = [None]*n_scales
    # coefs = []
    mean_fpr = np.linspace(0, 1, 500)

    for ns in range(n_scales):
        print('Scale: ' + str(ns))
        #for each repetition and each cv fold:
        # trim all data for the smallest number of samples for all scales.
        # create a poisson process with the smallest scale
        # predict multiple over scales with the correct label and ...
        # compute statistics and save it

        y = y_list[ns]
        y = y[0:max_idx]
        lab = np.copy(y)
        # print(y.shape, X.shape, lab.shape)
        lab[lab > 0] = 1
        # print(y.shape, X.shape, lab.shape)

        # a list of tprs and aucs for each scale
        tprs[ns] = []
        aucs[ns] = []

        for i in range(cv_reps):
            print('CV Repetition #' + str(i) + '/' + str(cv_reps))
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=(i+1)*(ns+1))
            # skf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            # skf.get_n_splits(X, y)

            for train_index, test_index in skf.split(X, lab):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                y_init_train = y_poisson_train[train_index]

                if o_perform_normalization:
                    print('Scaling data')
                    if scaler_model is None:
                        scaler = preprocessing.StandardScaler().fit(X_train)
                        # norm_const = fext.get_normalization_constants(X_train, axis=0)
                        # norm_const = fext.get_normalization_constants(np.concatenate((X_train, X_test)), axis=0)
                    else:
                        scaler = scaler_model

                    X_train = scaler.transform(X_train)
                    X_test = scaler.transform(X_test)
                    # X_train = fext.normalize_data(X_train, norm_const)
                    # X_test = fext.normalize_data(X_test, norm_const)

                if o_perform_PCA:
                    print('Performing PCA')
                    if pca_model is None:
                        pca = PCA(n_components=n_pcs, whiten=o_whiten)
                        pca = pca.fit(X_train)
                    else:
                        pca = pca_model
                    # pca = pca.fit(np.concatenate((X_train, X_test)))
                    X_train = pca.transform(X_train)
                    # X_test = pca.transform(X_test)

                # initialize the Poisson process with the smallest scale.
                nhpp = NonHomogeneousPoissonProcess(X_train, y_init_train, prediction_duration[0], estimation_method, model,
                                                    model_param)
                nhpp.fit_model()
                n_test = y_test.shape[0]
                p = np.zeros((n_test, 1))
                expected_agg_episodes = np.zeros((n_test, 1))

                for j in range(n_test):
                    if ns == 0:
                        X_pred = X_test[j].reshape(-1, 1).T
                    else:
                        # Using AR model to predict future data
                        n_future_inst = prediction_duration[ns] - prediction_duration[0]
                        # X_pred = prediction_of_future_data(X_test[j], ar_ord, n_bins, n_future_inst)
                        X_pred = (X_test[j].reshape(-1, 1).T).repeat(1 + n_future_inst, axis=0)


                    if o_perform_PCA:
                        X_pred = pca.transform(X_pred)

                    p[j], intensity_measure = nhpp.compute_arrival_probability(X_pred, -1)
                    expected_agg_episodes[j] = intensity_measure

                lab2 = np.copy(y_test)
                lab2[lab2 > 0] = 1

                fpr, tpr, thresholds = metrics.roc_curve(lab2, p)
                # plt.figure()
                # plt.plot(fpr, tpr)

                tprs[ns].append(interp(mean_fpr, fpr, tpr))
                tprs[ns][-1][0] = 0.0
                roc_auc = metrics.auc(fpr, tpr)
                if math.isnan(roc_auc):
                    print('!!!!!!NAN!!!!!!!!', roc_auc)

                # cv_roc_aucs.append(roc_auc)
                aucs[ns].append(roc_auc)

        mean_tpr[ns] = np.mean(tprs[ns], axis=0)
        mean_tpr[ns][-1] = 1.0
        mean_auc[ns] = metrics.auc(mean_fpr, mean_tpr[ns])
        std_auc[ns] = np.std(aucs[ns])

        # plt.figure()
        # plt.plot(mean_fpr, mean_tpr[ns])
        # plt.show()

    # return mean_fpr, mean_tpr, mean_auc, std_auc, tprs
    return [mean_fpr, mean_tpr, mean_auc, std_auc, tprs]


def prediction_of_future_data(x, ar_ord, n_bins, n_future_inst):
    """

    :param x: data ndarray (n,d)
    :param ar_ord: order of the autoregressive model
    :param n_bins: number of bins in x
    :param n_future_inst: number of future vectors
    :return: future instances (n_future_inst, d)
    """
    ar_coeffs, D, z = ar_fitting(x, ar_ord, n_bins)
    # predict n_future_inst
    d = x.shape[0]
    future_insts = np.zeros((n_future_inst + 1, d))
    for i in range(n_future_inst + 1):
        # D is a reshape of x (removing the std in the end) with shape (n_features, n_bins)
        # removing the 1st vector (to cope with the model order)
        D = D[:, 1:]
        # predicting future vector
        z = ar_predict(D, ar_coeffs)
        # reshaping as a 2d column array
        z = np.reshape(z, (-1, 1))
        # updating D with the prediction
        D = np.concatenate((D, z), axis=1)
        # computing the std array to complete the instance vector
        std_array = np.reshape(np.std(D, axis=1), (-1, 1))
        # concatenating and reshaping
        future_insts[i] = np.reshape(np.concatenate((D, std_array), axis=1).T, (1, d))

    return future_insts


def ar_predict(D, ar_coeffs):
    """
    ar_predict(D, ar_coeffs)
    :param D: N x L data matrix
    :param ar_coeffs: (L,) coefficient array
    :return: predicted vector D * ar_coeffs
    """
    return np.dot(D, ar_coeffs)


def ar_fitting(x, ar_ord, n_bins):

    # + 1 is for the feature std appended in the end (See gen_classifier_instances_from_session_data_frame at
    # featureextraction.py)
    # x = [x1 x2 x3 .... x12 x_stds]

    # get the last bin
    # reshape all previous bins in a matrix format
    # solve a LS problem
    # return coefficients

    total_n_bins = n_bins + 1
    if ar_ord is None:
        ar_ord = n_bins - 1
    # len_x = len(fext.select_feat_from_feat_code(feat_code))
    len_bin = x.shape[0]//total_n_bins
    X = x.reshape(total_n_bins, len_bin)
    D = X[:-2].T      # all but the last two vectors
    z = X[-2]         # get the penultimate vector (the last not std)
    # ar_coeffs = [np.linalg.lstsq(X[:][i], x_k[i], rcond=None)[0] for i in range(len_bin)]
    # print(D.shape)
    # print(z.shape)
    # ar_coeffs = np.linalg.lstsq(D, z, rcond=None)[0]
    ar_coeffs = FW(D, z, 1, max_iter=100, tol=1e-8, callback=None).toarray().reshape(ar_ord,)
    # print(ar_coeffs)
    # print(np.abs(np.roots(ar_coeffs)))
    return ar_coeffs, X[:-1].T, z


# def FW(A, b, alpha, max_iter=100, tol=1e-8, callback=None):
def FW(A, b, alpha, max_iter=20, tol=1e-3, callback=None):
  # .. initial estimate, could be any feasible point ..
  x_t = sparse.dok_matrix((A.shape[1], 1))

  # .. some quantities can be precomputed ..
  Atb = A.T.dot(b)
  for it in range(max_iter):
      # .. compute gradient. Slightly more involved than usual because ..
      # .. of the use of sparse matrices ..
      Ax = x_t.T.dot(A.T).ravel()
      grad = (A.T.dot(Ax) - Atb)

      # .. the LMO results in a vector that is zero everywhere except for ..
      # .. a single index. Of this vector we only store its index and magnitude ..
      idx_oracle = np.argmax(np.abs(grad))
      mag_oracle = alpha * np.sign(-grad[idx_oracle])
      d_t = -x_t.copy()
      d_t[idx_oracle] += mag_oracle
      g_t = - d_t.T.dot(grad).ravel()
      if g_t <= tol:
          break
      q_t = A[:, idx_oracle] * mag_oracle - Ax
      step_size = min(q_t.dot(b - Ax) / q_t.dot(q_t), 1.)
      x_t += step_size * d_t
      if callback is not None:
          callback(g_t)
  return x_t



#
# Test code
if __name__ == '__main__':

    np.random.seed(seed=1)
    # Test script
    # gradient and hessian test parameters
    n_runs = 100    # MC runs
    epsilon = 1e-6  # (f(x + epsilon*d) - f(x - epsilon*d) )/2*epsilon
    xi = 1e-1       # std of the point we are testing the gradient around the optimal point w

    # data gen parameters
    n = 1000
    d = 5
    noise_std = 0.1
    # estimation_method = 'MLE'
    # estimation_method = 'LS'
    estimation_method = 'WLS'
    model = 'linear'
    # model = 'explin'
    # model = 'svr'
    # model = 'dnn'

    model_parameters = {'gamma': 0.5, 'C': 1000, 'kernel': 'rbf'}#, 'probability': True}

    # data gen
    X = np.random.randn(n, d)
    # X[:, 10] = 1
    # w = 1e-1*np.random.rand(d,)
    w = np.random.rand(d, )

    # # Test MLE estimation with explin model
    # # y = np.exp(X.dot(w) + 1 + noise_std*np.random.randn(n,))
    # y = np.exp(X.dot(w) + 1) + noise_std * np.random.randn(n, )
    # prediction_duration = 1
    # nhpp = NonHomogeneousPoissonProcess(X, y, prediction_duration, estimation_method=estimation_method, model=model,
    #                                     model_param=model_parameters)

    # Test LS estimation with linear model
    y = X.dot(w) + 10
    # noise = noise_std * np.random.randn(y.shape[0], 1)
    # y += np.reshape(noise, (n,))
    noise_mean = np.zeros((n,))
    noise_cov = np.diag(np.maximum(np.random.rand(n), 0.5))
    noise = np.random.multivariate_normal(noise_mean, noise_cov)
    y += noise
    prediction_duration = 1
    nhpp = NonHomogeneousPoissonProcess(X, y, prediction_duration, estimation_method=estimation_method, model='linear')

    nhpp.fit_model()
    print('w_true = ', str(w))
    print('w_est = ', str(nhpp.w_opt))
    print('||w_error||^2 = ', str(np.linalg.norm(np.array(nhpp.w_opt) - np.concatenate((w, np.ones((1,)))))))
    print('||e||^2 = ', str(np.linalg.norm(y - np.dot(np.append(X, np.ones((n, 1)), axis=1), nhpp.w_opt))))

    intensity_vector = np.zeros(n)
    arrival_prob = np.zeros(n)
    for i in range(n):
        # p0, intensity_vector[i] = nhpp.compute_arrival_probability(X[i, :], 0)
        # arrival_prob[i] = 1 - p0
        arrival_prob[i], intensity_vector[i] = nhpp.compute_arrival_probability(X[i, :], -1)


    # plot stuff
    import matplotlib.pyplot as plt
    plt.figure()
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath},\usepackage{amssymb}')
    plt.rc('font', family='serif')
    plt.subplot(311)
    plt.plot(y, label=r'$N_k$')
    plt.legend()
    plt.subplot(312)
    plt.plot(intensity_vector, label=r'$\Lambda_k = \mathbb{E}(N_k)$')
    plt.legend()
    plt.subplot(313)
    plt.plot(arrival_prob, label=r'$P(N_k > 0)$')
    plt.legend()

    if estimation_method == 'WLS':
        plt.figure()
        plt.plot(nhpp.xi, label='xi')
        # plt.plot(1 / np.diag(noise_cov), label='noise_cov')
        plt.plot(1 / np.sqrt(np.diag(noise_cov)), label='noise_cov')

    # # Test gradient
    # sqr_grad_error_array = np.zeros((n_runs, 1))
    # norm_hess_error_array = np.zeros((n_runs, 1))
    # grad_direction = matrix(np.ones((d + 1, 1)))
    #
    # for i in range(n_runs):
    #     # gradient test
    #     # grad_direction = matrix(np.random.randn(d + 1, 1))
    #     w0_matrix = matrix(np.concatenate((w, np.ones(1, ))) + xi * np.random.randn(d + 1, ))
    #     f, g_transpose, H = nhpp.cvx_format_explin_cost(w0_matrix, matrix(1))
    #     f1, g1_transpose = nhpp.cvx_format_explin_cost(w0_matrix + epsilon * grad_direction)  # , matrix(1))
    #     f2, g2_transpose = nhpp.cvx_format_explin_cost(w0_matrix - epsilon * grad_direction)  # , matrix(1))
    #     g_hat_1 = (f1 - f2) / (2 * epsilon)
    #     true_grad = g_transpose * grad_direction
    #     sqr_grad_error_array[i] = (g_hat_1 - true_grad)**2
    #
    #     # Hessian test
    #     H_num = matrix(0.0, (d+1, d+1))
    #     grad_num_vec = matrix(0.0, (d+1, 1))
    #     for j in range(d+1):
    #         grad_dir_j = matrix(0.0, (d+1, 1))
    #         grad_dir_j[j] = grad_direction[j]
    #         f1, g1_transpose = nhpp.cvx_format_explin_cost(w0_matrix + epsilon * grad_dir_j)  # , matrix(1))
    #         f2, g2_transpose = nhpp.cvx_format_explin_cost(w0_matrix - epsilon * grad_dir_j)  # , matrix(1))
    #         grad_num_vec[j] = (f1-f2)/(2*epsilon)
    #         H_num[:, j] = (0.5/epsilon) * (g1_transpose.T - g2_transpose.T)
    #
    #     # print(w0_j)
    #     # print('H = ', H)
    #     # print('H_num = ', H_num)
    #     norm_hess_error_array[i] = np.linalg.norm(H-H_num, ord='fro')
    #
    # print('mean grad error = ', np.mean(sqr_grad_error_array))
    # print('std grad error = ', np.std(sqr_grad_error_array))
    # print('E(||H||_F) = ', np.mean(norm_hess_error_array))
    # print('std(||H||_F) = ', np.std(norm_hess_error_array))

    # int_seed = 1
    # agg_pred_cv(X, y, 2, 1, int_seed, prediction_duration, estimation_method, model)
