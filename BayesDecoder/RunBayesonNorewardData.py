import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from RunKeras import NaiveBayes as nb
from PlotDecodingResults import ModelPredictionPlots

import os
import scipy.io
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import pandas as pd


class PrepareBehaviorData(object):

    def __init__(self, BehaviorData, tracklength, trackbins, trackstart_index=0, figure_flag=1):
        self.BehaviorData = BehaviorData
        self.tracklength = tracklength
        self.trackbins = trackbins
        self.trackstart = np.min(self.BehaviorData)
        self.trackend = np.max(self.BehaviorData)
        self.numbins = int(self.tracklength / self.trackbins)

        # Bin and Convert position to binary
        self.create_trackbins()
        self.position_binary = self.convert_y_to_index(self.BehaviorData, trackstart_index, figure_flag)

    def create_trackbins(self):
        self.tracklengthbins = np.around(np.linspace(self.trackstart, self.trackend, self.numbins),
                                         decimals=5)

    def convert_y_to_index(self, Y, trackstart_index=0, figure_flag=1):
        Y_binary = np.zeros((np.size(Y, 0)))
        i = 0
        while i < np.size(Y, 0):
            current_y = np.around(Y[i], decimals=4)
            idx = self.find_nearest1(self.tracklengthbins, current_y)
            Y_binary[i] = idx
            if idx == self.numbins - 1:
                while self.find_nearest1(self.tracklengthbins, current_y) != trackstart_index and i < np.size(Y, 0):
                    current_y = np.around(Y[i], decimals=4)
                    idx = self.find_nearest1(self.tracklengthbins, current_y)
                    if idx == self.numbins - 1 or idx == 0:  # Correct for end of the track misses
                        Y_binary[i] = idx
                    else:
                        Y_binary[i] = self.numbins - 1
                    i += 1
            i += 1

        if figure_flag:
            plt.figure(figsize=(10, 3), dpi=80)
            plt.plot(Y_binary)
            plt.ylabel('Binned Position')
            plt.xlabel('Frames')
        return Y_binary

    @staticmethod
    def find_nearest1(array, value):
        idx = (np.abs(array - value)).argmin()
        return idx


class PreprocessData(object):
    # @staticmethod
    # def get_laps_of_equal_velocity(Imgobj, TaskA='Task1', TaskB='Task2'):

    @staticmethod
    def plot_velocity_distribution(Imgobj, Tasklist):
        plt.figure(dpi=80)
        for t in Tasklist:
            if t == 'Task2':
                stoplicklap = Imgobj.Parsed_Behavior['lick_stop'].item()
                label = Imgobj.TaskDict[t] + 'After Lick Stops'
            else:
                stoplicklap = 0
                label = Imgobj.TaskDict[t]
            sns.distplot(Imgobj.Parsed_Behavior['actuallaps_laptime'].item()[t][stoplicklap:], label=label, bins=10,
                         kde=False)
            plt.legend()
            plt.xlabel('Lap Time (seconds)')
            plt.ylabel('Number of laps')

    @staticmethod
    def plot_corrected_velocity_distribution(Imgobj, Tasklist):
        plt.figure(dpi=80)
        for t in Tasklist:
            if t == 'Task2':
                stoplicklap = Imgobj.Parsed_Behavior['lick_stop'].item()
                label = Imgobj.TaskDict[t] + 'After Lick Stops'
            else:
                stoplicklap = 0
                label = Imgobj.TaskDict[t]
            sns.distplot(Imgobj.Parsed_Behavior['goodlaps_laptime'].item()[t][stoplicklap:], label=label, bins=10,
                         kde=False)
            plt.legend()
            plt.xlabel('Lap Time (seconds)')
            plt.ylabel('Number of laps')

    @staticmethod
    def get_laps_of_similar_velocity(Imgobj, TaskA='Task1', TaskB='Task2', tol=0, after_stoplick=1):
        laptime_TaskA = np.asarray(Imgobj.Parsed_Behavior['actuallaps_laptime'].item()[TaskA])
        if after_stoplick:
            stoplicklap = Imgobj.Parsed_Behavior['lick_stop'].item()
            laptime_TaskB = np.asarray(Imgobj.Parsed_Behavior['actuallaps_laptime'].item()[TaskB][stoplicklap:])
        else:
            laptime_TaskB = np.asarray(Imgobj.Parsed_Behavior['actuallaps_laptime'].item()[TaskB])

        # Get velocity bins that are between the min of TaskB and max of TaskA
        thresholdedlapsA = \
            np.where((laptime_TaskA >= np.min(laptime_TaskA) - tol) & (laptime_TaskA <= np.max(laptime_TaskA) + tol))[0]
        thresholdedlapsB = \
            np.where((laptime_TaskB > np.min(laptime_TaskA) - tol) & (laptime_TaskB < np.max(laptime_TaskA) + tol))[0]

        lapvelocityA = laptime_TaskA[thresholdedlapsA]
        lapvelocityB = laptime_TaskB[thresholdedlapsB]

        print('Chosen speeds')
        print('Number of chosen laps : %s : %d, %s : %d' % (
            TaskA, np.size(thresholdedlapsA), TaskB, np.size(thresholdedlapsB)))
        print('Remainin laps in %d' % (np.size(laptime_TaskB) - np.size(thresholdedlapsB)))
        print(TaskA, lapvelocityA)
        print(TaskB, lapvelocityB)

        # Non overlapping laps
        worstlaps_TaskB = np.where(laptime_TaskB > np.max(laptime_TaskA) + tol)[0]
        print('Worst lap speeds %s' % TaskB)
        print(laptime_TaskB[worstlaps_TaskB])

        if after_stoplick:
            thresholdedlapsB += stoplicklap
            worstlaps_TaskB += stoplicklap

        return lapvelocityA, lapvelocityB, thresholdedlapsA, thresholdedlapsB, worstlaps_TaskB

    @staticmethod
    def compile_data_from_thresholdedlaps(Imgobj, Task, X_data, Y_data, thresholded_laps, E_correction=1,
                                          lapcorrectionflag=0, randomise=0, figureflag=1):

        if lapcorrectionflag:
            lapframes = \
                [scipy.io.loadmat(os.path.join(Imgobj.FolderName, 'Behavior', p))['E'].T for p in
                 Imgobj.PlaceFieldData if Task in p and 'Task2a' not in p][0]
        else:
            lapframes = \
                [scipy.io.loadmat(os.path.join(Imgobj.FolderName, 'Behavior', p))['bad_E'].T for p in
                 Imgobj.PlaceFieldData if Task in p and 'Task2a' not in p][0]

        X_new = np.array([])
        Y_new = np.array([])

        if randomise:
            np.random.shuffle(thresholded_laps)

        if figureflag:
            plt.figure(figsize=(10, 3), dpi=80)
            plt.plot(Y_data, color='b', linewidth=2)
            plt.title('Chosen laps')
        for l in thresholded_laps:
            laps1 = np.where(lapframes == l + E_correction)[0]
            X_new = np.vstack((X_new, X_data[laps1[0]:laps1[-1], :])) if len(X_new) else X_data[laps1[0]:laps1[-1], :]
            Y_new = np.vstack((Y_new, Y_data[laps1[0]:laps1[-1]])) if len(Y_new) else Y_data[laps1[0]:laps1[-1]]

            if figureflag:
                plt.plot(laps1[:-1], Y_data[laps1[0]:laps1[-1]], color='k', linewidth=2)
        print('Data shapes : ', np.shape(X_new), np.shape(Y_new))
        return X_new, Y_new

    @staticmethod
    def equalise_laps_with_numlaps_innorew(Imgobj, X, Y, Tasklabel, laps_current, numlaps_topick, E_correction=1,
                                           lapcorrectionflag=0, figureflag=1):

        if np.size(laps_current) > numlaps_topick:
            samplelaps = np.random.choice(laps_current, numlaps_topick, replace=False)
        else:
            samplelaps = laps_current
            np.random.shuffle(samplelaps)

        if lapcorrectionflag:
            lapframes = \
                [scipy.io.loadmat(os.path.join(Imgobj.FolderName, 'Behavior', p))['E'].T for p in
                 Imgobj.PlaceFieldData if Tasklabel in p and 'Task2a' not in p][0]
        else:
            lapframes = \
                [scipy.io.loadmat(os.path.join(Imgobj.FolderName, 'Behavior', p))['bad_E'].T for p in
                 Imgobj.PlaceFieldData if Tasklabel in p and 'Task2a' not in p][0]

        print(laps_current)
        print(samplelaps)

        X_eq = np.array([])
        Y_eq = np.array([])

        if figureflag:
            plt.figure(figsize=(10, 3), dpi=80)
            plt.plot(Y, color='b', linewidth=2)
            plt.title('Equal chosen laps')
        for l in samplelaps:
            laps1 = np.where(lapframes == l + E_correction)[0]
            X_eq = np.vstack((X_eq, X[laps1[0]:laps1[-1], :])) if len(X_eq) else X[laps1[0]:laps1[-1], :]
            Y_eq = np.vstack((Y_eq, Y[laps1[0]:laps1[-1]])) if len(Y_eq) else Y[laps1[0]:laps1[-1]]

            if figureflag:
                plt.plot(laps1[:-1], Y[laps1[0]:laps1[-1]], color='k', linewidth=2)

        print('New data shapes : ', np.shape(X_eq), np.shape(Y_eq))
        return X_eq, Y_eq


class RunNaiveBayes_ondata(object):
    def __init__(self):
        self.nb = nb()  # Load naive bayes classifier
        self.m = ModelPredictionPlots()

    def run_naivebayes(self, X_data, y_cat_data, testsize):
        # Split data into training and test sets and run bayes
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_cat_data, test_size=testsize, random_state=None,
                                                            shuffle=False)
        print('Data shapes : ', np.shape(X_train), np.shape(X_test), np.shape(y_train), np.shape(y_test))
        gaussianNB = self.nb.fit_naivebayes(X_train, y_train)
        scores, y_predicted, y_errorprob = nb.validate_model(gaussianNB, X_test, y_test)
        R2 = self.m.get_R2(y_test, y_predicted)
        rho = self.m.get_rho(y_test, y_predicted)
        print(f'R2: %.2f' % R2)
        print(f'rho: %.2f\n\n' % rho)
        return y_test, y_predicted

    def k_foldvalidation(self, X_data, y_cat_data, split_size=5, figureflag=1):
        # Define k-fold validation
        k = KFold(n_splits=split_size, random_state=None, shuffle=False)
        print(f'Performing %d-fold validation' % k.n_splits)
        count_cv = 1
        nbcv_dataframe = pd.DataFrame(
            columns=['CVIndex', 'ModelAccuracy', 'R2', 'rho', 'Y_diff', 'mean_errorprob', 'sem_errorprob'])
        for train_index, test_index in k.split(X_data):
            print(f'Validation %d' % count_cv)
            xcv_train, xcv_test = X_data[train_index], X_data[test_index]
            ycv_train, ycv_test = y_cat_data[train_index], y_cat_data[test_index]

            print(np.shape(xcv_train), np.shape(ycv_train),
                  np.shape(xcv_test), np.shape(ycv_test))
            cvsnbmodel = self.nb.fit_naivebayes(xcv_train, ycv_train)

            # evaluate the model and save evalautions
            scores, ycv_predict, ycv_probability = self.nb.validate_model(
                cvsnbmodel, xcv_test, ycv_test, plot_figure=figureflag)

            R2 = self.m.get_R2(ycv_test, ycv_predict)
            rho = self.m.get_rho(ycv_test, ycv_predict)

            nbcv_dataframe = nbcv_dataframe.append({'CVIndex': count_cv,
                                                    'ModelAccuracy': scores,
                                                    'Y_diff': np.abs(ycv_test - ycv_predict),
                                                    'R2': R2,
                                                    'rho': rho,
                                                    'mean_errorprob': np.mean(ycv_probability, 0),
                                                    'sem_errorprob': scipy.stats.sem(ycv_probability, 0)},
                                                   ignore_index=True)
            count_cv += 1

        return nbcv_dataframe

    def plotcrossvalidationresult(self, axis, cv_dataframe, trackbins, numsplits):
        # Difference
        meandiff = np.mean(cv_dataframe['Y_diff'].to_list(), 1) * trackbins
        sem = scipy.stats.sem(cv_dataframe['Y_diff'].to_list(), 1) * trackbins
        error1, error2 = meandiff - sem, meandiff + sem
        axis.plot(np.arange(numsplits), meandiff)
        axis.fill_between(np.arange(numsplits), error1, error2, alpha=0.5)
        axis.set_xlabel('CrossValidation #')
        axis.set_ylabel('Difference (cm)')
        axis.set_title(f'Difference in predicted position %.2f +/- %2f' % (
            np.mean(cv_dataframe['Y_diff'].to_list()) * trackbins,
            np.std(cv_dataframe['Y_diff'].to_list()) * trackbins))


