import os
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import sys
from collections import OrderedDict
from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)
import csv

sns.set_context('paper', font_scale=1.3)
import pandas as pd
import warnings

# For plotting styles
PlottingFormat_Folder = '/home/sheffieldlab/Desktop/NoReward/Scripts/PlottingTools/'
sys.path.append(PlottingFormat_Folder)
import plottingfunctions as pf

# Data Details
DataDetailsFolder = '/home/sheffieldlab/Desktop/NoReward/Scripts/AnimalDetails/'
sys.path.append(DataDetailsFolder)
import DataDetails


class BayesError(object):
    def __init__(self, ParentDataFolder, BayesFolder, controlflag=0):
        self.controlflag = controlflag
        self.ple = PlotErrorData
        self.colors = sns.color_palette(["#3498db", "#9b59b6"])
        self.tracklength = 200
        self.trackbins = 5
        self.BayesFolder = BayesFolder
        self.ParentDataFolder = ParentDataFolder
        self.accuracy_dict, self.numlaps_dict = self.get_lapwiseerror_peranimal()
        self.accuracy_dict_incm = self.get_lapwiseerror_incm()

    def load_lick_around_reward(self, taskstoplot):
        files = [f for f in os.listdir(self.ParentDataFolder) if 'Bayes' not in f and 'Lick' not in f]
        print(files)
        self.lick_data_dict = {keys: [] for keys in taskstoplot}
        for i in files:
            print('Loading..', i)
            data = np.load(os.path.join(self.ParentDataFolder, i, 'SaveAnalysed', 'behavior_data.npz'),
                           allow_pickle=True)
            for t in taskstoplot:
                licks = data['licks_bytimefromreward'].item()[t]
                self.lick_data_dict[t].append(np.mean(licks, 0))

    def create_accuaracy_datafame(self, taskstoplot):
        dataframe = pd.DataFrame(index=self.accuracy_dict.keys(), columns=taskstoplot)
        return dataframe

    def get_lapwiseerror_peranimal(self):
        files = [f for f in os.listdir(self.BayesFolder)]
        accuracy_dict = OrderedDict()
        numlaps_dict = OrderedDict()
        for f in files:
            print(f)
            animalname = f[:f.find('_')]
            animal_tasks = DataDetails.ControlAnimals(animalname)['task_dict']
            data = np.load(os.path.join(self.BayesFolder, f), allow_pickle=True)
            animal_accuracy = {k: [] for k in animal_tasks}
            animal_numlaps = {k: [] for k in animal_tasks}
            for t in animal_tasks:
                print(t)
                animal_accuracy[t] = self.calulate_lapwiseerror(y_actual=data['fit'].item()[t]['ytest'],
                                                                y_predicted=data['fit'].item()[t]['yang_pred'],
                                                                numlaps=data['numlaps'].item()[t],
                                                                lapframes=data['lapframes'].item()[t])

                animal_numlaps[t] = data['numlaps'].item()[t]
            accuracy_dict[animalname] = animal_accuracy
            numlaps_dict[animalname] = animal_numlaps

        return accuracy_dict, numlaps_dict

    def get_lapwiseerror_incm(self):
        files = [f for f in os.listdir(self.BayesFolder)]
        accuracy_dict = OrderedDict()
        for f in files:
            animalname = f[:f.find('_')]
            print(f)
            animal_tasks = DataDetails.ControlAnimals(animalname)['task_dict']
            data = np.load(os.path.join(self.BayesFolder, f), allow_pickle=True)
            animal_accuracy = {k: [] for k in animal_tasks}
            for t in animal_tasks:
                print(t)
                animal_accuracy[t] = self.calulate_lapwiseerror_incm(y_actual=data['fit'].item()[t]['ytest'],
                                                                     y_predicted=data['fit'].item()[t]['yang_pred'],
                                                                     numlaps=data['numlaps'].item()[t],
                                                                     lapframes=data['lapframes'].item()[t])
            accuracy_dict[animalname] = animal_accuracy
        return accuracy_dict

    def calulate_lapwiseerror(self, y_actual, y_predicted, numlaps, lapframes):
        lap_R2 = []
        for l in np.arange(numlaps - 1):
            laps = np.where(lapframes == l + 1)[0]
            lap_R2.append(self.get_R2(y_actual[laps], y_predicted[laps]))

        return np.asarray(lap_R2)

    def calulate_lapwiseerror_incm(self, y_actual, y_predicted, numlaps, lapframes):
        lap_error_incm = []
        for l in np.arange(numlaps):
            laps = np.where(lapframes == l + 1)[0]
            y_error = self.get_cm_error(y_actual[laps], y_predicted[laps])
            if ~np.all(np.isnan(y_error)):
                lap_error_incm.append(y_error)

        return np.asarray(lap_error_incm)

    @staticmethod
    def get_R2(y_actual, y_predicted):
        y_mean = np.mean(y_actual)
        R2 = 1 - np.sum((y_predicted - y_actual) ** 2) / np.sum((y_actual - y_mean) ** 2)
        if np.isinf(R2):
            R2 = 0
        return R2

    def get_cm_error(self, y_actual, y_predicted):
        numbins = int(self.tracklength / self.trackbins)
        y_diff = (np.abs(np.nan_to_num(y_actual) - np.nan_to_num(y_predicted))) * self.trackbins
        # Bin error by track
        y_diff_by_track = np.zeros(np.max(numbins))
        for i in np.arange(numbins):
            y_indices = np.where(y_actual == i)[0]
            y_diff_by_track[i] = np.nanmean(y_diff[y_indices])

        return y_diff_by_track


class GetErrordata(BayesError):
    def plot_shuffleerror_withlickstop(self, axis, taskstoplot):
        ## Go through lick data frame, shuffle to equalise laps by lick stop
        numiterations = 1000
        accuracy_dataframe = self.create_accuaracy_datafame(taskstoplot)
        shuffle_error = {k: np.zeros((len(self.accuracy_dict), numiterations)) for k in taskstoplot}
        for n1, animal in enumerate(self.accuracy_dict):
            for n2, t in enumerate(self.accuracy_dict[animal]):
                if t in taskstoplot:
                    decodererror = self.accuracy_dict[animal][t]
                    decodererror = decodererror[~np.isnan(decodererror)]
                    tasklap = np.size(decodererror)
                    for iter in np.arange(numiterations):
                        if t == 'Task1a':
                            randlaps = np.random.choice(np.arange(tasklap - 5, tasklap), 4,
                                                        replace=False)
                            shuffle_error[t][n1, iter] = np.nanmean(decodererror[randlaps])
                        else:
                            randlaps = np.random.choice(np.arange(0, tasklap), 4,
                                                        replace=False)
                            shuffle_error[t][n1, iter] = np.nanmean(decodererror[randlaps])

                    if t == 'Task2':
                        accuracy_dataframe.loc[animal, t] = np.nanmean(shuffle_error[t][n1, :])
                        accuracy_dataframe.loc[animal, 'Task2b'] = np.nanmean(shuffle_error['Task2b'][n1, :])
                    else:
                        accuracy_dataframe.loc[animal, t] = np.nanmean(shuffle_error[t][n1, :])
        self.ple.plot_accuracy_boxplot(axis, colors=self.colors, accuracy_dataframe=accuracy_dataframe,
                                       taskstoplot=taskstoplot)
        GetPValues().get_shuffle_pvalue(accuracy_dataframe, taskstocompare=['Task1a', 'Task1b'])

    def get_bayeserror_acrosstimebins(self, axis, taskstoplot, task2_bins=5):
        bin_df = pd.DataFrame()
        bin_compiled = {k: [] for k in ['Bin0', 'Bin1', 'Bin2', 'Bin3', 'Animal']}
        for animal in self.accuracy_dict:
            bayeserror_acrosstimebins = OrderedDict()
            bin_compiled['Animal'].append(animal)
            for n2, t in enumerate(self.accuracy_dict[animal]):
                lickstop = 5
                if t in taskstoplot:
                    decodererror = self.accuracy_dict[animal][t]
                    decodererror = decodererror[~np.isnan(decodererror)]
                    tasklap = np.size(decodererror)
                    if t == 'Task1a':
                        bayeserror_acrosstimebins['Bin0'] = np.nanmean(decodererror[tasklap - 5:])
                        bin_compiled['Bin0'].append(bayeserror_acrosstimebins['Bin0'])
                    else:
                        bayeserror_acrosstimebins['Bin1'] = np.nanmean(decodererror[:lickstop])
                        bin_compiled['Bin1'].append(bayeserror_acrosstimebins['Bin1'])
                        iteration = 2
                        for l in np.arange(lickstop, lickstop + (task2_bins * 2), task2_bins):
                            bayeserror_acrosstimebins[f'Bin%d' % iteration] = np.nanmean(
                                decodererror[l:l + task2_bins])
                            bin_compiled[f'Bin%d' % iteration].append(
                                bayeserror_acrosstimebins[f'Bin%d' % iteration])
                            iteration += 1

                bin_df = bin_df.append(pd.DataFrame(
                    {'Bin': list(bayeserror_acrosstimebins.keys()), 'Value': list(bayeserror_acrosstimebins.values()),
                     'Animal': animal}), ignore_index=True)
        bin_compiled_df = pd.DataFrame.from_dict(bin_compiled)
        bin_compiled_df = bin_compiled_df.set_index('Animal')
        self.ple.plot_bayeserror_bytimebins(axis, bin_df, bin_compiled_df)
        GetPValues().get_shuffle_pvalue(bin_compiled_df, taskstocompare=['Bin0', 'Bin1', 'Bin2', 'Bin3'])

    def get_bayes_error_withtracklength(self, ax, taskstoplot, lickthreshold=2):
        bayeserror_withtrack = {k: [] for k in taskstoplot}
        for n1, animal in enumerate(self.accuracy_dict_incm):
            # print(animal, laps_with_nolicks)
            for n2, t in enumerate(self.accuracy_dict_incm[animal]):
                if t in taskstoplot:
                    decodererror = self.accuracy_dict_incm[animal][t]
                    tasklap = np.size(decodererror, 0)
                    if t == 'Task1a':
                        bayeserror_withtrack[t].append(np.nanmean(decodererror[-10:, :], 0))
                    else:
                        bayeserror_withtrack[t].append(np.nanmean(decodererror, 0))
        for t in taskstoplot:
            bayeserror_withtrack[t] = np.asarray(bayeserror_withtrack[t])
        self.ple.plot_errorwith_tracklength(ax[0], self.colors, bayeserror_withtrack, taskstoplot)
        bayes_diff = self.ple.plot_errorwith_tracklength_difference(ax[1], bayeserror_withtrack)
        return bayeserror_withtrack, bayes_diff


class PlotErrorData:
    @staticmethod
    def plot_errorwith_tracklength(ax, colors, accuracy_dataframe, taskstoplot):
        for n, t in enumerate(taskstoplot):
            m = np.nanmean(accuracy_dataframe[t], 0)
            sem = scipy.stats.sem(accuracy_dataframe[t], 0, nan_policy='omit')
            ax.plot(m, color=colors[n], label=t)
            ax.fill_between(np.arange(np.size(m)), m - sem, m + sem, color=colors[n], alpha=0.5)
            # print(np.shape(accuracy_dataframe[t]))
        pf.set_axes_style(ax)
        ax.set_xlabel('Track Length (cm)')
        ax.set_ylabel('Accuracy (cm)')
        # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_xlim((1, 40))

        # # #Calculate p-value
        # for n in np.arange(np.size(accuracy_dataframe['Task1'], 1)):
        #     t, p = scipy.stats.mannwhitneyu(accuracy_dataframe['Task1'][:, n], accuracy_dataframe['Task2'][:, n])
        #     if p < 0.05:
        #         ax.plot(n, 35, 'k*')

    @staticmethod
    def plot_errorwith_tracklength_difference(ax, accuracy_dataframe):
        diff = []
        for i in np.arange(np.size(accuracy_dataframe['Task1a'], 0)):
            diff.append(np.abs(accuracy_dataframe['Task1a'][i, :] - accuracy_dataframe['Task1b'][i, :]))
        diff = np.asarray(diff)
        # print(np.shape(diff))
        m = np.nanmean(diff, 0)
        sem = scipy.stats.sem(diff, 0, nan_policy='omit')
        ax.errorbar(np.arange(np.size(m)), m, yerr=sem, marker='o', markerfacecolor='none', color='k')
        ax.set_xlim((1, 40))
        ax.set_xlabel('Track Length (cm)')
        pf.set_axes_style(ax)
        return diff

    @staticmethod
    def plot_accuracy_boxplot(ax, colors, accuracy_dataframe, taskstoplot):
        df_melt = accuracy_dataframe.melt(var_name='Task', value_name='Error')
        for index, row in accuracy_dataframe.iterrows():
            toplot = [r for r in row]
            ax.plot(toplot, 'ko-', markerfacecolor='none', zorder=2)
        df_melt['Error'] = df_melt['Error'].astype(float)
        sns.boxplot(x='Task', y='Error', data=df_melt, palette=colors, order=taskstoplot, ax=ax, showfliers=False,
                    zorder=1)
        ax.set_xlabel('')
        ax.set_ylabel('R-squared error')
        pf.set_axes_style(ax, numticks=4)

    @staticmethod
    def plot_bayeserror_bytimebins(ax, bin_df, bin_df_rowwise):
        colors = sns.cubehelix_palette(8, start=.5, rot=-.75)
        for index, row in bin_df_rowwise.iterrows():
            ax.plot([row['Bin0'], row['Bin1'], row['Bin2'], row['Bin3']], 'ko-', markerfacecolor='none', zorder=2)
        sns.boxplot(x='Bin', y='Value', data=bin_df, order=[f'Bin%d' % i for i in np.arange(4)], palette=colors,
                    ax=ax, showfliers=False)

        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_xlabel('')
        ax.set_ylabel('R-squared error')
        pf.set_axes_style(ax, numticks=4)

    @staticmethod
    def plot_bayeserror_with_slidingwindow(axis, colors, bayeserror, numlicks):
        zerolick = np.where(numlicks == 0)
        nonzerolick = np.where(numlicks > 0)
        print(np.shape(zerolick), np.shape(nonzerolick))
        label = ['With Licks', 'Without Licks']
        ax1 = axis[0].twinx()
        for n, l in enumerate([nonzerolick, zerolick]):
            weights = np.ones_like(bayeserror[l]) / float(len(bayeserror[l]))
            sns.distplot(bayeserror[l], bins=np.arange(-1, 1, 50), hist=False,
                         ax=ax1, color=colors[n], label=label[n])
            axis[0].hist(bayeserror[l], bins=np.linspace(-0.5, 1, 20), color=colors[n], alpha=0.5,
                         weights=weights)
            axis[1].hist(bayeserror[l], bins=np.linspace(-0.5, 1, 1000), color=colors[n],
                         normed=True, cumulative=True, histtype='step')

        axis[1].set_ylim((0, 1))
        axis[0].set_ylabel('Normalized laps')
        for a in axis:
            pf.set_axes_style(a, numticks=4)
            a.set_xlabel('R-squared')
        t, p = scipy.stats.ks_2samp(bayeserror[nonzerolick], bayeserror[zerolick])
        axis[0].set_title(f'P-value : %f' % p)

    @staticmethod
    def plot_lick_withtime(axis, numlicks):
        numlicks = numlicks / np.max(numlicks, 1)[:, np.newaxis]
        mean_licks = np.mean(numlicks, 0)
        sem_licks = scipy.stats.sem(numlicks, 0)
        axis.plot(mean_licks, 'k')
        axis.fill_between(np.arange(np.size(mean_licks)), mean_licks - sem_licks, mean_licks + sem_licks,
                          color='lightgrey')
        axis.set_ylabel('Normalized licks')
        axis.set_xlabel('Laps in time')
        pf.set_axes_style(axis, numticks=4)

    @staticmethod
    def plotlickdataaroundreward(axis, taskstoplot, lick_data, colors, frames_per_sec):
        for n, t in enumerate(taskstoplot):
            mean_all = np.mean(np.asarray(lick_data[t]), 0)
            std_all = np.std(np.asarray(lick_data[t]), 0)

            seconds_aroundrew = (np.size(mean_all) / frames_per_sec) / 2
            x = np.linspace(-seconds_aroundrew, seconds_aroundrew, np.size(mean_all))
            axis.plot(x, mean_all, linewidth=0.8, color=colors[n])
            axis.fill_between(x, mean_all - std_all, mean_all + std_all, color=colors[n], alpha=0.5)
            axis.set_xlim((x[0], x[-1]))
        axis.axvline(0, color='black', linestyle='--')
        axis.set_xlabel('Time (seconds)')
        axis.set_ylabel('Mean\nlicking signal')
        pf.set_axes_style(axis)


class GetPValues:
    def get_shuffle_pvalue(self, accuracy_dataframe, taskstocompare):
        # Get two p-values. One with outlier and one without
        accuracy_dataframe = accuracy_dataframe[taskstocompare]
        outlier_rem_df = self.remove_outlier(accuracy_dataframe, taskstocompare)

        print('\033[1mMultiple comparisons after removing Outliers\033[0m')

        self.get_multiplecomparisons_kruskal(outlier_rem_df)
        print('\n\n\n')
        print('\033[1mMultiple comparisons without removing outliers\033[0m')
        self.get_multiplecomparisons_kruskal(accuracy_dataframe)
        print('\n\n\n')

    def get_multiplecomparisons_kruskal(self, dataframe):
        # If distributions are different then do multiple comparisons
        dataframe = dataframe.dropna()
        print(dataframe)
        cleanbin = dataframe.melt(var_name='Bin', value_name='Value')
        MultiComp = MultiComparison(cleanbin['Value'],
                                    cleanbin['Bin'])
        comp = MultiComp.allpairtest(scipy.stats.kruskal, method='Bonf')
        print(comp[0])

    def remove_outlier(self, dataframe, taskstocompare):
        index_outlier = []
        for t in taskstocompare:
            taskdata = dataframe[t]
            taskdata = taskdata.dropna()
            Q1 = taskdata.quantile(0.25)
            Q3 = taskdata.quantile(0.75)
            IQR = Q3 - Q1
            outlier = (taskdata < (Q1 - 1.5 * IQR)) | (taskdata > (Q3 + 1.5 * IQR))
            index_outlier.extend(taskdata.index[outlier])
        # Drop outliers and drop NaNs
        print('Removing ouliers %s' % index_outlier)
        nooutlier_df = dataframe.drop(index_outlier)
        nooutlier_df = nooutlier_df.dropna(how='all')

        return nooutlier_df
