import os
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import sys
from collections import OrderedDict
from scipy.optimize import curve_fit

sns.set_context('paper', font_scale=1.3)
import pandas as pd

PvaluesFolder = '/home/sheffieldlab/Desktop/NoReward/Scripts/Figure1/'
sys.path.append(PvaluesFolder)
from Pvalues import GetPValues

# For plotting styles
PlottingFormat_Folder = '/home/sheffieldlab/Desktop/NoReward/Scripts/PlottingTools/'
sys.path.append(PlottingFormat_Folder)
import plottingfunctions as pf

# Data Details
DataDetailsFolder = '/home/sheffieldlab/Desktop/NoReward/Scripts/AnimalDetails/'
sys.path.append(DataDetailsFolder)
import DataDetails


class BayesError(object):
    def __init__(self, ParentDataFolder, BayesFolder, LickFolder, CFC12flag=0, controlflag=0):
        self.controlflag = controlflag
        self.ple = PlotErrorData
        if self.controlflag:
            self.colors = sns.color_palette(["#3498db", "#9b59b6"])
        else:
            colors = sns.color_palette('muted')
            self.colors = [colors[0], colors[1], colors[3], colors[2]]
            self.task2_colors = [self.colors[1], self.colors[2]]

        self.tracklength = 200
        self.trackbins = 5
        self.BayesFolder = BayesFolder
        self.ParentDataFolder = ParentDataFolder
        self.CFC12flag = CFC12flag
        self.LickFolder = LickFolder
        self.load_lick_data()
        self.velocity_slope = self.get_velocity_in_space()
        self.accuracy_dict, self.numlaps_dict = self.get_lapwiseerror_peranimal()
        self.accuracy_dict_incm = self.get_lapwiseerror_incm()

    def load_attention_data(self, animalname):
        required_laps = np.load(os.path.join(self.ParentDataFolder, animalname, 'SaveAnalysed', 'attentionlaps.npz'),
                                allow_pickle=True)
        return required_laps

    def load_lick_data(self):
        self.lickstop_df = pd.read_csv(os.path.join(self.LickFolder, 'Lickstops.csv'), index_col=0)
        self.lickstopcorrected_df = pd.read_csv(os.path.join(self.LickFolder, 'NormalizedLickstops.csv'), index_col=0)

    def load_lick_around_reward(self, taskstoplot):
        files = [f for f in os.listdir(self.ParentDataFolder) if
                 f not in ['LickData', 'BayesResults_All', 'SaveAnalysed']]
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
        dataframe = pd.DataFrame(index=self.lickstop_df.index, columns=taskstoplot)
        return dataframe

    def get_lapwiseerror_peranimal(self):
        files = [f for f in os.listdir(self.BayesFolder)]
        accuracy_dict = OrderedDict()
        numlaps_dict = OrderedDict()
        for f in files:
            print(f)
            animalname = f[:f.find('_')]
            if animalname in ['CFC12'] and self.CFC12flag == 0:
                continue
            animal_tasks = DataDetails.ExpAnimalDetails(animalname)['task_dict']
            data = np.load(os.path.join(self.BayesFolder, f), allow_pickle=True)
            animal_accuracy = {k: [] for k in animal_tasks}
            animal_numlaps = {k: [] for k in animal_tasks}
            for t in animal_tasks:
                animal_accuracy[t] = self.calulate_lapwiseerror(y_actual=data['fit'].item()[t]['ytest'],
                                                                y_predicted=data['fit'].item()[t]['yang_pred'],
                                                                numlaps=data['numlaps'].item()[t],
                                                                lapframes=data['lapframes'].item()[t])

                animal_numlaps[t] = data['numlaps'].item()[t]
            accuracy_dict[animalname] = animal_accuracy
            numlaps_dict[animalname] = animal_numlaps

        return accuracy_dict, numlaps_dict

    def get_velocity_in_space(self):
        velocity_file = np.load(os.path.join(self.ParentDataFolder, 'SaveAnalysed', 'velocity_in_space.npz'),
                                allow_pickle=True)
        velocity_slope = velocity_file['speed_ratio'].item()
        return velocity_slope

    def get_lapwiseerror_incm(self):
        files = [f for f in os.listdir(self.BayesFolder)]
        accuracy_dict = OrderedDict()
        for f in files:
            print(f)
            animalname = f[:f.find('_')]
            if animalname in ['CFC12'] and self.CFC12flag == 0:
                continue
            animal_tasks = DataDetails.ExpAnimalDetails(animalname)['task_dict']
            data = np.load(os.path.join(self.BayesFolder, f), allow_pickle=True)
            animal_accuracy = {k: [] for k in animal_tasks}
            for t in animal_tasks:
                animal_accuracy[t] = self.calulate_lapwiseerror_incm(y_actual=data['fit'].item()[t]['ytest'],
                                                                     y_predicted=data['fit'].item()[t]['yang_pred'],
                                                                     numlaps=data['numlaps'].item()[t],
                                                                     lapframes=data['lapframes'].item()[t])
            accuracy_dict[animalname] = animal_accuracy
        return accuracy_dict

    def calulate_lapwiseerror(self, y_actual, y_predicted, numlaps, lapframes):
        lap_R2 = []
        for l in np.arange(numlaps):
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
    def plot_shuffleerror_withanylicks(self, axis, taskstoplot, lickthreshold=0, removenan=False):
        numiterations = 1000
        accuracy_dataframe = self.create_accuaracy_datafame(taskstoplot)
        shuffle_error = {k: np.zeros((len(self.accuracy_dict), numiterations)) for k in taskstoplot}
        for n1, animal in enumerate(self.accuracy_dict):
            anylicks = np.load(os.path.join(self.ParentDataFolder, animal, 'SaveAnalysed', 'behavior_data.npz'),
                               allow_pickle=True)['numlicks_withinreward_alllicks'].item()['Task2']
            attentionlaps = self.load_attention_data(animal)
            laps_with_licks = np.where(anylicks > lickthreshold)[0]
            laps_with_nolicks = attentionlaps['attentivelaps_withoutlicks']

            for n2, t in enumerate(self.accuracy_dict[animal]):
                if t in taskstoplot:
                    decodererror = self.accuracy_dict[animal][t]
                    decodererror = decodererror[~np.isnan(decodererror)]
                    tasklap = np.size(decodererror)
                    laps_with_licks = laps_with_licks[laps_with_licks < tasklap]
                    laps_with_nolicks = laps_with_nolicks[laps_with_nolicks < tasklap]
                    # print(animal, laps_with_licks)
                    for iter in np.arange(numiterations):
                        if t == 'Task1':
                            randlaps = np.random.choice(np.arange(tasklap - 5, tasklap),
                                                        4, replace=False)
                            shuffle_error[t][n1, iter] = np.nanmean(decodererror[randlaps])
                        elif t == 'Task2':
                            randlaps = np.random.choice(laps_with_licks, np.minimum(4, np.size(laps_with_licks)),
                                                        replace=False)
                            shuffle_error[t][n1, iter] = np.nanmean(decodererror[randlaps])
                            randlaps = np.random.choice(laps_with_nolicks, np.minimum(4, np.size(laps_with_licks)),
                                                        replace=False)
                            shuffle_error['Attended'][n1, iter] = np.nanmean(decodererror[randlaps])
                        else:
                            randlaps = np.random.choice(np.arange(0, tasklap), np.minimum(4, np.size(laps_with_licks)),
                                                        replace=False)
                            shuffle_error[t][n1, iter] = np.nanmean(decodererror[randlaps])

                    if t == 'Task2':
                        accuracy_dataframe.loc[animal, t] = np.nanmean(shuffle_error[t][n1, :])
                        accuracy_dataframe.loc[animal, 'Attended'] = np.nanmean(shuffle_error['Attended'][n1, :])
                    else:
                        accuracy_dataframe.loc[animal, t] = np.nanmean(shuffle_error[t][n1, :])
        self.ple.plot_accuracy_boxplot(axis, colors=self.colors, accuracy_dataframe=accuracy_dataframe,
                                       taskstoplot=taskstoplot, removenan=removenan)
        GetPValues().get_shuffle_pvalue(accuracy_dataframe, taskstocompare=taskstoplot)

    def get_bayeserror_with_slidingwindow(self, axis, taskstoplot, totalnumlaps=15, windowsize=2):
        bayeserror_withslidingwindow = {k: np.zeros((len(self.accuracy_dict), totalnumlaps - windowsize)) for k in
                                        taskstoplot}
        numlicks_withslidingwindow = {k: np.zeros((len(self.accuracy_dict), totalnumlaps - windowsize)) for k in
                                      taskstoplot}
        for n1, animal in enumerate(self.accuracy_dict):
            attentionlaps = self.load_attention_data(animal)['attentivelaps'][:-1]
            anylicks = np.load(os.path.join(self.ParentDataFolder, animal, 'SaveAnalysed', 'behavior_data.npz'),
                               allow_pickle=True)['numlicks_withinreward_alllicks'].item()

            for n2, t in enumerate(self.accuracy_dict[animal]):
                if t in taskstoplot:
                    print(animal, np.shape(attentionlaps), np.shape(self.accuracy_dict[animal][t]))
                    decodererror = self.accuracy_dict[animal][t][attentionlaps]
                    # decodererror = decodererror[~np.isnan(decodererror)]
                    for i in np.arange(0, totalnumlaps - windowsize):
                        numlicks_withslidingwindow[t][n1, i] = np.sum(anylicks[t][attentionlaps][i:i + windowsize])
                        bayeserror_withslidingwindow[t][n1, i] = np.nanmedian(decodererror[i:i + windowsize])
        bayeserror_flat = {k: [] for k in taskstoplot}
        norm_lick = {k: [] for k in taskstoplot}
        # Flatten data from all animals into one array and remove NaNs
        for t in taskstoplot:
            bayeserror_flat[t] = bayeserror_withslidingwindow[t].flatten()
            notnan_index = np.where(~np.isnan(bayeserror_flat[t]))[0]
            bayeserror_flat[t] = bayeserror_flat[t][notnan_index]
            # Normalise licks per animals
            temp_lick = numlicks_withslidingwindow[t] / np.nanmax(numlicks_withslidingwindow[t], 1)[:, np.newaxis]
            norm_lick[t] = temp_lick.flatten()[notnan_index]

        self.ple.plot_bayeserror_with_slidingwindow(axis, self.task2_colors, bayeserror_flat['Task2'],
                                                    norm_lick['Task2'])
        return bayeserror_withslidingwindow, numlicks_withslidingwindow

    def get_bayeserror_with_slidingwindow_withvelocity(self, taskstoplot, totalnumlaps=15, windowsize=2):
        bayeserror_withslidingwindow = {k: np.zeros((len(self.accuracy_dict), totalnumlaps - windowsize)) for k in
                                        taskstoplot}
        numlicks_withslidingwindow = {k: np.zeros((len(self.accuracy_dict), totalnumlaps - windowsize)) for k in
                                      taskstoplot}
        slope_withslidingwindow = {k: np.zeros((len(self.accuracy_dict), totalnumlaps - windowsize)) for k in
                                   taskstoplot}
        for n1, animal in enumerate(self.accuracy_dict):
            attentionlaps = self.load_attention_data(animal)['attentivelaps'][:-1]
            anylicks = np.load(os.path.join(self.ParentDataFolder, animal, 'SaveAnalysed', 'behavior_data.npz'),
                               allow_pickle=True)['numlicks_withinreward_alllicks'].item()
            for n2, t in enumerate(self.accuracy_dict[animal]):
                if t in taskstoplot:
                    decodererror = self.accuracy_dict[animal][t][attentionlaps]
                    velocity_slope = np.asarray(self.velocity_slope[animal][t])[attentionlaps]
                    print(animal, np.shape(decodererror))
                    for i in np.arange(0, totalnumlaps - windowsize):
                        numlicks_withslidingwindow[t][n1, i] = np.sum(anylicks[t][attentionlaps][i:i + windowsize])
                        slope_withslidingwindow[t][n1, i] = np.nanmean(velocity_slope[i:i + windowsize])
                        bayeserror_withslidingwindow[t][n1, i] = np.nanmedian(decodererror[i:i + windowsize])

        return bayeserror_withslidingwindow, numlicks_withslidingwindow, slope_withslidingwindow

    def get_bayes_error_withtracklength(self, ax, taskstoplot, lickthreshold=2):
        bayeserror_withtrack = {k: [] for k in taskstoplot}
        for n1, animal in enumerate(self.accuracy_dict_incm):
            anylicks = np.load(os.path.join(self.ParentDataFolder, animal, 'SaveAnalysed', 'behavior_data.npz'),
                               allow_pickle=True)['numlicks_withinreward_alllicks'].item()['Task2']
            attentionlaps_withnolicks = self.load_attention_data(animal)['attentivelaps_withoutlicks']
            # print(animal, laps_with_nolicks)
            for n2, t in enumerate(self.accuracy_dict_incm[animal]):
                if t in taskstoplot:
                    decodererror = self.accuracy_dict_incm[animal][t]
                    if t == 'Task2':
                        bayeserror_withtrack[t].append(np.mean(decodererror[attentionlaps_withnolicks, :], 0))
                    elif t == 'Task1':
                        bayeserror_withtrack[t].append(np.nanmean(decodererror[-5:, :], 0))
                    else:
                        bayeserror_withtrack[t].append(np.mean(decodererror, 0))
        for t in taskstoplot:
            bayeserror_withtrack[t] = np.asarray(bayeserror_withtrack[t])
        self.ple.plot_errorwith_tracklength(ax, self.colors, bayeserror_withtrack, taskstoplot)
        return bayeserror_withtrack


class PlotErrorData:
    @staticmethod
    def plot_errorwith_tracklength(ax, colors, accuracy_dataframe, taskstoplot):
        for n, t in enumerate(taskstoplot):
            m = np.nanmean(accuracy_dataframe[t], 0)
            sem = scipy.stats.sem(accuracy_dataframe[t], 0, nan_policy='omit')
            ax.plot(m, color=colors[n], label=t)
            ax.fill_between(np.arange(np.size(m)), m - sem, m + sem, color=colors[n], alpha=0.5)
        pf.set_axes_style(ax)
        ax.set_xlabel('Track Length (cm)')
        ax.set_ylabel('Accuracy (cm)')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        for n in np.arange(np.size(accuracy_dataframe['Task1'], 1)):
            t, p = scipy.stats.mannwhitneyu(accuracy_dataframe['Task1'][:, n], accuracy_dataframe['Task2'][:, n])
            # print(p)
            if p <= 0.05:
                ax.plot(n, 35, 'k*')

    @staticmethod
    def plot_accuracy_boxplot(ax, colors, accuracy_dataframe, taskstoplot, removenan):
        if removenan:
            accuracy_dataframe = accuracy_dataframe.dropna()
        df_melt = accuracy_dataframe.melt(var_name='Task', value_name='Error')

        for index, row in accuracy_dataframe.iterrows():
            toplot = [r for r in row]
            ax.plot(toplot, 'ko-', markerfacecolor='none', zorder=2)
        df_melt['Error'] = df_melt['Error'].astype(float)
        sns.boxplot(x='Task', y='Error', data=df_melt, palette=colors, order=taskstoplot, ax=ax, showfliers=False,
                    zorder=1)
        ax.set_xlabel('')
        ax.set_ylabel('R-squared error')
        ax.set_ylim((0.35, 1.0))
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
    def plot_lick_withtime(axis, numlicks, axislabel, axislim=(0, 1), color='k', dashed=1):
        # num = (numlicks - np.min(numlicks, 1)[:, np.newaxis])
        # denom = (np.max(numlicks, 1) - np.min(numlicks, 1))[:, np.newaxis]
        numlicks = numlicks / np.nanmax(numlicks, 1)[:, np.newaxis]
        mean_licks = np.nanmean(numlicks, 0)
        sem_licks = scipy.stats.sem(numlicks, 0, nan_policy='omit')
        if dashed:
            axis.plot(mean_licks, '.-', color=color)
        else:
            axis.plot(mean_licks, '.', color=color)
        axis.fill_between(np.arange(np.size(mean_licks)), mean_licks - sem_licks, mean_licks + sem_licks,
                          color='lightgrey')
        axis.set_ylabel(axislabel)
        axis.set_xlabel('Laps in time')
        axis.set_ylim(axislim)
        pf.set_axes_style(axis, numticks=4)

    @staticmethod
    def fit_error_with_sigmoid(axis, data):
        normdata = data / np.nanmax(data, 1)[:, np.newaxis]
        mean_data = np.nanmean(normdata, 0)

        ydata = mean_data
        xdata = np.arange(np.size(mean_data))
        # axis.plot(xdata, ydata, 'o', label='data')

        popt, pcov = curve_fit(PlotErrorData.sigmoid, xdata, ydata)
        y = PlotErrorData.sigmoid(xdata, *popt)
        # print(popt)
        axis.plot(xdata, y, label='fit')

        yfirst = np.gradient(y, edge_order=2)
        # print(np.diff(yfirst))
        idx = np.where(np.diff(yfirst) > 0)[0][0]
        # print(idx)
        axis.axvline(idx)

    @staticmethod
    def sigmoid(x, a, b, c, d):
        """ General sigmoid function
        a adjusts amplitude
        b adjusts y offset
        c adjusts x offset
        d adjusts slope """
        y = ((a - b) / (1 + np.exp(x - (c / 2)) ** d)) + b
        return y

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
