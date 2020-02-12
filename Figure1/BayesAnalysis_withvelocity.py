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
from Pvalues import GetPValues
from sklearn.linear_model import LinearRegression

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
    def __init__(self, ParentDataFolder, BayesFolder, taskstoget, CFC12flag=0):
        colors = sns.color_palette('muted')
        self.colors = [colors[0], colors[1], colors[3], colors[2]]
        self.task2_colors = [self.colors[1], self.colors[2]]
        self.BayesFolder = BayesFolder
        self.ParentDataFolder = ParentDataFolder
        self.CFC12flag = CFC12flag
        self.taskstoget = taskstoget
        self.tracklength = 200
        self.actuallaptime, self.goodlaptime, self.licks = self.load_velocity_data()
        self.accuracy_dict, self.numlaps_dict = self.get_lapwiseerror_peranimal()

    def load_velocity_data(self):
        AnimalFolders = [f for f in os.listdir(self.ParentDataFolder) if f not in ['LickData', 'BayesResults_All']]
        actuallaptime = OrderedDict()
        goodlaptime = OrderedDict()
        licksinlaps = OrderedDict()

        for a in AnimalFolders:
            BehaviorData = np.load(os.path.join(self.ParentDataFolder, a, 'SaveAnalysed', 'behavior_data.npz'),
                                   allow_pickle=True)
            actuallaptime[a] = BehaviorData['actuallaps_laptime'].item()
            goodlaptime[a] = BehaviorData['goodlaps_laptime'].item()
            licksinlaps[a] = BehaviorData['numlicks_withinreward_alllicks'].item()
        return actuallaptime, goodlaptime, licksinlaps

    def get_lapwiseerror_peranimal(self):
        files = [f for f in os.listdir(self.BayesFolder)]
        accuracy_dict = OrderedDict()
        numlaps_dict = OrderedDict()
        for f in files:
            print(f)
            animalname = f[:f.find('_')]
            if animalname == 'CFC12' and self.CFC12flag == 0:
                continue
            animal_tasks = DataDetails.ExpAnimalDetails(animalname)['task_dict']
            trackbins = DataDetails.ExpAnimalDetails(animalname)['trackbins']
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

    def calulate_lapwiseerror(self, y_actual, y_predicted, numlaps, lapframes):
        lap_R2 = []
        for l in np.arange(numlaps - 1):
            laps = np.where(lapframes == l + 1)[0]
            lap_R2.append(self.get_R2(y_actual[laps], y_predicted[laps]))

        return np.asarray(lap_R2)

    @staticmethod
    def get_R2(y_actual, y_predicted):
        y_mean = np.mean(y_actual)
        R2 = 1 - np.sum((y_predicted - y_actual) ** 2) / np.sum((y_actual - y_mean) ** 2)
        if np.isinf(R2):
            R2 = 0
        return R2


class VelocityGraphs(BayesError):
    def threshold_velocity_peranimal(self, baselinetask='Task1', uselicks=0, tol=0, toplaptime=100):
        shuffle_iterations = 100
        df_velocity = pd.DataFrame(columns=['Animalname', 'Accuracy', 'Laptype'])
        df_for_ptest_gl = pd.DataFrame(index=self.actuallaptime.keys(),
                                       columns=['%s_goodlap' % t for t in self.taskstoget])
        df_for_ptest_wl = pd.DataFrame(index=self.actuallaptime.keys(),
                                       columns=['%s_worstlap' % t for t in self.taskstoget if t not in baselinetask])
        lapspeeds_gl = {k: [] for k in self.taskstoget}
        lapspeeds_wl = {k: [] for k in self.taskstoget}
        for animalname in self.actuallaptime.keys():
            print(animalname)
            base_laptime = np.asarray(self.actuallaptime[animalname][baselinetask])

            df_velocity = df_velocity.append({'Animalname': animalname, 'Task': baselinetask,
                                              'Accuracy': np.nanmean(
                                                  self.accuracy_dict[animalname][baselinetask][-5:]),
                                              'Laptype': '%s_Goodlaps' % baselinetask}, ignore_index=True)
            df_for_ptest_gl.loc[animalname, '%s_goodlap' % baselinetask] = np.nanmean(
                self.accuracy_dict[animalname][baselinetask][-5:])
            lapspeeds_gl[baselinetask].extend(base_laptime[-5:])
            # Use last five laps for baseline task
            for t in self.actuallaptime[animalname]:
                if t != baselinetask and t in self.taskstoget:
                    laptime = np.asarray(self.actuallaptime[animalname][t])[:-1]
                    taskthreshold_gl = \
                        np.where((laptime >= base_laptime.min() - tol) & (laptime <= base_laptime.max() + tol))[0]
                    taskthreshold_wl = np.where((laptime > np.max(base_laptime) + tol) & (laptime < toplaptime))[0]
                    lapspeeds_gl[t].extend(laptime[taskthreshold_gl])
                    lapspeeds_wl[t].extend(laptime[taskthreshold_wl])

                    minimum_shuffle = np.argmin([np.size(taskthreshold_gl), np.size(taskthreshold_wl)])

                    if t == 'Task2' and uselicks == 1:
                        print('Bla')
                        licks = self.licks[animalname][t]
                        licklaps = np.where(licks[taskthreshold_gl] <= 2)
                        goodlaps = np.nanmean(self.accuracy_dict[animalname][t][taskthreshold_gl][licklaps])

                        licklaps = np.where(licks[taskthreshold_wl] <= 2)
                        worstlaps = np.nanmean(self.accuracy_dict[animalname][t][taskthreshold_wl][licklaps])
                    else:
                        goodlaps = np.nanmean(self.accuracy_dict[animalname][t][taskthreshold_gl])
                        worstlaps = np.nanmean(self.accuracy_dict[animalname][t][taskthreshold_wl])

                    df_velocity = df_velocity.append({'Animalname': animalname,
                                                      'Accuracy': goodlaps, 'Laptype': '%s_Goodlaps' % t},
                                                     ignore_index=True)
                    df_velocity = df_velocity.append({'Animalname': animalname,
                                                      'Accuracy': worstlaps, 'Laptype': '%s_Badlaps' % t},
                                                     ignore_index=True)
                    df_for_ptest_gl.loc[animalname, '%s_goodlap' % t] = goodlaps
                    df_for_ptest_wl.loc[animalname, '%s_worstlap' % t] = worstlaps
                    print('%s : Total laps %d, Goodlaps %d, Worstlaps %d' % (
                        t, np.size(laptime), np.size(taskthreshold_gl), np.size(taskthreshold_wl)))
                    print(minimum_shuffle)

        df_for_ptest = pd.concat((df_for_ptest_gl, df_for_ptest_wl), axis=1)
        return df_velocity, df_for_ptest, lapspeeds_gl, lapspeeds_wl

    def plot_boxplot_of_velocity(self, axis, df_velocity, df_for_ptest):
        x = [0, 1, 2]
        for i, row in df_for_ptest.iterrows():
            axis.plot(x, row, 'ko-', markerfacecolor='none', zorder=2)
        sns.boxplot(x='Laptype', y='Accuracy', data=df_velocity, ax=axis, palette=self.colors, width=0.6,
                    showfliers=False, zorder=1)
        axis.legend().set_visible(False)
        GetPValues().get_shuffle_pvalue(df_for_ptest, taskstocompare=list(df_for_ptest.columns))
        pf.set_axes_style(axis, numticks=4)
        axis.set_xlabel('')
        axis.set_xticklabels(['Task1', 'Task2b_good', 'Task2b_bad'])

    def create_hist_of_velocity_pertask(self, axis, taskstoplot, toplaptime=50, uselicks=0):
        velocity_dict = {k: [] for k in self.taskstoget}
        accuracy_bytask = {k: [] for k in self.taskstoget}
        for animalname in self.actuallaptime.keys():
            for t, values in self.actuallaptime[animalname].items():
                if t in self.taskstoget:
                    licks = self.licks[animalname][t]
                    values = np.asarray(values)[:-1]
                    threshold = np.where(values < toplaptime)[0]

                    if t == 'Task2' and uselicks == 1:
                        print('Bla')
                        nolicklaps = np.where(licks[threshold] == 0)[0]
                        print(np.shape(nolicklaps))
                        velocity_dict[t].extend(values[threshold][nolicklaps])
                        accuracy_bytask[t].extend(np.asarray(self.accuracy_dict[animalname][t])[threshold][nolicklaps])
                    else:
                        velocity_dict[t].extend(values[threshold])
                        accuracy_bytask[t].extend(np.asarray(self.accuracy_dict[animalname][t])[threshold])

        ax1 = axis[0].twinx()
        for n, t in enumerate(taskstoplot):
            weights = np.ones_like(velocity_dict[t]) / float(len(velocity_dict[t]))
            sns.distplot(velocity_dict[t], hist=False, bins=np.arange(0, toplaptime, 25),
                         ax=ax1, color=self.colors[n], label=t)
            axis[0].hist(velocity_dict[t], bins=np.linspace(0, toplaptime, 25),
                         color=self.colors[n], linewidth=2, weights=weights, alpha=0.5)
        # Plot scatter
        axis[1].plot(accuracy_bytask['Task2'], velocity_dict['Task2'], 'o', color='grey', markerfacecolor='none')
        y_pred_linearreg, rsquared = self.linear_regression(np.nan_to_num(accuracy_bytask['Task2']),
                                                            np.nan_to_num(velocity_dict['Task2']))
        corrcoef = np.corrcoef(np.nan_to_num(accuracy_bytask['Task2']), np.nan_to_num(velocity_dict['Task2']))[0, 1]
        pearsonsr = scipy.stats.pearsonr(np.nan_to_num(accuracy_bytask['Task2']), np.nan_to_num(velocity_dict['Task2']))
        axis[1].plot(accuracy_bytask['Task2'], y_pred_linearreg, color='k', linewidth=1)

        for a in axis:
            pf.set_axes_style(a, numticks=3)
        pf.set_axes_style(ax1, numticks=3)

        # Axis labels
        axis[0].set_xlim((0, toplaptime))
        # axis[0].set_ylim((0, 0.5))
        axis[0].set_xlabel('Time to complete a lap (s)')
        axis[0].set_ylabel('Normalised laps')

        axis[1].set_title('r2=%0.3f, r=%0.3f, p=%0.3f' % (rsquared, corrcoef, pearsonsr[1]))
        axis[1].set_xlabel('Decoder R2')
        axis[1].set_ylabel('Time to complete a lap (s)')

        print(np.size(accuracy_bytask['Task2']))
        print(np.size(np.where(np.asarray(accuracy_bytask['Task2']) > 0.9)))


    def linear_regression(self, x, y):
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        linear_regressor = LinearRegression()  # create object for the class
        linear_regressor.fit(x, y)  # perform linear regression
        y_pred = linear_regressor.predict(x)  # make predictions
        r_sq = linear_regressor.score(x, y)
        return y_pred, r_sq
