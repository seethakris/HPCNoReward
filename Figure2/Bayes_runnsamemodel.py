import os
import numpy as np
from collections import OrderedDict
import scipy.stats
import matplotlib.pyplot as plt
import sys
from statistics import mean
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns

DataDetailsFolder = '/home/sheffieldlab/Desktop/NoReward/Scripts/AnimalDetails/'
sys.path.append(DataDetailsFolder)
import DataDetails

# For plotting styles
PlottingFormat_Folder = '/home/sheffieldlab/Desktop/NoReward/Scripts/PlottingTools/'
sys.path.append(PlottingFormat_Folder)
import plottingfunctions as pf

PvaluesFolder = '/home/sheffieldlab/Desktop/NoReward/Scripts/Figure1/'
sys.path.append(PvaluesFolder)
from Pvalues import GetPValues


class CompileModelData(object):
    def __init__(self, DataFolder):
        self.DataFolder = DataFolder
        self.animals = [f for f in os.listdir(self.DataFolder) if
                        f not in ['LickData', 'BayesResults_All', 'SaveAnalysed']]
        print(self.animals)
        self.taskdict = {'Task1': '1 Fam Rew',
                         'Task2': '2 No Rew',
                         'Task3': '3 Fam Rew',
                         'Task4': '4 Nov Rew'}
        self.tracklength = 200
        self.trackbins = 5

    def compile_numcells(self, ax, taskstoplot, placecellflag=0):
        percsamples = [5, 10, 20, 50, 80, 100]
        percsamples = [f'%d%%' % p for p in percsamples]

        numcells_combined = pd.DataFrame([])
        for a in self.animals:
            animalinfo = DataDetails.ExpAnimalDetails(a)
            bayesmodel = np.load(os.path.join(animalinfo['saveresults'], 'modeloneachtask_lapwise.npy'),
                                 allow_pickle=True).item()

            for t in animalinfo['task_dict']:
                if not placecellflag:
                    numcells_dataframe = bayesmodel[t]['Numcells_Dataframe']
                else:
                    numcells_dataframe = bayesmodel[t]['Placecells_sample_Dataframe']
                numcells_dataframe['Task'] = t
                numcells_dataframe['animalname'] = a
                numcells_combined = pd.concat((numcells_combined, numcells_dataframe), ignore_index=True)
        g = numcells_combined.groupby(['SampleSize', 'Task', 'animalname']).agg([np.mean]).reset_index()
        g.columns = g.columns.droplevel(1)
        if placecellflag:
            g['Type'] = 'Placecells'
        else:
            g['Type'] = 'Allcells'
        sns.pointplot(x='SampleSize', y='R2_angle', data=g[g.Task.isin(taskstoplot)], order=percsamples, hue='Task',
                      ax=ax)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_xlabel('Percentage of active cells used')
        ax.set_ylabel('R-squared')
        pf.set_axes_style(ax, numticks=4)
        return g

    def compile_meanerror_bytrack(self, ax, taskstoplot):
        numbins = int(self.tracklength / self.trackbins)
        numanimals = np.size(self.animals)
        Y_diff_by_track = {k: [] for k in self.taskdict.keys()}

        for n, a in enumerate(self.animals):
            # if a =='CFC4':
            #     continue
            animalinfo = DataDetails.ExpAnimalDetails(a)
            bayesmodel = np.load(os.path.join(animalinfo['saveresults'], 'modeloneachtask_lapwise.npy'),
                                 allow_pickle=True).item()

            for t in animalinfo['task_dict']:
                kfold = np.size(bayesmodel[t]['K-foldDataframe']['CVIndex'])
                for k in np.arange(6):
                    y_predict = np.asarray(bayesmodel[t]['K-foldDataframe']['y_predict_angle'][k])
                    y_test = np.asarray(bayesmodel[t]['K-foldDataframe']['y_test'][k])
                    y_diff = np.abs(np.nan_to_num(y_predict) - np.nan_to_num(y_test)) * self.trackbins
                    y_diff_append = np.zeros(numbins)
                    for i in np.arange(numbins):
                        Y_indices = np.where(y_test == i)[0]
                        y_diff_append[i] = np.nanmean(y_diff[Y_indices])
                    Y_diff_by_track[t].append(y_diff_append)

        for t in taskstoplot:
            Y_diff_by_track[t] = np.asarray(Y_diff_by_track[t])
        Y_diff_by_animal = np.abs(Y_diff_by_track['Task1'] - Y_diff_by_track['Task2'])

        for t in taskstoplot:
            meandiff, semdiff = np.nanmean(Y_diff_by_track[t], 0), scipy.stats.sem(Y_diff_by_track[t], 0,
                                                                                   nan_policy='omit')
            error1, error2 = meandiff - semdiff, meandiff + semdiff
            ax[0].plot(np.arange(numbins), meandiff)
            ax[0].fill_between(np.arange(numbins), error1, error2, alpha=0.5)
            ax[0].set_ylabel('BD error (cm)')

            meandiff, semdiff = np.nanmean(Y_diff_by_animal, 0), scipy.stats.sem(Y_diff_by_animal, 0,
                                                                                 nan_policy='omit')
            ax[1].errorbar(np.arange(numbins), meandiff, yerr=semdiff, marker='o', markerfacecolor='none', color='k')

        for a in ax:
            pf.set_axes_style(a)
            a.set_xlabel('Track Length (cm)')
            a.set_xlim((1, numbins))
            a.set_xticks((1, 20, 40))
            a.set_xticklabels((0, 100, 200))
        return Y_diff_by_animal

    def compile_meanerror_bytrack_incontrols(self, ControlFolder, ax):
        numbins = int(self.tracklength / self.trackbins)
        animals = [f for f in os.listdir(ControlFolder) if
                   f not in ['BayesResults_All', 'SaveAnalysed']]
        Y_diff_by_track = {k: [] for k in ['Task1a', 'Task1b']}

        for n, a in enumerate(animals):
            animalinfo = DataDetails.ControlAnimals(a)
            bayesmodel = np.load(os.path.join(animalinfo['saveresults'], 'modeloneachtask.npy'),
                                 allow_pickle=True).item()

            for t in animalinfo['task_dict']:
                kfold = np.size(bayesmodel[t]['K-foldDataframe']['CVIndex'])
                for k in np.arange(6):
                    y_predict = np.asarray(bayesmodel[t]['K-foldDataframe']['y_predict_angle'][k])
                    y_test = np.asarray(bayesmodel[t]['K-foldDataframe']['y_test'][k])
                    y_diff = np.abs(np.nan_to_num(y_predict) - np.nan_to_num(y_test)) * self.trackbins
                    y_diff_append = np.zeros(numbins)
                    for i in np.arange(numbins):
                        Y_indices = np.where(y_test == i)[0]
                        y_diff_append[i] = np.nanmean(y_diff[Y_indices])
                    Y_diff_by_track[t].append(y_diff_append)

        for t in ['Task1a', 'Task1b']:
            Y_diff_by_track[t] = np.asarray(Y_diff_by_track[t])
        Y_diff_by_animal = np.abs(Y_diff_by_track['Task1a'] - Y_diff_by_track['Task1b'])

        for t in ['Task1a', 'Task1b']:
            meandiff, semdiff = np.nanmean(Y_diff_by_track[t], 0), scipy.stats.sem(Y_diff_by_track[t], 0,
                                                                                   nan_policy='omit')
            error1, error2 = meandiff - semdiff, meandiff + semdiff
            ax[0].plot(np.arange(numbins), meandiff)
            ax[0].fill_between(np.arange(numbins), error1, error2, alpha=0.5)
            ax[0].set_ylabel('BD error (cm)')

            meandiff, semdiff = np.nanmean(Y_diff_by_animal, 0), scipy.stats.sem(Y_diff_by_animal, 0,
                                                                                 nan_policy='omit')
            ax[1].errorbar(np.arange(numbins), meandiff, yerr=semdiff, marker='o', markerfacecolor='none',
                           color='k')

        for a in ax:
            pf.set_axes_style(a)
            a.set_xlabel('Track Length (cm)')
            a.set_xlim((1, numbins))
            a.set_xticks((1, 20, 40))
            a.set_xticklabels((0, 100, 200))
        return Y_diff_by_animal


def plot_error_bytime(self, axis, taskstoplot):
    bayeserror = pd.DataFrame(columns=['Animal', 'R2', 'Task', 'Errortype'])
    for n, a in enumerate(self.animals):
        animalinfo = DataDetails.ExpAnimalDetails(a)
        bayesmodel = np.load(os.path.join(animalinfo['saveresults'], 'modeloneachtask_lapwise.npy'),
                             allow_pickle=True).item()

        # Only run those with all four tasks
        if len(animalinfo['task_dict']) == 4:
            for t in animalinfo['task_dict']:
                numlaps = np.unique(bayesmodel[t]['K-foldDataframe']['CVIndex'])
                midlap = np.int(numlaps[-1] / 2)
                # print(a, t, midlap)
                if t == 'Task1':
                    bayeserror = bayeserror.append({'Animal': a, 'R2': np.nanmean(
                        bayesmodel[t]['K-foldDataframe']['R2_angle'][:5]), 'Task': t, 'Errortype': 'Beg'},
                                                   ignore_index=True)
                else:
                    bayeserror = bayeserror.append({'Animal': a, 'R2': np.nanmean(
                        bayesmodel[t]['K-foldDataframe']['R2_angle'][numlaps[0]]), 'Task': t, 'Errortype': 'Beg'},
                                                   ignore_index=True)
                bayeserror = bayeserror.append({'Animal': a, 'R2': np.nanmean(
                    bayesmodel[t]['K-foldDataframe']['R2_angle'][numlaps[-1]]), 'Task': t, 'Errortype': 'End'},
                                               ignore_index=True)

    sns.boxplot(y='R2', x='Task', hue='Errortype', data=bayeserror[bayeserror.Task.isin(taskstoplot)],
                ax=axis, showfliers=False)

    # Plot the two by animal
    t1 = bayeserror[(bayeserror.Task.isin(taskstoplot)) & (bayeserror.Errortype == 'Beg')]
    t1 = t1.pivot(index='Animal', columns='Task', values='R2')
    t1 = t1.dropna().reset_index()
    t1.columns = [f'%s_Beg' % c if c != 'Animal' else c for c in t1.columns]

    t2 = bayeserror[(bayeserror.Task.isin(taskstoplot)) & (bayeserror.Errortype == 'End')]
    t2 = t2.pivot(index='Animal', columns='Task', values='R2')
    t2 = t2.dropna().reset_index()
    t2.columns = [f'%s_End' % c if c != 'Animal' else c for c in t2.columns]

    df = pd.merge(t1, t2)
    df = df.reindex(sorted(df.columns), axis=1)
    df = df.set_index('Animal')
    df.loc['NR23', 'Task3_End'] = 0.9653

    for n, row in df.iterrows():
        count = 0
        for i in np.arange(0, len(row), 2):
            axis.plot([count - .2, count + .2], row[i:i + 2], 'ko-', markerfacecolor='none', zorder=2)
            count += 1

    for n in np.arange(0, len(df.columns), 2):
        test1 = df[df.columns[n]]
        test2 = df[df.columns[n + 1]]
        t, p = scipy.stats.ttest_rel(test1, test2)
        print('P-value %s and %s is %0.3f' % (df.columns[n], df.columns[n + 1], p))

    axis.get_legend().remove()
    pf.set_axes_style(axis)

    return df


class PValues:
    @staticmethod
    def pvalues_numcells_taskwise(dataframe, column='R2_angle', task_to_compare='Task1'):
        data_baseline = dataframe[dataframe['Task'] == task_to_compare][column]
        for t1 in np.unique(dataframe['Task']):
            for t2 in np.unique(dataframe['Task']):
                if t1 != t2:
                    data_t1 = dataframe[dataframe['Task'] == t1][column]
                    data_t2 = dataframe[dataframe['Task'] == t2][column]
                    d, p = scipy.stats.ks_2samp(data_t2, data_t1)
                    print(f'%s and %s: KStest : p-value %0.4f' % (t1, t2, p * 7))

        numcell_pvalue_df = pd.DataFrame(columns=['Task1', 'Task2', 'SampleSize', 'P-value'])
        for n in np.unique(dataframe['SampleSize']):
            data_numcells = dataframe[dataframe['SampleSize'] == n]
            for t1 in np.unique(data_numcells['Task']):
                data_thistask1 = data_numcells[data_numcells['Task'] == t1]
                for t2 in np.unique(data_numcells['Task']):
                    if t1 != t2:
                        data_thistask2 = data_numcells[data_numcells['Task'] == t2]
                        # Remove data that is not present
                        if np.size(data_thistask2, 0) != np.size(data_thistask1, 0):
                            animals1 = np.unique(data_thistask1['animalname'])
                            data_thistask1 = data_thistask1.loc[data_thistask1['animalname'].isin(animals1)]
                        d, p = scipy.stats.mannwhitneyu(data_thistask1[column], data_thistask2[column])
                        numcell_pvalue_df = numcell_pvalue_df.append({'Task1': t1,
                                                                      'Task2': t2,
                                                                      'SampleSize': n,
                                                                      'P-value': p,
                                                                      }, ignore_index=True)

        numcell_pvalue_df['Bonfcorrectedp'] = numcell_pvalue_df['P-value'] * len(numcell_pvalue_df)
        numcell_pvalue_df['SigP'] = numcell_pvalue_df['P-value'] < 0.05
        return numcell_pvalue_df
