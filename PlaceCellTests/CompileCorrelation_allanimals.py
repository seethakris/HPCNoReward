import scipy.io
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import sys
import scipy.stats
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

DataDetailsFolder = '/home/sheffieldlab/Desktop/NoReward/Scripts/AnimalDetails/'
sys.path.append(DataDetailsFolder)
import DataDetails

PlottingFormat_Folder = '/home/sheffieldlab/Desktop/NoReward/Scripts/PlottingTools/'
sys.path.append(PlottingFormat_Folder)
import plottingfunctions as pf


class CombineData(object):
    def __init__(self, CombinedDataFolder, ParentDataFolder):
        self.CombinedDataFolder = CombinedDataFolder
        self.ParentDataFolder = ParentDataFolder
        self.npyfiles = [f for f in os.listdir(self.CombinedDataFolder) if f.endswith('.npz')]

    def get_correlation_data(self, f):
        data = np.load(os.path.join(self.CombinedDataFolder, f), allow_pickle=True)
        return data

    def get_animal_behaviordata(self, animalname):
        data = np.load(os.path.join(self.ParentDataFolder, animalname, 'SaveAnalysed', 'behavior_data.npz'),
                       allow_pickle=True)
        return data

    def get_mean_correlation_withshuffle(self, axis, taskstoplot):
        num_iterations = 1000
        shuffle_mean_corr = {k: np.zeros((num_iterations, np.size(self.npyfiles) - 2)) for k in taskstoplot}
        count = 0
        for n, f in enumerate(self.npyfiles):
            animalname = f[: f.find('_')]
            animal_tasks = DataDetails.ExpAnimalDetails(animalname)['task_dict']
            corr_data = self.get_correlation_data(f)
            corr_animal = corr_data['correlation_withTask1'].item()
            sigPFs = corr_data['sig_PFs_cellnum'].item()['Task1']
            lickstoplap = self.get_animal_behaviordata(animalname)['lick_stop'].item()['Task2']
            if lickstoplap > 2:
                for t in animal_tasks.keys():
                    if t in taskstoplot:
                        tasklap = np.size(corr_animal[t], 1)
                        corr_data_pfs = corr_animal[t][sigPFs, :]
                        for i in np.arange(num_iterations):
                            if t == 'Task2':
                                randlaps = np.random.choice(np.arange(0, lickstoplap), 4, replace=False)
                                shuffle_mean_corr[t][i, count] = np.mean(corr_data_pfs[:, randlaps].flatten())
                                randlaps = np.random.choice(np.arange(lickstoplap, tasklap), 4, replace=False)
                                shuffle_mean_corr['Task2b'][i, count] = np.mean(
                                    corr_data_pfs[:, randlaps].flatten())
                            else:
                                randlaps = np.random.choice(np.arange(0, tasklap - 5), 4, replace=False)
                                shuffle_mean_corr[t][i, count] = np.mean(corr_data_pfs[:, randlaps].flatten())
                count += 1

        # Get p-value
        p_value_task2 = []
        p_value_task2b = []
        for i in np.arange(num_iterations):
            t, p = scipy.stats.ttest_rel(shuffle_mean_corr['Task1'][i, :], shuffle_mean_corr['Task2'][i, :])
            p_value_task2.append(p > 0.01)
            t, p = scipy.stats.ttest_rel(shuffle_mean_corr['Task1'][i, :], shuffle_mean_corr['Task2b'][i, :])
            p_value_task2b.append(p > 0.01)
        print('Shuffled laps P-value with lick %0.3f, without lick %0.3f' % (
            np.size(np.where(p_value_task2)) / num_iterations, np.size(np.where(p_value_task2b)) / num_iterations))

        # Plot shuffle histogram
        # Remove zeros
        data = {k: [] for k in ['Task1', 'Task2', 'Task2b']}
        for t in ['Task1', 'Task2', 'Task2b']:
            temp = shuffle_mean_corr[t].flatten()
            data[t] = temp
            sns.distplot(data[t], label=t,
                         bins=np.linspace(0, 1, 50), ax=axis[1, 0])
        axis[1, 0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        t, p1 = scipy.stats.ks_2samp(data['Task1'], data['Task2'])
        t, p2 = scipy.stats.ks_2samp(data['Task1'], data['Task2b'])
        print('Flattened P-value with lick %f, without lick %f' % (p1, p2))

        # Get mean_correlation
        mean_correlation = {k: [] for k in taskstoplot}
        sem_correlation = {k: [] for k in taskstoplot}
        for t in taskstoplot:
            mean_correlation[t] = np.mean(shuffle_mean_corr[t], 0)
            sem_correlation[t] = scipy.stats.sem(shuffle_mean_corr[t], 0, nan_policy='omit')
        df = pd.DataFrame.from_dict(mean_correlation)
        df = df.replace(0, np.nan)
        df = df.dropna(how='all')
        for p in np.arange(2):
            if p == 0:
                df_melt = df.melt(var_name='Task', value_name='Error')
                for index, row in df.iterrows():
                    axis[0, p].plot([row['Task1'], row['Task2'], row['Task2b'], row['Task3']], 'k')
                print(df)
            else:
                df_div = df[df.columns].div(df['Task1'].values, axis=0)
                print(df_div)
                df_melt = df_div.melt(var_name='Task', value_name='Error')
                for index, row in df_div.iterrows():
                    axis[0, p].plot([row['Task1'], row['Task2'], row['Task2b'], row['Task3']], 'k')
            sns.boxplot(x='Task', y='Error', data=df_melt, palette='Blues', order=[
                'Task1', 'Task2', 'Task2b', 'Task3'], ax=axis[0, p])
            sns.stripplot(x='Task', y='Error', data=df_melt, color='k', size=5, order=[
                'Task1', 'Task2', 'Task2b', 'Task3'], ax=axis[0, p], dodge=False, jitter=False)
            axis[0, p].set_xlabel('')

        t, p1 = scipy.stats.ttest_rel(df['Task1'], df['Task2'])
        t, p2 = scipy.stats.ks_2samp(df['Task1'], df['Task2b'])
        print('Mean P-value with lick %f, without lick %f' % (p1, p2))

        axis[1, 1].axis('off')
        for a in axis.flatten():
            pf.set_axes_style(a)
        return shuffle_mean_corr, mean_correlation, data

    def get_lapwise_correlation_peranimal(self, taskstoplot, axis):
        numlaps = {'Task1': 5, 'Task2': 14, 'Task3': 11}
        correlation_data = np.zeros((len(self.npyfiles) - 2, sum(numlaps.values())))
        lick_data = np.zeros((len(self.npyfiles) - 2, sum(numlaps.values())))
        count = 0
        for n1, f in enumerate(self.npyfiles):
            print(f, count, np.shape(correlation_data))
            animalname = f[: f.find('_')]
            animal_tasks = DataDetails.ExpAnimalDetails(animalname)['task_dict']
            corr_data = self.get_correlation_data(f)
            corr_animal = corr_data['correlation_withTask1'].item()
            sigPFs = corr_data['sig_PFs_cellnum'].item()['Task1']
            lickstoplap = self.get_animal_behaviordata(animalname)['lick_stop'].item()['Task2']
            lick_per_lap = self.get_animal_behaviordata(animalname)['numlicks_withinreward_alllicks'].item()
            if lickstoplap > 2:
                count_lap = 0
                for n2, t in enumerate(animal_tasks.keys()):
                    if t in taskstoplot:
                        corr_sigPFs = corr_animal[t][sigPFs, :]
                        tasklap = np.size(corr_animal[t], 1)
                        if t == 'Task1':
                            randlaps = np.random.choice(np.arange(0, tasklap), numlaps[t], replace=False)
                            this_task_data = np.nanmedian(corr_sigPFs[:, np.arange(12, 12 + numlaps[t])], 0)
                            this_lick_data = lick_per_lap[t][-numlaps[t]:]
                        elif t == 'Task2':
                            this_task_data = np.nanmedian(corr_sigPFs[:, lickstoplap - 3:lickstoplap + 11], 0)
                            this_lick_data = lick_per_lap[t][lickstoplap - 3:lickstoplap + 11]
                        else:
                            this_task_data = np.nanmedian(corr_sigPFs[:, :numlaps[t]], 0)
                            this_lick_data = lick_per_lap[t][:numlaps[t]]

                        correlation_data[count, count_lap:count_lap + numlaps[t]] = this_task_data
                        lick_data[count, count_lap:count_lap + numlaps[t]] = this_lick_data
                        count_lap += numlaps[t]
                count += 1

        # Normalize and compare for p-value with Task1
        corr_norm = correlation_data / np.max(correlation_data[:, :numlaps['Task1']])
        lick_norm = lick_data / np.max(lick_data[:, :numlaps['Task1']])

        # Plot_traces
        plot_axis = [axis, axis.twinx()]
        colors = sns.color_palette('dark', 2)
        label = ['Mean Correlation', 'Mean Licks']
        for n, d in enumerate([corr_norm, lick_norm]):
            mean = np.mean(d, 0)
            sem = scipy.stats.sem(d, 0)
            if n == 0:
                plot_axis[n].errorbar(np.arange(np.size(mean)), mean, yerr=sem, color=colors[n])

            plot_axis[n].plot(np.arange(np.size(mean)), mean, '.-', color=colors[n])
            plot_axis[n].set_ylabel(label[n], color=colors[n])

        # Get p-values
        for l in np.arange(np.size(correlation_data, 1)):
            d, p = scipy.stats.ranksums(correlation_data[:, l], correlation_data[:, 0])
            if np.round(p, 3) < 0.05:
                axis.plot(l, 0.9, '*', color='k')
            print(l, p)
        for a in plot_axis:
            pf.set_axes_style(axis)
        axis.set_xlabel('Lap Number')

        return correlation_data, lick_data
