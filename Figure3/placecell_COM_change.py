import scipy.io
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import sys
import scipy.stats
from scipy.signal import savgol_filter
from _collections import OrderedDict
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# For plotting styles
PlottingFormat_Folder = '/home/sheffieldlab/Desktop/NoReward/Scripts/PlottingTools/'
sys.path.append(PlottingFormat_Folder)
import plottingfunctions as pf

DataDetailsFolder = '/home/sheffieldlab/Desktop/NoReward/Scripts/AnimalDetails/'
sys.path.append(DataDetailsFolder)
import DataDetails

PvaluesFolder = '/home/sheffieldlab/Desktop/NoReward/Scripts/Figure1/'
sys.path.append(PvaluesFolder)
from Pvalues import GetPValues


class GetData(object):
    def __init__(self, FolderName, CombinedDataFolder, LickFolder):
        self.FolderName = FolderName
        self.CombinedDataFolder = CombinedDataFolder
        self.LickFolder = LickFolder
        self.animals = [f for f in os.listdir(self.FolderName) if
                        f not in ['LickData', 'BayesResults_All', 'SaveAnalysed', 'PlaceCellResults_All']]
        self.trackbins = 5
        self.tracklength = 200

    def get_correlation_withtask(self, taskstoplot, basetask='Task1', laps=14, controlflag=0):
        combine_COM = {keys: [] for keys in taskstoplot}
        combine_correlation = {keys: [] for keys in taskstoplot}
        for a in self.animals:
            if controlflag:
                animalinfo = DataDetails.ControlAnimals(a)
            else:
                animalinfo = DataDetails.ExpAnimalDetails(a)
                lickstop_df = pd.read_csv(os.path.join(self.LickFolder, 'Lickstops.csv'), index_col=0)
                lickstop = lickstop_df.loc[a, lickstop_df.columns[1]]
                if lickstop < 4:
                    continue

            p = np.load(
                os.path.join(self.FolderName, a, 'PlaceCells', f'%s_placecell_data.npz' % a),
                allow_pickle=True)
            corr = p['correlation_withTask1'].item()
            if np.size(corr[taskstoplot[0]], 1) > 20:
                sig_PFs = p['sig_PFs_cellnum_revised'].item()

                params_PFs = pd.read_csv(
                    os.path.join(self.CombinedDataFolder, f'%s_placecellparams_df.csv' % a), index_col=0)

                for t in taskstoplot:
                    if t in animalinfo['task_dict']:
                        correlation_task = corr[t]
                        print('%s %s Laps: %d' % (a, t, np.size(corr[t], 1)))
                        com_task1 = params_PFs[(params_PFs.Task == basetask)]
                        for cell in sig_PFs[basetask]:
                            com_cell = np.asarray(com_task1.loc[com_task1['CellNumber'] == cell]['WeightedCOM'])[:,
                                       np.newaxis]
                            for i in np.arange(len(com_cell)):
                                if controlflag:
                                    thislaps = list(np.arange(laps + 1))
                                    thislaps.pop(7)
                                    combine_correlation[t].append(correlation_task[cell, thislaps])
                                elif t in ['Task1', 'Task3']:
                                    combine_correlation[t].append(correlation_task[cell, :laps])
                                elif t == 'Task2':
                                    combine_correlation[t].append(
                                        correlation_task[cell, lickstop - 4:lickstop + (laps - 4)])
                                combine_COM[t].extend(com_cell[i])

        combine_corr_array = {k: np.asarray([]) for k in taskstoplot}
        for t in taskstoplot:
            for i in combine_correlation[t]:
                combine_corr_array[t] = np.vstack((combine_corr_array[t], i)) if combine_corr_array[t].size else i
        return combine_COM, combine_corr_array

    def get_correlation_withslidingwindow(self, taskstoplot, totallaps=30, slidingwindow=2, basetask='Task1',
                                          controlflag=0):
        combine_COM = {keys: [] for keys in taskstoplot}
        combine_correlation = {keys: [] for keys in taskstoplot}
        count = 0
        for a in self.animals:
            print(a)
            if controlflag:
                animalinfo = DataDetails.ControlAnimals(a)

            else:
                animalinfo = DataDetails.ExpAnimalDetails(a)
                lickstop_df = pd.read_csv(os.path.join(self.LickFolder, 'Lickstops.csv'), index_col=0)
                lickstop = lickstop_df.loc[a, lickstop_df.columns[1]]
                if lickstop < 4:
                    continue
                # if a not in ['CFC17', 'NR14', 'NR15', 'CFC4', 'NR6', 'NR21']:
                #     continue

            p = np.load(
                os.path.join(self.FolderName, a, 'PlaceCells', f'%s_placecell_data.npz' % a),
                allow_pickle=True)
            corr = p['correlation_withTask1'].item()
            sig_PFs = p['sig_PFs_cellnum_revised'].item()

            params_PFs = pd.read_csv(
                os.path.join(self.CombinedDataFolder, f'%s_placecellparams_df.csv' % a), index_col=0)

            for t in taskstoplot:
                if t in animalinfo['task_dict']:
                    print('%s %s Laps: %d' % (a, t, np.size(corr[t], 1)))
                    correlation_task = corr[t]
                    if t == 'Task1b':
                        correlation_task[:, 6:8] = corr['Task1b'][:, 22:24]

                    com_task1 = params_PFs[(params_PFs.Task == basetask)]
                    numcells = len(com_task1.loc[com_task1['CellNumber'].isin(sig_PFs[basetask])])
                    corr_animal = np.zeros((numcells, totallaps - slidingwindow))
                    n = 0
                    print(a, t, np.mean(correlation_task[sig_PFs[basetask], 0]))
                    for cell in sig_PFs[basetask]:
                        com_cell = np.asarray(com_task1.loc[com_task1['CellNumber'] == cell]['WeightedCOM'])[:,
                                   np.newaxis]

                        for l in np.arange(len(com_cell)):

                            for b in np.arange(0, totallaps - slidingwindow):
                                if b == 0 and t == 'Task2':
                                    corr_animal[n, 0] = correlation_task[cell, 0]
                                else:
                                    corr_animal[n, b] = np.nanmean(correlation_task[cell, b:b + slidingwindow])
                            n += 1
                            combine_COM[t].extend(com_cell[l])
                    combine_correlation[t].append(corr_animal)

        combine_corr_array = {k: np.asarray([]) for k in taskstoplot}
        for t in taskstoplot:
            for i in combine_correlation[t]:
                combine_corr_array[t] = np.vstack((combine_corr_array[t], i)) if combine_corr_array[t].size else i

        return combine_COM, combine_corr_array


class PlotFigures(object):
    @staticmethod
    def plot_correlation_with_com(ax, com, corr, taskstoplot):
        com_task = np.asarray(com[taskstoplot])
        sort_com = np.argsort(com_task)

        sorted_corr = corr[taskstoplot][sort_com, :]
        sorted_corr_arranged = np.zeros_like(sorted_corr)
        count = 0
        for i in np.arange(1, 40):
            com_idx = np.where(np.round(com_task[sort_com]) == i)[0]
            temp = sorted_corr[com_idx, :]
            temp1 = np.argsort(np.max(temp, 1))[::-1]
            sorted_corr_arranged[count:count + len(com_idx), :] = temp[temp1, :]
            count += len(com_idx)

        ax[0].imshow(sorted_corr_arranged, aspect='auto', interpolation='bilinear',
                     cmap='YlGnBu', vmin=0, vmax=0.8)
        ax[1].plot(com_task[sort_com], np.arange(np.size(com_task)))
        ax[0].set_ylabel('Cell')
        ax[0].set_xlabel('Lap')
        ax[0].set_title(taskstoplot)
        ax[1].set_xlabel('Track length(cm)')

        for a in ax:
            a.set_ylim((np.size(com_task), 0))

        pf.set_axes_style(ax[1], numticks=4)

    @staticmethod
    def get_correlation_withlap(ax, com_data, correlation_data, tasktoplot, controlflag=0, sliding_window=0):
        com = np.asarray(com_data[tasktoplot])
        y = correlation_data[tasktoplot]
        # y[y < 0] = np.nan

        laps = np.size(y, 1)
        mean_binned = np.zeros(laps)
        error = np.zeros(laps)
        type = ['Beg', 'Mid', 'End']
        color = ['black', 'blue', 'red']
        for n, i in enumerate([[0, 10], [25, 35], [36, 40]]):
            ybin = y[np.where((com >= i[0]) & (com <= i[1]))[0], :]
            for l in np.arange(laps):
                mean_binned[l], error[l] = np.nanmean(ybin[:, l]), scipy.stats.sem(ybin[:, l], nan_policy='omit')
            mean_binned = (mean_binned - np.min(mean_binned)) / (np.max(mean_binned) - np.min(mean_binned))
            # print(mean_binned)

            if controlflag:
                ax.errorbar(np.arange(laps), mean_binned, yerr=error, fmt='o--', color=color[n], ecolor=color[n],
                            markerfacecolor='none',
                            label='Control  %s' % type[n])
            else:
                ax.errorbar(np.arange(laps), mean_binned, yerr=error, fmt='o-', color=color[n], ecolor=color[n],
                            label='Exp  %s' % type[n])
        if tasktoplot == 'Task2' and sliding_window:
            ax.axvline(9, color='k', linestyle='--', label='lickstop')
        if tasktoplot == 'Task2' and not sliding_window:
            ax.axvline(4, color='k', linestyle='--', label='lickstop')
        ax.set_title(tasktoplot)
        ax.set_xlabel('Laps')
