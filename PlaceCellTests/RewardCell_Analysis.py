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
from collections import OrderedDict
from copy import copy

# For plotting styles
PlottingFormat_Folder = '/home/sheffieldlab/Desktop/NoReward/Scripts/PlottingTools/'
sys.path.append(PlottingFormat_Folder)
import plottingfunctions as pf


class Combinedpfs:
    def __init__(self, CombinedDataFolder, ParentDataFolder, TaskDict, norewardtask, donotuseanimal, controlflag=0):
        self.CombinedDataFolder = CombinedDataFolder
        self.ParentDataFolder = ParentDataFolder
        self.TaskDict = TaskDict
        self.donotuse = donotuseanimal
        self.controlflag = controlflag
        self.csvfiles_pfs = [f for f in os.listdir(self.CombinedDataFolder) if f.endswith('.csv') if
                             'common' not in f and 'reward' not in f]
        self.npzfiles = [f for f in os.listdir(self.CombinedDataFolder) if f.endswith('.npz')]
        self.npyfiles = [f for f in os.listdir(self.CombinedDataFolder) if f.endswith('.npy')]
        self.trackbins = 5
        self.tracklength = 200
        self.nsecondsroundrew = 1
        self.framerate = 30.98
        self.numanimals = len(self.csvfiles_pfs)
        if not self.controlflag:
            self.norewardtask = norewardtask
            self.csvfiles_reward = [f for f in os.listdir(self.CombinedDataFolder) if f.endswith('.csv') if
                                    'common' not in f and 'place' not in f]
            self.rewardparam_combined = self.combineanimaldataframes(self.csvfiles_reward)
        self.pfparam_combined = self.combineanimaldataframes(self.csvfiles_pfs)

        # Add no lick to taskdict
        self.new_taskDict = copy(self.TaskDict)
        self.new_taskDict['Task2b'] = '3 No Rew No Lick'
        self.new_taskDict = OrderedDict(sorted(self.new_taskDict.items()))

    def combineanimaldataframes(self, csvfiles):
        for n, f in enumerate(csvfiles):
            animalname = f[:f.find('_')]
            if animalname not in self.donotuse:
                print(f)
                df = pd.read_csv(os.path.join(self.CombinedDataFolder, f), index_col=0)
                if n == 0:
                    combined_dataframe = df
                else:
                    combined_dataframe = combined_dataframe.append(df, ignore_index=True)
        return combined_dataframe

    def get_com_allanimal(self, taskA, taskB, vmax=0):
        com_all_animal = np.array([])
        for n, f in enumerate(self.csvfiles_pfs):
            animalname = f[:f.find('_')]
            if animalname not in self.donotuse:
                df = pd.read_csv(os.path.join(self.CombinedDataFolder, f), index_col=0)
                t1 = df[df['Task'] == taskA]
                t2 = df[df['Task'] == taskB]
                combined = pd.merge(t1, t2, how='inner', on=['CellNumber'],
                                    suffixes=(f'_%s' % taskA, f'_%s' % taskB))

                if n == 0:
                    com_all_animal = np.vstack((combined[f'WeightedCOM_%s' % taskA] * self.trackbins,
                                                combined[f'WeightedCOM_%s' % taskB] * self.trackbins))
                else:
                    com_all_animal = np.hstack(
                        (com_all_animal, np.vstack((combined[f'WeightedCOM_%s' % taskA] * self.trackbins,
                                                    combined[f'WeightedCOM_%s' % taskB] * self.trackbins))))

        self.plot_com_scatter_heatmap(com_all_animal, taskA, taskB, vmax=vmax)

    def plot_com_scatter_heatmap(self, combined_dataset, taskA, taskB, bins=10, vmax=0, datatype='array'):
        # Scatter plots
        fs, ax = plt.subplots(1, 2, figsize=(8, 4))
        if datatype == 'array':
            y = combined_dataset[0, :]
            x = combined_dataset[1, :]
        ax[0].scatter(y, x, color='k')
        ax[0].plot([0, self.tracklength], [0, self.tracklength], linewidth=2, color=".3")
        ax[0].set_xlabel(taskB)
        ax[0].set_ylabel(taskA)
        ax[0].set_title('Center of Mass')

        # Heatmap of scatter plot
        heatmap, xedges, yedges = np.histogram2d(y, x, bins=bins)
        heatmap = (heatmap / np.size(y)) * 100
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        if vmax == 0:
            img = ax[1].imshow(heatmap.T, cmap='gray_r', extent=extent, interpolation='bilinear', origin='lower',
                               vmin=0, vmax=np.max(heatmap))
        else:
            img = ax[1].imshow(heatmap.T, cmap='gray_r', extent=extent, interpolation='bilinear', origin='lower',
                               vmin=0, vmax=vmax)

        ax[1].plot([0 + bins, self.tracklength - bins], [0 + bins, self.tracklength - bins], linewidth=2,
                   color=".3")
        axins = CommonFunctions.add_colorbar_as_inset(axes=ax[1])
        if vmax == 0:
            cb = fs.colorbar(img, cax=axins, pad=0.2, ticks=[0, np.int(np.max(heatmap))])
        else:
            cb = fs.colorbar(img, cax=axins, pad=0.2, ticks=[0, vmax])
        cb.set_label('% Field Density', rotation=270, labelpad=12)

        for a in ax:
            pf.set_axes_style(a, numticks=5)

    def plot_mean_rewardcell_pertask(self, reward_data, taskA, taskB):
        fs, ax = plt.subplots(1, len(self.TaskDict), figsize=(10, 3), sharex='all', sharey='all')
        mean_pf_allanimals = {k: [] for k in self.TaskDict.keys()}
        for n, a in enumerate(np.unique(reward_data['AnimalName'])):
            Behaviorfiles, PlaceFieldfiles = self.get_data_folders(os.path.join(self.ParentDataFolder, a))
            df = reward_data[reward_data['AnimalName'] == a]
            lick_stop = Behaviorfiles['lick_stop'].item()[self.norewardtask[0]]
            # Loop through tasks and plot heatmap of cell in all tasks
            x = OrderedDict()
            for t in self.TaskDict.keys():
                pfile = [p for p in PlaceFieldfiles if t in p][0]
                x[t] = scipy.io.loadmat(os.path.join(self.ParentDataFolder, a, 'Behavior', pfile))
            for n1, t in enumerate(self.TaskDict.keys()):
                # Loop through cells
                for l in df.CellNumber:
                    if t in [taskA, taskB]:
                        pfnum = np.asarray(df.loc[df.index[df['CellNumber'] == l]][f'PFNumber_%s' % t])[0]
                        meanpf_thistask = np.nanmean(x[t]['sig_PFs_with_noise'][pfnum - 1][l], 1)
                    elif t in 'Task2':
                        meanpf_thistask = np.nanmean(x[t]['Allbinned_F'][0, l][:, lick_stop:], 1)
                    else:
                        meanpf_thistask = np.nanmean(x[t]['Allbinned_F'][0, l], 1)
                    mean_pf_allanimals[t].append(meanpf_thistask)
                    ax[n1].plot(meanpf_thistask, alpha=0.5, color='grey')
                ax[n1].set_title(t)

        for n1, t in enumerate(self.TaskDict.keys()):
            ax[n1].plot(np.mean(mean_pf_allanimals[t], 0), 'k', lw=2)
        return mean_pf_allanimals

    def plot_all_placefields(self, tasks_to_plot):
        pf_data_all = {k: np.array([]) for k in tasks_to_plot}
        for n, f in enumerate(self.npyfiles):
            animalname = f[:f.find('_')]
            pcdata = np.load(os.path.join(self.CombinedDataFolder, f)).item()
            for t in pcdata.keys():
                if t in tasks_to_plot:
                    if pf_data_all[t].size:
                        pf_data_all[t] = np.vstack((pf_data_all[t], pcdata[t]))
                    else:
                        pf_data_all[t] = pcdata[t]
        # Sort data and plot
        pcsortednum = {k: [] for k in tasks_to_plot}
        for t in tasks_to_plot:
            pcsortednum[t] = np.argsort(np.nanargmax(pf_data_all[t], 1))
        PlottingFunctions.plot_placecells_with_track_pertask(tasks_to_plot, pf_data_all, pcsortednum,
                                                             figsize=(6, 4))

    def plot_histogram_of_com(self, combined_dataframe, tasks_to_plot, bins=20, figsize=(10, 6)):
        fs, ax = plt.subplots(2, len(tasks_to_plot), figsize=figsize, dpi=100, sharex='all', sharey='row')
        # Plot percentage of place fields
        normhist_all = {k: [] for k in tasks_to_plot}
        for a in np.unique(combined_dataframe.animalname):
            for n, taskname in enumerate(tasks_to_plot):
                # Get number of active cells for calculating percentage
                # Best way to normalise?
                normfactor = np.sum(
                    np.load(os.path.join(self.CombinedDataFolder, [f for f in self.npzfiles if a in f][0]))[
                        'numPFs_incells'].item()[taskname])
                # print(a, taskname, normfactor)
                data = combined_dataframe[(combined_dataframe.Task == taskname) & (combined_dataframe.animalname == a)][
                           'WeightedCOM'] * self.trackbins
                hist_com, bins_com, center, width = CommonFunctions.make_histogram(data, bins, normfactor,
                                                                                   self.tracklength)
                normhist_all[taskname].append(hist_com)
                ax[0, n].bar(center, hist_com, align='center', width=width, color='grey')
                ax[0, n].set_title(taskname)
                ax[0, n].set_xlabel('Track Length')

        # Plot absolute numbers
        for n, taskname in enumerate(tasks_to_plot):
            task_data = combined_dataframe[(combined_dataframe.Task == taskname)]['WeightedCOM'] * self.trackbins
            sns.distplot(task_data, ax=ax[1, n], bins=40, color='black',
                         kde=True,
                         kde_kws={'kernel': 'gau', 'bw': 70, 'shade': True, 'cut': 0, 'lw': 0, 'color': [0.6, 0.6, 0.6],
                                  'alpha': 0.9},
                         hist_kws={'histtype': 'step', 'color': 'k', 'lw': 3})
            sns.kdeplot(task_data, ax=ax[1, n], kernel='gau', bw=2, lw=0, cut=0, shade=True, color='k', alpha=0.2)
            ax[0, n].set_ylabel('Percentage of fields')
            ax[1, n].legend_.remove()
        for a in ax.flatten():
            pf.set_axes_style(a, numticks=4)
            # a.set_xlim((0 + 7, self.tracklength - 7))
        fs.tight_layout()
        return normhist_all, center

    def calculate_ratiofiring_atrewzone(self, combined_dataframe, tasks_to_compare, ranges):
        cellratio_df = pd.DataFrame(columns=['Mid', 'End', 'Animal', 'TaskName'])
        cellratio_dict = {k: [] for k in tasks_to_compare}
        for n1, a in enumerate(np.unique(combined_dataframe.animalname)):
            print(a)
            for n2, taskname in enumerate(tasks_to_compare):
                normfactor = np.sum(
                    np.load(os.path.join(self.CombinedDataFolder, [f for f in self.npzfiles if a in f][0]))[
                        'numPFs_incells'].item()[taskname])
                data = combined_dataframe[(combined_dataframe.Task == taskname) & (combined_dataframe.animalname == a)]
                g = data.groupby(pd.cut(data.WeightedCOM * 5, ranges)).count()['WeightedCOM'].tolist()
                cellratio_df = cellratio_df.append({'Mid': np.mean(g[:-1]) / normfactor,
                                                    'End': g[-1] / normfactor,
                                                    'Animal': a, 'TaskName': taskname},
                                                   ignore_index=True)
                cellratio_dict[taskname].append(g / normfactor)
        df = cellratio_df.melt(id_vars=['Animal', 'TaskName'], var_name='Track', value_name='Ratio')
        for taskname in tasks_to_compare:
            cellratio_dict[taskname] = np.asarray(cellratio_dict[taskname])

        fs, ax = plt.subplots(1, dpi=100, figsize=(5, 3))
        sns.barplot(x='TaskName', y='Ratio', data=df[df.Track != 'Beg'],
                    hue='Track', palette='Set2', ax=ax)
        # Plot individual datapoints with a line
        startpnt = -0.25
        for i in tasks_to_compare:
            x = df[(df.Track == 'Mid') & (df.TaskName == i)]['Ratio']
            y = df[(df.Track == 'End') & (df.TaskName == i)]['Ratio']
            ax.plot([startpnt, startpnt + 0.5], [x, y], 'k.-', alpha=0.5)
            startpnt += 1

        pf.set_axes_style(ax)
        ax.set_ylabel('Percentage of fields')
        ax.legend(bbox_to_anchor=(1.2, 1), loc='upper right')
        for i in tasks_to_compare:
            x = df[(df.Track == 'Mid') & (df.TaskName == i)]['Ratio']
            y = df[(df.Track == 'End') & (df.TaskName == i)]['Ratio']
            d, p = scipy.stats.ttest_rel(x, y)
            print(f'%s: T-test : p-value %0.4f' % (i, p))

        return df, cellratio_df, cellratio_dict

    def get_data_folders(self, FolderName):
        PlaceFieldfiles = \
            [f for f in os.listdir(os.path.join(FolderName, 'Behavior')) if
             (f.endswith('.mat') and 'PlaceFields' in f and 'GoodBehavior' in f and 'Task2b' not in f)]
        Behaviorfiles = np.load(os.path.join(FolderName, 'SaveAnalysed', 'behavior_data.npz'),
                                allow_pickle=True)

        return Behaviorfiles, PlaceFieldfiles


class FindRewardCells(Combinedpfs):
    def find_plot_common_end_ofzone_cells(self, taskA, taskB, difference=5):
        com_df = pd.DataFrame(
            columns=['AnimalName', 'CellNumber', f'PFNumber_%s' % taskA, f'PFNumber_%s' % taskB,
                     f'WeightedCOM_%s' % taskA, f'WeightedCOM_%s' % taskB, 'COMdifference'])
        numpcs_inTaskA = 0
        for n, f in enumerate(self.csvfiles_pfs):
            animalname = f[:f.find('_')]
            if animalname not in self.donotuse:
                df = pd.read_csv(os.path.join(self.CombinedDataFolder, f), index_col=0)
                numpcs_inTaskA += len(df[df['Task'] == taskA])
                t1 = df[df['Task'] == taskA]
                t2 = df[df['Task'] == taskB]
                combined = pd.merge(t1, t2, how='inner', on=['CellNumber'],
                                    suffixes=(f'_%s' % taskA, f'_%s' % taskB))
                comtaskA, comtaskB = combined[f'WeightedCOM_%s' % taskA], combined[f'WeightedCOM_%s' % taskB]
                dictionary = {'AnimalName': np.asarray(combined[f'animalname_%s' % taskA]),
                              'CellNumber': np.asarray(combined['CellNumber']),
                              f'PFNumber_%s' % taskA: np.asarray(combined[f'PlaceCellNumber_%s' % taskA]),
                              f'PFNumber_%s' % taskB: np.asarray(combined[f'PlaceCellNumber_%s' % taskB]),
                              f'WeightedCOM_%s' % taskA: np.asarray(comtaskA),
                              f'WeightedCOM_%s' % taskB: np.asarray(comtaskB),
                              'COMdifference': np.asarray(np.abs(comtaskA - comtaskB))}

                com_df = com_df.append(pd.DataFrame.from_dict(dictionary), ignore_index=True)
        # get reward zone firing cell details
        endzone = (self.tracklength / self.trackbins) - 10  # 10 bins or 50cm from end zone
        reward_df = com_df[
            (com_df[f'WeightedCOM_%s' % taskA] > endzone) & (com_df[f'WeightedCOM_%s' % taskB] > endzone) & (
                    com_df['COMdifference'] < difference)].reset_index(drop=True)

        # Remove reward cells that also have a place cell in all the tasks
        temp_reward_df = copy(reward_df)
        for index, r in temp_reward_df.iterrows():
            animal = r['AnimalName']
            CellNum = r['CellNumber']
            thisanimal = self.pfparam_combined[(self.pfparam_combined.animalname == animal) &
                                               (self.pfparam_combined.CellNumber == CellNum)]['NumPlacecells']

            if np.any(thisanimal > 1):
                reward_df = reward_df.drop(index)

        reward_df = reward_df.reset_index(drop=True)
        # Scatter plot and heatmap reward cells to verify
        fs, ax = plt.subplots(1, figsize=(4, 4))
        self.plot_rewardcoms_on_allcoms(com_df, ax, taskA, taskB, color='k')
        self.plot_rewardcoms_on_allcoms(reward_df, ax, taskA, taskB, color='r')

        print(f'Percentage of reward cells %0.2f' % ((len(reward_df) / numpcs_inTaskA) * 100))
        self.numpcs_inTaskA = numpcs_inTaskA
        return reward_df

    def plot_rewardcoms_on_allcoms(self, combined_data, ax, taskA, taskB, color, reward_flag=0):
        if reward_flag:
            y = (combined_data[f'WeightedCOM_%s' % taskA] / self.framerate) - self.nsecondsroundrew
            x = (combined_data[f'WeightedCOM_%s' % taskB] / self.framerate) - self.nsecondsroundrew
            ax.plot([-self.nsecondsroundrew, self.nsecondsroundrew * 2],
                    [-self.nsecondsroundrew, self.nsecondsroundrew * 2], linewidth=2, color=".3")
        else:
            y, x = combined_data[f'WeightedCOM_%s' % taskA] * self.trackbins, combined_data[
                f'WeightedCOM_%s' % taskB] * self.trackbins
            ax.plot([0, self.tracklength], [0, self.tracklength], linewidth=2, color=".3")

        ax.scatter(y, x, color=color)
        ax.set_xlabel(taskB)
        ax.set_ylabel(taskA)
        ax.set_title('Selected Reward Cells')
        pf.set_axes_style(ax, numticks=3)

    def find_rewardcells_atrewardzone(self, taskA, taskB, difference=15):
        # Conditions
        # 1. Cell has COM around reward zone
        # 2. Cell has relatively high precision
        com_df = pd.DataFrame(
            columns=['AnimalName', 'CellNumber', 'Precision_%s' % taskA, 'Precision_%s' % taskB,
                     'Stability_%s' % taskA, 'Stability_%s' % taskB,
                     f'WeightedCOM_%s' % taskA, f'WeightedCOM_%s' % taskB, 'COMdifference'])

        for n, f in enumerate(self.csvfiles_reward):
            animalname = f[:f.find('_')]
            rewardzone = self.nsecondsroundrew * self.framerate
            if animalname not in self.donotuse:
                df = pd.read_csv(os.path.join(self.CombinedDataFolder, f), index_col=0)
                df = df.dropna()
                df = df[(df['Precision'] > 0.01) & (df['Stability'] > 0.1)]
                t1 = df[df['Task'] == taskA]
                t2 = df[df['Task'] == taskB]
                combined = pd.merge(t1, t2, how='inner', on=['CellNumber'],
                                    suffixes=(f'_%s' % taskA, f'_%s' % taskB))
                ptaskA, ptaskB = combined[f'Precision_%s' % taskA], combined[f'Precision_%s' % taskB]
                dictionary = {'AnimalName': np.asarray(combined[f'animalname_%s' % taskA]),
                              'CellNumber': np.asarray(combined['CellNumber']),
                              f'WeightedCOM_%s' % taskA: np.asarray(combined[f'WeightedCOM_%s' % taskA]),
                              f'WeightedCOM_%s' % taskB: np.asarray(combined[f'WeightedCOM_%s' % taskB]),
                              f'Precision_%s' % taskA: np.asarray(combined[f'Precision_%s' % taskA]),
                              f'Precision_%s' % taskB: np.asarray(combined[f'Precision_%s' % taskB]),
                              f'Stability_%s' % taskA: np.asarray(combined[f'Stability_%s' % taskA]),
                              f'Stability_%s' % taskB: np.asarray(combined[f'Stability_%s' % taskB]),
                              'COMdifference': np.asarray(
                                  np.abs(combined[f'WeightedCOM_%s' % taskA] - combined[f'WeightedCOM_%s' % taskB]))}
                com_df = com_df.append(pd.DataFrame.from_dict(dictionary), ignore_index=True)
        reward_df = com_df
        # reward_df = com_df[
        #     (com_df[f'WeightedCOM_%s' % taskA] > rewardzone - rewardzone / 2) & (
        #             com_df[f'WeightedCOM_%s' % taskA] < rewardzone * 2 + rewardzone / 2) & (
        #             com_df[f'WeightedCOM_%s' % taskB] > rewardzone - rewardzone / 2) & (
        #             com_df[f'WeightedCOM_%s' % taskB] < rewardzone * 2 + rewardzone / 2) & (
        #             com_df['COMdifference'] < difference)].reset_index(drop=True)
        reward_df = reward_df.reset_index(drop=True)
        # Scatter plot and heatmap reward cells to verify
        fs, ax = plt.subplots(1, figsize=(4, 4))
        self.plot_rewardcoms_on_allcoms(com_df, ax, taskA, taskB, color='k', reward_flag=1)
        self.plot_rewardcoms_on_allcoms(reward_df, ax, taskA, taskB, color='r', reward_flag=1)

        return reward_df

    def plot_binmean_of_rewardcells(self, reward_data, endzoneflag=0):
        # Load data per animal
        plot_count, axis_count1, axis_count2 = 0, 0, 0
        numaxis = 3
        if endzoneflag:
            plotpdf = PdfPages(os.path.join(self.CombinedDataFolder, 'RewardCellsaroundrewardzone.pdf'))
        else:
            plotpdf = PdfPages(os.path.join(self.CombinedDataFolder, 'RewardCells.pdf'))
        rewarddata_dff = []
        for n, a in enumerate(np.unique(reward_data['AnimalName'])):
            Behaviorfiles, PlaceFieldfiles = self.get_data_folders(os.path.join(self.ParentDataFolder, a))
            df = reward_data[reward_data['AnimalName'] == a]
            numlaps = Behaviorfiles['numlaps'].item()
            lick_stop = Behaviorfiles['lick_stop'].item()[self.norewardtask[0]]
            # Loop through tasks and plot heatmap of cell in all tasks
            if endzoneflag:
                npzfile = np.load(os.path.join(self.CombinedDataFolder, [f for f in self.npzfiles if a in f][0]),
                                  allow_pickle=True)
                reward_dff = npzfile['rewarddata_percell'].item()
            else:
                x = OrderedDict()
                for t in self.TaskDict.keys():
                    pfile = [p for p in PlaceFieldfiles if t in p][0]
                    x[t] = scipy.io.loadmat(os.path.join(self.ParentDataFolder, a, 'Behavior', pfile))

            for l in df.CellNumber:
                # Loop through cells
                data_percell = np.array([])
                for t in self.TaskDict.keys():
                    if endzoneflag:
                        pf_thistask = reward_dff[t][l].T
                        pf_thistask = (pf_thistask - pf_thistask.min()) / (pf_thistask.max() - pf_thistask.min())
                    else:
                        pf_thistask = np.nan_to_num(x[t]['Allbinned_F'][0, l])
                        # normalisePf_task
                        pf_thistask = (pf_thistask - pf_thistask.min()) / (pf_thistask.max() - pf_thistask.min())
                    data_percell = np.hstack((data_percell, pf_thistask)) if data_percell.size else pf_thistask
                data_percell = np.nan_to_num(data_percell)
                rewarddata_dff.append(data_percell)
                # Plot heatmap of cell
                if plot_count == 0 or plot_count == (numaxis ** 2):
                    if plot_count == (numaxis ** 2):
                        plotpdf.savefig(fs, bbox_inches='tight')
                        plt.close()
                    fs, ax = plt.subplots(numaxis, numaxis, figsize=(8, 10))  # Make a new plot for each 9 plots
                    for aa in ax.flatten():
                        aa.axis('off')
                    plot_count, axis_count1, axis_count2 = 0, 0, 0
                ax[axis_count1, axis_count2].imshow(data_percell.T, aspect='auto', cmap='jet',
                                                    vmin=0, vmax=0.5, interpolation='bilinear')
                ax[axis_count1, axis_count2].set_title(f'Animal: %s, Cell: %s' % (a, l))
                if endzoneflag:
                    ax[axis_count1, axis_count2].axvline(self.nsecondsroundrew * self.framerate, ymin=0,
                                                         ymax=np.size(data_percell, 0),
                                                         color='k')
                # Plot a vertical line for end of each task
                lap = 0
                for t in self.TaskDict.keys():
                    lap += numlaps[t]
                    if t == 'Task1':
                        licklap = lap + lick_stop
                        ax[axis_count1, axis_count2].axhline(licklap, color='r', linewidth=2)
                    ax[axis_count1, axis_count2].axhline(lap, color='gray', linewidth=2)

                axis_count1 += 1
                if np.remainder(plot_count + 1, numaxis) == 0:
                    axis_count2 += 1
                    axis_count1 = 0
                plot_count += 1
        # Save last figure
        plotpdf.savefig(fs, bbox_inches='tight')
        plt.close()
        plotpdf.close()

    def get_data_for_correlation(self, reward_data, task_to_correlate='Task1', endzoneflag=0):
        corr_animal = np.array([])
        numlaps_intask_touse = {'Task1': 15, 'Task2': 15, 'Task3': 15,
                                'Task4': 15}  # Get random 10 laps and first 15 for task2
        lickstop_cell = []
        for n, a in enumerate(np.unique(reward_data['AnimalName'])):
            Behaviorfiles, PlaceFieldfiles = self.get_data_folders(os.path.join(self.ParentDataFolder, a))
            df = reward_data[reward_data['AnimalName'] == a]
            numlaps = Behaviorfiles['numlaps'].item()
            lick_stop = Behaviorfiles['lick_stop'].item()[self.norewardtask[0]]
            # Loop through tasks and plot heatmap of cell in all tasks
            if endzoneflag:
                npzfile = np.load(os.path.join(self.CombinedDataFolder, [f for f in self.npzfiles if a in f][0]),
                                  allow_pickle=True)
                reward_dff = npzfile['rewarddata_percell'].item()
            else:
                x = OrderedDict()
                for t in self.TaskDict.keys():
                    pfile = [p for p in PlaceFieldfiles if t in p][0]
                    x[t] = scipy.io.loadmat(os.path.join(self.ParentDataFolder, a, 'Behavior', pfile))

            corr_all = np.array([])
            for c in df.CellNumber:
                # Loop through cells
                if endzoneflag:
                    correlation_mean = np.nanmean(reward_dff[task_to_correlate][c].T, 1)
                else:
                    correlation_mean = np.nanmean(x[task_to_correlate]['Allbinned_F'][0, c], 1)
                corr = []
                for t in ['Task1', 'Task2', 'Task3', 'Task4']:
                    if endzoneflag:
                        pf_thistask = np.nan_to_num(reward_dff[t][c].T)
                    else:
                        pf_thistask = np.nan_to_num(x[t]['Allbinned_F'][0, c])
                    # Lap by lap correlation
                    if t == 'Task2':
                        laps = np.arange(numlaps_intask_touse[t])
                    else:
                        laps = np.random.choice(np.arange(np.size(pf_thistask, 1)), numlaps_intask_touse[t])
                    for l in laps:
                        temp = np.corrcoef(pf_thistask[:, l], correlation_mean)[0, 1]
                        if ~np.isnan(temp):
                            corr.append(temp)
                        else:
                            corr.append(0)
                lickstop_cell.append(lick_stop)
                corr_all = np.vstack((corr_all, np.asarray(corr))) if corr_all.size else np.asarray(corr)
            corr_animal = np.vstack((corr_animal, corr_all)) if corr_animal.size else corr_all
        print(np.shape(corr_animal), np.shape(lickstop_cell))
        # Plot all animals separatly for now
        # corr_animal = corr_animal[np.argsort(np.nanmean(corr_animal, 1))[::-1], :]
        fs, ax = plt.subplots(1, figsize=(5, 3))
        ax.imshow(corr_animal, aspect='auto', interpolation='nearest',
                  vmin=0, vmax=1)
        ax.axis('off')
        # Plot a vertical line for end of each task
        lap = 0
        for t in ['Task1', 'Task2', 'Task3']:
            lap += numlaps_intask_touse[t]
            ax.axvline(lap, color='blue', linewidth=2)
        for i, l in enumerate(lickstop_cell):
            ax.plot(numlaps_intask_touse['Task1'] + l, i, '|', color='r', markersize=7, markeredgewidth=3)

        return corr_animal

    def get_mean_correlation_withtaskA(self, reward_data, task_to_correlate='Task1', endzoneflag=0):
        mean_corr_task = {k: [] for k in self.new_taskDict}
        lickstop_cell = []
        for n, a in enumerate(np.unique(reward_data['AnimalName'])):
            Behaviorfiles, PlaceFieldfiles = self.get_data_folders(os.path.join(self.ParentDataFolder, a))
            df = reward_data[reward_data['AnimalName'] == a]
            numlaps = Behaviorfiles['numlaps'].item()
            lick_stop = Behaviorfiles['lick_stop'].item()[self.norewardtask[0]]
            # Loop through tasks and plot heatmap of cell in all tasks
            if endzoneflag:
                npzfile = np.load(os.path.join(self.CombinedDataFolder, [f for f in self.npzfiles if a in f][0]),
                                  allow_pickle=True)
                reward_dff = npzfile['rewarddata_percell'].item()
            else:
                x = OrderedDict()
                for t in self.TaskDict.keys():
                    pfile = [p for p in PlaceFieldfiles if t in p][0]
                    x[t] = scipy.io.loadmat(os.path.join(self.ParentDataFolder, a, 'Behavior', pfile))

            corr_all = np.array([])
            for c in df.CellNumber:
                # Loop through cells
                if endzoneflag:
                    correlation_mean = np.nanmean(reward_dff[task_to_correlate][c].T, 1)
                else:
                    correlation_mean = np.nanmean(x[task_to_correlate]['Allbinned_F'][0, c], 1)
                corr = []
                for t in self.TaskDict.keys():
                    if endzoneflag:
                        pf_thistask = np.nan_to_num(reward_dff[t][c].T)
                    else:
                        pf_thistask = np.nan_to_num(x[t]['Allbinned_F'][0, c])
                    # Lap by lap correlation
                    for l in np.arange(np.size(pf_thistask, 1)):
                        temp = np.corrcoef(pf_thistask[:, l], correlation_mean)[0, 1]
                        if ~np.isnan(temp):
                            corr.append(temp)
                    if t == 'Task2':
                        mean_corr_task[t].append(np.nanmean(corr[:lick_stop]))
                        mean_corr_task['Task2b'].append(np.nanmean(corr[lick_stop:]))
                    else:
                        mean_corr_task[t].append(np.nanmean(corr))

        return mean_corr_task

    def plot_stability_precision_of_reward_cells(self, reward_data):
        for index, row in reward_data.iterrows():
            animalname = row['AnimalName']
            cellnumber = row['CellNumber']
            thiscell_data = self.rewardparam_combined[(self.rewardparam_combined['animalname'] == animalname) & (
                    self.rewardparam_combined['CellNumber'] == cellnumber)]
            if index == 0:
                combinedf = thiscell_data
            else:
                combinedf = pd.concat((combinedf, thiscell_data))
        columns_to_plot = ['Precision', 'Stability']
        fs, ax = plt.subplots(1, len(columns_to_plot), figsize=(7, 3))
        for n, c in enumerate(columns_to_plot):
            sns.boxplot(x='Task', y=c, data=combinedf, ax=ax[n], order=self.TaskDict.keys(), palette='Blues',
                        showfliers=False)
            # sns.stripplot(x='Task', y=c, data=combinedf, ax=ax[n], color='k', size=2)
            pf.set_axes_style(ax[n])
        fs.tight_layout()
        return combinedf

    def plot_mean_correlation_with_taskA(self, reward_data1, reward_data2):
        # Plot mean_correlation
        df1, df2 = pd.DataFrame.from_dict(reward_data1), pd.DataFrame.from_dict(reward_data1)
        df1 = pd.melt(df1, value_name='mean_correlation', var_name='Task')
        df2 = pd.melt(df2, value_name='mean_correlation', var_name='Task')
        df = pd.concat((df1, df2))
        fs, ax = plt.subplots(1, figsize=(5, 3))
        sns.boxplot(x='Task', y='mean_correlation', data=df, palette='Blues', ax=ax, showfliers=False)
        sns.stripplot(x='Task', y='mean_correlation', data=df, color='k', size=2, ax=ax)
        pf.set_axes_style(ax)

class CommonFunctions:
    @staticmethod
    def create_data_dict(TaskDict):
        data_dict = {keys: [] for keys in TaskDict.keys()}
        return data_dict

    @staticmethod
    def add_colorbar_as_inset(axes):
        axins = inset_axes(axes,
                           width="5%",  # width = 5% of parent_bbox width
                           height="50%",  # height : 50%
                           loc='lower left',
                           bbox_to_anchor=(1.05, 0., 1, 1),
                           bbox_transform=axes.transAxes,
                           borderpad=0.5,
                           )
        return axins

    @staticmethod
    def make_histogram(com, bins, normalisefactor, tracklength):
        hist_com, bins_com = np.histogram(com, bins=np.arange(0, tracklength + 5, bins))
        hist_com = (hist_com / np.sum(normalisefactor)) * 100
        width = np.diff(bins_com)
        center = (bins_com[:-1] + bins_com[1:]) / 2
        return hist_com, bins_com, center, width
