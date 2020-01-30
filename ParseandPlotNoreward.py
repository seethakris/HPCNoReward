import scipy.io
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from copy import copy
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import find_peaks
import pandas as pd
from scipy.stats import norm
import pickle

sns.set_context('paper', font_scale=1.4)
sns.set_palette(sns.color_palette('muted'))
sns.set_color_codes('muted')

""" There are 4 types of data in this paradigm - VR1 Reward, VR1 Noreward, VR1 Reward and Novel Reward 
"""


class GetData(object):
    def __init__(self, FolderName, Task_NumFrames, TaskDict):

        self.FolderName = FolderName
        self.Task_Numframes = Task_NumFrames
        self.FigureFolder = os.path.join(self.FolderName, 'Figures')
        self.SaveFolder = os.path.join(self.FolderName, 'SaveAnalysed')

        if not os.path.exists(self.FigureFolder):
            os.mkdir(self.FigureFolder)
        if not os.path.exists(self.SaveFolder):
            os.mkdir(self.SaveFolder)

        self.TaskDict = TaskDict
        self.ImgFileName = [f for f in os.listdir(FolderName) if f.endswith('.mat')]
        self.BehFileName = [f for f in os.listdir(os.path.join(FolderName, 'Behavior')) if
                            f.endswith('.mat') and 'PlaceFields' not in f and 'plain1' not in f]

        self.PlaceFieldData = \
            [f for f in os.listdir(os.path.join(FolderName, 'Behavior')) if
             (f.endswith('.mat') and 'PlaceFields' in f)]

        # Create a number of dicts for storing files trial wise

        self.Fcdata_dict = self.create_data_dict()
        self.Fdata_dict = self.create_data_dict()
        self.Fc3data_dict = self.create_data_dict()
        self.sig_PFs_cellnum_dict = self.create_data_dict()
        self.running_data = self.create_data_dict()
        self.lick_data = self.create_data_dict()
        self.timestamps = self.create_data_dict()


        self.numlaps = self.find_numlaps_by_task()
        self.numcells = self.load_images()
        self.load_behdata()

        self.lick_flag, self.lap_time = self.find_lick_parameters()

        for n, i in enumerate(self.lick_flag['Task2']):
            if i == 0 and n > 0:
                self.lickstop_lap = n
                break
        print('Animal stops licking in Task2 at lap %d' % self.lickstop_lap)

        self.reward_frame, self.rewardregion_fluor_data, self.cells_firing_to_reward = self.create_data_during_reward()
        self.PFParams_df = self.create_df_placefield_params()
        self.find_sig_PFs_cellnum_bytask()

        self.save_parsed_data()

    def load_images(self):
        # Open calcium data and store in dicts per trial
        data = scipy.io.loadmat(os.path.join(self.FolderName, self.ImgFileName[0]))
        numcells = np.size(data['data'].item()[1], 1)
        count = 0
        for i in range(0, len(self.Task_Numframes)):
            self.Fcdata_dict[f'Task%d' % (i + 1)] = data['data'].item()[1].T[:, count:count + self.Task_Numframes[i]]
            self.Fc3data_dict[f'Task%d' % (i + 1)] = data['data'].item()[2].T[:, count:count + self.Task_Numframes[i]]
            count += self.Task_Numframes[i]

        return numcells

    def load_behdata(self):
        for i in self.BehFileName:
            taskname = i[i.find('Task'):i.find('Task') + 5]
            x = scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', i))
            self.lick_data[taskname] = x['session'].item()[0][0][0][3]
            self.running_data[taskname] = x['session'].item()[0][0][0][0]
            # self.timestamps[taskname] = x['session'].item()[0][0][0][5].T



    def find_lick_parameters(self):
        # find start and end of a lap and see if there is a lick
        lickflag = {keys: [] for keys in self.TaskDict.keys()}
        laptime = {keys: [] for keys in self.TaskDict.keys()}
        for i in self.PlaceFieldData:
            taskname = i[i.find('Task'):i.find('Task') + 5]
            x = scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', i))

            lapframes = x['E'].T
            for l in range(1, np.max(lapframes) + 1):
                laps = np.where(lapframes == l)[0]
                licks = self.lick_data[taskname][laps]
                laptime[taskname].append(np.size(laps) / 30.98)
                if np.any(licks > 2):
                    lickflag[taskname].append(1)
                else:
                    lickflag[taskname].append(0)

        return lickflag, laptime

    def create_data_dict(self):
        data_dict = {f'Task%d' % keys: [] for keys in range(1, len(self.Task_Numframes) + 1)}
        return data_dict

    def find_numlaps_by_task(self):
        laps = {keys: [] for keys in self.TaskDict.keys()}
        for i in self.PlaceFieldData:
            taskname = i[i.find('Task'):i.find('Task') + 5]
            x = scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', i))
            laps[taskname] = np.size(x['Allbinned_F'][0, 0], 1)

            print('Number of laps in %s is %d' % (taskname, laps[taskname]))
        return laps

    def find_sig_PFs_cellnum_bytask(self):
        for i in self.PlaceFieldData:
            taskname = i[i.find('Task'):i.find('Task') + 5]
            x = scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', i))
            tempx = np.asarray(np.nan_to_num(x['number_of_PFs'])).T
            # Correct tempx
            for n in range(0, np.size(tempx)):
                if len(x['sig_PFs'][0, n]) == 1:
                    tempx[n] = 0
            tempx = tempx.T

            print('Number of PlaceCells in %s is %d' % (taskname, np.size(np.where(tempx == 1)[1])))
            self.sig_PFs_cellnum_dict[taskname] = np.where(tempx == 1)[1]

    def create_df_placefield_params(self):
        pf_dict = {keys: [] for keys in
                   ['Task', 'CellNum', 'PF_width', 'PF_startbin', 'PF_endbin', 'Place_field_num', 'Center_of_mass']}

        for i in self.PlaceFieldData:
            x = scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', i))
            self.numcells = np.size(x['number_of_PFs'], 1)

            for n in range(0, self.numcells):  # Loop by cell
                if np.nan_to_num(x['number_of_PFs'][0][n]) != 0 and len(x['sig_PFs'][0, n]) > 1:
                    num_pfs = np.int(x['number_of_PFs'][0][n])
                    for p in range(0, num_pfs):  # Loop by number of placefields in cell
                        # print(i, n, p, num_pfs, len(x['sig_PFs'][p, n]))
                        pf_dict['Task'].append(self.TaskDict[i[i.find('Task'):i.find('Task') + 5]])
                        pf_dict['CellNum'].append(n)
                        pf_dict['PF_width'].append(x['PF_width'][p, n])
                        pf_dict['PF_startbin'].append(x['PF_start_bins'][p, n])
                        pf_dict['PF_endbin'].append(x['PF_end_bins'][p, n])
                        pf_dict['Place_field_num'].append(f'PF_%d' % (p + 1))
                        pf_dict['Center_of_mass'].append(np.nanargmax(np.mean(np.nan_to_num(x['sig_PFs'][p, n]), 1)))

        return pd.DataFrame.from_dict(pf_dict)

    def plot_pf_params(self):
        sns.catplot(x="PF_width", y="Task", order=list(self.TaskDict.values()),
                    kind="box", orient="h", height=3, aspect=2,
                    data=self.PFParams_df[self.PFParams_df['PF_width'] < 100])
        sns.catplot(x="Center_of_mass", y="Task", order=list(self.TaskDict.values()),
                    kind="box", orient="h", height=3, aspect=2,
                    data=self.PFParams_df)
        g = sns.catplot(x="Place_field_num", hue='Task', hue_order=list(self.TaskDict.values()),
                        kind="count", height=3, aspect=2, data=self.PFParams_df)
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle(f'Number of active cells = %d' % self.numcells, fontsize=10)

        fs, ax = plt.subplots(1, figsize=(8, 4))
        g = sns.pointplot(x="Task", y="Center_of_mass", hue="CellNum", kind="point",
                          order=list(self.TaskDict.values()), data=self.PFParams_df, palette="Blues", ci=None, ax=ax)
        g.legend_.remove()

    def plot_behavior(self):
        fs, axes = plt.subplots(len(self.TaskDict), 1, figsize=(10, 10), sharex='all', sharey='all')
        for n, i in enumerate(self.TaskDict.keys()):
            axes[n].plot(self.running_data[i], linewidth=2)
            axes[n].plot(self.lick_data[i] / 4, linewidth=1, alpha=0.5)
            axes[n].set_title(self.TaskDict[i])

        axes[n].set_xlabel('Time (seconds)')
        fs.tight_layout()

    def plot_remapping_withTaskA(self, TaskA='Task1'):
        placecells_formapping = self.sig_PFs_cellnum_dict[TaskA]
        img_dict = {keys: [] for keys in self.TaskDict.keys()}
        for i in self.PlaceFieldData:
            taskname = i[i.find('Task'):i.find('Task') + 5]
            x = scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', i))
            bins = np.size(x['Allbinned_F'][0, 0], 0)
            img = np.zeros((np.size(placecells_formapping), bins))

            for n, c in enumerate(placecells_formapping):
                img[n, :] = np.mean(np.nan_to_num(x['Allbinned_F'][0, c]), 1)

            img_dict[taskname] = img

        # Sort by task
        img_argsort = np.argsort(np.nanargmax(img_dict[TaskA], 1))
        fs, ax1 = plt.subplots(1, len(self.TaskDict), figsize=(10, 4))
        for n, i in enumerate(self.TaskDict.keys()):
            ax1[n].imshow(img_dict[i][img_argsort, :], aspect='auto', cmap='cool', interpolation='nearest', vmin=0,
                          vmax=0.5)
            if n > 0:
                ax1[n].axis('off')
            ax1[n].set_title(self.TaskDict[i])
        ax1[0].set_xlabel('Bins')
        ax1[0].set_ylabel('Cell #')
        ax1[0].locator_params(nbins=4)
        fs.suptitle(f'Remapping aligned to %s' % TaskA, fontsize=10)
        plt.savefig(f'Remapping_with_%s.pdf' % TaskA, bbox_inches='tight', dpi=300)
        plt.show()

        return img_dict

    def plot_cell_by_cell_activity_duringlaps(self):
        Pdf = PdfPages(os.path.join(self.FigureFolder, 'BinMean_velocity_percell.pdf'))
        task_axes = {'Task1': 0, 'Task2': 2, 'Task3': 4, 'Task4': 6}

        with sns.axes_style('dark'):
            for c in range(0, self.numcells):
                fs = plt.figure(figsize=(10, 3))
                gs = plt.GridSpec(1, len(self.TaskDict) * 2, width_ratios=[3, 1, 3, 1, 3, 1, 3, 1])
                for i in self.PlaceFieldData:
                    taskname = i[i.find('Task'):i.find('Task') + 5]
                    x = scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', i))

                    # Plot bin mean
                    ax1 = fs.add_subplot(gs[task_axes[taskname]])
                    ax1.imshow(np.nan_to_num(x['Allbinned_F'][0, c]).T, interpolation='nearest',
                               aspect='auto', cmap='cool', vmin=0, vmax=0.5)
                    ax1.set_ylim((self.numlaps[taskname] - 1, -0.1))

                    # Plot lap traversal time
                    ax2 = fs.add_subplot(gs[task_axes[taskname] + 1])
                    ax2.barh(np.arange(self.numlaps[taskname]), self.lap_time[taskname])
                    ax2.set_ylim((self.numlaps[taskname] - 1, -0.1))
                    ax2.set_xlim((0, max(self.lap_time['Task2'])))
                    ax2.set_yticklabels([])
                    ax2.tick_params(left=False, right=False, bottom=False)

                    for ax in [ax1, ax2]:
                        ax.spines['right'].set_visible(False)
                        ax.spines['top'].set_visible(False)
                        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                                     ax.get_xticklabels() + ax.get_yticklabels()):
                            item.set_fontsize(7)

                    # Pretify
                    if task_axes[taskname] == 0:
                        ax1.set_xlabel('Bins')
                        ax1.set_ylabel('Laps')
                        ax2.set_xlabel('Lap Time(s)')
                    else:
                        ax1.spines['left'].set_visible(False)
                        ax1.tick_params(left=False, right=False, bottom=False)
                        ax1.set_yticklabels([])
                        ax1.set_xticklabels([])

                    if taskname == 'Task2':
                        ax1.axhline(self.lickstop_lap, linewidth=2, color='k')

                    if c in self.sig_PFs_cellnum_dict[taskname]:
                        ax1.set_title(f'%s : Cell %d is a placecell' % (self.TaskDict[taskname], c), fontsize=8)
                    else:
                        ax1.set_title(f'%s : Cell %d ' % (self.TaskDict[taskname], c), fontsize=8)

                fs.subplots_adjust(wspace=0.05)
                Pdf.savefig(fs, bbox_inches='tight')
                plt.close()
        Pdf.close()

    def create_data_during_reward(self):
        reward_frame = {keys: [] for keys in self.TaskDict.keys()}
        reward_data = {keys: [] for keys in self.TaskDict.keys()}
        reward_cell_num = {keys: [] for keys in self.TaskDict.keys()}

        for i in self.PlaceFieldData:
            taskname = i[i.find('Task'):i.find('Task') + 5]
            x = scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', i))
            lap_frame = x['E'].T
            data = self.Fc3data_dict[taskname]

            # Get frames where reward was presented first
            lap_end_frame = np.zeros((np.max(lap_frame)), dtype=np.int)
            for l in range(1, np.max(lap_frame) + 1):
                lap_end_frame[l - 1] = np.where(lap_frame == l)[0][-1] - 50  # About 1.5s from end of lap
                reward_frame[taskname].append(lap_end_frame[l - 1])

            # Parse and store data around reward
            rewdata_lap = np.zeros((self.numcells, np.max(lap_frame), 120))  # About 4 seconds of data
            for c in range(0, self.numcells):
                for l in range(0, np.max(lap_frame)):
                    if np.size(data[:, lap_end_frame[l] - 60:lap_end_frame[l] + 60], 1) == 120:
                        rewdata_lap[:, l, :] = data[:, lap_end_frame[l] - 60:lap_end_frame[l] + 60]

                x_axis = np.arange(-60, 60, 1)
                gaussian_at_reward = norm.pdf(x_axis, 0, 10) * 10

                corr_at_reward = np.corrcoef(gaussian_at_reward, np.mean(rewdata_lap[c, :, :], 0))[0, 1]
                if corr_at_reward > 0.5:
                    reward_cell_num[taskname].append(c)
            reward_data[taskname] = rewdata_lap
        return reward_frame, reward_data, reward_cell_num

    def plot_cell_by_cell_activity_duringreward(self):
        Pdf = PdfPages(os.path.join(self.FigureFolder, 'Rewardregion_activity_percell.pdf'))
        task_axes = {'Task1': 0, 'Task2': 1, 'Task3': 2, 'Task4': 3}
        for c in range(0, self.numcells):
            fs, axes = plt.subplots(1, len(self.TaskDict), figsize=(10, 3))

            for i in self.TaskDict.keys():
                taskname = i
                data = self.rewardregion_fluor_data[taskname][c, :, :]
                ax1 = axes[task_axes[taskname]]
                ax1.imshow(np.nan_to_num(data), interpolation='nearest',
                           aspect='auto', cmap='cool', vmin=0, vmax=0.5)
                ax1.set_ylim((self.numlaps[taskname] - 1, -0.1))
                ax1.axvline(60, linewidth=2, color='grey')
                ax1.set_title(f'%s : Cell %d ' % (self.TaskDict[taskname], c), fontsize=8)
                for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                             ax1.get_xticklabels() + ax1.get_yticklabels()):
                    item.set_fontsize(7)
                if taskname == 'Task2':  # Lap where lick stops
                    ax1.axhline(self.lickstop_lap, linewidth=2, color='k')

                # Pretify
                if task_axes[taskname] == 0:
                    ax1.set_xlabel('Time (seconds)')
                    ax1.set_ylabel('Laps')
                    ax1.set_xticks([0, 60, 119])
                    ax1.set_xticklabels([-2, 0, 2])
                else:
                    ax1.spines['right'].set_visible(False)
                    ax1.spines['top'].set_visible(False)
                    ax1.tick_params(left=False, right=False, bottom=False)
                    ax1.set_yticklabels([])
                    ax1.set_xticklabels([])

            Pdf.savefig(fs, bbox_inches='tight')
            plt.close()
        Pdf.close()

    def correlate_acivity_of_placecellbytask(self, TaskA='Task1'):
        placecells_formapping = self.sig_PFs_cellnum_dict[TaskA]
        data_formapping = [i for i in self.PlaceFieldData if TaskA in i][0]
        data_formapping = scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', data_formapping))['sig_PFs']

        correlation_per_task = {keys: [] for keys in self.TaskDict.keys()}
        for i in self.PlaceFieldData:
            taskname = i[i.find('Task'):i.find('Task') + 5]
            x = scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', i))
            laps = np.size(x['Allbinned_F'][0, 0], 1)
            corr = np.zeros((np.size(placecells_formapping), laps))

            for n, c in enumerate(placecells_formapping):
                data1 = np.mean(np.nan_to_num(data_formapping[0, c]), 1)
                data2 = np.nan_to_num(x['Allbinned_F'][0, c])
                if len(data2) > 1:
                    for l in range(0, laps):
                        corr[n, l] = np.corrcoef(data2[:, l], data1)[0, 1]

            corr = np.nan_to_num(corr)
            correlation_per_task[taskname] = corr

        self.plot_correlation_by_task(correlation_per_task, Task=TaskA)
        return correlation_per_task

    def correlate_acivity_of_allcellsbytask(self, TaskA='Task1'):
        data_formapping = [i for i in self.PlaceFieldData if TaskA in i][0]
        data_formapping = scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', data_formapping))['Allbinned_F']

        correlation_per_task = {keys: [] for keys in self.TaskDict.keys()}
        for i in self.PlaceFieldData:
            taskname = i[i.find('Task'):i.find('Task') + 5]
            x = scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', i))
            laps = np.size(x['Allbinned_F'][0, 0], 1)
            corr = np.zeros((self.numcells, laps))

            for c in range(0, self.numcells):
                data1 = np.mean(np.nan_to_num(data_formapping[0, c]), 1)
                data2 = np.nan_to_num(x['Allbinned_F'][0, c])
                for l in range(0, laps):
                    temp = np.corrcoef(data2[:, l], data1)[0, 1]
                    if ~np.isnan(temp):
                        corr[c, l] = temp

            corr = np.nan_to_num(corr)
            correlation_per_task[taskname] = corr

        self.plot_correlation_by_task(correlation_per_task, Task=TaskA)
        return correlation_per_task

    def correlate_activity_of_rewardzonebytask(self, TaskA='Task1'):
        data_formapping = self.rewardregion_fluor_data[TaskA]

        correlation_per_task = {keys: [] for keys in self.TaskDict.keys()}
        for i in self.TaskDict:
            taskname = i
            laps = self.numlaps[taskname]
            corr = np.zeros((self.numcells, laps))

            for c in range(0, self.numcells):
                data1 = np.nan_to_num(np.mean(data_formapping[c, :, :], 0))
                data2 = np.nan_to_num(self.rewardregion_fluor_data[taskname][c, :, :])

                for l in range(0, laps):
                    # print(np.shape(data2), np.shape(data1), np.shape(corr))
                    # print(np.corrcoef(data2[l, :], data1)[0, 1])
                    temp = np.corrcoef(data2[l, :], data1)[0, 1]
                    if ~np.isnan(temp):
                        corr[c, l] = temp

            corr = np.nan_to_num(corr)
            correlation_per_task[taskname] = corr

        self.plot_correlation_by_task(correlation_per_task, Task=TaskA)
        return correlation_per_task

    def correlation_activity_inrewardcells(self, TaskA='Task1'):
        cells_formapping = self.cells_firing_to_reward[TaskA]
        data_formapping = self.rewardregion_fluor_data[TaskA]

        correlation_per_task = {keys: [] for keys in self.TaskDict.keys()}
        for i in self.TaskDict:
            taskname = i
            laps = self.numlaps[taskname]
            corr = np.zeros((np.size(cells_formapping), laps))

            for n, c in enumerate(cells_formapping):
                data1 = np.nan_to_num(np.mean(data_formapping[c, :, :], 0))
                data2 = np.nan_to_num(self.rewardregion_fluor_data[taskname][c, :, :])

                for l in range(0, laps):
                    # print(np.shape(data2), np.shape(data1), np.shape(corr))
                    # print(np.corrcoef(data2[l, :], data1)[0, 1])
                    temp = np.corrcoef(data2[l, :], data1)[0, 1]
                    if ~np.isnan(temp):
                        corr[n, l] = temp

            corr = np.nan_to_num(corr)
            correlation_per_task[taskname] = corr

        self.plot_correlation_by_task(correlation_per_task, Task=TaskA)
        return correlation_per_task

    def plot_correlation_by_task(self, data_to_plot, Task):
        fs, axes = plt.subplots(2, 4, sharex='col', sharey='row',
                                gridspec_kw={'height_ratios': [2, 1]},
                                figsize=(10, 6))

        for n, i in enumerate(self.TaskDict.keys()):
            ax1 = axes[0, n]
            ax1.imshow(data_to_plot[i], interpolation='nearest', aspect='auto', cmap='viridis', vmin=0,
                       vmax=1)
            ax1.set_xlim((0, self.numlaps[i]))
            ax1.set_title(self.TaskDict[i])
            ax1.spines['right'].set_visible(False)
            ax1.spines['top'].set_visible(False)
            ax1.spines['bottom'].set_visible(False)
            ax1.tick_params(bottom=False)

            ax2 = axes[1, n]
            ax2.plot(np.mean(data_to_plot[i], 0), '-o', linewidth=2)
            # ax2.set_xlim((0, self.numlaps[i]))
            ax2.set_xlabel('Lap number')

            ax3 = ax2.twinx()  # instantiate a second axes that shares the same x-axis
            ax3.plot(self.lap_time[i], '-o', color='r', alpha=0.5, label='Lap time')
            ax3.set_ylim((0, max(self.lap_time['Task2'])))

            # Pretify
            for a in [ax2, ax3]:
                a.spines['right'].set_visible(False)
                a.spines['top'].set_visible(False)
                a.spines['left'].set_visible(False)
                a.tick_params(left=False, right=False)

            if n == len(self.TaskDict.keys()) - 1:
                ax3.set_ylabel('Lap Time (s)', color='r')
            else:
                ax3.set_yticklabels([])

            for l in range(0, self.numlaps[i]):
                if self.lick_flag[i][l]:
                    ax2.axvline(l, linewidth=0.25, color='k')
        axes[0, 0].set_ylabel('Cell Number')
        axes[1, 0].set_ylabel('Mean Correlation', color='b')

        fs.subplots_adjust(wspace=0.1, hspace=0.1)
        fs.suptitle(f'Correlation with %s' % Task, fontsize=10)

        # plt.tight_layout()
        plt.show()

    def plot_common_PF_heatmap(self):
        PF_sort = {keys: [] for keys in self.TaskDict.keys()}
        PF_sort.update({'Cellnum': []})
        for i in range(0, self.numcells):
            PF_sort['Cellnum'].append(i)
            for t in self.TaskDict:
                if i in self.sig_PFs_cellnum_dict[t]:
                    PF_sort[t].append(1)
                else:
                    PF_sort[t].append(0)

        PF_sort_df = pd.DataFrame.from_dict(PF_sort)
        PF_sort_df = PF_sort_df.sort_values(by=['Task1', 'Task2', 'Task3'], ascending=False)

        sns.heatmap(PF_sort_df.drop(['Cellnum'], axis=1), cbar=False)

    def plot_rewardcell_heatmap(self):
        Rew_sort = {keys: [] for keys in self.TaskDict.keys()}
        Rew_sort.update({'Cellnum': []})
        for i in range(0, self.numcells):
            Rew_sort['Cellnum'].append(i)
            for t in self.TaskDict:
                if i in self.cells_firing_to_reward[t]:
                    Rew_sort[t].append(1)
                else:
                    Rew_sort[t].append(0)

        Rew_sort_df = pd.DataFrame.from_dict(Rew_sort)
        Rew_sort_df = Rew_sort_df.sort_values(by=['Task1', 'Task2', 'Task3'], ascending=False)

        sns.heatmap(Rew_sort_df.drop(['Cellnum'], axis=1), cbar=False)

        return Rew_sort_df

    def save_parsed_data(self):
        with open(os.path.join(self.SaveFolder, 'Fc3data.pickle'), 'wb') as handle:
            pickle.dump(self.Fc3data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.SaveFolder, 'Fcdata.pickle'), 'wb') as handle:
            pickle.dump(self.Fcdata_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.SaveFolder, 'rewardregion_fluor_data.pickle'), 'wb') as handle:
            pickle.dump(self.rewardregion_fluor_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.SaveFolder, 'sig_PFs_cellnum_dict.pickle'), 'wb') as handle:
            pickle.dump(self.sig_PFs_cellnum_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.SaveFolder, 'behavior_data.pickle'), 'wb') as handle:
            pickle.dump(
                (self.running_data, self.numlaps, self.lick_data, self.lap_time, self.lickstop_lap, self.lick_flag,
                 self.reward_frame), handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.SaveFolder, 'PF_params.pickle'), 'wb') as handle:
            pickle.dump(self.PFParams_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def plot_place_cell_location(self):
        # Plot Place cells
        data = scipy.io.loadmat(os.path.join(self.FolderName, self.ImgFileName[0]))
        img_masks = data['data'].item()[3]
        task_axes = {'Task1': 0, 'Task2': 1, 'Task3': 2, 'Task4': 3}
        task_value = {'Task1': 1, 'Task2': 2, 'Task3': 2, 'Task4': 2}

        cell_mask = {keys: [] for keys in self.TaskDict}

        for keys, pfs in self.sig_PFs_cellnum_dict.items():
            task_mask = np.zeros_like(img_masks[:, :, 0])
            for p in pfs:
                task_mask += img_masks[:, :, p]
            mask = np.ma.masked_where(task_mask > 0, task_mask)
            mask.set_fill_value(task_value[keys])

            cell_mask[keys].extend(mask.filled())

        # Plot Task1 with other tasks
        fs, axes = plt.subplots(1, len(self.TaskDict), figsize=(30, 10))
        axes[task_axes['Task1']].imshow(cell_mask['Task1'])
        axes[task_axes['Task1']].axis('off')
        cmap = plt.cm.get_cmap('Dark2')
        cmap.set_under('black')
        for keys in self.TaskDict:
            if 'Task1' not in keys:
                ax1 = axes[task_axes[keys]]
                c = ax1.imshow(np.add(cell_mask['Task1'], cell_mask[keys]), cmap=cmap, vmin=0.5)
                ax1.set_title(f'Task1 and %s' % keys)
                ax1.axis('off')
        fs.subplots_adjust(wspace=0.1, hspace=0.1, top=0.9)
        fs.suptitle('Place cells')
        fs.tight_layout()
        return cell_mask

    def plot_reward_pf_cells_location(self):
        # Overlay reward and place cells
        data = scipy.io.loadmat(os.path.join(self.FolderName, self.ImgFileName[0]))
        img_masks = data['data'].item()[3]

        task_axes = {'Task1': 0, 'Task2': 1, 'Task3': 2, 'Task4': 3}

        cell_mask = {keys: [] for keys in self.TaskDict}

        for t in self.TaskDict:
            pf_mask = np.zeros_like(img_masks[:, :, 0])
            rew_mask = np.zeros_like(img_masks[:, :, 0])
            for c in range(0, self.numcells):
                if c in self.sig_PFs_cellnum_dict[t]:
                    pf_mask += img_masks[:, :, c]

                if c in self.cells_firing_to_reward[t]:
                    rew_mask += img_masks[:, :, c]

            mask1 = np.ma.masked_where(pf_mask > 0, pf_mask)
            mask1.set_fill_value(1)

            mask2 = np.ma.masked_where(rew_mask > 0, rew_mask)
            mask2.set_fill_value(2)

            cell_mask[t].extend(np.add(mask1.filled(), mask2.filled()))

        # Plot Task1 with other tasks
        fs, axes = plt.subplots(1, len(self.TaskDict), figsize=(30, 10))
        cmap = plt.cm.get_cmap('Set1')
        cmap.set_under('black')
        for keys in self.TaskDict:
            ax1 = axes[task_axes[keys]]
            c = ax1.imshow(cell_mask[keys], cmap=cmap, vmin=0.5)
            ax1.set_title(keys)
            ax1.axis('off')

        fs.subplots_adjust(wspace=0.1, hspace=0.1, top=0.9)
        fs.suptitle('Place and Reward cells')
        fs.tight_layout()
        return cell_mask

    def location_of_commoncells_with_taskchange(self, TaskA='Task1'):
        # Create a data frame for center of mask comparison with TaskA
        TaskA_df = self.PFParams_df[self.PFParams_df['Task'] == self.TaskDict[TaskA]][['Center_of_mass', 'CellNum']]

        fs, ax = plt.subplots(1, len(self.TaskDict) - 1, figsize=(20, 5))
        ax_count = 0
        for t in self.TaskDict.keys():
            if t != TaskA:
                TaskB_df = self.PFParams_df[self.PFParams_df['Task'] == self.TaskDict[t]][['Center_of_mass', 'CellNum']]
                c = pd.merge(TaskA_df, TaskB_df, how='inner', on=['CellNum'],
                             suffixes=(f'_%s' % self.TaskDict[TaskA], f'_%s' % self.TaskDict[t]))

                sns.scatterplot(x=f'Center_of_mass_%s' % self.TaskDict[t],
                                y=f'Center_of_mass_%s' % self.TaskDict[TaskA], data=c, s=70, color=".2",
                                ax=ax[ax_count])

                ax_count += 1

        fs.tight_layout()
