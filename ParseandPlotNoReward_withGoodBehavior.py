import scipy.io
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import pickle
import random
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


sns.set_context('paper', font_scale=1.2)
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
        self.GoodBehFileName = [f for f in os.listdir(os.path.join(FolderName, 'Behavior')) if
                                f.endswith('.mat') and 'PlaceFields' not in f and 'good_behavior' in f]

        self.PlaceFieldData = \
            [f for f in os.listdir(os.path.join(FolderName, 'Behavior')) if
             (f.endswith('.mat') and 'PlaceFields' in f and 'GoodBehavior' in f)]

        # Create a number of dicts for storing files trial wise
        self.Fcdata_dict = self.create_data_dict()
        self.Fdata_dict = self.create_data_dict()
        self.Fc3data_dict = self.create_data_dict()
        self.sig_PFs_cellnum_dict = self.create_data_dict()
        self.running_data = self.create_data_dict()
        self.good_running_data = self.create_data_dict()
        self.good_running_index = self.create_data_dict()
        self.reward_data = self.create_data_dict()
        self.lick_data = self.create_data_dict()
        self.timestamps = self.create_data_dict()
        self.numlicks_withinreward = self.create_data_dict()
        self.numlicks_outsidereward = self.create_data_dict()

        self.numlaps = self.find_numlaps_by_task()
        self.numcells = self.load_images()
        self.load_behdata()

        self.good_lap_time, self.actual_lap_time = self.find_lap_parameters()
        self.lick_perlap_inspace, self.lick_bin_edge = self.find_lick_parameters()
        self.quantify_lickstop_lickperlap()

        #
        self.PFParams_df = self.create_df_placefield_params()
        self.find_sig_PFs_cellnum_bytask()

        # self.save_parsed_data()

    def create_data_dict(self):
        data_dict = {f'Task%d' % keys: [] for keys in range(1, len(self.Task_Numframes) + 1)}
        return data_dict

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
            self.reward_data[taskname] = x['session'].item()[0][0][0][1]
            self.running_data[taskname] = x['session'].item()[0][0][0][0]

        for i in self.GoodBehFileName:
            taskname = i[i.find('Task'):i.find('Task') + 5]
            x = scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', i))
            self.good_running_data[taskname] = x['good_behavior'].item()[0].T
            self.good_running_index[taskname] = x['good_behavior'].item()[1][0]

    def find_lick_parameters(self):
        numBins = np.linspace(0, 0.7, 40)
        numlicks_spacelap_dict = {keys: [] for keys in self.TaskDict.keys()}
        numlicks_dict = {keys: [] for keys in ['TaskName', 'LapNumber', 'Numlicks', ]}
        # Find space at which each lick is present
        # Find number of licks per lap
        for i in self.PlaceFieldData:
            taskname = i[i.find('Task'):i.find('Task') + 5]
            x = scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', i))
            lapframes = x['E'].T

            numlicks_perspace_perlap = np.zeros((np.max(lapframes) - 1, np.size(numBins) - 1))
            for this, next in zip(range(1, np.max(lapframes)), range(2, np.max(lapframes) + 1)):
                [thislap, nextlap] = [np.where(lapframes == this)[0], np.where(lapframes == next)[0]]
                [thislap_start, thislap_end] = [self.good_running_index[taskname][thislap[0]],
                                                self.good_running_index[taskname][thislap[-1]]]
                [nextlap_start, nextlap_end] = [self.good_running_index[taskname][nextlap[0]],
                                                self.good_running_index[taskname][nextlap[-1]]]

                if taskname not in 'Task2':
                    reward_lap = self.reward_data[taskname][thislap_start:nextlap_start]
                    reward_frame = np.where(np.diff(reward_lap, axis=0) > 4)[0][0]
                    # print(thisrun[reward_frame])
                    laprun = self.running_data[taskname][thislap_start:thislap_start + reward_frame]
                    prelicks = self.lick_data[taskname][thislap_start:thislap_start + reward_frame]
                else:
                    laprun = self.running_data[taskname][thislap_start:nextlap_start]
                    prelicks = self.lick_data[taskname][thislap_start:nextlap_start]

                prelicks_startframes = np.where(np.diff(prelicks, axis=0) > 1)[0]
                prelick_space = laprun[prelicks_startframes]

                numlicks_perspace_perlap[this - 1, :], bin_edges = np.histogram(prelick_space, numBins)

            numlicks_spacelap_dict[taskname] = numlicks_perspace_perlap

        return numlicks_spacelap_dict, bin_edges

    def quantify_lickstop_lickperlap(self):
        # Find the region in the familiar environemnt where most licks occur
        # Number of out of region licks versus in region licks for each environment each lap
        control_lick = self.lick_perlap_inspace['Task1']
        highlicks = np.where(np.sum(control_lick, 0) > self.numlaps['Task1'] * 1.5)[
            0]  # If licks exceep numlaps ran by 1.5
        print(f'Space with high licks %0.2f' % self.lick_bin_edge[highlicks[0]])
        for n, i in enumerate(self.TaskDict.keys()):
            licks = self.lick_perlap_inspace[i]
            self.numlicks_withinreward[i] = np.sum(licks[:, highlicks[0]:], axis=1)
            self.numlicks_outsidereward[i] = np.sum(licks[:, :highlicks[0]], axis=1)
            if i in 'Task2':
                self.lick_stop = np.where(self.numlicks_withinreward[i] == 0)[0][0]
                print(f'Pre lick stops at %dth lap' % self.lick_stop)

    def find_lap_parameters(self):

        # find start and end of a lap and see if there is a lick
        good_laptime = {keys: [] for keys in self.TaskDict.keys()}
        actual_laptime = {keys: [] for keys in self.TaskDict.keys()}
        for i in self.PlaceFieldData:
            taskname = i[i.find('Task'):i.find('Task') + 5]
            x = scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', i))

            lapframes = x['E'].T
            for l in range(1, np.max(lapframes) + 1):
                laps = np.where(lapframes == l)[0]
                good_laptime[taskname].append(np.size(laps) / 30.98)
                actual_laptime[taskname].append(
                    (self.good_running_index[taskname][laps[-1]] - self.good_running_index[taskname][laps[0]]) / 30.98)

        return good_laptime, actual_laptime

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

    def plot_behavior(self):
        # Plot behavior traces and
        fs, axes = plt.subplots(len(self.TaskDict), 3, figsize=(15, 8), sharex='col', sharey='col')

        for n, i in enumerate(self.TaskDict.keys()):
            axes[n, 0].plot(self.running_data[i], linewidth=2)
            axes[n, 0].plot(self.lick_data[i] / 4, linewidth=1, alpha=0.5)
            axes[n, 1].plot(self.good_running_data[i], linewidth=2)
            axes[n, 2].bar(np.arange(np.size(self.lick_perlap_inspace[i], 1)),
                           np.sum(self.lick_perlap_inspace[i], axis=0))
            axes[n, 0].set_title(self.TaskDict[i])

            ax2 = axes[n, 2].twinx()
            ax2.plot(np.arange(np.size(self.lick_perlap_inspace[i], 1)), np.mean(self.lick_perlap_inspace[i], axis=0),
                     linewidth=2, color='r')
            ax2.set_ylim((0, 5))

        axes[-1, 2].set_xlabel('Spatial bins')
        axes[-1, 2].set_ylabel('Number of licks')
        fs.tight_layout()

        # Plot number of laps ran per task
        fs, axes = plt.subplots(1, figsize=(5, 2))
        for n, i in enumerate(self.TaskDict.keys()):
            axes.bar(n, self.numlaps[i])
        axes.set_xticks(range(len(self.numlaps)))
        axes.set_xticklabels(list(self.TaskDict.keys()))
        axes.set_ylabel('Number of laps \n per task')

    def plot_velocity(self):
        fs, axes = plt.subplots(1, 2, figsize=(12, 3), sharex='all', sharey='all')
        labels, data = [*zip(*self.actual_lap_time.items())]
        bp = axes[0].boxplot(data, patch_artist=True)
        plt.xticks(range(1, len(labels) + 1), labels)
        for median in bp['medians']:
            median.set(color='black', linewidth=1)

        labels, data = [*zip(*self.good_lap_time.items())]
        bp = axes[1].boxplot(data, patch_artist=True)
        plt.xticks(range(1, len(labels) + 1), labels)
        for median in bp['medians']:
            median.set(color='black', linewidth=1)

        axes[0].set_title('Actual Lap Time')
        axes[0].set_ylabel(f'Time taken to complete lap \n (seconds)')
        axes[1].set_title('Corrected Lap Time')
        fs.tight_layout()

    def plot_lick_per_lap(self):

        fs, axes = plt.subplots(1, len(self.TaskDict), figsize=(15, 3), sharex='all', sharey='all')
        for n, i in enumerate(self.TaskDict.keys()):
            axes[n].plot(self.numlicks_withinreward[i], linewidth=2, marker='.', markersize=10)
            axes[n].set_title(self.TaskDict[i])
            if i in 'Task2':
                axes[n].axvline(self.lick_stop, color='k', alpha=0.5, linewidth=2)

        axes[0].set_ylabel('Number of pre licks \n in reward zone')
        axes[0].set_xlabel('Lap Number')
        fs.tight_layout()

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

    def plot_remapping_withTaskA(self, TaskA='Task1'):
        # Divide task2 into everything
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
            ax1[n].imshow(img_dict[i][img_argsort, :], aspect='auto', cmap='jet', interpolation='nearest', vmin=0,
                          vmax=1)
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

    def plot_remapping_withTaskA_splitbylickstop(self, TaskA='Task1'):
        placecells_formapping = self.sig_PFs_cellnum_dict[TaskA]
        img_dict = {keys: [] for keys in self.TaskDict.keys()}
        for i in self.PlaceFieldData:
            taskname = i[i.find('Task'):i.find('Task') + 5]
            x = scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', i))
            bins = np.size(x['Allbinned_F'][0, 0], 0)

            if 'Task2' in taskname:
                img = np.zeros((np.size(placecells_formapping), bins))
                for n, c in enumerate(placecells_formapping):
                    img[n, :] = np.mean(np.nan_to_num(x['Allbinned_F'][0, c][:, :self.lick_stop]), 1)
                img_dict[taskname + '_Beflick'] = img

                img = np.zeros((np.size(placecells_formapping), bins))
                for n, c in enumerate(placecells_formapping):
                    img[n, :] = np.mean(np.nan_to_num(x['Allbinned_F'][0, c][:, self.lick_stop:]), 1)
                img_dict[taskname + '_Aftlick'] = img

                img = np.zeros((np.size(placecells_formapping), bins))
                for n, c in enumerate(placecells_formapping):
                    img[n, :] = np.mean(np.nan_to_num(x['Allbinned_F'][0, c]), 1)
                img_dict[taskname] = img

            else:
                img = np.zeros((np.size(placecells_formapping), bins))
                for n, c in enumerate(placecells_formapping):
                    img[n, :] = np.mean(np.nan_to_num(x['Allbinned_F'][0, c]), 1)
                img_dict[taskname] = img

        # Sort by task
        img_argsort = np.argsort(np.nanargmax(img_dict[TaskA], 1))
        fs, ax1 = plt.subplots(1, len(self.TaskDict) + 1, figsize=(10, 3.5), dpi=300)
        axes_count = 0
        for n, i in enumerate(self.TaskDict.keys()):
            if 'Task2' in i:
                ax1[axes_count].imshow(img_dict[i + '_Beflick'][img_argsort, :], aspect='auto', cmap='jet',
                                       interpolation='nearest', vmin=0, vmax=1)
                ax1[axes_count].set_title('No Reward \n Before Lick Stops')
                ax1[axes_count + 1].imshow(img_dict[i + '_Aftlick'][img_argsort, :], aspect='auto', cmap='jet',
                                           interpolation='nearest', vmin=0, vmax=1)
                ax1[axes_count + 1].set_title('No Reward \n After Lick Stops', fontsize=10)

                if n > 0:
                    ax1[axes_count].axis('off')
                    ax1[axes_count + 1].axis('off')
                axes_count += 2
            else:
                img = ax1[axes_count].imshow(img_dict[i][img_argsort, :], aspect='auto', cmap='jet', interpolation='nearest',
                                       vmin=0, vmax=1)
                ax1[axes_count].set_title(self.TaskDict[i], fontsize=10)

                if n > 0:
                    ax1[axes_count].axis('off')

                axes_count += 1

        ax1[0].spines['right'].set_visible(False)
        ax1[0].spines['top'].set_visible(False)
        ax1[0].set_xlabel('Track Length (cm)', fontsize=10)
        ax1[0].set_ylabel('Cell #', fontsize=10)
        ax1[0].locator_params(nbins=4, tight=True)
        ax1[0].set_xticks([0, 20, 39])
        ax1[0].set_xticklabels([0, 100, 200])

        axins = inset_axes(ax1[-1],
                           width="5%",  # width = 5% of parent_bbox width
                           height="50%",  # height : 50%
                           loc='lower left',
                           bbox_to_anchor=(1.05, 0., 1, 1),
                           bbox_transform=ax1[-1].transAxes,
                           borderpad=0.5,
                           )
        cb = fs.colorbar(img, cax=axins, pad=0.2, ticks=[0, 0.5, 1])
        cb.set_label('\u0394F/F', rotation=270, labelpad=12)

        plt.tight_layout()

        return img_dict

    def plot_remapping_withTaskA_splitbylickstop_normalizenumlaps(self, TaskA='Task1'):
        placecells_formapping = self.sig_PFs_cellnum_dict[TaskA]
        img_dict = {keys: [] for keys in self.TaskDict.keys()}
        del img_dict['Task2']  # So it can be divided into licks
        for i in self.PlaceFieldData:
            taskname = i[i.find('Task'):i.find('Task') + 5]
            x = scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', i))
            bins = np.size(x['Allbinned_F'][0, 0], 0)

            if 'Task2' in taskname:
                img = np.zeros((np.size(placecells_formapping), bins))
                for n, c in enumerate(placecells_formapping):
                    img[n, :] = np.mean(np.nan_to_num(x['Allbinned_F'][0, c][:, :self.lick_stop]), 1)
                img_dict[taskname + '_Beflick'] = img

                img = np.zeros((np.size(placecells_formapping), bins))
                randomizelaps = random.sample(range(self.lick_stop, self.numlaps['Task2']), self.lick_stop)
                print(taskname, randomizelaps)
                for n, c in enumerate(placecells_formapping):
                    img[n, :] = np.mean(np.nan_to_num(x['Allbinned_F'][0, c][:, randomizelaps]), 1)
                img_dict[taskname + '_Aftlick'] = img

            else:
                randomizelaps = random.sample(range(0, self.numlaps[taskname]), self.lick_stop)
                print(taskname, randomizelaps)
                img = np.zeros((np.size(placecells_formapping), bins))
                for n, c in enumerate(placecells_formapping):
                    img[n, :] = np.mean(np.nan_to_num(x['Allbinned_F'][0, c][:, randomizelaps]), 1)
                img_dict[taskname] = img

        # Sort by task
        img_argsort = np.argsort(np.nanargmax(img_dict[TaskA], 1))
        fs, ax1 = plt.subplots(1, len(self.TaskDict) + 1, figsize=(10, 4))
        axes_count = 0
        for n, i in enumerate(self.TaskDict.keys()):
            if 'Task2' in i:
                ax1[axes_count].imshow(img_dict[i + '_Beflick'][img_argsort, :], aspect='auto', cmap='jet',
                                       interpolation='nearest', vmin=0, vmax=1)
                ax1[axes_count].set_title('No Reward \n Before Lick Stops')
                ax1[axes_count + 1].imshow(img_dict[i + '_Aftlick'][img_argsort, :], aspect='auto', cmap='jet',
                                           interpolation='nearest', vmin=0, vmax=1)
                ax1[axes_count + 1].set_title('No Reward \n After Lick Stops')
                if n > 0:
                    ax1[axes_count].axis('off')
                    ax1[axes_count + 1].axis('off')
                axes_count += 2
            else:
                ax1[axes_count].imshow(img_dict[i][img_argsort, :], aspect='auto', cmap='jet', interpolation='nearest',
                                       vmin=0, vmax=1)
                ax1[axes_count].set_title(self.TaskDict[i])

                if n > 0:
                    ax1[axes_count].axis('off')

                axes_count += 1

        ax1[0].set_xlabel('Bins')
        ax1[0].set_ylabel('Cell #')
        ax1[0].locator_params(nbins=4)
        plt.show()

        return img_dict

    def significance_test_on_mapstability_bytask(self, TaskA='Task1', iterations=2):
        # Run iterations of picking random laps and judging correlation for each cell in space
        # in comparison with task1
        img_dict = {}

        # Get data to compare from control task
        placecells_formapping = self.sig_PFs_cellnum_dict[TaskA]
        filename = [i for i in self.PlaceFieldData if TaskA in i][0]
        x = scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', filename))
        bins = np.size(x['Allbinned_F'][0, 0], 0)
        img = np.zeros((np.size(placecells_formapping), bins))
        for n, c in enumerate(placecells_formapping):
            img[n, :] = np.mean(np.nan_to_num(x['Allbinned_F'][0, c]), 1)

        img_argsort = np.argsort(np.nanargmax(img, 1))
        img_dict[TaskA + '_Control'] = img[img_argsort, :]

        # plt.figure()
        # plt.imshow(img_dict[TaskA + '_Control'], aspect='auto', cmap='jet', interpolation='nearest',
        #            vmin=0, vmax=1)

        task_correlation = {}
        for i in self.PlaceFieldData:

            taskname = i[i.find('Task'):i.find('Task') + 5]
            x = scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', i))
            bins = np.size(x['Allbinned_F'][0, 0], 0)
            img = np.zeros((np.size(placecells_formapping), bins))

            # Get cell data with multiple iterations
            if 'Task2' in taskname:

                # After lick stops
                map_correlation = np.zeros((iterations, np.size(placecells_formapping)))
                iter1 = 0
                while iter1 < iterations:
                    randomizelaps = random.sample(range(self.lick_stop, self.numlaps['Task2']), self.lick_stop)
                    map_correlation[iter1, :] = self.remapping_correlation_withcontrol(
                        dataA=img_dict[TaskA + '_Control'],
                        dataB=x['Allbinned_F'],
                        placecells=placecells_formapping,
                        sorted_cells=img_argsort,
                        laps=randomizelaps,
                        bins=bins)

                    iter1 += 1

                task_correlation['Task2 after_lick_stops'] = np.nan_to_num(map_correlation)

                # Before lick stops
                randomizelaps = random.sample(range(self.lick_stop, self.numlaps['Task2']), self.lick_stop)
                map_correlation = self.remapping_correlation_withcontrol(dataA=img_dict[TaskA + '_Control'],
                                                                         dataB=x['Allbinned_F'],
                                                                         placecells=placecells_formapping,
                                                                         sorted_cells=img_argsort,
                                                                         laps=range(0, self.lick_stop),
                                                                         bins=bins)

                task_correlation['Task2 before_lick_stops'] = np.nan_to_num(map_correlation)

            else:
                map_correlation = np.zeros((iterations, np.size(placecells_formapping)))
                iter1 = 0
                while iter1 < iterations:
                    randomizelaps = random.sample(range(0, self.numlaps[taskname]), self.lick_stop)
                    map_correlation[iter1, :] = self.remapping_correlation_withcontrol(
                        dataA=img_dict[TaskA + '_Control'],
                        dataB=x['Allbinned_F'],
                        placecells=placecells_formapping,
                        sorted_cells=img_argsort,
                        laps=randomizelaps,
                        bins=bins)

                    iter1 += 1

                task_correlation[taskname] = np.nan_to_num(map_correlation)

        df_combine = pd.DataFrame(task_correlation['Task2 before_lick_stops'], columns=['Task2 before_lick_stops'])
        for i in task_correlation.keys():
            if i not in 'Task2 before_lick_stops':
                df = pd.DataFrame(np.mean(task_correlation[i], 0), columns=[i])
                df_combine = pd.concat([df_combine, df], axis=1)

        fs, axes = plt.subplots(1, figsize=(15, 5))
        ax = sns.boxplot(data=df_combine, linewidth=2.5, ax=axes,
                         order=['Task1', 'Task2 before_lick_stops', 'Task2 after_lick_stops',
                                'Task3', 'Task4'])
        sns.stripplot(data=df_combine, order=['Task1', 'Task2 before_lick_stops', 'Task2 after_lick_stops',
                                              'Task3', 'Task4'], color="0.3")
        ax.set_ylabel('Place Field Stability')
        fs.tight_layout()

        fs, axes = plt.subplots(1, 4, figsize=(15, 3), sharex='all', sharey='all')
        count_axis = 0
        for n, i in enumerate(['Task2 before_lick_stops', 'Task2 after_lick_stops',
                               'Task3', 'Task4']):
            sns.distplot(task_correlation['Task1'].ravel(), ax=axes[count_axis])
            if i == 'Task2 before_lick_stops':
                sns.distplot(task_correlation[i], ax=axes[n])
            else:
                sns.distplot(task_correlation[i].ravel(), ax=axes[count_axis], label=i)

            axes[count_axis].set_title(i)
            count_axis += 1
        axes[0].set_ylabel('Number of cells')
        fs.tight_layout()

        return task_correlation

    def remapping_correlation_withcontrol(self, dataA, dataB, placecells, sorted_cells, laps, bins):
        img = np.zeros((np.size(placecells), bins))
        for n, c in enumerate(placecells):
            img[n, :] = np.mean(np.nan_to_num(dataB[0, c][:, laps]), 1)

        sorted_img = img[sorted_cells, :]

        map_correlation = np.zeros(np.size(placecells))
        for c in range(0, np.size(placecells)):  # Find correlation cell by cell
            map_correlation[c] = np.corrcoef(dataA[c, :], sorted_img[c, :])[0, 1]

        return map_correlation

    def plot_place_cell_location(self):
        # Plot Place cells
        data = scipy.io.loadmat(os.path.join(self.FolderName, self.ImgFileName[0]))
        img_masks = data['data'].item()[3]
        mean_image = data['data'].item()[5]
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

        # # Plot Task1 with other tasks
        # fs, axes = plt.subplots(1, len(self.TaskDict), figsize=(30, 10))
        # axes[task_axes['Task1']].imshow(cell_mask['Task1'])
        # axes[task_axes['Task1']].axis('off')
        # cmap = plt.cm.get_cmap('Dark2')
        # cmap.set_under('black')
        # for keys in self.TaskDict:
        #     if 'Task1' not in keys:
        #         ax1 = axes[task_axes[keys]]
        #         c = ax1.imshow(np.add(cell_mask['Task1'], cell_mask[keys]), cmap=cmap, vmin=0.5)
        #         ax1.set_title(f'Task1 and %s' % keys)
        #         ax1.axis('off')
        # fs.subplots_adjust(wspace=0.1, hspace=0.1, top=0.9)
        # fs.suptitle('Place cells')
        # fs.tight_layout()
        return cell_mask

    def local_spatial_stability(self):
        #Calculate local stability with time
        #Correlate cell activity with previous lap
        # Do this for every cell

        this_data = []
        next_data = []
        for i in self.PlaceFieldData[0:1]:
            taskname = i[i.find('Task'):i.find('Task') + 5]
            x = scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', i))
            bins = np.size(x['Allbinned_F'][0, 0], 0)
            img = np.zeros((self.numcells, bins))
            for c in range(0, self.numcells):
                img = np.nan_to_num(x['Allbinned_F'][0, c])
                print(taskname, np.shape(img))

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

        sort_cells = np.argsort(np.mean(correlation_per_task[TaskA], 1))[::-1]
        for i in correlation_per_task.keys():
            correlation_per_task[i] = correlation_per_task[i][sort_cells, :]
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
            ax3.plot(self.numlicks_withinreward[i], '-o', color='r', alpha=0.5, label='Lap time')
            # ax3.set_ylim((0, max(self.lap_time['Task2'])))

            # Pretify
            for a in [ax2, ax3]:
                a.spines['right'].set_visible(False)
                a.spines['top'].set_visible(False)
                a.spines['left'].set_visible(False)
                a.tick_params(left=False, right=False)

            if n == len(self.TaskDict.keys()) - 1:
                ax3.set_ylabel('Pre Licks', color='r')
            else:
                ax3.set_yticklabels([])

            for l in range(0, self.numlaps[i] - 1):
                if self.numlicks_withinreward[i][l]:
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
