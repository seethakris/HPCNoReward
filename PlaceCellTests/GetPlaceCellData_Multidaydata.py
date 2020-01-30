import scipy.io
import os
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
import seaborn as sns
from collections import OrderedDict
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import sys
import scipy.stats
import h5py

# For plotting styles
PlottingFormat_Folder = '/home/sheffieldlab/Desktop/NoReward/Scripts/PlottingTools/'
sys.path.append(PlottingFormat_Folder)
import plottingfunctions as pf


class GetData:
    def __init__(self, animalinfo, FolderName, noreward_task, v73_flag=0):
        self.FolderName = FolderName
        self.FigureFolder = os.path.join(self.FolderName, 'Figures')
        self.SaveFolder = os.path.join(self.FolderName, 'PlaceCells')
        self.TaskDict = animalinfo['task_dict']
        self.Task_Numframes = animalinfo['task_numframes']
        self.tracklength = animalinfo['tracklength']
        self.trackbins = animalinfo['trackbins']
        self.animalname = animalinfo['animal']
        self.noreward_task = noreward_task

        if not os.path.exists(self.FigureFolder):
            os.mkdir(self.FigureFolder)
        if not os.path.exists(self.SaveFolder):
            os.mkdir(self.SaveFolder)

        self.get_data_folders()
        if v73_flag:
            self.load_v73_Data()
        else:
            self.load_fluorescentdata()

        self.get_lapframes_numcells()
        self.lickstoplap = self.Parsed_Behavior['lick_stop'].item()[self.noreward_task]
        self.lickstopframe = np.where(self.good_lapframes['Task3'] == self.lickstoplap + 1)[0][
            0]  # Task3 is no reward for multiday animal
        print(self.lickstoplap, self.lickstopframe)

        # Find significant place cells
        # Add no lick data where exists
        self.new_taskDict = copy(self.TaskDict)
        self.new_taskDict[f'%sb' % self.noreward_task] = '3 No Rew No Lick'
        self.new_taskDict = OrderedDict(sorted(self.new_taskDict.items()))
        self.find_sig_PFs_cellnum_bytask()
        # self.calculate_pfparameters()
        self.correlate_acivity_of_allcellsbytask()
        # self.common_droppedcells_withTask1()
        # self.save_analyseddata()

    def create_data_dict(self):
        data_dict = {keys: [] for keys in self.TaskDict.keys()}
        return data_dict

    def get_data_folders(self):
        self.ImgFileName = [f for f in os.listdir(self.FolderName) if f.endswith('.mat')]
        self.Parsed_Behavior = np.load(os.path.join(self.FolderName, 'SaveAnalysed', 'behavior_data.npz'),
                                       allow_pickle=True)
        self.PlaceFieldData = \
            [f for f in os.listdir(os.path.join(self.FolderName, 'Behavior')) if
             (f.endswith('.mat') and 'PlaceFields' in f and 'GoodBehavior' in f)]

    def find_sig_PFs_cellnum_bytask(self):
        self.sig_PFs_cellnum = self.create_data_dict()
        self.numPFS_incells = self.create_data_dict()
        for i in self.PlaceFieldData:
            ft = i.find('Task')
            taskname = i[ft:ft + i[ft:].find('_')]
            x = scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', i))
            tempx = np.squeeze(np.asarray(np.nan_to_num(x['number_of_PFs'])).T).astype(int)
            print(f'%s : Place cells: %d PlaceFields: %d' % (
                taskname, np.size(np.where(tempx > 0)[0]), np.sum(tempx[tempx > 0])))

            self.sig_PFs_cellnum[taskname] = np.where(tempx > 0)[0]
            self.numPFS_incells[taskname] = tempx[np.where(tempx > 0)[0]]

    def common_droppedcells_withTask1(self):
        self.droppedcells = self.create_data_dict()
        self.commoncells = self.create_data_dict()
        for i in self.new_taskDict.keys():
            if i not in 'Task1':
                self.droppedcells[i] = np.asarray(
                    [l for l in self.sig_PFs_cellnum['Task1'] if l not in self.sig_PFs_cellnum[i]])
                self.commoncells[i] = list(
                    set(self.sig_PFs_cellnum['Task1']).intersection(self.sig_PFs_cellnum[i]))

    def load_fluorescentdata(self):
        self.Fc3data_dict = self.create_data_dict()
        # Open calcium data and store in dicts per trial
        data = scipy.io.loadmat(os.path.join(self.FolderName, self.ImgFileName[0]))
        count = 0
        for i in self.TaskDict.keys():
            self.Fc3data_dict[i] = data['data'].item()[2].T[:,
                                   count:count + self.Task_Numframes[i]]
            print(f'%s : Number of Frames: %d' % (i, np.size(self.Fc3data_dict[i], 1)))
            count += self.Task_Numframes[i]

    def get_lapframes_numcells(self):
        self.good_lapframes = self.create_data_dict()
        for t in self.TaskDict.keys():
            self.good_lapframes[t] = [scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', p))['E'].T for p in
                                      self.PlaceFieldData if t in p][0]

        self.numcells = np.size(self.Fc3data_dict['Task1'], 0)
        print(f'Total number of cells: %d' % self.numcells)

    def load_v73_Data(self):
        self.Fc3data_dict = self.create_data_dict()
        f = h5py.File(os.path.join(self.FolderName, self.ImgFileName[0]), 'r')
        for k, v in f.items():
            print(k, np.shape(v))

        count = 0
        for i in self.TaskDict.keys():
            self.Fc3data_dict[i] = f['Fc3'][:, count:count + self.Task_Numframes[i]]
            count += self.Task_Numframes[i]

    def get_and_sort_placeactivity(self):
        pc_activity_dict = {keys: [] for keys in self.TaskDict.keys()}
        pcsortednum = {keys: [] for keys in self.TaskDict.keys()}
        for i in self.PlaceFieldData:
            ft = i.find('Task')
            taskname = i[ft:ft + i[ft:].find('_')]
            print(taskname)
            x = scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', i))
            pc_activity = np.zeros(
                (np.int(np.sum(self.numPFS_incells[taskname])), np.int(self.tracklength / self.trackbins)))
            cellcount = 0
            for n in np.arange(np.size(self.sig_PFs_cellnum[taskname])):
                for i in np.arange(self.numPFS_incells[taskname][n]):
                    pc_activity[cellcount, :] = np.nanmean(x['sig_PFs'][i][self.sig_PFs_cellnum[taskname][n]], 1)
                    cellcount += 1
            pcsortednum[taskname] = np.argsort(np.nanargmax(pc_activity, 1))
            pc_activity_dict[taskname] = pc_activity
        return pc_activity_dict, pcsortednum

    def calculate_remapping_with_task(self, taskA):
        pc_activity_dict = {keys: [] for keys in self.new_taskDict.keys()}
        pcsortednum = {keys: [] for keys in self.new_taskDict.keys()}
        cells_to_plot = list(self.sig_PFs_cellnum[taskA])
        for i in self.PlaceFieldData:
            ft = i.find('Task')
            taskname = i[ft:ft + i[ft:].find('_')]
            x = scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', i))
            pc_activity = np.zeros(
                (np.int(np.sum(self.numPFS_incells[taskA])), np.int(self.tracklength / self.trackbins)))
            cellcount = 0
            for n, c in enumerate(cells_to_plot):
                for l in np.arange(self.numPFS_incells[taskA][n]):  # Iterate over how many ever place cells are there
                    # Separate lick and no lick in noreward task
                    if taskname == self.noreward_task:
                        pc_activity[cellcount, :] = np.nanmean((x['Allbinned_F'][0, c][:, :self.lickstoplap]), 1)
                    else:
                        pc_activity[cellcount, :] = np.nanmean((x['Allbinned_F'][0, c]), 1)
                    cellcount += 1
            pc_activity_dict[taskname] = pc_activity

        # Sort by taskA
        pcsorted = np.argsort(np.nanargmax(pc_activity_dict[taskA], 1))
        for taskname in self.new_taskDict.keys():
            pcsortednum[taskname] = pcsorted

        return pc_activity_dict, pcsortednum

    def correlate_acivity_of_allcellsbytask(self, TaskA='Task1'):
        data_formapping = [i for i in self.PlaceFieldData if TaskA in i][0]
        data_formapping = scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', data_formapping))['Allbinned_F']

        correlation_per_task = {keys: [] for keys in self.TaskDict.keys()}
        for i in self.PlaceFieldData:
            ft = i.find('Task')
            taskname = i[ft:ft + i[ft:].find('_')]
            print(taskname)
            x = scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', i))
            laps = np.size(x['Allbinned_F'][0, 0], 1)
            corr = np.zeros((self.numcells, laps))

            for c in range(0, self.numcells):
                data1 = np.nanmean(data_formapping[0, c], 1)
                data2 = np.nan_to_num(x['Allbinned_F'][0, c])
                for l in range(0, laps):

                    temp = np.corrcoef(data2[:, l], data1)[0, 1]
                    if ~np.isnan(temp):
                        corr[c, l] = temp
            correlation_per_task[taskname] = corr
        return correlation_per_task


class PlottingFunctions(GetData):
    def plot_placecells_with_track_pertask(self, pc_activity, sorted_pcs, figsize=(10, 4)):
        taskaxis = {'Task1': 0, 'Task2': 1, 'Task3': 2, 'Task4': 3, 'Task5': 4}
        fs, ax1 = plt.subplots(1, len(self.TaskDict), figsize=figsize, dpi=100, sharex='all', sharey='all')
        for taskname in self.TaskDict.keys():
            task_data = pc_activity[taskname][sorted_pcs[taskname], :]
            normalise_data = (task_data - np.min(task_data.flatten())) / (
                    np.max(task_data.flatten()) - np.min(task_data.flatten()))
            ax1[taskaxis[taskname]].imshow(normalise_data,
                                           aspect='auto', cmap='jet', interpolation='nearest', vmin=0, vmax=0.5)

            ax1[taskaxis[taskname]].set_title(self.new_taskDict[taskname])
            ax1[taskaxis[taskname]].set_xticks([0, 20, 39])
            ax1[taskaxis[taskname]].set_xticklabels([0, 100, 200])
            ax1[taskaxis[taskname]].set_xlim((0, 39))
            pf.set_axes_style(ax1[taskaxis[taskname]], numticks=4)
        ax1[0].set_xlabel('Track Length (cm)')
        ax1[0].set_ylabel('Cell')

        ax1[0].locator_params(nbins=4)
        fs.tight_layout()
        plt.show()

    def plot_correlation_by_task(self, data_to_plot, placecell_flag=0, taskA='Task1', figsize=(10, 6)):
        numlaps = self.Parsed_Behavior['numlaps'].item()
        numlicks_withinreward = self.Parsed_Behavior['numlicks_withinreward'].item()
        fs, axes = plt.subplots(2, len(self.TaskDict.keys()), sharex='col', sharey='row',
                                gridspec_kw={'height_ratios': [2, 1]},
                                figsize=figsize)

        count_axis = 0
        if placecell_flag:
            cells_to_plot = self.sig_PFs_cellnum[taskA]
            celldata = data_to_plot[taskA][cells_to_plot, :]
            sort_cells = np.argsort(np.nanmean(celldata, 1))[::-1]
        else:
            celldata = data_to_plot[taskA]
            sort_cells = np.argsort(np.nanmean(celldata, 1))[::-1]

        for n, i in enumerate(self.TaskDict.keys()):
            if placecell_flag:
                celldata = data_to_plot[i][cells_to_plot, :]
            else:
                celldata = data_to_plot[i]
            celldata = celldata[sort_cells, :]

            ax1 = axes[0, count_axis]
            ax1.imshow(celldata, interpolation='nearest', aspect='auto', cmap='viridis', vmin=0,
                       vmax=1)
            ax1.set_xlim((0, numlaps[i]))
            ax1.set_title(self.TaskDict[i])

            ax2 = axes[1, count_axis]
            ax2.plot(np.nanmean(celldata, 0), '-o', linewidth=2, color='b')
            # print(np.shape(celldata))
            print(f'%s : %0.3f' % (i, np.nanmean(celldata[:, 1:])))
            ax2.set_xlabel('Lap number')
            ax3 = ax2.twinx()  # instantiate a second axes that shares the same x-axis
            ax3.plot(numlicks_withinreward[i], '-o', color='r', alpha=0.5,
                     label='Lap time')
            ax3.set_ylim((0, 7))

            if n == len(self.TaskDict.keys()) - 1:
                ax3.set_ylabel('Pre Licks', color='r')
            else:
                ax3.set_yticklabels([])

            for l in range(0, numlaps[i] - 1):
                if numlicks_withinreward[i][l]:
                    ax2.axvline(l, linewidth=0.25, color='k')
            count_axis += 1
            for a in [ax1, ax2, ax3]:
                pf.set_axes_style(a)
        axes[0, 0].set_ylabel('Cell Number')
        axes[1, 0].set_ylabel('Mean Correlation', color='b')
        fs.subplots_adjust(wspace=0.1, hspace=0.1)
