from sklearn.model_selection import train_test_split
import os
import numpy as np
from collections import OrderedDict
import scipy.stats
import matplotlib.pyplot as plt
import sys
from statistics import mean
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import h5py
from scipy.ndimage import gaussian_filter1d

sys.path.append('rastermap/rastermap/')
import mapping, importlib

importlib.reload(mapping)

DataDetailsFolder = '/home/sheffieldlab/Desktop/NoReward/Scripts/AnimalDetails/'
sys.path.append(DataDetailsFolder)
import DataDetails

# For plotting styles
PlottingFormat_Folder = '/home/sheffieldlab/Desktop/NoReward/Scripts/PlottingTools/'
sys.path.append(PlottingFormat_Folder)
import plottingfunctions as pf

pf.set_style()


class LoadData(object):
    def __init__(self, AnimalName, FolderName, CompiledFolderName,
                 classifier_type, taskstoplot):
        print('Loading Data')
        self.animalname = AnimalName
        self.animalinfo = DataDetails.ExpAnimalDetails(self.animalname)
        self.FolderName = os.path.join(FolderName, self.animalname)
        self.CompiledFolderName = CompiledFolderName  # For importing Bayes results
        self.Task_Numframes = self.animalinfo['task_numframes']
        self.TaskDict = self.animalinfo['task_dict']
        self.classifier_type = classifier_type
        self.framespersec = 30.98
        self.taskstoplot = taskstoplot
        self.trackbins = 5

        # Run functions
        self.get_data_folders()
        if self.animalinfo['v73_flag']:
            self.load_v73_Data()
        else:
            self.load_fluorescentdata()
        self.get_place_cells()
        self.load_behaviordata()
        self.load_lapparams()
        self.rasterdata = self.combinedata_forraster(self.taskstoplot)
        self.colors = sns.color_palette('deep', len(self.taskstoplot))

    def get_data_folders(self):
        self.ImgFileName = [f for f in os.listdir(self.FolderName) if f.endswith('.mat')]
        self.Parsed_Behavior = np.load(os.path.join(self.FolderName, 'SaveAnalysed', 'behavior_data.npz'),
                                       allow_pickle=True)
        self.PlaceCells = np.load(
            os.path.join(self.FolderName, 'PlaceCells', f'%s_placecell_data.npz' % self.animalname), allow_pickle=True)
        BayesFile = \
            [f for f in os.listdir(self.CompiledFolderName) if self.animalname in f and self.classifier_type in f][0]
        self.BayesData = np.load(os.path.join(self.CompiledFolderName, BayesFile), allow_pickle=True)
        self.PlaceFieldData = \
            [f for f in os.listdir(os.path.join(self.FolderName, 'Behavior')) if
             (f.endswith('.mat') and 'PlaceFields' in f and 'GoodBehavior' in f and 'Lick' not in f)]

    def load_fluorescentdata(self):
        self.Fcdata_dict = CommonFunctions.create_data_dict(self.TaskDict)
        self.Fc3data_dict = CommonFunctions.create_data_dict(self.TaskDict)
        # Open calcium data and store in dicts per trial
        data = scipy.io.loadmat(os.path.join(self.FolderName, self.ImgFileName[0]))
        self.numcells = np.size(data['data'].item()[1], 1)
        self.meanimg = data['data'].item()[5]
        count = 0
        for i in self.TaskDict.keys():
            self.Fcdata_dict[i] = data['data'].item()[1].T[:,
                                  count:count + self.Task_Numframes[i]]
            self.Fc3data_dict[i] = data['data'].item()[2].T[:,
                                   count:count + self.Task_Numframes[i]]
            count += self.Task_Numframes[i]

    def load_v73_Data(self):
        self.Fcdata_dict = CommonFunctions.create_data_dict(self.TaskDict)
        self.Fc3data_dict = CommonFunctions.create_data_dict(self.TaskDict)
        f = h5py.File(os.path.join(self.FolderName, self.ImgFileName[0]), 'r')
        for k, v in f.items():
            print(k, np.shape(v))

        count = 0
        for i in self.TaskDict.keys():
            self.Fcdata_dict[i] = f['Fc'][:, count:count + self.Task_Numframes[i]]
            self.Fc3data_dict[i] = f['Fc3'][:, count:count + self.Task_Numframes[i]]
            count += self.Task_Numframes[i]

    def load_behaviordata(self):
        # Load required behavior data
        self.good_running_data = CommonFunctions.create_data_dict(self.TaskDict)
        self.good_running_index = CommonFunctions.create_data_dict(self.TaskDict)
        self.actual_laptime = CommonFunctions.create_data_dict(self.TaskDict)
        self.lick_data = CommonFunctions.create_data_dict(self.TaskDict)
        self.numlaps = CommonFunctions.create_data_dict(self.TaskDict)
        self.numlicksperlap = CommonFunctions.create_data_dict(self.TaskDict)

        for keys in self.TaskDict.keys():
            self.good_running_data[keys] = self.Parsed_Behavior['good_running_data'].item()[keys]
            self.good_running_index[keys] = self.Parsed_Behavior['good_running_index'].item()[keys]
            self.actual_laptime[keys] = self.Parsed_Behavior['actuallaps_laptime'].item()[keys]
            self.lick_data[keys] = self.Parsed_Behavior['corrected_lick_data'].item()[keys]
            self.lick_data[keys][self.lick_data[keys] == 0] = np.nan
            self.numlicksperlap[keys] = self.Parsed_Behavior['numlicks_withinreward'].item()[keys]
            self.numlaps[keys] = self.Parsed_Behavior['numlaps'].item()[keys]

    def load_lapparams(self):
        self.good_lapframes = CommonFunctions.create_data_dict(self.TaskDict)
        for t in self.TaskDict.keys():
            self.good_lapframes[t] = [scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', p))['E'].T for p in
                                      self.PlaceFieldData if t in p and 'Task2a' not in p][0]
            self.good_lapframes[t] = self.good_lapframes[t][:self.animalinfo['task_framestokeep'][t]]

    def combinedata_forraster(self, taskstoplot, placecell_flag=0):
        xdata = np.array([])
        for t in taskstoplot:
            index = self.good_running_index[t]
            if placecell_flag:
                data_task = self.Fc3data_placecells[t][:, index][:, :self.animalinfo['task_framestokeep'][t]]
            else:
                data_task = self.Fc3data_dict[t][:, index][:, :self.animalinfo['task_framestokeep'][t]]
            xdata = np.hstack((xdata, data_task)) if xdata.size else data_task
            # Correct other data
            self.good_running_data[t] = self.good_running_data[t][:self.animalinfo['task_framestokeep'][t]]
            self.lick_data[t] = self.lick_data[t][:self.animalinfo['task_framestokeep'][t]]
            self.Fc3data_dict[t] = self.Fc3data_dict[t][:, index][:, :self.animalinfo['task_framestokeep'][t]]
            self.Fcdata_dict[t] = self.Fcdata_dict[t][:, index][:, :self.animalinfo['task_framestokeep'][t]]
        return xdata

    def get_bayes_data(self, axis, accuracy_measure='R2'):
        decoderfit = self.BayesData['fit'].item()
        count = 0
        for n, t in enumerate(self.taskstoplot):
            y_actual = decoderfit[t]['ytest']
            y_predicted = decoderfit[t]['yang_pred']
            R2 = self.calulate_lapwiseerror(y_actual, y_predicted, self.good_lapframes[t], accuracy_measure)
            if t == 'Task1':
                R2 = R2[-5:]
            numlaps = np.size(R2)
            print(numlaps)
            x = np.arange(count, count + numlaps)
            axis.plot(x, R2, '.-', color=self.colors[n], linewidth=2, markeredgecolor='black', zorder=2)
            for nl in np.arange(numlaps):
                if self.numlicksperlap[t][nl] > 0:
                    axis.axvline(nl + count, color='k', alpha=0.5, linewidth=1, markersize=10, zorder=1)
            count += numlaps
        if accuracy_measure == 'R2':
            axis.set_ylabel('Goodness of fit')
        else:
            axis.set_ylabel('Decoder Error (cm)')
        pf.set_axes_style(axis, numticks=3)

    def calulate_lapwiseerror(self, y_actual, y_predicted, lapframes, accuracy_measure='R2'):
        lap_error = []
        for l in np.arange(np.max(lapframes) - 1):
            laps = np.where(lapframes == l + 1)[0]
            if accuracy_measure == 'R2':
                error = CommonFunctions.get_R2(y_actual[laps], y_predicted[laps])
            else:
                error = CommonFunctions.get_y_difference(y_actual[laps], y_predicted[laps]) * self.trackbins
            lap_error.append(error)
        return np.asarray(lap_error)

    def get_place_cells(self):
        self.Fc3data_placecells = CommonFunctions.create_data_dict(self.TaskDict)
        pcells = self.PlaceCells['sig_PFs_cellnum'].item()['Task1']
        for t in self.TaskDict:
            self.Fc3data_placecells[t] = self.Fc3data_dict[t][pcells, :]
            print(np.shape(self.Fc3data_placecells[t]))


class plotraster(LoadData):
    def make_rastermap(self, ncomp=1, nx=30, npc=200):
        model = mapping.Rastermap(n_components=ncomp, n_X=nx, nPC=npc).fit(self.rasterdata)
        self.isort = np.argsort(model.embedding[:, 0])
        Sm = gaussian_filter1d(self.rasterdata[self.isort, :].T,
                               np.minimum(1, int(self.rasterdata.shape[0] * 0.005)),
                               axis=1)
        self.Sm = Sm.T
        print(np.shape(self.Sm))

    def plot_rastermap(self, ax, ylim=0):
        if ylim == 0:
            ylim = self.numcells

        ax[0].imshow(self.Sm, vmin=0, vmax=0.5, cmap='jet', aspect='auto',
                     extent=[0, self.Sm.shape[1] / self.framespersec, 0, self.Sm.shape[0]])
        ax[0].set_ylim((0, ylim))
        count = 0
        for n, i in enumerate(self.taskstoplot):
            x = np.linspace(count, count + np.size(self.good_running_data[i]) / self.framespersec,
                            np.size(self.good_running_data[i]))
            ax[0].axvline(x[-1], color='k')
            ax[1].plot(x, self.good_running_data[i], color=self.colors[n])
            count += np.size(self.good_running_data[i]) / self.framespersec
            ax[1].plot(x, self.lick_data[i] * 0.7, '|', color='grey', linewidth=0.5, markersize=10)
        ax[1].set_yticklabels([])

        for a in ax:
            pf.set_axes_style(a, numticks=2)

    def plot_samplecells(self, numcells, cellnumber, **kwargs):
        print(cellnumber)
        if 'ax' not in kwargs.keys():
            f, ax = plt.subplots(numcells + 1, 1, figsize=(15, (numcells + 1) / 2), sharex='all',
                                 gridspec_kw={'hspace': 0.1})
        # # Plot lick
        # count = 0
        # for n2, t in enumerate(self.taskstoplot):
        #     x = np.linspace(count, count + np.size(self.good_running_data[t]) / self.framespersec,
        #                     np.size(self.good_running_data[t]))
        #     ax[0].plot(x, self.lick_data[t], '|', color='k',
        #                linewidth=0.8, markersize=10)
        #     count += np.size(self.good_running_data[t]) / self.framespersec
        # ax[0].axis('off')

        for n1, c in enumerate(cellnumber):
            count = 0
            if 'ax' in kwargs.keys():
                axis = kwargs['ax']
            else:
                axis = ax[n1]
            for n2, t in enumerate(self.taskstoplot):
                x = np.linspace(count, count + np.size(self.good_running_data[t]) / self.framespersec,
                                np.size(self.good_running_data[t]))
                axis.plot(x, self.Fcdata_dict[t][c, :], color=self.colors[n2])
                axis.set_ylabel(c)
                axis.set_ylim((0, 2))
                axis.set_yticklabels('')
                count += np.size(self.good_running_data[t]) / self.framespersec

            pf.set_axes_style(axis, numticks=1)

    def plot_mean_image_and_cellmask(self):
        plt.imshow(self.meanimg, cmap='gray')
        plt.axis('off')


class CommonFunctions(object):

    @staticmethod
    def create_data_dict(TaskDict):
        data_dict = {keys: [] for keys in TaskDict.keys()}
        return data_dict

    @staticmethod
    def get_R2(y_actual, y_predicted):
        y_mean = np.mean(y_actual)
        R2 = 1 - np.sum((y_predicted - y_actual) ** 2) / np.sum((y_actual - y_mean) ** 2)
        return R2

    @staticmethod
    def accuracy_metric(y_actual, y_predicted):
        correct = 0
        for i in range(len(y_actual)):
            if y_actual[i] == y_predicted[i]:
                correct += 1
        if len(y_actual) == 0:
            return np.nan
        else:
            return correct / float(len(y_actual))

    @staticmethod
    def get_y_difference(y_actual, y_predicted):
        y_diff = np.mean(np.abs(y_predicted - y_actual))
        return y_diff
