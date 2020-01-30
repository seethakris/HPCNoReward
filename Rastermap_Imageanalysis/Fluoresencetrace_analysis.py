import os
import numpy as np
import scipy.stats
import sys
import seaborn as sns
import h5py
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import cm
from matplotlib.colors import hsv_to_rgb
from scipy.signal import savgol_filter

sys.path.append('/home/sheffieldlab/Desktop/NoReward/Scripts/rastermap/rastermap/')
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


class PlotRaster(object):
    def __init__(self, AnimalName, FolderName, SaveFigureFolder, taskstoplot):
        self.taskstoplot = taskstoplot
        self.SaveFigureFolder = SaveFigureFolder
        self.colors = sns.color_palette('deep')
        self.task2_colors = [self.colors[1], self.colors[3]]

        self.animalname = AnimalName
        self.animalinfo = DataDetails.ExpAnimalDetails(self.animalname)
        self.ParentFolderName = FolderName
        self.FolderName = os.path.join(FolderName, self.animalname)
        self.Task_Numframes = self.animalinfo['task_numframes']
        self.removeframesforbayes = self.animalinfo['task_framestokeep']
        self.TaskDict = self.animalinfo['task_dict']
        self.framespersec = 30.98
        self.trackbins = 5

        self.get_data_folders()
        if self.animalinfo['v73_flag']:
            self.load_v73_Data()
        else:
            self.load_fluorescentdata()
        self.load_lapparams()
        self.load_behaviordata()
        self.raster_fdata, self.raster_cdata = self.combinedata_correct_forraster()
        self.make_rastermap(self.raster_fdata, self.raster_cdata)
        fs, ax = plt.subplots(3, sharex='all', dpi=300, gridspec_kw={'height_ratios': [2, 0.5, 0.5], 'hspace': 0.3})
        self.plot_rastermap(ax, fdata=self.raster_fdata, crop_cellflag=0, ylim_meandff=0.1)

    def get_data_folders(self):
        self.ImgFileName = [f for f in os.listdir(self.FolderName) if f.endswith('.mat')]
        self.Parsed_Behavior = np.load(os.path.join(self.FolderName, 'SaveAnalysed', 'behavior_data.npz'),
                                       allow_pickle=True)
        self.PlaceFieldFolder = \
            [f for f in os.listdir(os.path.join(self.FolderName, 'Behavior')) if
             (f.endswith('.mat') and 'PlaceFields' in f and 'GoodBehavior' in f and 'Lick' not in f)]
        self.PlaceCells = np.load(
            os.path.join(self.FolderName, 'PlaceCells', f'%s_placecell_data.npz' % self.animalname), allow_pickle=True)
        self.Correlation_Data = self.PlaceCells['correlation_withTask1'].item()

    def load_fluorescentdata(self):
        self.Fcdata_dict = CommonFunctions.create_data_dict(self.TaskDict)
        self.Fc3data_dict = CommonFunctions.create_data_dict(self.TaskDict)
        # Open calcium data and store in dicts per trial
        data = scipy.io.loadmat(os.path.join(self.FolderName, self.ImgFileName[0]))
        self.numcells = np.size(data['data'].item()[1], 1)
        # self.meanimg = data['data'].item()[5]
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
        self.numcells = np.size(f['Fc'], 0)
        for i in self.TaskDict.keys():
            self.Fcdata_dict[i] = f['Fc'][:, count:count + self.Task_Numframes[i]]
            self.Fc3data_dict[i] = f['Fc3'][:, count:count + self.Task_Numframes[i]]
            count += self.Task_Numframes[i]

    def load_lapparams(self):
        self.good_lapframes = CommonFunctions.create_data_dict(self.TaskDict)
        for t in self.TaskDict.keys():
            self.good_lapframes[t] = [scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', p))['E'].T for p in
                                      self.PlaceFieldFolder if t in p and 'Task2a' not in p][0]
            self.good_lapframes[t] = self.good_lapframes[t]

    def load_behaviordata(self):
        # Load required behavior data
        self.good_running_data = CommonFunctions.create_data_dict(self.TaskDict)
        self.good_running_index = CommonFunctions.create_data_dict(self.TaskDict)
        self.lick_data = CommonFunctions.create_data_dict(self.TaskDict)
        self.numlaps = CommonFunctions.create_data_dict(self.TaskDict)
        self.numlicksperlap = CommonFunctions.create_data_dict(self.TaskDict)

        for keys in self.TaskDict.keys():
            self.good_running_data[keys] = self.Parsed_Behavior['good_running_data'].item()[keys]
            self.good_running_index[keys] = self.Parsed_Behavior['good_running_index'].item()[keys]
            self.lick_data[keys] = self.Parsed_Behavior['corrected_lick_data'].item()[keys]
            self.lick_data[keys][self.lick_data[keys] == 0] = np.nan
            self.numlicksperlap[keys] = self.Parsed_Behavior['numlicks_withinreward_alllicks'].item()[keys]
            self.numlaps[keys] = self.Parsed_Behavior['numlaps'].item()[keys]
            self.lickstoplap = self.Parsed_Behavior['lick_stop'].item()

    def combinedata_correct_forraster(self):
        xdata = np.array([])
        cdata = np.array([])
        for t in self.taskstoplot:
            # Combine raster data
            index = self.good_running_index[t]
            data_task = self.Fc3data_dict[t][:, index]
            xdata = np.hstack((xdata, data_task)) if xdata.size else data_task
            data_task = self.Correlation_Data[t]
            cdata = np.hstack((cdata, data_task)) if cdata.size else data_task
            print(np.shape(self.Fc3data_dict[t][:, index]), np.shape(self.Correlation_Data[t]),
                  np.shape(self.good_running_data[t]))
        return xdata, cdata

    def make_rastermap(self, xdata, cdata, ncomp=1, nx=30, npc=200):
        xdata = xdata[np.argsort(np.max(cdata, 1)), :]
        model = mapping.Rastermap(n_components=ncomp, n_X=nx, nPC=npc).fit(xdata)
        self.isort = np.argsort(model.embedding[:, 0])
        Sm = gaussian_filter1d(xdata[self.isort, :].T,
                               0.5,
                               axis=1)
        # np.minimum(1, int(xdata.shape[0] * 0.001)),
        self.Sm = Sm.T
        print(np.shape(self.Sm))

    def crop_Sm_and_sort(self, cells_to_use, cdata):
        self.crpdSm = np.array([])
        cdata_crpd = np.array([])
        for c in cells_to_use:
            cdata_crpd = np.vstack((cdata_crpd, cdata[c[0]:c[1], :])) if cdata_crpd.size else cdata[c[0]:c[1], :]
            self.crpdSm = np.vstack((self.crpdSm, self.Sm[c[0]:c[1], :])) if self.crpdSm.size else self.Sm[c[0]:c[1], :]

    def plot_rastermap(self, ax, fdata, ylim=0, crop_cellflag=0, ylim_meandff=0.6):
        if ylim == 0 and crop_cellflag == 0:
            ylim = np.size(fdata, 0)
        elif ylim == 0 and crop_cellflag == 1:
            ylim = np.size(self.crpdSm, 0)
        jet = cm.get_cmap('jet', 256)

        if crop_cellflag:
            ax[0].imshow(self.crpdSm, vmin=0, vmax=0.7, cmap=jet, aspect='auto', rasterized=True,
                         interpolation='nearest',
                         extent=[0, self.crpdSm.shape[1] / self.framespersec, 0, self.crpdSm.shape[0]])
        else:
            ax[0].imshow(self.Sm, vmin=0, vmax=0.7, cmap='jet', aspect='auto',
                         extent=[0, self.Sm.shape[1] / self.framespersec, 0, self.Sm.shape[0]])
        ax[0].set_ylim((0, ylim))
        ax[0].set_xlim((0, self.Sm.shape[1] / self.framespersec))
        ax[0].set_ylabel('Cell Number')

        count = 0
        count_dff = 0
        for n, i in enumerate(self.taskstoplot):
            x = np.linspace(count, count + np.size(self.good_running_data[i]) / self.framespersec,
                            np.size(self.good_running_data[i]))
            ax[0].axvline(x[-1], color='k', linewidth=0.5)
            ax[2].plot(x, self.good_running_data[i], color=self.colors[n], linewidth=0.5)
            ax[2].plot(x, self.lick_data[i] * 0.75, '|', color='grey', markeredgewidth=0.05, markersize=5,
                       alpha=0.5)
            ax[1].plot(x, savgol_filter(
                np.nanmean(fdata[:, count_dff:count_dff + np.size(self.good_running_data[i])], 0), 101, 2),
                       color=self.colors[n], linewidth=0.5)
            count += np.size(self.good_running_data[i]) / self.framespersec
            count_dff += np.size(self.good_running_data[i])

        ax[2].set_yticklabels([])
        ax[1].set_ylim((0, ylim_meandff))
        pf.set_axes_style(ax[0], numticks=4)
        pf.set_axes_style(ax[1], numticks=1)


class CommonFunctions(object):
    @staticmethod
    def create_data_dict(TaskDict):
        data_dict = {keys: [] for keys in TaskDict.keys()}
        return data_dict
