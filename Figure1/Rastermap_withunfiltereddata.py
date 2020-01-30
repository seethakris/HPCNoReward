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
        self.Task_Numframes['Task3'] = 14999
        self.removeframesforbayes = self.animalinfo['task_framestokeep']
        self.TaskDict = self.animalinfo['task_dict']
        self.framespersec = 30.98
        self.trackbins = 5

        self.get_data_folders()
        self.load_lapparams()
        self.load_behaviordata()

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
        self.CellDataFolder = os.path.join(self.FolderName, 'Suite2psegmentation')
        if self.animalinfo['v73_flag']:
            self.load_v73_Data()
        else:
            self.load_fluorescentdata()

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
        self.numcells = np.size(f['Fc'], 0)
        for i in self.TaskDict.keys():
            self.Fcdata_dict[i] = f['Fc'][:, count:count + self.Task_Numframes[i]]
            self.Fc3data_dict[i] = f['Fc3'][:, count:count + self.Task_Numframes[i]]
            count += self.Task_Numframes[i]

    def load_lapparams(self):
        self.actual_lapframes = CommonFunctions.create_data_dict(self.TaskDict)
        for t in self.TaskDict.keys():
            self.actual_lapframes[t] = \
                [scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', p))['bad_E'].T for p in
                 self.PlaceFieldFolder if t in p and 'Task2a' not in p][0]

    def load_behaviordata(self):
        # Load required behavior data
        self.running_data = CommonFunctions.create_data_dict(self.TaskDict)
        self.lick_data = CommonFunctions.create_data_dict(self.TaskDict)
        self.numlaps = CommonFunctions.create_data_dict(self.TaskDict)

        for keys in self.TaskDict.keys():
            self.running_data[keys] = self.Parsed_Behavior['running_data'].item()[keys]
            self.lick_data[keys] = self.Parsed_Behavior['lick_data'].item()[keys]
            self.lick_data[keys][self.lick_data[keys] < 0.1] = np.nan
            self.lick_data[keys][np.where(~np.isnan(self.lick_data[keys]))[0]] = 1
            self.numlaps[keys] = self.Parsed_Behavior['numlaps'].item()[keys]

    def filter_cells_withhighskew(self, threshold=4):
        celldata = np.load(os.path.join(self.CellDataFolder, 'stat.npy'), allow_pickle=True)
        iscell = np.load(os.path.join(self.CellDataFolder, 'iscell.npy'), allow_pickle=True)
        skew = []
        for i in np.arange(len(celldata)):
            if iscell[i, 0] == 1:
                skew.append(celldata[i]['skew'])
        skew = np.asarray(skew)
        cells_withlowskew = np.where(skew < threshold)[0]
        return skew, cells_withlowskew, iscell

    def combinedata_correct_forraster(self, taskframes, remove_laps):
        xdata = np.array([])
        for t in self.taskstoplot:
            if remove_laps[t] > 0:
                remove_index = np.where(self.actual_lapframes[t] != remove_laps[t] + 2)[0]
                data_task = self.Fc3data_dict[t][:, remove_index][:, taskframes[t][0]:taskframes[t][1]]
                xdata = np.hstack((xdata, data_task)) if xdata.size else data_task
                self.running_data[t] = self.running_data[t][remove_index][taskframes[t][0]:taskframes[t][1]]
                self.lick_data[t] = self.lick_data[t][remove_index][taskframes[t][0]:taskframes[t][1]]
                self.Fc3data_dict[t] = self.Fc3data_dict[t][:, remove_index][:, taskframes[t][0]:taskframes[t][1]]
                self.Fcdata_dict[t] = self.Fcdata_dict[t][:, remove_index][:, taskframes[t][0]:taskframes[t][1]]
            else:
                data_task = self.Fc3data_dict[t][:, taskframes[t][0]:taskframes[t][1]]
                xdata = np.hstack((xdata, data_task)) if xdata.size else data_task
                self.running_data[t] = self.running_data[t][taskframes[t][0]:taskframes[t][1]]
                self.lick_data[t] = self.lick_data[t][taskframes[t][0]:taskframes[t][1]]
                self.Fc3data_dict[t] = self.Fc3data_dict[t][:, taskframes[t][0]:taskframes[t][1]]
                self.Fcdata_dict[t] = self.Fcdata_dict[t][:, taskframes[t][0]:taskframes[t][1]]
            print(t)
            print(np.shape(self.running_data[t]))
            print(np.shape(self.lick_data[t]))
            print(np.shape(data_task))
        # xdata = xdata[:, :-1]
        # self.Fcdata_dict['Task3'] = self.Fcdata_dict['Task3'][:-1]
        return xdata

    def make_rastermap(self, xdata, ncomp=1, nx=30, npc=200):
        model = mapping.Rastermap(n_components=ncomp, n_X=nx, nPC=npc).fit(xdata)
        self.isort = np.argsort(model.embedding[:, 0])
        Sm = gaussian_filter1d(xdata[self.isort, :].T,
                               0.5,
                               axis=1)
        self.Sm = xdata
        print(np.shape(self.Sm))

    def crop_Sm_and_sort(self, cells_to_use, **kwargs):
        if 'tosort' in kwargs.keys():
            crpdSm = np.array([])
            for c in cells_to_use:
                if crpdSm.size:
                    crpdSm = np.vstack((crpdSm, kwargs['tosort'][c[0]:c[1], :]))
                else:
                    crpdSm = kwargs['tosort'][c[0]:c[1], :]
            return crpdSm
        else:
            self.crpdSm = np.array([])
            for c in cells_to_use:
                self.crpdSm = np.vstack((self.crpdSm, self.Sm[c[0]:c[1], :])) if self.crpdSm.size else self.Sm[
                                                                                                       c[0]:c[1], :]
        # self.crpdSm = self.crpdSm / np.max(self.crpdSm, 1)[:, np.newaxis]

    def plot_rastermap(self, fighandle, ax, fdata, rasterdata, ylim=0, crop_cellflag=0, ylim_meandff=0.6):
        if ylim == 0 and crop_cellflag == 0:
            ylim = np.size(rasterdata, 0)
        elif ylim == 0 and crop_cellflag == 1:
            ylim = np.size(self.crpdSm, 0)
        jet = cm.get_cmap('jet', 256)

        if crop_cellflag:
            im = ax[0].imshow(self.crpdSm, vmin=0, vmax=0.1, cmap='plasma', aspect='auto', interpolation='hanning',
                              extent=[0, self.crpdSm.shape[1] / self.framespersec, 0, self.crpdSm.shape[0]])
            CommonFunctions.plot_colorbar(fighandle, ax[0], im, title=f'\u0394F/F', ticks=[0, 0.7])
        else:
            ax[0].imshow(rasterdata, vmin=0, vmax=0.3, cmap='plasma', aspect='auto', interpolation='hanning',
                         extent=[0, rasterdata.shape[1] / self.framespersec, 0, rasterdata.shape[0]])
        ax[0].set_ylim((0, ylim))
        ax[0].set_xlim((0, self.Sm.shape[1] / self.framespersec))
        ax[0].set_ylabel('Cell Number')

        count = 0
        count_dff = 0
        for n, i in enumerate(self.taskstoplot):
            x = np.linspace(count, count + np.size(self.running_data[i]) / self.framespersec,
                            np.size(self.running_data[i]))
            ax[0].axvline(x[-1], color='k', linewidth=0.5)
            ax[2].plot(x, self.running_data[i], color=self.colors[n], linewidth=0.5)
            ax[2].plot(x, self.lick_data[i] * 0.75, '|', color='grey', markeredgewidth=0.5, markersize=5)
            ax[1].plot(x, savgol_filter(
                np.nanmean(fdata[:, count_dff:count_dff + np.size(self.running_data[i])], 0), 31, 2),
                       color=self.colors[n], linewidth=0.5)
            count += np.size(self.running_data[i]) / self.framespersec
            count_dff += np.size(self.running_data[i])

        ax[2].set_yticklabels([])
        ax[1].set_ylim((0, ylim_meandff))

        pf.set_axes_style(ax[0], numticks=4)
        pf.set_axes_style(ax[1], numticks=1)

    def plot_samplecells(self, cellnumber, axis):
        count = 0
        for n2, t in enumerate(self.taskstoplot):
            x = np.linspace(count, count + np.size(self.running_data[t]) / self.framespersec,
                            np.size(self.running_data[t]))
            axis.plot(x, np.squeeze(self.Fcdata_dict[t][cellnumber, :]), color=self.colors[n2], linewidth=0.5)
            count += np.size(self.running_data[t]) / self.framespersec
        axis.set_xlabel('Time (seconds)')
        axis.set_ylim((0, 2.5))
        axis.set_yticklabels('')
        pf.set_axes_style(axis, numticks=1)


class CommonFunctions(object):
    @staticmethod
    def create_data_dict(TaskDict):
        data_dict = {keys: [] for keys in TaskDict.keys()}
        return data_dict

    @staticmethod
    def calulate_lapwiseerror(y_actual, y_predicted, numlaps, lapframes):
        lap_R2 = []
        for l in np.arange(numlaps - 1):
            laps = np.where(lapframes == l + 1)[0]
            lap_R2.append(CommonFunctions.get_R2(y_actual[laps], y_predicted[laps]))

        return np.asarray(lap_R2)

    @staticmethod
    def get_R2(y_actual, y_predicted):
        y_mean = np.mean(y_actual)
        R2 = 1 - np.sum((y_predicted - y_actual) ** 2) / np.sum((y_actual - y_mean) ** 2)
        if np.isinf(R2):
            R2 = 0
        return R2

    @staticmethod
    def plot_colorbar(fighandle, ax, imobject, title, ticks, cwidth='3%', cheight='60%'):
        axins = inset_axes(ax,
                           width=cwidth,
                           height=cheight,
                           loc='lower left',
                           bbox_to_anchor=(1.02, 0., 1, 1),
                           bbox_transform=ax.transAxes,
                           borderpad=0,
                           )
        cb = fighandle.colorbar(imobject, cax=axins, pad=0.01, ticks=ticks)
        cb.outline.set_visible(False)
        cb.set_label(title, rotation=270, labelpad=8)
        cb.ax.tick_params(size=0)
