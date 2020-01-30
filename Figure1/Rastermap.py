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
from itertools import groupby

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
    def __init__(self, AnimalName, FolderName, SaveFigureFolder, taskstoplot, controlflag=0):
        print('Loading Data')
        self.taskstoplot = taskstoplot
        self.SaveFigureFolder = SaveFigureFolder
        self.controlflag = controlflag
        if self.controlflag:
            self.colors = sns.color_palette(["#3498db", "#9b59b6"])
        else:
            self.colors = sns.color_palette('deep')
            self.task2_colors = [self.colors[1], self.colors[3]]

        self.animalname = AnimalName
        if self.controlflag:
            self.animalinfo = DataDetails.ControlAnimals(self.animalname)
        else:
            self.animalinfo = DataDetails.ExpAnimalDetails(self.animalname)

        self.ParentFolderName = FolderName
        self.FolderName = os.path.join(FolderName, self.animalname)
        self.Task_Numframes = self.animalinfo['task_numframes']
        self.removeframesforbayes = self.animalinfo['task_framestokeep']
        self.TaskDict = self.animalinfo['task_dict']
        self.framespersec = 30.98
        self.trackbins = 5

        # Run functions
        self.get_data_folders()
        if self.animalinfo['v73_flag']:
            self.load_v73_Data()
        else:
            self.load_fluorescentdata()
        self.load_Bayesfit()
        self.load_behaviordata()
        self.load_lapparams()

        if not self.controlflag:
            self.lickstoplap = np.int(self.lickstoplap['Task2'] - 1)
            self.lickstopframe = np.where(self.good_lapframes['Task2'] == self.lickstoplap)[0][0]

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

    def load_Bayesfit(self):
        BayesFolder = os.path.join(self.ParentFolderName, 'BayesResults_All')
        Bayesfile = [f for f in os.listdir(BayesFolder) if self.animalname in f][0]
        self.bayesdata = np.load(os.path.join(BayesFolder, Bayesfile), allow_pickle=True)
        bayesfit = self.bayesdata['fit'].item()
        self.bayesR2 = {k: [] for k in self.taskstoplot}
        self.bayes_yactual = {k: [] for k in self.taskstoplot}
        self.bayes_ypred = {k: [] for k in self.taskstoplot}
        for t in self.taskstoplot:
            self.bayesR2[t] = CommonFunctions.calulate_lapwiseerror(y_actual=bayesfit[t]['ytest'],
                                                                    y_predicted=bayesfit[t]['yang_pred'],
                                                                    numlaps=self.bayesdata['numlaps'].item()[t],
                                                                    lapframes=self.bayesdata['lapframes'].item()[t])

            self.bayes_yactual[t] = bayesfit[t]['ytest']
            self.bayes_ypred[t] = bayesfit[t]['yang_pred']

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

        if not self.controlflag:
            self.lickstoplap = self.Parsed_Behavior['lick_stop'].item()

    def combinedata_correct_forraster(self, taskframes, remove_laps):
        xdata = np.array([])
        cdata = np.array([])
        bayesdata = np.array([])

        for t in self.taskstoplot:
            index = self.good_running_index[t]
            # print(np.shape(index))
            if remove_laps[t] > 0:
                remove_index = np.where(self.good_lapframes[t] != remove_laps[t] + 1)[0]
                index = index[remove_index]
                self.good_running_data[t] = self.good_running_data[t][remove_index][taskframes[t][0]:taskframes[t][1]]
                self.lick_data[t] = self.lick_data[t][remove_index][taskframes[t][0]:taskframes[t][1]]
                remove_index = remove_index[:self.removeframesforbayes[t]]
                self.bayes_ypred[t] = self.bayes_ypred[t][remove_index][
                                      taskframes[t][0]:taskframes[t][1] - self.removeframesforbayes[t]]
                self.bayes_yactual[t] = self.bayes_yactual[t][remove_index][
                                        taskframes[t][0]:taskframes[t][1] - self.removeframesforbayes[t]]
            else:
                self.good_running_data[t] = self.good_running_data[t][taskframes[t][0]:taskframes[t][1]]
                self.lick_data[t] = self.lick_data[t][taskframes[t][0]:taskframes[t][1]]
                self.bayes_ypred[t] = self.bayes_ypred[t][
                                      taskframes[t][0]:taskframes[t][1] - self.removeframesforbayes[t]]
                self.bayes_yactual[t] = self.bayes_yactual[t][
                                        taskframes[t][0]:taskframes[t][1] - self.removeframesforbayes[t]]
            # Combine raster data
            data_task = self.Fc3data_dict[t][:, index][:, taskframes[t][0]:taskframes[t][1]]
            xdata = np.hstack((xdata, data_task)) if xdata.size else data_task

            # Combine correlation and numlaps
            laps_used = np.unique(self.good_lapframes[t][taskframes[t][0]:taskframes[t][1]])
            laps_used = laps_used[laps_used > 0] - 1
            # print(laps_used)
            if remove_laps[t] > 0:
                laps_used = laps_used[np.where(laps_used != remove_laps[t])[0]]
            # print(laps_used)
            data_task = self.Correlation_Data[t][:, laps_used]
            cdata = np.hstack((cdata, data_task)) if cdata.size else data_task
            self.numlicksperlap[t] = self.numlicksperlap[t][laps_used]

            # Combine BayesData
            R2_data = self.bayesR2[t][laps_used]
            bayesdata = np.hstack((bayesdata, R2_data)) if bayesdata.size else R2_data

            # Correct other data
            self.Fc3data_dict[t] = self.Fc3data_dict[t][:, index][:, taskframes[t][0]:taskframes[t][1]]
            self.Fcdata_dict[t] = self.Fcdata_dict[t][:, index][:, taskframes[t][0]:taskframes[t][1]]

        return xdata, cdata, bayesdata

    def make_rastermap(self, xdata, cdata, ncomp=1, nx=30, npc=200):
        xdata = xdata[np.argsort(np.max(cdata, 1)), :]
        self.corr_sort = np.argsort(np.max(cdata, 1))
        model = mapping.Rastermap(n_components=ncomp, n_X=nx, nPC=npc).fit(xdata)
        self.isort = np.argsort(model.embedding[:, 0])
        # Sm = gaussian_filter1d(xdata[self.isort, :].T,
        #                        0.5,
        #                        axis=1)
        # np.minimum(1, int(xdata.shape[0] * 0.001)),
        self.Sm = xdata[self.isort, :]
        print(np.shape(self.Sm))

    def sortcells_toplotfov(self, cdata, cells_to_use):
        celldata = np.load(os.path.join(self.CellDataFolder, 'stat.npy'), allow_pickle=True)
        iscell = np.load(os.path.join(self.CellDataFolder, 'iscell.npy'), allow_pickle=True)
        # mimg = np.load(os.path.join(self.CellDataFolder, 'ops.npy'), allow_pickle=True).item()['meanImg']
        refimg = np.load(os.path.join(self.CellDataFolder, 'ops.npy'), allow_pickle=True).item()['refImg']

        # Normalize mean_image
        mimg = refimg.astype('float64')
        mimg1 = np.percentile(mimg, 10)
        mimg99 = np.percentile(mimg, 99.9)
        mimg = (mimg - mimg1) / (mimg99 - mimg1)
        mimg = np.maximum(0, np.minimum(1, mimg))

        # Sort cells
        cellids = np.where(iscell[:, 0])[0]
        # Sort by correlation then by isort
        cellorder = cellids[np.argsort(np.max(cdata, 1))]
        # cellorder = cellorder[self.isort]
        # cellorder = cellorder[cells_to_use[0][0]:cells_to_use[0][1]]
        mean_img = mimg

        H = np.zeros_like(mean_img)
        S = np.zeros_like(mean_img)
        count = 0
        for n in np.arange(np.shape(cellorder)[0]):
            if iscell[n, 0] == 1:
                if celldata[n]['med'][0] < 250 or celldata[n]['med'][1] < 450:
                    ypix = celldata[n]['ypix'].flatten()
                    xpix = celldata[n]['xpix'].flatten()
                    H[ypix, xpix] = np.random.random()
                    S[ypix, xpix] = 0.5
                    count += 1

        print(count)
        pix = np.concatenate(((H[:, :, np.newaxis]),
                              S[:, :, np.newaxis],
                              mean_img[:, :, np.newaxis]), axis=-1)
        pix = hsv_to_rgb(pix)

        dummy_img = np.ones_like(mean_img)
        onlypix = np.concatenate(((H[:, :, np.newaxis]),
                                  S[:, :, np.newaxis],
                                  dummy_img[:, :, np.newaxis]), axis=-1)
        onlypix = hsv_to_rgb(onlypix)
        return pix, mimg, refimg, onlypix

    def find_transient_distribution(self, fdata, threshold=0.5, transthreshold=10):
        numtransients = np.zeros(np.size(fdata, 0))
        numtime = (np.size(fdata, 1) / self.framespersec)
        print(numtime)
        for i in np.arange(np.size(fdata, 0)):
            # filterdata
            smdata = savgol_filter(fdata[i, :], 31, 2)
            bw_trans = smdata > threshold
            numtransients[i] = np.size(self.consecutive_one(bw_trans))
        numtransients_persec = (numtransients / numtime) * 100
        ax = sns.distplot(numtransients_persec, bins=10, kde=False)
        ax.set_xlabel('Transients per second')
        ax.set_ylabel('Number of cells')

        activecells = np.where(numtransients_persec > transthreshold)[0]
        return activecells

    def len_iter(self, items):
        return sum(1 for _ in items)

    def consecutive_one(self, data):
        return [self.len_iter(run) for val, run in groupby(data) if val]

    def crop_Sm_and_sort(self, cells_to_use, cdata):
        self.crpdSm = np.array([])
        cdata_crpd = np.array([])
        for c in cells_to_use:
            cdata_crpd = np.vstack((cdata_crpd, cdata[c[0]:c[1], :])) if cdata_crpd.size else cdata[c[0]:c[1], :]
            self.crpdSm = np.vstack((self.crpdSm, self.Sm[c[0]:c[1], :])) if self.crpdSm.size else self.Sm[c[0]:c[1], :]

    def plot_rastermap(self, fighandle, ax, fdata, ylim=0, crop_cellflag=0, ylim_meandff=0.6):
        if ylim == 0 and crop_cellflag == 0:
            ylim = np.size(fdata, 0)
        elif ylim == 0 and crop_cellflag == 1:
            ylim = np.size(self.crpdSm, 0)

        if crop_cellflag:
            im = ax[0].imshow(self.crpdSm, vmin=0, vmax=0.3, cmap='plasma', aspect='auto', interpolation='hanning',
                              extent=[0, self.crpdSm.shape[1] / self.framespersec, 0, self.crpdSm.shape[0]])
            CommonFunctions.plot_colorbar(fighandle, ax[0], im, title=f'\u0394F/F', ticks=[0, 0.3])
        else:
            ax[0].imshow(self.Sm, vmin=0, vmax=0.5, cmap='plasma', aspect='auto',
                         extent=[0, self.Sm.shape[1] / self.framespersec, 0, self.Sm.shape[0]])
        ax[0].set_ylim((0, ylim))
        ax[0].set_xlim((0, self.Sm.shape[1] / self.framespersec))
        ax[0].set_ylabel('Cell Number')

        # Plot average across population
        # x = np.linspace(0, np.size(fdata, 1) / self.framespersec, np.size(fdata, 1))
        # ax[1].plot(x, savgol_filter(np.nanmean(fdata, 0), 101, 2), color='k', linewidth=0.5)
        count = 0
        count_dff = 0
        for n, i in enumerate(self.taskstoplot):
            x = np.linspace(count, count + np.size(self.good_running_data[i]) / self.framespersec,
                            np.size(self.good_running_data[i]))
            ax[0].axvline(x[-1], color='k', linewidth=0.5)
            if i == 'Task2':
                ax[2].plot(x[:self.lickstopframe], self.good_running_data[i][:self.lickstopframe],
                           color=self.task2_colors[0], linewidth=0.5)
                ax[2].plot(x[self.lickstopframe:], self.good_running_data[i][self.lickstopframe:],
                           color=self.task2_colors[1], linewidth=0.5)
            else:
                ax[2].plot(x, self.good_running_data[i], color=self.colors[n], linewidth=0.5)
            ax[2].plot(x, self.lick_data[i] * 0.75, '|', color='grey', markeredgewidth=0.05, markersize=5,
                       alpha=0.5)
            ax[1].plot(x, savgol_filter(
                np.nanmean(fdata[:, count_dff:count_dff + np.size(self.good_running_data[i])], 0), 31, 2),
                       color=self.colors[n], linewidth=0.5)
            count += np.size(self.good_running_data[i]) / self.framespersec
            count_dff += np.size(self.good_running_data[i])

        ax[2].set_yticklabels([])
        ax[1].set_ylim((0, ylim_meandff))

        pf.set_axes_style(ax[0], numticks=4)
        pf.set_axes_style(ax[1], numticks=1)

    def saveimshow_as_tiff(self, filename, set_inches=(4, 2)):
        fs, ax = plt.subplots(1, dpi=300)
        ax.imshow(self.crpdSm, vmin=0, vmax=0.7, cmap='jet', aspect='auto', interpolation='hanning',
                  extent=[0, self.crpdSm.shape[1] / self.framespersec, 0, self.crpdSm.shape[0]])
        count = 0
        for n, i in enumerate(self.taskstoplot):
            x = np.linspace(count, count + np.size(self.good_running_data[i]) / self.framespersec,
                            np.size(self.good_running_data[i]))
            ax.axvline(x[-1], color='k', linewidth=0.5)
            count += np.size(self.good_running_data[i]) / self.framespersec
        ax.axis('off')
        fs.set_size_inches(set_inches)
        fs.savefig(os.path.join(self.SaveFigureFolder, f'%s.tif' % filename), bbox_inches='tight', transparent=True)

    def plot_samplecells(self, cellnumber, axis):
        count = 0
        for n2, t in enumerate(self.taskstoplot):
            x = np.linspace(count, count + np.size(self.good_running_data[t]) / self.framespersec,
                            np.size(self.good_running_data[t]))
            if t == 'Task2':
                axis.plot(x[:self.lickstopframe], np.squeeze(self.Fcdata_dict[t][cellnumber, :self.lickstopframe]),
                          color=self.task2_colors[0], linewidth=0.5)
                axis.plot(x[self.lickstopframe:], np.squeeze(self.Fcdata_dict[t][cellnumber, self.lickstopframe:]),
                          color=self.task2_colors[1], linewidth=0.5)
            else:
                axis.plot(x, np.squeeze(self.Fcdata_dict[t][cellnumber, :]), color=self.colors[n2], linewidth=0.5)
            count += np.size(self.good_running_data[t]) / self.framespersec
        axis.set_xlabel('Time (seconds)')
        axis.set_ylim((0, 2.5))
        axis.set_yticklabels('')
        pf.set_axes_style(axis, numticks=1)

    def plot_bayesR2(self, r2data, ax, traininglaps=20):
        count = 0
        for n, i in enumerate(self.taskstoplot):
            x = np.arange(count, count + np.size(self.numlicksperlap[i]))
            print(i)
            if i == 'Task2':
                print(self.lickstoplap)
                ax.plot(x[:self.lickstoplap], r2data[x[:self.lickstoplap]], '.-', color=self.task2_colors[0],
                        linewidth=0.5, markersize=1, zorder=2)
                ax.plot(x[self.lickstoplap:], r2data[x[self.lickstoplap:]], '.-', color=self.task2_colors[1],
                        linewidth=0.5, markersize=1, zorder=2)
            elif i in ['Task1', 'Task1a']:
                ax.plot(x[traininglaps:], r2data[x[traininglaps:]], '.-', color=self.colors[n],
                        linewidth=0.5, markersize=1, zorder=2)
            else:
                ax.plot(x, r2data[x], '.-', color=self.colors[n],
                        linewidth=0.5, markersize=1, zorder=2)

            for l in np.arange(np.size(self.numlicksperlap[i])):
                if i in ['Task1', 'Task1a'] and l < traininglaps:
                    continue

                if self.numlicksperlap[i][l] > 0:
                    ax.axvline(count + l, color='grey', linewidth=0.2, zorder=1)
            count += np.size(self.numlicksperlap[i])
        ax.set_xlabel('Lap #')
        ax.set_ylabel('R-squared')
        ax.set_ylim((-0.1, 1))
        pf.set_axes_style(ax)

    def plot_bayesfit(self, ax):
        for n, t in enumerate(self.taskstoplot):
            x = np.linspace(0, np.size(self.good_running_data[t]) / self.framespersec,
                            np.size(self.good_running_data[t]))
            ax1 = ax[n].twinx()
            if t == 'Task2':
                ax1.plot(x[:self.lickstopframe], self.good_running_data[t][:self.lickstopframe],
                         color=self.task2_colors[0], linewidth=0.5, zorder=2)
                ax1.plot(x[self.lickstopframe:], self.good_running_data[t][self.lickstopframe:],
                         color=self.task2_colors[1], linewidth=0.5, zorder=2)
                ax[n].plot(x, self.bayes_ypred[t], 'o', markeredgewidth=0.5, fillstyle='none', markersize=2,
                           markeredgecolor='grey', zorder=1)
            else:
                ax1.plot(x, self.good_running_data[t], color=self.colors[n], linewidth=0.5, zorder=2)
                ax[n].plot(x, self.bayes_ypred[t], 'o', markeredgewidth=0.5, fillstyle='none', markersize=2,
                           markeredgecolor='grey', zorder=1)
            ax1.set_yticklabels('')
            ax[n].set_yticklabels('')
            pf.set_axes_style(ax[n])

    def plot_correlation_with_lick_data(self, fighandle, cdata, ax, cmap='viridis', ylim=0.25, **kwargs):
        cdata = cdata[np.argsort(np.max(cdata, 1)), :]
        cdata = cdata[self.isort, :]
        if 'cells_to_use' in kwargs.keys():
            cdata_crpd = np.array([])
            for c in kwargs['cells_to_use']:
                cdata_crpd = np.vstack((cdata_crpd, cdata[c[0]:c[1], :])) if cdata_crpd.size else cdata[c[0]:c[1], :]
            im = ax[0].imshow(cdata_crpd, interpolation='nearest', aspect='auto', cmap=cmap, vmin=0,
                              vmax=1)
            CommonFunctions.plot_colorbar(fighandle, ax[0], im, title=f'correlation', ticks=[0, 1], cheight='50%')
        else:
            ax[0].imshow(cdata, interpolation='bilinear', aspect='auto', cmap='viridis', vmin=0,
                         vmax=1)
            ax[1].plot(np.mean(cdata, 0), '.-', color='k', linewidth=0.5)

        count = 0
        for n, i in enumerate(self.taskstoplot):
            x = np.arange(count, count + np.size(self.numlicksperlap[i]))
            ax[0].axvline(x[-1], color='k', linewidth=0.5)
            if 'cells_to_use' in kwargs.keys():
                if i == 'Task2':
                    ax[1].plot(x[:self.lickstoplap], np.mean(cdata_crpd[:, x[:self.lickstoplap]], 0), '.-',
                               color=self.task2_colors[0],
                               linewidth=0.5, markersize=1, zorder=2)
                    ax[1].plot(x[self.lickstoplap:], np.mean(cdata_crpd[:, x[self.lickstoplap:]], 0), '.-',
                               color=self.task2_colors[1],
                               linewidth=0.5, markersize=1, zorder=2)
                else:
                    ax[1].plot(x, np.mean(cdata_crpd[:, x], 0), '.-', color=self.colors[n], linewidth=0.5,
                               markersize=1, zorder=2)
            else:
                ax[1].plot(x, np.mean(cdata[:, x], 0), '.-', color=self.colors[n], linewidth=0.5,
                           markersize=1, zorder=2)
            for l in np.arange(np.size(self.numlicksperlap[i])):
                if self.numlicksperlap[i][l] > 0:
                    ax[1].axvline(count + l, color='grey', linewidth=0.2, zorder=1)
            count += np.size(self.numlicksperlap[i])

        ax[1].set_ylim((0.0, ylim))
        ax[1].set_xlabel('Lap #')
        ax[0].set_yticklabels([])
        for a in ax:
            pf.set_axes_style(a, numticks=1)


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
