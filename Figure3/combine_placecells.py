import scipy.io
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import sys
import scipy.stats
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
    def __init__(self, FolderName, CombinedDataFolder):
        self.FolderName = FolderName
        self.CombinedDataFolder = CombinedDataFolder
        self.animals = [f for f in os.listdir(self.FolderName) if
                        f not in ['LickData', 'BayesResults_All', 'SaveAnalysed', 'PlaceCellResults_All']]
        self.trackbins = 5
        self.tracklength = 200
        self.plpc = PlotPCs()

    def get_data_folders(self, animalname):
        imgfilename = [f for f in os.listdir(os.path.join(self.FolderName, animalname)) if f.endswith('.mat')]
        parsed_behavior = np.load(os.path.join(self.FolderName, 'SaveAnalysed', 'behavior_data.npz'),
                                  allow_pickle=True)
        pf_data = \
            [f for f in os.listdir(os.path.join(self.FolderName, animalname, 'Behavior')) if
             (f.endswith('.mat') and 'PlaceFields' in f and 'GoodBehavior' in f)]
        pf_params = np.load(
            os.path.join(self.FolderName, 'PlaceCells', f'%s_placecell_data.npz' % animalname), allow_pickle=True)

        pf_remapping_dict = np.load(
            os.path.join(self.FolderName, animalname, 'PlaceCells', '%s_pcs_sortedbyTask1', animalname),
            allow_pickle=True)
        return imgfilename, parsed_behavior, pf_data, pf_params, pf_remapping_dict

    def combine_placecells_withtask(self, fig, axis, taskstoplot, tasktocompare='Task1'):
        pc_activity_dict = {keys: np.asarray([]) for keys in taskstoplot}
        for a in self.animals:
            animalinfo = DataDetails.ExpAnimalDetails(a)
            if len(animalinfo['task_dict']) == 4:
                pf_remapping = np.load(
                    os.path.join(self.FolderName, a, 'PlaceCells', '%s_pcs_sortedbyTask1.npy' % a),
                    allow_pickle=True).item()

                for t in taskstoplot:
                    pc_activity_dict[t] = np.vstack((pc_activity_dict[t], pf_remapping[t])) if pc_activity_dict[
                        t].size else pf_remapping[t]

        pcsortednum = {keys: [] for keys in taskstoplot}
        pcsorted = np.argsort(np.nanargmax(pc_activity_dict[tasktocompare], 1))
        for t in taskstoplot:
            pcsortednum[t] = pcsorted

        # Correlate place cells
        corrcoef_dict = self.find_correlation(pc_activity_dict, taskstoplot, tasktocompare)

        task_data = pc_activity_dict['Task1'][pcsorted, :]
        normalise_data = np.nanmax(task_data, 1)[:, np.newaxis]
        self.plpc.plot_placecells_pertask(fig, axis, taskstoplot, pc_activity_dict, pcsortednum,
                                          normalise_data=normalise_data)

        return corrcoef_dict

    def combine_control_placecells(self, fig, axis, ControlFolder, taskstoplot, tasktocompare='Task1a', controlflag=0):
        pc_activity_dict = {keys: np.asarray([]) for keys in taskstoplot}
        animals = [f for f in os.listdir(ControlFolder) if
                   f not in ['LickData', 'BayesResults_All', 'SaveAnalysed', 'PlaceCellResults_All']]
        for a in animals:
            pf_remapping = np.load(
                os.path.join(ControlFolder, a, 'PlaceCells', '%s_pcs_sortedbyTask1.npy' % a),
                allow_pickle=True).item()

            for t in taskstoplot:
                pc_activity_dict[t] = np.vstack((pc_activity_dict[t], pf_remapping[t])) if pc_activity_dict[
                    t].size else pf_remapping[t]

        pcsortednum = {keys: [] for keys in taskstoplot}
        pcsorted = np.argsort(np.nanargmax(pc_activity_dict[tasktocompare], 1))
        for t in taskstoplot:
            pcsortednum[t] = pcsorted

        task_data = pc_activity_dict['Task1a'][pcsorted, :]
        normalise_data = np.nanmax(task_data, 1)[:, np.newaxis]
        self.plpc.plot_placecells_pertask(fig, axis, taskstoplot, pc_activity_dict, pcsortednum, controlflag,
                                          normalise_data=normalise_data)

        # Correlate place cells
        corrcoef_dict = self.find_correlation(pc_activity_dict, taskstoplot, tasktocompare)
        return corrcoef_dict

    def find_correlation(self, pc_dict, taskstoplot, tasktocompare):
        # Correlate place cells
        basetask = pc_dict[tasktocompare]
        corrcoef_withtask = {keys: [] for keys in taskstoplot}
        for t in taskstoplot:
            for c in np.arange(np.size(pc_dict[t], 0)):
                c = np.corrcoef(pc_dict[t][c, :], basetask[c, :])[0, 1]
                corrcoef_withtask[t].append(c)
        return corrcoef_withtask

    def combine_placecells_pertask(self, fig, axis, taskstoplot):
        pc_activity_dict = {keys: np.asarray([]) for keys in taskstoplot}
        perccells_peranimal = {keys: [] for keys in taskstoplot + ['animal']}
        pcsortednum = {keys: [] for keys in taskstoplot}
        for a in self.animals:
            animalinfo = DataDetails.ExpAnimalDetails(a)
            if len(animalinfo['task_dict']) == 4:
                pf_remapping = np.load(
                    os.path.join(self.FolderName, a, 'PlaceCells', '%s_pcs_pertask.npy' % a),
                    allow_pickle=True).item()
                pfparams = np.load(
                    os.path.join(self.FolderName, a, 'PlaceCells', f'%s_placecell_data.npz' % a), allow_pickle=True)
                perccells_peranimal['animal'].append(a)
                for t in taskstoplot:
                    perccells_peranimal[t].append(
                        (np.sum(pfparams['numPFs_incells'].item()[t]) / pfparams['numcells']) * 100)
                    print(t, np.sum(pfparams['numPFs_incells'].item()[t]), np.shape(pf_remapping[t]))
                    pc_activity_dict[t] = np.vstack((pc_activity_dict[t], pf_remapping[t])) if pc_activity_dict[
                        t].size else pf_remapping[t]

        for t in taskstoplot:
            print(t, np.shape(pc_activity_dict[t]))
            pcsortednum[t] = np.argsort(np.nanargmax(pc_activity_dict[t], 1))

        self.plpc.plot_placecells_pertask(fig, axis, taskstoplot, pc_activity_dict, pcsortednum)
        perccells_peranimal = pd.DataFrame.from_dict(perccells_peranimal)
        perccells_peranimal = perccells_peranimal.set_index('animal')
        return perccells_peranimal

    def get_com_allanimal(self, fig, axis, taskA, taskB, vmax=0):
        csvfiles_pfs = [f for f in os.listdir(self.CombinedDataFolder) if f.endswith('.csv') and 'reward' not in f]
        com_all_animal = np.array([])
        count = 0
        for n, f in enumerate(csvfiles_pfs):
            a = f[:f.find('_')]
            animalinfo = DataDetails.ExpAnimalDetails(a)
            if len(animalinfo['task_dict']) == 4:
                print(f)
                df = pd.read_csv(os.path.join(self.CombinedDataFolder, f), index_col=0)
                t1 = df[df['Task'] == taskA]
                t2 = df[df['Task'] == taskB]
                combined = pd.merge(t1, t2, how='inner', on=['CellNumber'],
                                    suffixes=(f'_%s' % taskA, f'_%s' % taskB))

                if count == 0:
                    com_all_animal = np.vstack((combined[f'WeightedCOM_%s' % taskA] * self.trackbins,
                                                combined[f'WeightedCOM_%s' % taskB] * self.trackbins))
                else:
                    com_all_animal = np.hstack(
                        (com_all_animal, np.vstack((combined[f'WeightedCOM_%s' % taskA] * self.trackbins,
                                                    combined[f'WeightedCOM_%s' % taskB] * self.trackbins))))
                count += 1
        self.plpc.plot_com_scatter_heatmap(fig, axis, com_all_animal, taskA, taskB, self.tracklength, vmax=vmax)
        return np.abs(np.subtract(com_all_animal[0, :], com_all_animal[1, :]))


class PlotPCs(object):
    @staticmethod
    def plot_numcells(axis, numcells_df, taskstoplot):
        numcells_df = numcells_df[taskstoplot]
        df = numcells_df.melt(var_name='Task', value_name='numcells')
        sns.boxplot(x='Task', y='numcells', data=df, ax=axis, width=0.5)
        for n, row in numcells_df.iterrows():
            axis.plot(row, 'o-', color='k', markerfacecolor='none')
        axis.set_ylabel('Percentage of place cells')
        GetPValues().get_shuffle_pvalue(numcells_df, taskstocompare=taskstoplot)
        pf.set_axes_style(axis, numticks=4)

    @staticmethod
    def plot_com_scatter_heatmap(fig, axis, combined_dataset, taskA, taskB, tracklength, bins=10, vmax=0):
        # Scatter plots
        y = combined_dataset[0, :]
        x = combined_dataset[1, :]
        axis[0].scatter(y, x, color='k', s=3)
        axis[0].plot([0, tracklength], [0, tracklength], linewidth=2, color=".3")
        axis[0].set_xlabel(taskB)
        axis[0].set_ylabel(taskA)
        axis[0].set_title('Center of Mass')

        # Heatmap of scatter plot
        heatmap, xedges, yedges = np.histogram2d(y, x, bins=bins)
        heatmap = (heatmap / np.size(y)) * 100
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        if vmax == 0:
            img = axis[1].imshow(heatmap.T, cmap='gray_r', extent=extent, interpolation='bilinear', origin='lower',
                                 vmin=0, vmax=np.max(heatmap))
        else:
            img = axis[1].imshow(heatmap.T, cmap='gray_r', extent=extent, interpolation='bilinear', origin='lower',
                                 vmin=0, vmax=vmax)

        axis[1].plot([0 + bins, tracklength - bins], [0 + bins, tracklength - bins], linewidth=2,
                     color=".3")
        axins = PlotPCs.add_colorbar_as_inset(axes=axis[1])
        if vmax == 0:
            cb = fig.colorbar(img, cax=axins, pad=0.2, ticks=[0, np.int(np.max(heatmap))])
        else:
            cb = fig.colorbar(img, cax=axins, pad=0.2, ticks=[0, vmax])
        cb.set_label('% Field Density', rotation=270, labelpad=12)

        for a in axis:
            pf.set_axes_style(a, numticks=4)
            a.set_ylim((0, tracklength))
            a.set_xlim((0, tracklength))

    @staticmethod
    def plot_placecells_pertask(fig, axis, taskstoplot, pc_activity, sorted_pcs, controlflag=0, **kwargs):
        for n, taskname in enumerate(taskstoplot):
            task_data = pc_activity[taskname][sorted_pcs[taskname], :]
            if 'normalise_data' in kwargs.keys():
                normalise_data = task_data / kwargs['normalise_data']
            else:
                normalise_data = task_data / np.nanmax(task_data, 1)[:, np.newaxis]
            normalise_data = np.nan_to_num(normalise_data)

            img = axis[n].imshow(normalise_data,
                                 aspect='auto', cmap='jet', interpolation='nearest', vmin=0, vmax=1.0)

            axis[n].set_xticks([0, 20, 39])
            axis[n].set_xticklabels([0, 100, 200])
            axis[n].set_xlim((0, 39))
            if controlflag:
                axis[n].set_title('Cntrl: %s' % taskname)
            else:
                axis[n].set_title('Exp: %s' % taskname)

            pf.set_axes_style(axis[n], numticks=4)
        axis[0].set_xlabel('Track Length (cm)')
        axis[0].set_ylabel('Cell')
        axins = PlotPCs.add_colorbar_as_inset(axis[-1])
        cb = fig.colorbar(img, cax=axins, pad=0.2, ticks=[0, 1])
        cb.set_label('Delta f/f')
        cb.ax.tick_params(size=0)

    @staticmethod
    def scatter_of_common_center_of_mass(self, taskA, taskB, bins=10):
        combined_dataset = self.pfparam_common_combined
        fs, ax = plt.subplots(1, 2, figsize=(8, 4))
        x = combined_dataset[f'%s_%s' % ('WeightedCOM', taskA)] * self.trackbins
        y = combined_dataset[f'%s_%s' % ('WeightedCOM', taskB)] * self.trackbins
        # Scatter plot
        ax[0].scatter(y, x, color='k')
        ax[0].plot([0, self.tracklength], [0, self.tracklength], linewidth=2, color=".3")
        ax[0].set_xlabel(taskB)
        ax[0].set_ylabel(taskA)
        ax[0].set_title('Center of Mass')
        # Heatmap of scatter plot
        heatmap, xedges, yedges = np.histogram2d(y, x, bins=bins)
        heatmap = (heatmap / np.size(y)) * 100
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        img = ax[1].imshow(heatmap.T, cmap='gray_r', extent=extent, interpolation='bilinear', origin='lower', vmin=0)
        ax[1].plot([0 + bins, self.tracklength - bins], [0 + bins, self.tracklength - bins], linewidth=2, color=".3")
        axins = PlotPCs.add_colorbar_as_inset(axes=ax[1])
        cb = fs.colorbar(img, cax=axins, pad=0.2, ticks=[0, np.int(np.max(heatmap))])
        cb.set_label('Field Density', rotation=270, labelpad=12)

        for a in ax:
            pf.set_axes_style(a, numticks=5)

    @staticmethod
    def plot_barplot_correlation(ax, c_cntrl, c_exp):
        df1 = pd.DataFrame.from_dict(c_exp)
        df1 = df1.dropna()
        df1 = df1.melt(var_name='Task', value_name='Correlation')

        df2 = pd.DataFrame.from_dict(c_cntrl)
        df2 = df2.dropna()
        df2 = df2.melt(var_name='Task', value_name='Correlation')
        df = pd.concat((df1, df2))

        sns.boxplot(x='Task', y='Correlation', data=df[~df['Task'].isin(['Task1', 'Task1a'])], ax=ax,
                    order=['Task1b', 'Task2b', 'Task3', 'Task4'], showfliers=False, width=0.5)

        for t in ['Task2b', 'Task3', 'Task4']:
            t1, p = scipy.stats.ttest_ind(df[df.Task == t]['Correlation'], df[df.Task == 'Task1b']['Correlation'])
            print('Task1 with %s : %0.3f' % (t, p))

        pf.set_axes_style(ax)

    @staticmethod
    def add_colorbar_as_inset(axes):
        axins = inset_axes(axes,
                           width="5%",  # width = 5% of parent_bbox width
                           height="40%",  # height : 50%
                           loc='lower left',
                           bbox_to_anchor=(1.05, 0., 1, 1),
                           bbox_transform=axes.transAxes,
                           borderpad=0.5,
                           )
        return axins
