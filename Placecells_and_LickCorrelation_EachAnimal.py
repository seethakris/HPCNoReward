import os
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import seaborn as sns
from copy import copy
import h5py

sns.set_context('paper', font_scale=1.2)
sns.set_palette(sns.color_palette('muted'))
sns.set_color_codes('muted')

""" There are 4 types of data in this paradigm - VR1 Reward, VR1 Noreward, VR1 Reward and Novel Reward 
"""


class GetData(object):

    def __init__(self, FolderName, Task_NumFrames, TaskDict, lick_stop, v73_flag=0):
        self.FolderName = FolderName
        self.Task_Numframes = Task_NumFrames
        self.lick_stop_frame = lick_stop
        self.v73_flag = v73_flag
        self.FigureFolder = os.path.join(self.FolderName, 'Figures')
        self.SaveFolder = os.path.join(self.FolderName, 'SaveAnalysed')

        if not os.path.exists(self.FigureFolder):
            os.mkdir(self.FigureFolder)
        if not os.path.exists(self.SaveFolder):
            os.mkdir(self.SaveFolder)

        self.TaskDict = TaskDict
        self.ImgFileName = [f for f in os.listdir(FolderName) if f.endswith('.mat')]
        self.Parsed_Behavior = np.load(os.path.join(FolderName, 'SaveAnalysed', 'behavior_data.npz'), allow_pickle=True)
        self.lick_stop_lap = self.Parsed_Behavior['lick_stop'].item()

        self.PlaceFieldData = \
            [f for f in os.listdir(os.path.join(FolderName, 'Behavior')) if
             (f.endswith('.mat') and 'PlaceFields' in f and 'GoodBehavior' in f)]

        self.Fcdata_dict = self.create_data_dict()
        self.Fdata_dict = self.create_data_dict()
        self.Fc3data_dict = self.create_data_dict()
        self.sig_PFs_cellnum_dict = self.create_data_dict()

        self.numcells = self.load_images()
        self.find_sig_PFs_cellnum_bytask()
        self.COM_df = self.calculate_center_of_mass()

    def create_data_dict(self):
        data_dict = {keys: [] for keys in self.TaskDict}
        return data_dict

    def load_images(self):
        if self.v73_flag:
            f = h5py.File(os.path.join(self.FolderName, self.ImgFileName[0]), 'r')
            numcells = np.size(f['Fc3'], 0)
            for k, v in f.items():
                print(k, np.shape(v))

            count = 0
            for i in self.TaskDict.keys():
                self.Fcdata_dict[i] = f['Fc'][:, count:count + self.Task_Numframes[i]]
                self.Fc3data_dict[i] = f['Fc3'][:, count:count + self.Task_Numframes[i]]
                self.Fcdata_dict[i] = self.Fcdata_dict[i].T
                self.Fc3data_dict[i] = self.Fc3data_dict[i].T

                count += self.Task_Numframes[i]
        else:
            # Open calcium data and store in dicts per trial
            data = scipy.io.loadmat(os.path.join(self.FolderName, self.ImgFileName[0]))
            numcells = np.size(data['data'].item()[1], 1)
            count = 0
            for i in self.TaskDict:
                self.Fcdata_dict[i] = data['data'].item()[1].T[:,
                                      count:count + self.Task_Numframes[i]]
                self.Fc3data_dict[i] = data['data'].item()[2].T[:,
                                       count:count + self.Task_Numframes[i]]
                count += self.Task_Numframes[i]

        return numcells

    def find_sig_PFs_cellnum_bytask(self):
        for i in self.PlaceFieldData:
            ft = i.find('Task')
            taskname = i[ft:ft + i[ft:].find('_')]
            x = scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', i))
            tempx = np.asarray(np.nan_to_num(x['number_of_PFs'])).T
            # Correct tempx
            # for n in range(0, np.size(tempx)):
            #     if len(x['sig_PFs'][0, n]) == 1:
            #         tempx[n] = 0
            tempx = tempx.T

            print('Number of PlaceCells in %s is %d' % (taskname, np.size(np.where(tempx >= 1)[1])))
            self.sig_PFs_cellnum_dict[taskname] = np.where(tempx >= 1)[1]

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
        # Dont use Task2 for now
        PF_sort_df = PF_sort_df.drop(['Task2'], axis=1)
        PF_sort_df = PF_sort_df.sort_values(by=['Task1', 'Task2a', 'Task3'], ascending=False)
        sns.heatmap(PF_sort_df.drop(['Cellnum'], axis=1), cbar=False)

    def plot_placecells_with_track_pertask(self):
        taskaxis = {'Task1': 0, 'Task2': 1, 'Task2a': 2, 'Task3': 3, 'Task4': 4}
        fs, ax1 = plt.subplots(1, len(self.TaskDict), figsize=(10, 3), dpi=100)

        img_dict_pfs, img_sorted_pfs = self.get_and_sort_cellactivity(self.sig_PFs_cellnum_dict)
        for taskname in self.TaskDict.keys():
            ax1[taskaxis[taskname]].imshow(img_dict_pfs[taskname][img_sorted_pfs[taskname], :], aspect='auto',
                                           cmap='jet',
                                           interpolation='nearest', vmin=0, vmax=1)

            ax1[taskaxis[taskname]].set_title(self.TaskDict[taskname])
            ax1[taskaxis[taskname]].set_xticks([0, 20, 39])
            ax1[taskaxis[taskname]].set_xticklabels([0, 100, 200])
            ax1[taskaxis[taskname]].set_xlim((0, 39))

        ax1[0].set_xlabel('Track Length (cm)')
        ax1[0].set_ylabel('Cell')

        ax1[0].locator_params(nbins=4)

        fs.tight_layout()
        plt.show()

    def plot_place_and_nonplacecells_withtrack_pertask(self):
        nonplacecells = {keys: [] for keys in self.TaskDict.keys()}
        for t in self.TaskDict.keys():
            pfs = self.sig_PFs_cellnum_dict[t]
            temp_shape = np.arange(self.numcells)
            nonplacecells[t] = np.setxor1d(temp_shape, pfs)

        img_dict_pfs, img_sorted_pfs = self.get_and_sort_cellactivity(self.sig_PFs_cellnum_dict)
        img_dict_notpfs, img_sorted_notpfs = self.get_and_sort_cellactivity(nonplacecells)
        taskaxis = {'Task1': 0, 'Task2': 1, 'Task2a': 2, 'Task3': 3, 'Task4': 4}
        fs, ax1 = plt.subplots(1, len(self.TaskDict), figsize=(10, 3), dpi=100)
        for taskname in self.TaskDict.keys():
            sorted_pfs = img_dict_pfs[taskname][img_sorted_pfs[taskname], :]
            sorted_notpfs = img_dict_notpfs[taskname][img_sorted_notpfs[taskname], :]
            sorted_all = np.vstack((sorted_pfs, sorted_notpfs))
            ax1[taskaxis[taskname]].imshow(sorted_all, aspect='auto',
                                           cmap='jet',
                                           interpolation='nearest', vmin=0, vmax=1)

            ax1[taskaxis[taskname]].set_title(self.TaskDict[taskname])
            ax1[taskaxis[taskname]].set_xticks([0, 20, 39])
            ax1[taskaxis[taskname]].set_xticklabels([0, 100, 200])
            ax1[taskaxis[taskname]].set_xlim((0, 39))

        ax1[0].set_xlabel('Track Length (cm)')
        ax1[0].set_ylabel('Cell')

        ax1[0].locator_params(nbins=4)

        fs.tight_layout()
        plt.show()

    def get_and_sort_cellactivity(self, cells_to_iterate):
        img_dict = {keys: [] for keys in self.TaskDict.keys()}
        img_argsort = {keys: [] for keys in self.TaskDict.keys()}
        for i in self.PlaceFieldData:
            ft = i.find('Task')
            taskname = i[ft:ft + i[ft:].find('_')]
            x = scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', i))
            bins = np.size(x['Allbinned_F'][0, 0], 0)
            img = np.zeros((np.size(cells_to_iterate[taskname]), bins))

            for n, c in enumerate(cells_to_iterate[taskname]):
                if taskname == 'Task2':
                    img[n, :] = np.mean(np.nan_to_num(x['Allbinned_F'][0, c][:, :self.lick_stop_lap]), 1)
                else:
                    img[n, :] = np.mean(np.nan_to_num(x['Allbinned_F'][0, c]), 1)

            img_dict[taskname] = img

            # Sort by activity along track
            img_argsort[taskname] = np.argsort(np.nanargmax(img_dict[taskname], 1))

        return img_dict, img_argsort

    def calculate_center_of_mass(self):
        # Go through place cells for each task and get center of mass for each lap traversal
        # Algorithm from Marks paper
        center_of_mass_df = pd.DataFrame(
            columns=['Task', 'CellNumber', 'PlaceCellNumber', 'COM', 'WeightedCOM', 'Precision'])
        for i in self.PlaceFieldData:
            ft = i.find('Task')
            taskname = i[ft:ft + i[ft:].find('_')]
            x = scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', i))

            # Get data dimensions for saving
            if taskname == 'Task2a':
                numlaps = self.Parsed_Behavior['numlaps'].item()['Task2']
            else:
                numlaps = self.Parsed_Behavior['numlaps'].item()[taskname]

            for n, p in enumerate(self.sig_PFs_cellnum_dict[taskname]):
                if np.nan_to_num(x['number_of_PFs'][0][p]) != 0 and len(x['sig_PFs'][0, p]) > 1:
                    num_pfs = np.int(x['number_of_PFs'][0][p])
                    for p1 in range(0, num_pfs):  # Loop by number of placefields in cell
                        # print(f'Place cell %d has %d fields' %(p, x['number_of_PFs'][0][p]))
                        if taskname == 'Task2':
                            data = x['sig_PFs'][p1, p][:, :self.lick_stop_lap]
                        else:
                            data = x['sig_PFs'][p1, p]
                        # print(np.shape(data))
                        num_com = np.zeros(np.size(data, 1))
                        denom_com = np.zeros(np.size(data, 1))
                        peak_f = np.zeros(np.size(data, 1))
                        COM = np.zeros(np.size(data, 1))
                        weighted_com_num = np.zeros(np.size(data, 1))
                        weighted_com_denom = np.zeros(np.size(data, 1))

                        xbin = np.linspace(0, 40, 40, endpoint=False)
                        for i in np.arange(np.size(data, 1)):
                            f_perlap = data[:, i]
                            f_perlap = np.nan_to_num(f_perlap)
                            num_com[i] = np.sum(np.multiply(f_perlap, xbin))
                            denom_com[i] = np.sum(f_perlap)
                            peak_f[i] = np.max(f_perlap)
                            COM[i] = num_com[i] / denom_com[i]
                            COM[i] = np.nan_to_num(COM[i])
                            weighted_com_num[i] = np.max(f_perlap) * COM[i]
                            weighted_com_denom[i] = np.max(f_perlap)

                        weighted_com = np.sum(weighted_com_num) / np.sum(weighted_com_denom)
                        precision_num = np.zeros(np.size(data, 1))
                        precision_denom = np.zeros(np.size(data, 1))
                        for i in np.arange(np.size(data, 1)):
                            f_perlap = data[:, i]
                            f_perlap = np.nan_to_num(f_perlap)
                            precision_num[i] = np.max(f_perlap) * np.square(COM[i] - weighted_com)
                            precision_denom[i] = np.max(f_perlap)

                        precision = 1 / (np.sqrt((np.sum(precision_num) / np.sum(precision_denom))))
                        if precision > 10:
                            precision = np.nan
                        center_of_mass_df = center_of_mass_df.append({'Task': taskname,
                                                                      'CellNumber': p,
                                                                      'PlaceCellNumber': p1 + 1,
                                                                      'COM': COM,
                                                                      'WeightedCOM': weighted_com,
                                                                      'Precision': precision}, ignore_index=True)
        center_of_mass_df['Task'] = center_of_mass_df['Task'].map(self.TaskDict)
        return center_of_mass_df

    def plot_percent_PFs_bytracklength(self):
        fs, ax = plt.subplots(2, figsize=(10, 5))
        for i in self.TaskDict.keys():
            if i == 'Task2' or i == 'Task4':
                continue

            sns.distplot(self.COM_df[self.COM_df.Task == self.TaskDict[i]]['WeightedCOM'] * 5, kde=False, rug=False,
                         hist_kws={"histtype": "step", "linewidth": 3, "label": self.TaskDict[i]}, ax=ax[0])
            ax[0].set_xlabel('Weighted center of mass (cm)')
            ax[0].set_ylabel('Number of Place Cells')
            ax[0].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        e = copy(list(self.TaskDict.values()))
        e.pop(1)
        sns.barplot(y='Precision', x='Task', order=e,
                    data=self.COM_df[self.COM_df.Task != self.TaskDict['Task2']])
        fs.tight_layout()

    def plot_remapping_withTaskA(self, TaskA='Task1'):
        # Divide task2 into everything
        placecells_formapping = self.sig_PFs_cellnum_dict[TaskA]
        img_dict = {keys: [] for keys in self.TaskDict.keys()}
        for i in self.PlaceFieldData:
            ft = i.find('Task')
            taskname = i[ft:ft + i[ft:].find('_')]
            x = scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', i))
            bins = np.size(x['Allbinned_F'][0, 0], 0)
            img = np.zeros((np.size(placecells_formapping), bins))

            for n, c in enumerate(placecells_formapping):
                if taskname == 'Task2':
                    img[n, :] = np.mean(np.nan_to_num(x['Allbinned_F'][0, c][:, :self.lick_stop_frame]), 1)
                else:
                    img[n, :] = np.mean(np.nan_to_num(x['Allbinned_F'][0, c]), 1)

            img_dict[taskname] = img
        # Sort by task
        img_argsort = np.argsort(np.nanargmax(img_dict[TaskA], 1))
        fs, ax1 = plt.subplots(1, len(self.TaskDict), figsize=(12, 4))
        for n, i in enumerate(self.TaskDict.keys()):
            ax1[n].imshow(img_dict[i][img_argsort, :], aspect='auto', cmap='jet', interpolation='nearest', vmin=0,
                          vmax=0.5)
            if n > 0:
                ax1[n].axis('off')
            if i == 'Task2':
                ax1[n].set_title(self.TaskDict[i] + ' Bef Lick Stop')
            else:
                ax1[n].set_title(self.TaskDict[i])
        ax1[0].set_xlabel('Bins')
        ax1[0].set_ylabel('Cell #')
        ax1[0].locator_params(nbins=4)
        ax1[0].set_xticks([0, 20, 39])
        ax1[0].set_xticklabels([0, 100, 200])
        fs.suptitle(f'Remapping aligned to %s' % TaskA, fontsize=10)
        plt.savefig(f'Remapping_with_%s.pdf' % TaskA, bbox_inches='tight', dpi=300)
        plt.show()

        return img_dict

    def plot_correlation_withTaskA(self, TaskA='Task1'):
        placecells_formapping = self.sig_PFs_cellnum_dict[TaskA]
        data_formapping = [i for i in self.PlaceFieldData if TaskA in i][0]
        data_formapping = scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', data_formapping))['sig_PFs']

        correlation_per_task = {keys: [] for keys in self.TaskDict.keys()}
        for i in self.PlaceFieldData:
            ft = i.find('Task')
            taskname = i[ft:ft + i[ft:].find('_')]
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
            ft = i.find('Task')
            taskname = i[ft:ft + i[ft:].find('_')]
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
        numlaps = self.Parsed_Behavior['numlaps'].item()
        numlicks_withinreward = self.Parsed_Behavior['numlicks_withinreward'].item()
        fs, axes = plt.subplots(2, 4, sharex='col', sharey='row',
                                gridspec_kw={'height_ratios': [2, 1]},
                                figsize=(10, 6))

        count_axis = 0
        for n, i in enumerate(self.TaskDict.keys()):
            if i != 'Task2a':
                ax1 = axes[0, count_axis]
                ax1.imshow(data_to_plot[i], interpolation='nearest', aspect='auto', cmap='viridis', vmin=0,
                           vmax=1)
                ax1.set_xlim((0, numlaps[i]))
                ax1.set_title(self.TaskDict[i])
                ax1.spines['right'].set_visible(False)
                ax1.spines['top'].set_visible(False)
                ax1.spines['bottom'].set_visible(False)
                ax1.tick_params(bottom=False)

                ax2 = axes[1, count_axis]
                ax2.plot(np.mean(data_to_plot[i], 0), '-o', linewidth=2)
                # ax2.set_xlim((0, self.numlaps[i]))
                ax2.set_xlabel('Lap number')

                ax3 = ax2.twinx()  # instantiate a second axes that shares the same x-axis
                ax3.plot(numlicks_withinreward[i], '-o', color='r', alpha=0.5,
                         label='Lap time')
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

                for l in range(0, numlaps[i] - 1):
                    if numlicks_withinreward[i][l]:
                        ax2.axvline(l, linewidth=0.25, color='k')
                count_axis += 1
        axes[0, 0].set_ylabel('Cell Number')
        axes[1, 0].set_ylabel('Mean Correlation', color='b')

        fs.subplots_adjust(wspace=0.1, hspace=0.1)
        fs.suptitle(f'Correlation with %s' % Task, fontsize=10)

        # plt.tight_layout()
        plt.show()

    def location_of_commoncells_with_taskchange(self, TaskA='Task1', column_To_plot='WeightedCOM'):
        # Create a data frame for center of mask comparison with TaskA
        TaskA_df = self.COM_df[self.COM_df['Task'] == self.TaskDict[TaskA]][[column_To_plot, 'CellNumber']]
        difference = {keys: [] for keys in self.TaskDict.keys()}
        # Plot scatter plot
        fs, ax = plt.subplots(1, len(self.TaskDict) - 2, figsize=(10, 3), dpi=100, sharey='all', sharex='all')
        ax_count = 0
        for t in self.TaskDict.keys():
            if t != TaskA and t != 'Task2':
                TaskB_df = self.COM_df[self.COM_df['Task'] == self.TaskDict[t]][[column_To_plot, 'CellNumber']]

                c = pd.merge(TaskA_df, TaskB_df, how='inner', on=['CellNumber'],
                             suffixes=(f'_%s' % self.TaskDict[TaskA], f'_%s' % self.TaskDict[t]))

                perc = c.shape[0] / TaskA_df.shape[0]
                print(f'Percentage of common cells between %s and %s: %0.2f %s' % (TaskA, t, perc * 100, '%'))

                sns.scatterplot(x=f'%s_%s' % (column_To_plot, self.TaskDict[t]),
                                y=f'%s_%s' % (column_To_plot, self.TaskDict[TaskA]), data=c, s=70, color=".2",
                                ax=ax[ax_count])
                difference[t] = np.abs(c[f'%s_%s' % (column_To_plot, self.TaskDict[t])] - c[
                    f'%s_%s' % (column_To_plot, self.TaskDict[TaskA])])

                ax_count += 1

        fs.tight_layout()
        # Plot mean differences
        for key in ['Task1', 'Task2']:
            del difference[key]
        d_df = pd.DataFrame.from_dict(difference)
        d_df = pd.melt(d_df, col_level=0, var_name='TaskName', value_name=f'%s_Difference' % column_To_plot)
        fs = plt.figure(figsize=(5, 3), dpi=100)
        sns.barplot(x='TaskName', y=f'%s_Difference' % column_To_plot, data=d_df)

    def plot_COM_difference_betweentasks(self, TaskA, TaskB):
        # Plot difference in COM
        fs, ax = plt.subplots(2, 1, figsize=(6, 5), dpi=100)

        TaskA_df = self.COM_df[self.COM_df['Task'] == self.TaskDict[TaskA]][['WeightedCOM', 'Precision', 'CellNumber']]
        TaskB_df = self.COM_df[self.COM_df['Task'] == self.TaskDict[TaskB]][['WeightedCOM', 'Precision', 'CellNumber']]
        c = pd.merge(TaskA_df, TaskB_df, how='inner', on=['CellNumber'],
                     suffixes=(f'_%s' % self.TaskDict[TaskA], f'_%s' % self.TaskDict[TaskB]))

        ax[0].bar(c[f'%s_%s' % ('WeightedCOM', self.TaskDict[TaskA])],
                  (c[f'%s_%s' % ('WeightedCOM', self.TaskDict[TaskB])] - c[
                      f'%s_%s' % ('WeightedCOM', self.TaskDict[TaskA])]) * 5)
        ax[0].set_ylabel('Difference in COM')

        ax[1].bar(c[f'%s_%s' % ('WeightedCOM', self.TaskDict[TaskA])],
                  (c[f'%s_%s' % ('Precision', self.TaskDict[TaskB])] - c[
                      f'%s_%s' % ('Precision', self.TaskDict[TaskA])]))
        ax[1].set_ylabel('Difference in Precision')
        for a in ax:
            a.set_title('%s vs %s' % (self.TaskDict[TaskA], self.TaskDict[TaskB]))
            a.locator_params(axis='x', nbins=4)

            a.set_xticks([0, 20, 39])
            a.set_xticklabels([0, 100, 200])
            a.set_xlabel('Track (cm)')

        fs.tight_layout()

    def plot_binned_activity_for_allcells_pertask(self):
        axrow = 4
        axcol = 4
        for i in self.PlaceFieldData:
            ft = i.find('Task')
            taskname = i[ft:ft + i[ft:].find('_')]
            print(taskname)
            Pdf = PdfPages(os.path.join(self.FigureFolder, 'Cellactivity_' + taskname))
            x = scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', i))
            fs, ax = plt.subplots(axrow, axcol, figsize=(10, 10), sharex='all', sharey='all')
            counter = 0
            while counter < self.numcells:
                for n, ax1 in enumerate(ax.flatten()):
                    print(counter)
                    if counter < self.numcells:
                        data = np.nan_to_num(x['Allbinned_F'][0, counter]).T

                        ax1.imshow(data, aspect='auto', cmap='jet',
                                   interpolation='nearest', vmin=0, vmax=0.5)
                        if counter in self.sig_PFs_cellnum_dict[taskname]:
                            ax1.set_title(f'Cell %d is a placecell' % counter, fontsize=5)
                        else:
                            ax1.set_title(f'Cell %d is not a placecell' % counter, fontsize=5)
                        ax1.axis('off')
                        counter += 1
                    else:
                        for ax1 in ax.flatten()[n:]:
                            ax1.axis('off')

                fs.tight_layout()
                Pdf.savefig(fs, bbox_inches='tight')
                plt.close()
                fs, ax = plt.subplots(axrow, axcol, figsize=(10, 10), sharex='all', sharey='all')
            Pdf.close()
