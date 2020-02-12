import os
import numpy as np
import scipy.stats
import sys
import seaborn as sns
import h5py
import matplotlib.pyplot as plt
from Pvalues import GetPValues
from scipy.signal import savgol_filter
from itertools import groupby
import pandas as pd
from numpy import trapz
from collections import OrderedDict

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


class AnalyseDff(object):
    def __init__(self, Foldername, taskstoplot, placecellflag):
        self.Foldername = Foldername
        self.taskstoplot = taskstoplot
        self.framespersec = 30.98
        self.dffthreshold = 0.1
        self.transthreshold = 10
        self.placecellflag = placecellflag
        self.animalname = [f for f in os.listdir(self.Foldername) if f not in ['LickData', 'BayesResults_All']]
        self.transient_dff = pd.DataFrame(columns=['Animalname', 'Task', 'Amplitude', 'Frequency', 'Length', 'Number'])
        self.gd = GetData(self.Foldername)

    def get_data_peranimal(self):
        for a in self.animalname:
            print(a)
            animalinfo = DataDetails.ExpAnimalDetails(a)
            Fcdata, SmFcdata = self.gd.get_dff(a, animalinfo)
            lapframes = self.gd.get_lapframes(a, animalinfo)
            good_running_index, laps_with_licks, laps_without_licks = self.gd.get_behavior_params(a)
            placecell = self.gd.load_placecells(a)
            placecell = placecell['Task1']

            # activecells = self.get_cells_with_transients(animalinfo, Fcdata, SmFcdata)
            auc, amplitude, length, frequency, numtransients = self.get_data_transients_pertask(animalinfo, placecell,
                                                                                                Fcdata, SmFcdata,
                                                                                                lapframes=lapframes,
                                                                                                laps_withlicks=laps_with_licks,
                                                                                                laps_withoutlicks=laps_without_licks,
                                                                                                threshold=self.dffthreshold,
                                                                                                transthreshold=self.transthreshold)
            self.save_transient_properties(a, auc, amplitude, length, frequency, numtransients)
            self.compile_to_dataframe(a, auc, amplitude, length, frequency, numtransients)
            # return amplitude, length, frequency, numtransients

    def compile_to_dataframe(self, animalname, auc, amplitude, length, frequency, numtransients):
        for t in self.taskstoplot + ['Task2a', 'Task2b']:
            self.transient_dff = self.transient_dff.append(
                {'Amplitude': np.mean(amplitude[t]), 'Length': np.mean(length[t]),
                 'Frequency': np.mean(frequency[t]), 'Areaundercurve': np.mean(amplitude[t]),
                 'Number': np.mean(numtransients[t]), 'Task': t, 'Animalname': animalname},
                ignore_index=True)

    def save_transient_properties(self, animalname, auc, amplitude, length, frequency, numtransients):

        SaveFolder = os.path.join(self.Foldername, animalname, 'SaveAnalysed')
        if self.placecellflag:
            np.savez(os.path.join(SaveFolder, 'transient_properties_placecells.npz'), amplitude=amplitude,
                     length=length, areaundercurve=auc,
                     frequency=frequency, numtransients=numtransients)
        else:
            np.savez(os.path.join(SaveFolder, 'transient_properties.npz'), amplitude=amplitude, areaundercurve=auc,
                     length=length, frequency=frequency, numtransients=numtransients)

    def get_data_transients_pertask(self, animalinfo, placecell, fdata, smdata, lapframes, laps_withlicks,
                                    laps_withoutlicks,
                                    threshold, transthreshold):
        animal_tasks = animalinfo['task_dict']
        trans_amplitude = {k: [] for k in self.taskstoplot + ['Task2a', 'Task2b']}
        trans_auc = {k: [] for k in self.taskstoplot + ['Task2a', 'Task2b']}
        trans_length = {k: [] for k in self.taskstoplot + ['Task2a', 'Task2b']}
        trans_frequency = {k: [] for k in self.taskstoplot + ['Task2a', 'Task2b']}
        numtransients = {k: [] for k in self.taskstoplot + ['Task2a', 'Task2b']}

        for t in animal_tasks:
            print(t)
            if t in self.taskstoplot:
                if self.placecellflag:
                    fluor = smdata[t][placecell, :]
                    f_fluor = fdata[t][placecell, :]
                else:
                    fluor = smdata[t]
                    f_fluor = fdata[t]

                for i in np.arange(np.size(fluor, 0)):
                    # print(i)
                    bw_trans = fluor[i, :] > threshold  # Only pick up transients above a threshold
                    if t == 'Task2':
                        # With licks
                        lapframes_withlicks = np.where(lapframes[t] == laps_withlicks)[0]
                        trans_start, trans_end = CommonFunctions.find_start_end_index(bw_trans[lapframes_withlicks])
                        # print('Licks', np.shape(trans_start))
                        if len(trans_start) > 1:
                            auc, a, l, f, n = self.find_transient_features(f_fluor[i, lapframes_withlicks], trans_start,
                                                                           trans_end,
                                                                           np.size(bw_trans[lapframes_withlicks]))
                            trans_auc, trans_amplitude, trans_length, trans_frequency, numtransients = self.add_data_to_lists(
                                auc, a, l, f, n, trans_auc, trans_amplitude, trans_length, trans_frequency,
                                numtransients,
                                task='Task2a')
                        # Without licks
                        lapframes_withoutlicks = np.where(lapframes[t] == laps_withoutlicks)[0]
                        trans_start, trans_end = CommonFunctions.find_start_end_index(bw_trans[lapframes_withoutlicks])

                        # print('W/oLicks', np.shape(trans_start))
                        if len(trans_start) > 1:
                            auc, a, l, f, n = self.find_transient_features(f_fluor[i, lapframes_withoutlicks],
                                                                           trans_start,
                                                                           trans_end,
                                                                           np.size(bw_trans[lapframes_withoutlicks]))
                            trans_auc, trans_amplitude, trans_length, trans_frequency, numtransients = self.add_data_to_lists(
                                auc, a, l, f, n, trans_auc, trans_amplitude, trans_length, trans_frequency,
                                numtransients,
                                task='Task2b')

                    # All other tasks
                    trans_start, trans_end = CommonFunctions.find_start_end_index(bw_trans)
                    if len(trans_start) > transthreshold:
                        auc, a, l, f, n = self.find_transient_features(f_fluor[i, :], trans_start, trans_end,
                                                                       np.size(bw_trans))
                        trans_auc, trans_amplitude, trans_length, trans_frequency, numtransients = self.add_data_to_lists(
                            auc, a, l, f, n, trans_auc, trans_amplitude, trans_length, trans_frequency, numtransients,
                            task=t)

        return trans_auc, trans_amplitude, trans_length, trans_frequency, numtransients

    def add_data_to_lists(self, auc, a, l, f, n, trans_auc, trans_amplitude, trans_length, trans_frequency,
                          numtransients, task):
        trans_auc[task].append(auc)
        trans_amplitude[task].append(a)
        trans_length[task].append(l)
        trans_frequency[task].append(f)
        numtransients[task].append(n)
        return trans_auc, trans_amplitude, trans_length, trans_frequency, numtransients

    def find_transient_features(self, data, start_index, end_index, datalength):
        amplitude = []
        auc = []
        length = []
        for n, (s, e) in enumerate(zip(start_index, end_index)):
            # Give some leeway so you can get a good area
            thistransdata = data[s - np.int(self.framespersec):e + np.int(self.framespersec)]
            # Do not append the giant first transient
            if n == 0 and trapz(thistransdata) > 100:
                print('Big transient %0.3f, %0.3f' % (np.max(data[s:e]), trapz(thistransdata)))
                # plt.plot(thistransdata)
                # plt.title(trapz(thistransdata))
                # plt.show()
                continue
            amplitude.append(np.max(data[s:e]))
            auc.append(trapz(thistransdata))
            length.append(e - s)

        # Calculate mean of everything
        # print('sum', np.sum(auc), datalength)
        auc = np.sum(auc) / datalength
        amplitude = np.mean(amplitude)
        length = np.mean(length) / self.framespersec
        frequency = np.mean(np.diff(start_index) / self.framespersec)
        number = len(start_index) / datalength
        return auc, amplitude, length, frequency, number


class PlotTransientProps(object):
    def __init__(self, Foldername):
        self.Foldername = Foldername
        self.animalname = [f for f in os.listdir(self.Foldername) if f not in ['LickData', 'BayesResults_All']]

    def plot_boxplot_of_transproperty(self, ax, df, property='Amplitude', normalize=True):
        group_params = df[['Animalname', 'Task', property]]
        property_df = pd.DataFrame(index=self.animalname, columns=[
            'Task1', 'Task2a', 'Task2b', 'Task3', 'Task4'])
        for a in self.animalname:
            denom = np.array((group_params[(group_params['Animalname'] == a) & (group_params['Task'] == 'Task1')][
                property]))[0]
            for t in ['Task1', 'Task2a', 'Task2b', 'Task3', 'Task4']:
                # print(group_params)
                item = np.array((group_params[(group_params['Animalname'] == a) & (group_params['Task'] == t)][
                    property]))[0]
                if normalize:
                    item = item / denom
                property_df.loc[a, t] = item
        if normalize:
            data = property_df.melt(var_name='Task', value_name=property)
            data[property] = data[property].astype(float)
        else:
            data = df[['Animalname', 'Task', property]]
        sns.boxplot(x='Task', y=property, data=data, ax=ax, order=[
            'Task1', 'Task2a', 'Task2b', 'Task3', 'Task4'], showfliers=False)

        for index, row in property_df.iterrows():
            toplot = [r for r in row]
            ax.plot(toplot, 'ko-', markerfacecolor='none')

        pf.set_axes_style(ax)

        GetPValues().get_shuffle_pvalue(property_df, taskstocompare=['Task1', 'Task2a', 'Task2b'])


class GetData(object):
    def __init__(self, Foldername):
        self.Foldername = Foldername

    def get_dff(self, animalname, animalinfo):
        animal_tasks = animalinfo['task_dict']
        animal_tasknumframes = animalinfo['task_numframes']
        ImgFileName = [f for f in os.listdir(os.path.join(self.Foldername, animalname)) if f.endswith('.mat')][0]

        if animalinfo['v73_flag']:
            Fcdata, SmFcdata = self.load_v73_Data(ImgFileName, animalname, animal_tasks, animal_tasknumframes)
        else:
            Fcdata, SmFcdata = self.load_fluorescentdata(ImgFileName, animalname, animal_tasks, animal_tasknumframes)
        return Fcdata, SmFcdata

    def load_fluorescentdata(self, ImgFileName, animalname, TaskDict, Task_NumFrames):
        Fcdata_dict = CommonFunctions.create_data_dict(TaskDict)
        SmFcdata_dict = CommonFunctions.create_data_dict(TaskDict)
        # Open calcium data and store in dicts per trial
        data = scipy.io.loadmat(os.path.join(self.Foldername, animalname, ImgFileName))

        # Smooth data
        fdata = data['data'].item()[1].T
        smdata = np.zeros_like(fdata)
        for i in np.arange(np.size(fdata, 0)):
            smdata[i, :] = savgol_filter(fdata[i, :], 31, 2)

        count = 0
        for i in TaskDict.keys():
            Fcdata_dict[i] = data['data'].item()[1].T[:, count:count + Task_NumFrames[i]]
            SmFcdata_dict[i] = smdata[:, count:count + Task_NumFrames[i]]
            count += Task_NumFrames[i]
        return Fcdata_dict, SmFcdata_dict

    def load_v73_Data(self, ImgFileName, animalname, TaskDict, Task_NumFrames):
        Fcdata_dict = CommonFunctions.create_data_dict(TaskDict)
        SmFcdata_dict = CommonFunctions.create_data_dict(TaskDict)
        f = h5py.File(os.path.join(self.Foldername, animalname, ImgFileName), 'r')
        for k, v in f.items():
            print(k, np.shape(v))

        # Smooth data
        fdata = f['Fc']
        smdata = np.zeros_like(fdata)
        for i in np.arange(np.size(fdata, 0)):
            smdata[i, :] = savgol_filter(fdata[i, :], 31, 2)

        count = 0
        for i in TaskDict.keys():
            Fcdata_dict[i] = f['Fc'][:, count:count + Task_NumFrames[i]]
            SmFcdata_dict[i] = smdata[:, count:count + Task_NumFrames[i]]
            count += Task_NumFrames[i]
        return Fcdata_dict, SmFcdata_dict

    def load_placecells(self, animalname):
        PlaceCells = np.load(
            os.path.join(self.Foldername, animalname, 'PlaceCells', f'%s_placecell_data.npz' % animalname),
            allow_pickle=True)
        PlaceCells = PlaceCells['sig_PFs_cellnum'].item()
        return PlaceCells

    def get_lapframes(self, animalname, animalinfo):
        TaskDict = animalinfo['task_dict']
        PlaceFieldFolder = \
            [f for f in os.listdir(os.path.join(self.Foldername, animalname, 'Behavior')) if
             (f.endswith('.mat') and 'PlaceFields' in f and 'GoodBehavior' in f and 'Lick' not in f)]
        good_lapframes = CommonFunctions.create_data_dict(TaskDict)
        for t in TaskDict.keys():
            good_lapframes[t] = \
                [scipy.io.loadmat(os.path.join(self.Foldername, animalname, 'Behavior', p))['E'].T for p in
                 PlaceFieldFolder if t in p and 'Task2a' not in p][0]
        return good_lapframes

    def get_behavior_params(self, animalname):
        behaviorfile = np.load(os.path.join(self.Foldername, animalname, 'SaveAnalysed', 'behavior_data.npz'),
                               allow_pickle=True)
        good_running_index = behaviorfile['good_running_index'].item()
        anylicks = behaviorfile['numlicks_withinreward_alllicks'].item()['Task2']
        laps_with_licks = np.where(anylicks > 1)[0]
        laps_with_nolicks = np.where(anylicks <= 1)[0]
        return good_running_index, laps_with_licks, laps_with_nolicks


class CommonFunctions(object):
    @staticmethod
    def create_data_dict(TaskDict):
        data_dict = {keys: [] for keys in TaskDict.keys()}
        return data_dict

    @staticmethod
    def clean_up_index(start_index, end_index, minlength=5):
        new_start_index, new_end_index = [], []
        for s, e in zip(start_index, end_index):
            # print(e, s, e-s)
            if e - s > minlength:
                new_start_index.append(s)
                new_end_index.append(e)
        return new_start_index, new_end_index

    @staticmethod
    def find_start_end_index(data):
        start_index, end_index = [], []
        # If first index is 0
        zeroflag = 0
        if data[0] == 1:
            start_index.append(0)
            end_index.append(np.where(np.diff(data) == 1)[0][0])
            zeroflag = 1

        for n, t in enumerate(data):
            if zeroflag and n <= end_index[0]:
                continue
            if (n + 1) != len(data):
                if (t == 0) & (data[n + 1] == 1):
                    start_index.append(n)
                if (t == 1) & (data[n + 1] == 0):
                    end_index.append(n)
        # print(len(start_index))
        start_index, end_index = CommonFunctions.clean_up_index(start_index, end_index)
        # print(len(start_index))
        return start_index, end_index

    def len_iter(self, items):
        return sum(1 for _ in items)

    def consecutive_one(self, data):
        return [self.len_iter(run) for val, run in groupby(data) if val]
