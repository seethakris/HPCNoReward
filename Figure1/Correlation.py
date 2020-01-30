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

DataDetailsFolder = '/home/sheffieldlab/Desktop/NoReward/Scripts/AnimalDetails/'
sys.path.append(DataDetailsFolder)
import DataDetails

# For plotting styles
PlottingFormat_Folder = '/home/sheffieldlab/Desktop/NoReward/Scripts/PlottingTools/'
sys.path.append(PlottingFormat_Folder)
import plottingfunctions as pf

pf.set_style()


class PlotCorrelation(object):
    def __init__(self, AnimalName, FolderName, taskstoplot, controlflag=0):
        print('Loading Data')
        self.taskstoplot = taskstoplot
        self.colors = sns.color_palette('deep', len(self.taskstoplot))
        self.animalname = AnimalName
        if controlflag:
            self.animalinfo = DataDetails.ControlAnimals(self.animalname)
        else:
            self.animalinfo = DataDetails.ExpAnimalDetails(self.animalname)
        self.FolderName = os.path.join(FolderName, self.animalname)
        self.Task_Numframes = self.animalinfo['task_numframes']
        self.TaskDict = self.animalinfo['task_dict']

        # Run functions
        self.get_data_folders()
        self.load_behaviordata()
        self.load_lapparams()

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

    def load_behaviordata(self):
        # Load required behavior data
        self.good_running_data = CommonFunctions.create_data_dict(self.TaskDict)
        self.lick_data = CommonFunctions.create_data_dict(self.TaskDict)
        self.numlicksperlap = CommonFunctions.create_data_dict(self.TaskDict)

        for keys in self.TaskDict.keys():
            self.good_running_data[keys] = self.Parsed_Behavior['good_running_data'].item()[keys]
            self.lick_data[keys] = self.Parsed_Behavior['corrected_lick_data'].item()[keys]
            self.lick_data[keys][self.lick_data[keys] == 0] = np.nan
            self.numlicksperlap[keys] = self.Parsed_Behavior['numlicks_withinreward_alllicks'].item()[keys]

    def load_lapparams(self):
        self.good_lapframes = CommonFunctions.create_data_dict(self.TaskDict)
        for t in self.TaskDict.keys():
            self.good_lapframes[t] = [scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', p))['E'].T for p in
                                      self.PlaceFieldFolder if t in p and 'Task2a' not in p][0]
            self.good_lapframes[t] = self.good_lapframes[t]

    def combine_data_for_correlation(self, taskframes):
        cdata = np.array([])
        for t in self.taskstoplot:
            laps_used = np.unique(self.good_lapframes[t][taskframes[t][0]:taskframes[t][1]])
            laps_used = laps_used[laps_used > 0] - 1
            data_task = self.Correlation_Data[t][:, laps_used]
            cdata = np.hstack((cdata, data_task)) if cdata.size else data_task

            # Correct other data
            self.numlicksperlap[t] = self.numlicksperlap[t][laps_used]
        return cdata

    def plot_correlation_with_lick_data(self, cdata, ax):
        ax[0].imshow(cdata, interpolation='nearest', aspect='auto', cmap='viridis', vmin=0,
                       vmax=1)



class CommonFunctions(object):
    @staticmethod
    def create_data_dict(TaskDict):
        data_dict = {keys: [] for keys in TaskDict.keys()}
        return data_dict
