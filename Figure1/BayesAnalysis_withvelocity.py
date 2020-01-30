import os
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import sys
from collections import OrderedDict
from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)
import csv

sns.set_context('paper', font_scale=1.3)
import pandas as pd
import warnings

# For plotting styles
PlottingFormat_Folder = '/home/sheffieldlab/Desktop/NoReward/Scripts/PlottingTools/'
sys.path.append(PlottingFormat_Folder)
import plottingfunctions as pf

# Data Details
DataDetailsFolder = '/home/sheffieldlab/Desktop/NoReward/Scripts/AnimalDetails/'
sys.path.append(DataDetailsFolder)
import DataDetails


class BayesError(object):
    def __init__(self, ParentDataFolder, BayesFolder, taskstoget, CFC12flag=0):
        colors = sns.color_palette('muted')
        self.colors = [colors[0], colors[1], colors[3], colors[2]]
        self.task2_colors = [self.colors[1], self.colors[2]]
        self.BayesFolder = BayesFolder
        self.ParentDataFolder = ParentDataFolder
        self.CFC12flag = CFC12flag
        self.taskstoget = taskstoget
        self.load_velocity_data(self.taskstoget)
        # self.accuracy_dict, self.numlaps_dict = self.get_lapwiseerror_peranimal()

    def load_velocity_data(self, taskstoplot):
        AnimalFolders = [f for f in os.listdir(self.ParentDataFolder) if f not in ['LickData', 'BayesResults_All']]
        actuallaptime = OrderedDict()
        goodlaptime = OrderedDict()

        for a in AnimalFolders:
            BehaviorData = np.load(os.path.join(self.ParentDataFolder, a, 'SaveAnalysed', 'behavior_data.npz'),
                                   allow_pickle=True)
            actuallaptime[a] = BehaviorData['actuallaps_laptime'].item()
            goodlaptime[a] = BehaviorData['goodlaps_laptime'].item()

    def get_lapwiseerror_peranimal(self):
        files = [f for f in os.listdir(self.BayesFolder)]
        accuracy_dict = OrderedDict()
        numlaps_dict = OrderedDict()
        for f in files:
            print(f)
            animalname = f[:f.find('_')]
            if animalname == 'CFC12' and self.CFC12flag == 0:
                continue
            animal_tasks = DataDetails.ExpAnimalDetails(animalname)['task_dict']
            trackbins = DataDetails.ExpAnimalDetails(animalname)['trackbins']
            data = np.load(os.path.join(self.BayesFolder, f), allow_pickle=True)
            animal_accuracy = {k: [] for k in animal_tasks}
            animal_numlaps = {k: [] for k in animal_tasks}
            for t in animal_tasks:
                animal_accuracy[t] = self.calulate_lapwiseerror(y_actual=data['fit'].item()[t]['ytest'],
                                                                y_predicted=data['fit'].item()[t]['yang_pred'],
                                                                numlaps=data['numlaps'].item()[t],
                                                                lapframes=data['lapframes'].item()[t])

                animal_numlaps[t] = data['numlaps'].item()[t]
            accuracy_dict[animalname] = animal_accuracy
            numlaps_dict[animalname] = animal_numlaps

        return accuracy_dict, numlaps_dict

    def calulate_lapwiseerror(self, y_actual, y_predicted, numlaps, lapframes):
        lap_R2 = []
        for l in np.arange(numlaps - 1):
            laps = np.where(lapframes == l + 1)[0]
            lap_R2.append(self.get_R2(y_actual[laps], y_predicted[laps]))

        return np.asarray(lap_R2)

    @staticmethod
    def get_R2(y_actual, y_predicted):
        y_mean = np.mean(y_actual)
        R2 = 1 - np.sum((y_predicted - y_actual) ** 2) / np.sum((y_actual - y_mean) ** 2)
        if np.isinf(R2):
            R2 = 0
        return R2
