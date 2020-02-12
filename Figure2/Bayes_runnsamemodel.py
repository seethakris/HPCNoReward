import os
import numpy as np
from collections import OrderedDict
import scipy.stats
import matplotlib.pyplot as plt
import sys
from statistics import mean
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns

DataDetailsFolder = '/home/sheffieldlab/Desktop/NoReward/Scripts/AnimalDetails/'
sys.path.append(DataDetailsFolder)
import DataDetails

# For plotting styles
PlottingFormat_Folder = '/home/sheffieldlab/Desktop/NoReward/Scripts/PlottingTools/'
sys.path.append(PlottingFormat_Folder)
import plottingfunctions as pf


class CompileModelData(object):
    def __init__(self, DataFolder):
        self.DataFolder = DataFolder
        self.tracklength = 200
        self.trackbins = 5

    def compile_numcells(self, ax, taskstoplot):
        percsamples = [1, 5, 10, 20, 50, 80, 100]
        percsamples = [f'%d%%' % p for p in percsamples]
        animals = [f for f in os.listdir(self.DataFolder) if
                   f not in ['LickData', 'BayesResults_All', 'SaveAnalysed']]
        # numcells_combined = pd.DataFrame([])
        # for a in animals:
        #     print(a)
        #     animalinfo = DataDetails.ExpAnimalDetails(a)
        #     bayesmodel = np.load(os.path.join(animalinfo['saveresults'], 'modeloneachtask.npy'),
        #                          allow_pickle=True).item()
        #
        #     for t in animalinfo['task_dict']:
        #         numcells_dataframe = bayesmodel[t]['Numcells_Dataframe']
        #         numcells_dataframe['Task'] = t
        #         numcells_dataframe['animalname'] = a
        #         numcells_combined = pd.concat((numcells_combined, numcells_dataframe), ignore_index=True)
        # g = numcells_combined.groupby(['SampleSize', 'Task', 'animalname']).agg([np.mean]).reset_index()
        # g.columns = g.columns.droplevel(1)
        # sns.pointplot(x='SampleSize', y='R2_angle', data=g[g.Task.isin(taskstoplot)], order=percsamples, hue='Task',
        #               ax=ax)
        # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # ax.set_xlabel('Percentage of active cells used')
        # ax.set_ylabel('R-squared')
        # # ax.set_aspect(aspect=1.6)
        # pf.set_axes_style(ax, numticks=4)
