import os
import numpy as np
import scipy.stats
import sys
import seaborn as sns
import h5py
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

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

class LickBehavior(object):
    def __init__(self, ExpFolderName, ControlFolderName,  TaskDict, TaskColors):

        self.ExpFolderName = ExpFolderName
        self.ControlFolderName = ControlFolderName
        self.TaskDict = TaskDict
        self.TaskColors = TaskColors
        self.frames_per_sec = 30.98