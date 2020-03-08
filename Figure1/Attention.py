import scipy.io
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import scipy.stats

PvaluesFolder = '/home/sheffieldlab/Desktop/NoReward/Scripts/Figure1/'
sys.path.append(PvaluesFolder)
from Pvalues import GetPValues

DataDetailsFolder = '/home/sheffieldlab/Desktop/NoReward/Scripts/AnimalDetails/'
sys.path.append(DataDetailsFolder)
import DataDetails

# For plotting styles
PlottingFormat_Folder = '/home/sheffieldlab/Desktop/NoReward/Scripts/PlottingTools/'
sys.path.append(PlottingFormat_Folder)
import plottingfunctions as pf


class CompileData(object):
    def __init__(self, FolderName):
        self.FolderName = FolderName
        self.Darkdatafolder = '/home/sheffieldlab/Desktop/NoReward/Dark/'
        self.SaveFolder = os.path.join(self.FolderName, 'SaveAnalysed')
        colors = sns.color_palette('muted')
        self.colors = [colors[0], colors[1], colors[3], colors[2]]
        self.animalname = [f for f in os.listdir(self.FolderName) if
                           f not in ['LickData', 'BayesResults_All', 'SaveAnalysed']]
        self.velocity_slope, self.speed_ratio = self.compile_slope_data()
        self.velocity_slope_dark, self.speed_ratio_dark = self.get_dark_behavior()

    def compile_slope_data(self):
        data = np.load(os.path.join(self.SaveFolder, 'velocity_in_space_withlicks.npz'), allow_pickle=True)
        velocity_slope = data['velocity_in_space'].item()
        speed_ratio = data['speed_ratio'].item()
        return velocity_slope, speed_ratio

    def get_dark_behavior(self, ):
        data = np.load(os.path.join(self.Darkdatafolder, 'SaveAnalysed', 'velocity_in_space.npz'), allow_pickle=True)
        velocity_slope = data['velocity_in_space'].item()
        speed_ratio = data['speed_ratio'].item()
        return velocity_slope, speed_ratio

    def plot_velocity_slope_bytask(self, data, axis, **kwargs):
        vinspace = np.asarray([])
        for a in data.keys():
            if 'tasktoplot' in kwargs.keys():
                vinspace = np.vstack(
                    (vinspace, data[a][kwargs['tasktoplot']])) if vinspace.size else data[a][kwargs['tasktoplot']]
                axis.set_title(kwargs['tasktoplot'])
            else:
                vinspace = np.vstack(
                    (vinspace, data[a])) if vinspace.size else data[a]
                axis.set_title(kwargs['title'])
        axis.plot(vinspace.T, color='lightgrey')
        axis.plot(np.nanmean(vinspace, 0), color='k')
        axis.set_xticks([0, 18, 36])
        axis.set_xticklabels([0, 100, 200])
        axis.set_xlabel('Track Length (cm)')
        pf.set_axes_style(axis)

    def plot_histogram_speed_ratio(self, axis, taskstoplot, bins=20):
        # Plot taskdata
        speed_ratio_exp = []
        axis1 = axis.twinx()
        for n, t in enumerate(taskstoplot):
            for a in self.speed_ratio.keys():
                speed_ratio_exp.extend(self.speed_ratio[a][t])
            weights = np.ones_like(speed_ratio_exp) / float(len(speed_ratio_exp))
            sns.distplot(speed_ratio_exp, bins=np.linspace(0, 2, bins), color=self.colors[n], hist=False,
                         kde=True, ax=axis1)
            axis.hist(speed_ratio_exp, bins=np.linspace(0, 2, bins),
                      color=self.colors[n], linewidth=2, weights=weights, label=t, alpha=0.5)
        # plot dark data
        speed_ratio_dark = []
        for a in self.speed_ratio_dark.keys():
            speed_ratio_dark.extend(self.speed_ratio_dark[a])
        weights = np.ones_like(speed_ratio_dark) / float(len(speed_ratio_dark))
        sns.distplot(speed_ratio_dark, bins=np.linspace(0, 2, bins), kde=True, ax=axis1, hist=False,
                     color='grey')
        axis.hist(speed_ratio_dark, bins=np.linspace(0, 2, bins),
                  color='grey', linewidth=2, weights=weights, label='Dark', alpha=0.5)

        axis.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        axis.set_xlim((0, 2))
        axis.set_xticks((0, 1.00, 2.00))
        axis.set_xticklabels((0, 1.00, 2.00))
        axis.axvline(1, linestyle='--', color='k')
        axis1.set_yticklabels('')
        axis.set_xlabel('Ratio of speed in middle by speed at end')
        pf.set_axes_style(axis, numticks=3)
        pf.set_axes_style(axis1)
        axis.set_ylabel('Lap bins')

    def velocity_inspace_byattention(self, axis):
        plot_title = ['With Lick', 'Attention withoutlick', 'Without attention']
        for n, laptype in enumerate(['lapswithlicks', 'attentivelaps_withoutlicks', 'notattentivelaps_withoutlicks']):
            vinspace = np.asarray([])
            for a in self.animalname:
                required_laps = np.load(os.path.join(self.FolderName, a, 'SaveAnalysed', 'attentionlaps.npz'),
                                        allow_pickle=True)

                laps = required_laps[laptype]
                if n == 1:
                    speed_ratio = np.asarray(self.speed_ratio[a]['Task2'])
                    weird = np.where(speed_ratio > 1.06)[0]
                    laps = np.intersect1d(weird, laps)

                vinspace = np.vstack(
                    (vinspace, self.velocity_slope[a]['Task2'][laps, :])) if vinspace.size else \
                    self.velocity_slope[a]['Task2'][laps, :]

            axis[n].plot(vinspace.T, color='lightgrey')
            axis[n].plot(np.mean(vinspace, 0), color='k')
            axis[n].set_title(plot_title[n])
            axis[n].set_xticks([0, 18, 36])
            axis[n].set_xticklabels([0, 100, 200])
            axis[n].set_xlabel('Track Length (cm)')
            pf.set_axes_style(axis[n])
        axis[0].set_ylabel('Normalized velocity')

    def get_lap_percentage(self, axis):
        lapswithattention, lapswithoutattention = [], []
        for n, a in enumerate(self.animalname):
            laps = np.load(
                os.path.join(self.FolderName, a, 'SaveAnalysed', 'attentionlaps.npz'),
                allow_pickle=True)
            totallaps = laps['totallaps'] - np.size(laps['lapswithlicks'])
            lapswithattention.append((np.size(laps['attentivelaps_withoutlicks']) / totallaps) * 100)
            lapswithoutattention.append((np.size(laps['notattentivelaps_withoutlicks']) / totallaps) * 100)
            print('%s: Laps with attn: %0.3f, Laps without attn: %0.3f' % (
                a, lapswithattention[n], lapswithoutattention[n]))

        for i, j in zip(lapswithattention, lapswithoutattention):
            axis.plot([1, 2], [i, j], 'ko-', markerfacecolor='none', zorder=2)
        axis.boxplot([lapswithattention, lapswithoutattention])
        axis.set_xticklabels(('With Attention', 'Withoutattention'))
        axis.set_ylabel('Percentage of laps')
        print('Percentage of attentive laps %0.3f +/- %0.3f' % (
            np.mean(lapswithattention), scipy.stats.sem(lapswithattention)))
        print('Percentage of not attentive laps %0.3f +/- %0.3f' % (
            np.mean(lapswithoutattention), scipy.stats.sem(lapswithoutattention)))
