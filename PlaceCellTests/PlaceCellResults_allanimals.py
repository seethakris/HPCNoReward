import scipy.io
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import sys
import scipy.stats
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# For plotting styles
PlottingFormat_Folder = '/home/sheffieldlab/Desktop/NoReward/Scripts/PlottingTools/'
sys.path.append(PlottingFormat_Folder)
import plottingfunctions as pf




class Combinedplots:
    def __init__(self, CombinedDataFolder):
        self.CombinedDataFolder = CombinedDataFolder
        csvfiles_pfs = [f for f in os.listdir(self.CombinedDataFolder) if f.endswith('.csv') if
                        'common' not in f and 'reward' not in f]
        csvfiles_commonpfs = [f for f in os.listdir(self.CombinedDataFolder) if f.endswith('.csv') if 'common' in f]

        self.npyfiles = [f for f in os.listdir(self.CombinedDataFolder) if f.endswith('.npz')]
        self.trackbins = 5
        self.tracklength = 200
        self.numanimals = len(csvfiles_pfs)
        # Combined pf dataframes into one big dataframe
        self.pfparam_combined = self.combineanimaldataframes(csvfiles_pfs)
        self.pfparam_common_combined = self.combineanimaldataframes(csvfiles_commonpfs)

    def combineanimaldataframes(self, csvfiles):
        for n, f in enumerate(csvfiles):
            df = pd.read_csv(os.path.join(self.CombinedDataFolder, f), index_col=0)
            if n == 0:
                combined_dataframe = df
            else:
                combined_dataframe = combined_dataframe.append(df, ignore_index=True)
        return combined_dataframe

    def plot_pfparams(self, tasks_to_plot):
        # Plot a combined historgram and a boxplot of means
        columns_to_plot = ['Precision', 'Width', 'FiringRatio', 'Firingintensity', 'Stability']
        df_plot = self.pfparam_combined[self.pfparam_combined['Task'].isin(tasks_to_plot)]
        fs, ax = plt.subplots(len(columns_to_plot), 3, figsize=(7, 8), gridspec_kw={'width_ratios': [2, 2, 1]})
        for n1, c in enumerate(columns_to_plot):
            # Plot boxplot
            sns.boxplot(x='Task', y=c, data=df_plot, ax=ax[n1, 2], order=tasks_to_plot, showfliers=False)
            ax[n1, 2].set_xlabel('')
            # For significance test
            x, y = df_plot[df_plot['Task'] == tasks_to_plot[0]][c], df_plot[df_plot['Task'] == tasks_to_plot[1]][c]
            d, p = CommonFunctions.significance_test(x, y, type_of_test='KStest')
            print(f'%s: KStest : p-value %0.4f' % (c, p))
            d, p = CommonFunctions.significance_test(x, y, type_of_test='Wilcoxon')
            print(f'%s: Wilcoxon : p-value %0.4f' % (c, p))
            for n2, t in enumerate(tasks_to_plot):
                # Plot histogram
                d = df_plot[df_plot['Task'] == t][c]
                d = d[~np.isnan(d)]
                sns.distplot(d, ax=ax[n1, 0], kde=False)
                ax[n1, 1].hist(d, bins=1000, normed=True, cumulative=True, label='CDF',
                               histtype='step', linewidth=1, alpha=0.8)
                # Does the sample come from a null distribution
                s, p = CommonFunctions.normalitytest(d)
                print('%s in Task %s against normal distribution p-value %0.4f' % (c, t, p))

        for a in ax.flatten():
            pf.set_axes_style(a, numticks=3)
        fs.tight_layout()
        fs.savefig('Place Cell params.pdf', bbox_inches='tight')

    def plot_pf_params_commoncells(self, tasks_to_plot):
        columns_to_plot = ['Precision', 'Width', 'FiringRatio', 'Firingintensity', 'Stability']
        fs, ax = plt.subplots(len(columns_to_plot), 3, figsize=(7, 8), gridspec_kw={'width_ratios': [2, 2, 1]})
        for n1, c in enumerate(columns_to_plot):
            # Plot boxplot
            p1 = self.pfparam_common_combined[f'%s_%s' % (c, tasks_to_plot[0])]
            p2 = self.pfparam_common_combined[f'%s_%s' % (c, tasks_to_plot[1])]
            df_plot = pd.concat([p1, p2], keys=tasks_to_plot, ).reset_index(level=[0]).rename(columns={0: c})
            sns.boxplot(x='level_0', y=c, data=df_plot, ax=ax[n1, 2],
                        order=tasks_to_plot, showfliers=False)
            ax[n1, 2].set_xlabel('')
            d, p = CommonFunctions.significance_test(np.asarray(p1), np.asarray(p2), type_of_test='KStest')
            print(f'%s: KStest : p-value %0.4f' % (c, p))
            for n2, t in enumerate(tasks_to_plot):
                # Plot histogram
                d = self.pfparam_common_combined[f'%s_%s' % (c, t)]
                d = d[~np.isnan(d)]
                sns.distplot(d, ax=ax[n1, 0], kde=False)
                ax[n1, 0].set_xlabel(c)
                ax[n1, 1].hist(d, bins=1000, normed=True, cumulative=True, label='CDF',
                               histtype='step', linewidth=1, alpha=0.8)
        for a in ax.flatten():
            pf.set_axes_style(a, numticks=3)
        fs.tight_layout()

    def plot_percentpfs_withtrackposition(self, tasks_to_plot, bins=20):
        color = sns.color_palette('dark', len(tasks_to_plot))
        df_plot = self.pfparam_combined[self.pfparam_combined['Task'].isin(tasks_to_plot)]
        fs, ax = plt.subplots(2, len(tasks_to_plot), sharex='row', sharey='row', figsize=(10, 5))
        # Create dataset to plot histogram from all animals
        hist_all = {keys: [] for keys in tasks_to_plot}
        for i in np.unique(df_plot.animalname):
            npy_file = [f for f in self.npyfiles if i in f][0]
            data_animal = np.load(os.path.join(self.CombinedDataFolder, npy_file))
            numplacecells_task1 = np.sum(data_animal['numPFs_incells'].item()[tasks_to_plot[0]])
            hist_com_animal = []
            for n, t in enumerate(tasks_to_plot):
                com_animal = df_plot[(df_plot.Task == t) & (df_plot.animalname == i)]['WeightedCOM'] * self.trackbins
                print(t, len(com_animal))
                hist_com, bins_com, center, width = CommonFunctions.make_histogram(com_animal, bins,
                                                                                   len(com_animal),
                                                                                   self.tracklength)
                hist_all[t].append(hist_com)
                hist_com_animal.append(hist_com)
                ax[0, n].bar(center, hist_com, align='center', width=width, alpha=0.5, label=t)
                ax[0, n].set_title(t)

            ax[1, 0].bar(center, hist_com_animal[0] - hist_com_animal[1], align='center', width=width, alpha=0.5)
            ax[1, 0].axhline(0, color='k')
            ax[1, 0].set_title(f'Difference between %s and %s' % (tasks_to_plot[0], tasks_to_plot[1]))
        for a in ax.flatten():
            pf.set_axes_style(a, numticks=4)
        ax[1, 1].axis('off')
        fs.tight_layout()

    def plot_percentpfs_ofdroppedcells(self, tasks_to_plot, bins=10):
        fs, ax = plt.subplots(1, figsize=(6, 3), dpi=100, sharey='all', sharex='all')
        for n in np.unique(self.pfparam_combined.animalname):
            # Plot COM of place cells along track
            df = self.pfparam_combined[self.pfparam_combined.animalname == n]
            numpyfile = [i for i in self.npyfiles if n in i][0]
            droppedcells = np.load(os.path.join(self.CombinedDataFolder, numpyfile), allow_pickle=True)[
                'dropped_cells'].item()[tasks_to_plot[1]]
            if n == 'CFC4':
                normfactor = 678
            else:
                normfactor = np.load(os.path.join(self.CombinedDataFolder, numpyfile), allow_pickle=True)[
                    'numcells']
            df_plot = df[df['Task'] == tasks_to_plot[0]]
            com = df_plot[df_plot['CellNumber'].isin(droppedcells)]['WeightedCOM'] * self.trackbins
            hist_com, bins_com, center, width = CommonFunctions.make_histogram(com, bins, len(df_plot),
                                                                               self.tracklength)
            ax.bar(center, hist_com, align='center', width=width, color=sns.color_palette('dark', 1), alpha=0.5)
            ax.set_title('Fields in %s but not in %s' % (tasks_to_plot[0], tasks_to_plot[1]))
            pf.set_axes_style(ax, numticks=4)

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
        axins = CommonFunctions.add_colorbar_as_inset(axes=ax[1])
        cb = fs.colorbar(img, cax=axins, pad=0.2, ticks=[0, np.int(np.max(heatmap))])
        cb.set_label('Field Density', rotation=270, labelpad=12)

        for a in ax:
            pf.set_axes_style(a, numticks=5)

    def plot_pfparams_bytracklength(self, tasks_to_plot, nbins):
        # Plot a combined historgram and a boxplot of means
        columns_to_plot = ['Precision', 'Width', 'FiringRatio', 'Firingintensity']
        df_plot = self.pfparam_combined[self.pfparam_combined['Task'].isin(tasks_to_plot)]
        color = sns.color_palette('deep', len(tasks_to_plot))
        fs, ax = plt.subplots(len(columns_to_plot), 2, figsize=(10, 8), dpi=100, sharex='all', sharey='row')

        for i in np.unique(df_plot.animalname):
            npy_file = [f for f in self.npyfiles if i in f][0]
            data_animal = np.load(os.path.join(self.CombinedDataFolder, npy_file))
            numplacecells_task1 = np.sum(data_animal['numPFs_incells'].item()[tasks_to_plot[0]])
            for n, t in enumerate(tasks_to_plot):
                com_animal = df_plot[(df_plot.Task == t) & (df_plot.animalname == i)]['WeightedCOM'] * self.trackbins
                bins = np.linspace(0, self.tracklength, nbins + 1)
                ind = np.digitize(com_animal, bins)
                ax[0, n].set_title(t)
                for n2, c in enumerate(columns_to_plot):
                    y = np.asarray(df_plot[(df_plot.Task == t) & (df_plot.animalname == i)][c])
                    mean_binned = np.zeros(nbins)
                    error = np.zeros(nbins)
                    for b in np.arange(1, nbins + 1):
                        m, ci = CommonFunctions.mean_confidence_interval(y[ind == b])
                        mean_binned[b - 1], error[b - 1] = m, ci
                    width = np.diff(bins)
                    center = (bins[:-1] + bins[1:]) / 2
                    ax[n2, n].bar(center, mean_binned, width=width, alpha=0.5)
                    ax[n2, n].set_ylabel(c)

        for a in ax.flatten():
            a.set_xlabel('Track Length (cm)')
            pf.set_axes_style(a, numticks=4)

        fs.tight_layout()


class CommonFunctions:
    @staticmethod
    def mean_confidence_interval(data, confidence=0.95):
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.nanmean(a), scipy.stats.sem(a, nan_policy='omit')
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
        return m, h

    @staticmethod
    def add_inset_with_cdf(axins, y):
        axins.hist(y, bins=1000, normed=True, cumulative=True, label='CDF',
                   histtype='step', linewidth=2, alpha=0.5)
        axins.set_yticks([0, 1])

    @staticmethod
    def make_histogram(com, bins, normalisefactor, tracklength):
        hist_com, bins_com = np.histogram(com, bins=np.arange(0, tracklength + 5, bins))
        hist_com = (hist_com / np.sum(normalisefactor)) * 100
        width = np.diff(bins_com)
        center = (bins_com[:-1] + bins_com[1:]) / 2
        return hist_com, bins_com, center, width

    @staticmethod
    def significance_test(x, y, type_of_test='KStest'):
        if type_of_test == 'KStest':
            d, p = scipy.stats.ks_2samp(x, y)
            return d, p
        elif type_of_test == 'Wilcoxon':
            s, p = scipy.stats.ranksums(x, y)
            return s, p

    @staticmethod
    def normalitytest(x):
        d, p = scipy.stats.shapiro(x)
        return d, p

    @staticmethod
    def add_colorbar_as_inset(axes):
        axins = inset_axes(axes,
                           width="5%",  # width = 5% of parent_bbox width
                           height="50%",  # height : 50%
                           loc='lower left',
                           bbox_to_anchor=(1.05, 0., 1, 1),
                           bbox_transform=axes.transAxes,
                           borderpad=0.5,
                           )
        return axins
