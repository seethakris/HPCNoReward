import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


class ModelPredictionPlots(object):
    def __init__(self):
        print('Validation functions')

    def PlotCVResult(self, cv_dataframe, trackbins, numsplits):
        # Scatter plot of Y_test and Y_predict and difference between test and prediction
        fs, ax1 = plt.subplots(1, 2, figsize=(10, 3), dpi=100)
        R2 = []
        rho = []
        # Scatter
        for index, row in cv_dataframe.iterrows():
            Y_actual = row['Y_test']
            Y_predicted = row['Y_predict']
            R2.append(self.get_R2(Y_actual, Y_predicted))
            rho.append(self.get_rho(Y_actual, Y_predicted))
            sns.scatterplot(Y_actual * trackbins, Y_predicted * trackbins, color='gray', ax=ax1[0])
            ax1[0].set_ylim(ax1[0].get_ylim()[::-1])
            ax1[0].locator_params(nbins=4)
            ax1[0].set_xlabel('Actual Track Postion')
            ax1[0].set_ylabel('Predicted Track Postion')
        ax1[0].set_title(f'Final Accuracy : %.2f%% +/- %.2f%%'
                         % (np.mean(cv_dataframe['ModelAccuracy']),
                            np.std(cv_dataframe['ModelAccuracy'])))

        # Difference
        meandiff = np.mean(cv_dataframe['Y_diff'].to_list(), 1) * trackbins
        sem = scipy.stats.sem(cv_dataframe['Y_diff'].to_list(), 1) * trackbins
        error1, error2 = meandiff - sem, meandiff + sem
        ax1[1].plot(np.arange(numsplits), meandiff)
        ax1[1].fill_between(np.arange(numsplits), error1, error2, alpha=0.5)
        ax1[1].set_xlabel('CrossValidation #')
        ax1[1].set_ylabel('Difference (cm)')
        ax1[1].set_title(f'Difference in predicted position %.2f +/- %2f' % (
            np.mean(cv_dataframe['Y_diff'].to_list()) * trackbins,
            np.std(cv_dataframe['Y_diff'].to_list()) * trackbins))

        # Print mean correlation and mean R2
        print(f'R2: %.2f +/- %2f' % (np.mean(R2), np.std(R2)))
        print(f'Rho: %.2f +/- %2f' % (np.mean(rho), np.std(rho)))

    @staticmethod
    def PlotLapwiseAccuracy(lapframes, Y_actual, Y_predicted, numlaps, licks, velocity, stoplicklap=0,
                            good_running_flag=1):
        average_lap_diff = []
        standard_error_lap_diff = []
        for l in range(1, np.max(lapframes) + 1):
            laps = np.where(lapframes == l)[0]
            if good_running_flag:
                difference = np.abs(Y_actual[laps] - Y_predicted[laps])
            else:
                difference = np.abs(Y_actual[laps[0]:laps[-1]] - Y_predicted[laps[0]:laps[-1]])
            average_lap_diff.append(np.mean(difference))
            standard_error_lap_diff.append(scipy.stats.sem(difference))

        error1 = np.asarray(average_lap_diff) - np.asarray(standard_error_lap_diff)
        error2 = np.asarray(average_lap_diff) + np.asarray(standard_error_lap_diff)

        fs, ax1 = plt.subplots(1, 2, figsize=(15, 3), dpi=100)
        ax1[0].plot(np.arange(numlaps), average_lap_diff, linewidth=2)
        ax1[0].fill_between(np.arange(numlaps), error1, error2, alpha=0.5)
        ax2 = ax1[0].twinx()
        ax2.plot(np.arange(numlaps), velocity, color='grey', linewidth=2, alpha=0.5)
        ax2.set_ylabel('Velocity')

        ax1[1].plot(np.arange(numlaps), average_lap_diff, linewidth=2)
        ax1[1].fill_between(np.arange(numlaps), error1, error2, alpha=0.5)
        ax2 = ax1[1].twinx()
        ax2.plot(np.arange(numlaps - 1), licks, color='grey', linewidth=2, alpha=0.5)
        ax2.set_ylabel('Pre Licks')

        ax1[0].set_xlabel('Laps')
        ax1[0].set_ylabel('Prediction Error (cm)')

        if stoplicklap > 0:
            ax1[0].axvline(stoplicklap, color='black')
            ax1[1].axvline(stoplicklap, color='black')

        fs.tight_layout()
        return average_lap_diff, standard_error_lap_diff

    @staticmethod
    def plot_bayeserrorprob_beforeandaft_lick(Y_probability, lickstopframe, trackbins):
        plt.figure(figsize=(10, 3), dpi=80)
        y_prob_toplot = [Y_probability[:lickstopframe], Y_probability[lickstopframe:]]
        plot_label = ['Before Lick Stops', 'After Lick Stops']
        for index, y_prob in enumerate(y_prob_toplot):
            mean_error = np.mean(y_prob, 0)
            sem_error = scipy.stats.sem(y_prob, 0)
            error1 = mean_error - sem_error
            error2 = mean_error + sem_error
            plt.bar(np.arange(np.size(mean_error, 0)), mean_error, label=plot_label[index], alpha=0.5)
            # plt.fill_between(np.arange(np.size(mean_error, 0)), error1, error2, alpha=0.5)
            plt.xticks([0, 10, 20, 30, 39], [0, 50, 100, 150, 200])
            plt.ylabel('BD accuracy')
            plt.xlabel('Track Position (cm)')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()

    @staticmethod
    def plot_accuracy_beforeandaft_lickstops(Y_actual, Y_predicted, lickstopframe, tracklength, trackbins):
        fs = plt.figure(figsize=(10, 8), dpi=100)
        gs = plt.GridSpec(2, 2)
        y_actual_toplot = [Y_actual[:lickstopframe], Y_actual[lickstopframe:]]
        y_predicted_toplot = [Y_predicted[:lickstopframe], Y_predicted[lickstopframe:]]
        plot_title = ['Before Lick Stops', 'After Lick Stops']
        for axisnum, (actual, pred) in enumerate(zip(y_actual_toplot, y_predicted_toplot)):
            ax = fs.add_subplot(gs[0, axisnum])
            sns.scatterplot(actual * trackbins, pred * trackbins, color='gray', ax=ax)
            ax.set_ylim(ax.get_ylim()[::-1])
            ax.locator_params(nbins=4)
            ax.set_title(plot_title[axisnum])
            ax.set_xlabel('Actual Track Postion')
            ax.set_ylabel('Predicted Track Postion')

            # Get accuracy by tracklength
            y_diff = (actual - pred) * trackbins
            numbins = int(tracklength / trackbins)
            Y_diff_by_track_mean = np.zeros(numbins)
            Y_diff_by_track_sem = np.zeros(numbins)
            for i in np.arange(numbins):
                Y_indices = np.where(actual == i)[0]
                Y_diff_by_track_mean[i] = np.mean(y_diff[Y_indices])
                Y_diff_by_track_sem[i] = scipy.stats.sem(y_diff[Y_indices])
            ax = fs.add_subplot(gs[1, :])
            ax.bar(np.arange(numbins), Y_diff_by_track_mean, yerr=Y_diff_by_track_sem,
                   label=plot_title[axisnum], alpha=0.5)
            plt.xticks([0, 10, 20, 30, 39], [0, 50, 100, 150, 200])
            ax.locator_params(axis='y', nbins=4)
            ax.set_xlabel('Track Length (cm)')
            ax.set_ylabel('Difference in decoder accuracy (cm)')
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        fs.tight_layout()
        plt.show()

    @staticmethod
    def plot_accuracy_scatterplot(Y_actual, Y_predicted, trackbins):
        fs, ax1 = plt.subplots(1, figsize=(5, 3), dpi=100)
        sns.scatterplot(Y_actual * trackbins, Y_predicted * trackbins, color='gray', ax=ax1)
        ax1.set_ylim(ax1.get_ylim()[::-1])
        ax1.locator_params(nbins=4)
        ax1.set_xlabel('Actual Track Postion')
        ax1.set_ylabel('Predicted Track Postion')
        plt.show()

    @staticmethod
    def plot_bayes_probability(cv_dataframe):
        plt.figure(figsize=(10, 3), dpi=80)
        for index, row in cv_dataframe.iterrows():
            mean_error = row['mean_errorprob']
            sem_error = row['sem_errorprob']
            error1 = mean_error - sem_error
            error2 = mean_error + sem_error
            plt.plot(np.arange(np.size(mean_error, 0)), mean_error, label=f'Run%d' % index)
            plt.fill_between(np.arange(np.size(mean_error, 0)), error1, error2, alpha=0.5)
            plt.xticks([0, 10, 20, 30, 39], [0, 50, 100, 150, 200])
            plt.ylabel('BD accuracy')
            plt.xlabel('Track Position (cm)')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()

    @staticmethod
    def get_R2(Y_actual, Y_predicted):
        y_mean = np.mean(Y_actual)
        R2 = 1 - np.sum((Y_predicted - Y_actual) ** 2) / np.sum((Y_actual - y_mean) ** 2)
        return R2

    @staticmethod
    def get_rho(Y_actual, Y_predicted):
        rho = np.corrcoef(Y_actual, Y_predicted)[0, 1]
        return rho
