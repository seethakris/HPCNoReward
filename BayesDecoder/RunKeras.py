import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.utils.random import sample_without_replacement
import scipy.stats
from sklearn.model_selection import train_test_split
from PlotDecodingResults import ModelPredictionPlots
import os
import scipy.io

# fix random seed for reproducibility
np.random.seed(7)


class PrepareBehaviorData(object):

    def __init__(self, BehaviorData, tracklength, trackbins, trackstart_index=0, figure_flag=1):
        self.BehaviorData = BehaviorData
        self.tracklength = tracklength
        self.trackbins = trackbins
        self.trackstart = np.min(self.BehaviorData)
        self.trackend = np.max(self.BehaviorData)
        self.numbins = int(self.tracklength / self.trackbins)

        # Bin and Convert position to binary
        self.create_trackbins()
        self.position_binary = self.convert_y_to_index(self.BehaviorData, trackstart_index, figure_flag)

    def create_trackbins(self):
        self.tracklengthbins = np.around(np.linspace(self.trackstart, self.trackend, self.numbins),
                                         decimals=5)

    def convert_y_to_index(self, Y, trackstart_index=0, figure_flag=1):
        Y_binary = np.zeros((np.size(Y, 0)))
        i = 0
        while i < np.size(Y, 0):
            current_y = np.around(Y[i], decimals=4)
            idx = self.find_nearest1(self.tracklengthbins, current_y)
            Y_binary[i] = idx
            if idx == self.numbins - 1:
                while self.find_nearest1(self.tracklengthbins, current_y) != trackstart_index and i < np.size(Y, 0):
                    current_y = np.around(Y[i], decimals=4)
                    idx = self.find_nearest1(self.tracklengthbins, current_y)
                    if idx == self.numbins - 1 or idx == 0:  # Correct for end of the track misses
                        Y_binary[i] = idx
                    else:
                        Y_binary[i] = self.numbins - 1
                    i += 1
            i += 1

        if figure_flag:
            plt.figure(figsize=(10, 3), dpi=80)
            plt.plot(Y_binary)
            plt.ylabel('Binned Position')
            plt.xlabel('Frames')
        return Y_binary

    @staticmethod
    def find_nearest1(array, value):
        idx = (np.abs(array - value)).argmin()
        return idx


class PreprocessData(object):
    @staticmethod
    def get_frames_afterlickstops(Imgobj, X_norew, Y_norew):
        lapframes = \
            [scipy.io.loadmat(os.path.join(Imgobj.FolderName, 'Behavior', p))['E'].T for p in Imgobj.PlaceFieldData if
             'Task2' in p and 'Task2a' not in p][0]
        stoplicklap = Imgobj.Parsed_Behavior['lick_stop'].item()
        lickstop_goodbeh_frame = np.where(lapframes == stoplicklap)[0][-1]
        X_norew = X_norew[lickstop_goodbeh_frame:, :]
        Y_norew = Y_norew[lickstop_goodbeh_frame:]

        return X_norew, Y_norew



    @staticmethod
    def equalise_laps_with_numlaps_innorew(Imgobj, X, Y, Tasklabel):
        stoplicklap = Imgobj.Parsed_Behavior['lick_stop'].item()
        numlaps_afterlickstops = Imgobj.Parsed_Behavior['numlaps'].item()['Task2'] - stoplicklap
        print('Number of laps being chosen', numlaps_afterlickstops)
        numlaps_currenttask = Imgobj.Parsed_Behavior['numlaps'].item()[Tasklabel] - 3

        samplelaps = sample_without_replacement(numlaps_currenttask, numlaps_afterlickstops)
        lapframes = \
            [scipy.io.loadmat(os.path.join(Imgobj.FolderName, 'Behavior', p))['E'].T for p in Imgobj.PlaceFieldData if
             Tasklabel in p][0]
        print(samplelaps)
        X_eq = X[np.where(lapframes == samplelaps)[0], :]
        Y_eq = Y[np.where(lapframes == samplelaps)[0]]

        return X_eq, Y_eq


class SimpleDenseNN(object):

    def __init__(self, dropout=0, epochs=50, hiddenlayer=0, units=300, batch_size=10, verbose=0):
        # Model Parameters
        self.dropout = dropout
        self.epochs = epochs
        self.hiddenlayer = hiddenlayer
        self.units = units
        self.batch_size = batch_size
        self.verbose = verbose

    def fit_model(self, X_train, Y_train):
        model = Sequential()

        if self.hiddenlayer:
            # Add a single Dense NN input layer
            model.add(Dense(self.units[0], input_dim=np.size(X_train, 1), activation='relu'))
            if self.dropout != 0:
                model.add(Dropout(self.dropout))

            # Add hidden layers
            for i in range(np.size(self.units) - 1):
                model.add(Dense(self.units[i + 1], activation='relu'))
                if self.dropout != 0:
                    model.add(Dropout(self.dropout))

        else:
            # Add a single Dense NN layer
            model.add(Dense(self.units, input_dim=np.size(X_train, 1), activation='relu'))
            if self.dropout != 0:
                model.add(Dropout(self.dropout))

        # output layer
        model.add(Dense(np.size(Y_train, 1)))
        # Compile model
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        # Fit model
        print('Fitting Model')
        model.fit(X_train, Y_train, epochs=self.epochs, batch_size=self.batch_size,
                  verbose=self.verbose)

        # evaluate the model
        scores = model.evaluate(X_train, Y_train, batch_size=self.batch_size)
        print("\nAccuracy of training set: %.2f%%" % (scores[1] * 100))
        return model

    def validate_model(self, model, X_test, Y_test, plot_figure=1):
        scores = model.evaluate(X_test, Y_test, batch_size=self.batch_size)
        print("\nAccuracy of test set:: %.2f%%" % (scores[1] * 100))
        ynew = model.predict_classes(X_test)
        if plot_figure:
            plt.figure(figsize=(10, 3), dpi=80)
            plt.plot(ynew, alpha=0.5, linewidth=1, label='Predicted position')
            plt.plot(np.argmax(Y_test, 1), alpha=0.5, label='Actual position')
            plt.ylabel('Binned Position')
            plt.xlabel('Frames')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.show()

        return scores[1] * 100, ynew


class SimpleRNN(object):
    @staticmethod
    def create_X_Withhistory(neural_data, bins_before, bins_after, bins_current=1):
        num_examples = neural_data.shape[0]  # Number of total time bins we have neural data for
        num_neurons = neural_data.shape[1]  # Number of neurons
        surrounding_bins = bins_before + bins_after + bins_current  # Number of surrounding time bins used for prediction
        X = np.zeros([num_examples, surrounding_bins, num_neurons])  # Initialize covariate matrix with NaNs
        start_idx = 0
        for i in range(
                num_examples - bins_before - bins_after):  # The first bins_before and last bins_after bins don't get filled in
            end_idx = start_idx + surrounding_bins  # The bins of neural data we will be including are between start_idx and end_idx (which will have length "surrounding_bins")
            X[i + bins_before, :, :] = neural_data[start_idx:end_idx,
                                       :]  # Put neural data from surrounding bins in X, starting at row "bins_before"
            start_idx = start_idx + 1
        return X

    @staticmethod
    def fit_LSTM(X_train, y_train, epochs):
        model = Sequential()  # Declare model
        model.add(LSTM(X_train.shape[2],
                       input_shape=(X_train.shape[1], X_train.shape[2])))  # Within recurrent layer, include dropout
        model.add(Dense(y_train.shape[1]))
        model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])  # Set loss function and optimizer
        model.fit(X_train, y_train, nb_epoch=epochs, verbose=1)  # Fit the model
        # evaluate the model
        scores = model.evaluate(X_train, y_train)
        print("\nAccuracy of training set: %.2f%%" % (scores[1] * 100))

        return model

    @staticmethod
    def validate_model(model, X_test, y_test):
        scores = model.evaluate(X_test, y_test)
        print("\nAccuracy of test set:: %.2f%%" % (scores[1] * 100))
        ynew = model.predict_classes(X_test)
        plt.figure(figsize=(10, 3), dpi=80)
        plt.plot(ynew, alpha=0.5, linewidth=1, label='Predicted position')
        plt.plot(np.argmax(y_test, 1), alpha=0.5, label='Actual position')
        plt.ylabel('Binned Position')
        plt.xlabel('Frames')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()

        return ynew


class NaiveBayes(object):
    @staticmethod
    def fit_naivebayes(X_train, y_train):
        # Fit Bayes
        clf = GaussianNB()
        clf.fit(X_train, y_train)

        return clf

    @staticmethod
    def validate_model(model, X_test, y_test, plot_figure=1):
        scores = model.score(X_test, y_test)
        print("\nAccuracy of test set:: %.2f%%" % scores)

        ynew = model.predict(X_test)
        yprob = model.predict_proba(X_test)

        if plot_figure:
            plt.figure(figsize=(10, 3), dpi=80)
            plt.plot(ynew, alpha=0.5, linewidth=1, label='Predicted position')
            plt.plot(y_test, alpha=0.5, label='Actual position')
            plt.ylabel('Binned Position')
            plt.xlabel('Frames')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.show()

        return scores, ynew, yprob

    @staticmethod
    def plot_error_probability(yprob):
        mean_error = np.mean(yprob, 0)
        sem_error = scipy.stats.sem(yprob, 0)
        error1 = mean_error - sem_error
        error2 = mean_error + sem_error
        plt.figure(figsize=(10, 3), dpi=80)
        plt.plot(np.arange(np.size(yprob, 1)), np.mean(yprob, 0))
        plt.fill_between(np.arange(np.size(yprob, 1)), error1, error2, alpha=0.5)
        plt.xticks([0, 10, 20, 30, 39], [0, 50, 100, 150, 200])
        plt.ylabel('BD accuracy')
        plt.xlabel('Track Position (cm)')
        plt.show()

    def decoderaccuracy_wtih_numcells(self, X_data, Y_data, iterations, plot_figure=0):
        m = ModelPredictionPlots()
        numcells = np.size(X_data, 1)
        numsamples = [np.int(numcells * 0.1), np.int(numcells * 0.2), np.int(numcells * 0.5),
                      np.int(numcells * 0.8), numcells]
        R2_numcells_dict = {str(k): [] for k in numsamples}
        rho_numcells_dict = {str(k): [] for k in numsamples}
        for ns in numsamples:
            print(f'Fitting on %d neurons' % ns)
            for i in np.arange(iterations):
                cells = sample_without_replacement(numcells, ns)
                X_resample = X_data[:, cells]
                X_rs_train, X_rs_test, y_rs_train, y_rs_test = train_test_split(X_resample, Y_data, test_size=0.10,
                                                                                random_state=None, shuffle=False)
                nbpfmodel = self.fit_naivebayes(X_rs_train, y_rs_train)
                scores, prediction, probability = self.validate_model(nbpfmodel, X_rs_test, y_rs_test,
                                                                      plot_figure=plot_figure)
                R2_numcells_dict[str(ns)].append(m.get_R2(y_rs_test, prediction))
                rho_numcells_dict[str(ns)].append(m.get_rho(y_rs_test, prediction))
        return numsamples, R2_numcells_dict, rho_numcells_dict

    @staticmethod
    def plot_decoderaccuracy_with_numcells(R2_numcells_dict, numsamples):
        mean_R2 = []
        sem_R2 = []
        for key, item in R2_numcells_dict.items():
            mean_R2.append(np.mean(item))
            sem_R2.append(scipy.stats.sem(item))
        plt.figure(figsize=(5, 3), dpi=100)
        plt.errorbar(np.arange(np.size(numsamples)), mean_R2, yerr=sem_R2, color='k', fmt='o-',
                     ecolor='gray', capthick=5, linewidth=2)
        plt.xticks(np.arange(np.size(numsamples)), ['10%', '20%', '50%', '80%', '100%'])
        plt.xlabel('Percentage of place cells used')
        plt.ylabel('Decoding accuracy')

        return mean_R2, sem_R2
