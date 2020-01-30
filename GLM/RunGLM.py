import numpy as np
import os
import sys
import scipy.io
import h5py
from scipy.signal import savgol_filter
import seaborn as sns
from itertools import groupby
import matplotlib.pyplot as plt
from scipy.stats import zscore
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import log_loss
from sklearn.linear_model import Ridge

DataDetailsFolder = '/home/sheffieldlab/Desktop/NoReward/Scripts/AnimalDetails/'
sys.path.append(DataDetailsFolder)
import DataDetails


class createbehavioralparams(object):
    def __init__(self, FolderName, animalname):
        self.FolderName = FolderName
        self.animalname = animalname
        self.animalinfo = DataDetails.ExpAnimalDetails(self.animalname)
        self.Task_Numframes = self.animalinfo['task_numframes']
        self.TaskDict = self.animalinfo['task_dict']
        self.framespersec = 30.98
        self.trackbins = 5
        self.tracklength = 200
        self.trackstartindex = self.animalinfo['trackstart_index']

        self.get_fluorescence_data()
        self.get_behavior()
        self.load_lapparams()
        self.calculate_velocity()

    def get_fluorescence_data(self):
        ImgFileName = [f for f in os.listdir(os.path.join(self.FolderName, self.animalname)) if f.endswith('.mat')][0]
        if self.animalinfo['v73_flag']:
            self.load_v73_Data(ImgFileName)
        else:
            self.load_fluorescentdata(ImgFileName)

    def get_behavior(self):
        self.Parsed_Behavior = np.load(
            os.path.join(self.FolderName, self.animalname, 'SaveAnalysed', 'behavior_data.npz'),
            allow_pickle=True)
        self.runningdata = self.Parsed_Behavior['running_data'].item()
        self.goodrunningindex = self.Parsed_Behavior['good_running_index'].item()
        self.runningdata_incm = CommonFunctions.create_data_dict(self.TaskDict.keys())
        for t in self.TaskDict:
            self.runningdata_incm[t] = (self.runningdata[t] * self.tracklength) / np.max(self.runningdata[t])
        self.lickdata = self.Parsed_Behavior['lick_data'].item()
        self.rewardata = self.Parsed_Behavior['reward_data'].item()

    def load_fluorescentdata(self, ImgFileName):
        self.Fcdata_dict = CommonFunctions.create_data_dict(self.TaskDict.keys())
        self.Fc3data_dict = CommonFunctions.create_data_dict(self.TaskDict.keys())
        # Open calcium data and store in dicts per trial
        data = scipy.io.loadmat(os.path.join(self.FolderName, self.animalname, ImgFileName))
        self.numcells = np.size(data['data'].item()[1], 1)
        self.meanimg = data['data'].item()[5]
        count = 0
        for i in self.TaskDict.keys():
            self.Fcdata_dict[i] = data['data'].item()[1].T[:,
                                  count:count + self.Task_Numframes[i]]
            self.Fc3data_dict[i] = data['data'].item()[2].T[:,
                                   count:count + self.Task_Numframes[i]]
            count += self.Task_Numframes[i]

    def load_v73_Data(self, ImgFileName):
        self.Fcdata_dict = CommonFunctions.create_data_dict(self.TaskDict.keys())
        self.Fc3data_dict = CommonFunctions.create_data_dict(self.TaskDict.keys())
        f = h5py.File(os.path.join(self.FolderName, self.animalname, ImgFileName), 'r')
        for k, v in f.items():
            print(k, np.shape(v))

        count = 0
        self.numcells = np.size(f['Fc'], 0)
        for i in self.TaskDict.keys():
            self.Fcdata_dict[i] = f['Fc'][:, count:count + self.Task_Numframes[i]]
            self.Fc3data_dict[i] = f['Fc3'][:, count:count + self.Task_Numframes[i]]
            count += self.Task_Numframes[i]

    def load_lapparams(self):
        self.PlaceFieldFolder = \
            [f for f in os.listdir(os.path.join(self.FolderName, self.animalname, 'Behavior')) if
             (f.endswith('.mat') and 'PlaceFields' in f and 'GoodBehavior' in f and 'Lick' not in f)]
        self.good_lapframes = CommonFunctions.create_data_dict(self.TaskDict.keys())
        for t in self.TaskDict.keys():
            self.good_lapframes[t] = \
                [scipy.io.loadmat(os.path.join(self.FolderName, self.animalname, 'Behavior', p))['E'].T for p in
                 self.PlaceFieldFolder if t in p and 'Task2a' not in p][0]
            self.good_lapframes[t] = self.good_lapframes[t]

    def calculate_velocity(self):
        self.velocity_bytime = CommonFunctions.create_data_dict(self.TaskDict.keys())
        self.inst_accelaration = CommonFunctions.create_data_dict(self.TaskDict.keys())

        time = 1 / self.framespersec
        for t in self.TaskDict:
            velocity_diff = np.zeros((np.size(self.runningdata_incm[t])))
            accelartion_diff = np.zeros((np.size(self.runningdata_incm[t])))

            lapframes = self.good_lapframes[t]
            for this in range(1, np.max(lapframes) + 1):
                lap_start, lap_end = np.where(lapframes == this)[0][0], np.where(lapframes == this)[0][-1]
                r_start, r_end = self.goodrunningindex[t][lap_start], self.goodrunningindex[t][lap_end]
                runs = self.runningdata_incm[t][r_start:r_end]
                velocity_diff[r_start:r_end - 1] = np.diff(np.squeeze(runs))
                accelartion_diff[r_start:r_end - 1] = np.diff(velocity_diff[r_start:r_end])

            self.velocity_bytime[t] = velocity_diff / time
            self.inst_accelaration[t] = accelartion_diff / time

    def find_transient_distribution(self, fdata, threshold=0.5, transthreshold=10):
        numtransients = np.zeros(np.size(fdata, 0))
        numtime = (np.size(fdata, 1) / self.framespersec)
        for i in np.arange(np.size(fdata, 0)):
            # filterdata
            smdata = savgol_filter(fdata[i, :], 31, 2)
            bw_trans = smdata > threshold
            numtransients[i] = np.size(CommonFunctions.consecutive_one(bw_trans))
        numtransients_persec = (numtransients / numtime) * 100
        ax = sns.distplot(numtransients_persec, bins=10, kde=False)
        ax.set_xlabel('Transients per second * 100')
        ax.set_ylabel('Number of cells')

        activecells = np.where(numtransients_persec > transthreshold)[0]
        return activecells

    def get_variables_for_GLM(self, taskstotest):
        fluor, position, velocity, accelaration, lick = {}, {}, {}, {}, {}
        for t in taskstotest:
            print(t)
            fluor[t], position[t], velocity[t], accelaration[t], lick[t] = self.compile_task_variables_for_GLM(task=t)

        # Combine fluorescence for tcalculating activecells
        combined_fluor = np.asarray([])
        for t in taskstotest:
            combined_fluor = np.hstack((combined_fluor, fluor[t])) if combined_fluor.size else fluor[t]
        return combined_fluor, fluor, position, velocity, accelaration, lick

    def compile_task_variables_for_GLM(self, task, smoothing_window=31):
        # Get only frames with vr start and end
        lapframes = np.where(self.good_lapframes[task])[0]
        vrstart, vrend = self.goodrunningindex[task][lapframes[0]], self.goodrunningindex[task][lapframes[-1]]
        fluor = self.Fc3data_dict[task][:, vrstart:vrend]
        position = self.runningdata[task][vrstart:vrend]
        velocity = self.velocity_bytime[task][vrstart:vrend]
        accelaration = self.inst_accelaration[task][vrstart:vrend]
        lick = self.lickdata[task][vrstart:vrend]

        # Filter everything
        for i in np.arange(np.size(fluor, 0)):
            fluor[i, :] = savgol_filter(fluor[i, :], smoothing_window, 2)
        velocity = savgol_filter(velocity, smoothing_window, 2)
        accelaration = savgol_filter(accelaration, smoothing_window, 2)
        velocity = savgol_filter(velocity, smoothing_window, 2)

        return fluor, position, velocity, accelaration, lick


class GLM(object):
    def __init__(self, animalname, F_data, position, velocity, accelaration, lick, taskstotest):
        self.animalname = animalname
        self.taskstotest = taskstotest
        self.F_data = F_data
        self.position = position
        self.velocity = velocity
        self.accelaration = accelaration
        self.lick = lick

        self.F_norm = self.normalize_F()
        self.scale_all_behavioralvariables()

    def normalize_F(self):
        # Normalize F and ensure there are no negative going transients
        F_norm = {}
        for t in self.taskstotest:
            num = self.F_data[t]
            denom = (np.max(self.F_data[t], 1)[:, np.newaxis])
            F_norm[t] = num / denom
            F_norm[t][F_norm[t] < 0] = 0
        return F_norm

    def scale_all_behavioralvariables(self):
        for t in self.taskstotest:
            self.velocity[t] = (self.velocity[t] - np.min(self.velocity[t])) / (
                    np.max(self.velocity[t]) - np.min(self.velocity[t]))
            self.accelaration[t] = (self.accelaration[t] - np.min(self.accelaration[t])) / (
                    np.max(self.accelaration[t]) - np.min(self.accelaration[t]))
            self.lick[t] = (self.lick[t] - np.min(self.lick[t])) / (np.max(self.lick[t]) - np.min(self.lick[t]))
            self.position[t] = (self.position[t] - np.min(self.position[t])) / (
                    np.max(self.position[t]) - np.min(self.position[t]))

    def plot_behavior_variables(self, xlim=5000):
        fs, ax = plt.subplots(len(self.taskstotest), figsize=(12, 6))
        for n, t in enumerate(self.taskstotest):
            ax[n].plot(self.velocity[t], label='velocity', alpha=0.5)
            ax[n].plot(self.position[t], label='position', alpha=0.5)
            ax[n].plot(self.lick[t], label='lick', alpha=0.5)
            ax[n].plot(self.accelaration[t], label='accelaration', alpha=0.5)
            ax[n].set_xlim((0, xlim))
            ax[n].legend()

    def behdataframe_for_glm(self, behvariables, task):
        behdataframe = pd.DataFrame({'pos': list(np.squeeze(self.position[task])),
                                     'vel': list(self.velocity[task]), 'lick': list(np.squeeze(self.lick[task])),
                                     'acc': list(self.accelaration[task])})
        behdataframe = behdataframe[behvariables]
        return behdataframe

    def run_glm(self, behvariables, activecells):
        residuals = CommonFunctions.create_data_dict(self.taskstotest)
        score = CommonFunctions.create_data_dict(self.taskstotest)
        coefficients = pd.DataFrame()
        for t in self.taskstotest:
            print(t)
            beh_df = self.behdataframe_for_glm(behvariables=behvariables, task=t)
            exog = sm.add_constant(beh_df)
            for a in activecells:
                endog = np.squeeze(self.F_norm[t][a, :])
                try:
                    res, res_coeffs, res_score = self.glm_function(endog, exog, task=t)
                except:
                    print('Coundnt run ', a)
                    continue
                residuals[t].append(res)
                score[t].append(res_score)
                coefficients = coefficients.append(res_coeffs, ignore_index=True)
        return residuals, score, coefficients

    def glm_function(self, endog, exog, task, loo=np.nan):
        glm_poisson = sm.GLM(endog, exog, family=sm.families.Poisson((sm.families.links.log)))
        res = glm_poisson.fit()
        score = CommonFunctions.get_R2(endog, res.predict())
        res_coeffs = res.params
        res_coeffs['Task'] = task
        res_coeffs['animalname'] = self.animalname
        if not np.isnan(loo):
            res_coeffs['Loo_Split'] = loo

        return res, res_coeffs, score

    def leave_one_out_test(self, behvariables, activecells):
        coefficients = pd.DataFrame()
        loo = LeaveOneOut()
        numsplits = loo.get_n_splits(behvariables)
        ratio = {k: np.zeros((numsplits, len(activecells))) for k in self.taskstotest}
        score = {k: np.zeros((numsplits, len(activecells))) for k in self.taskstotest}

        for t in self.taskstotest:
            print(t)
            beh_df = self.behdataframe_for_glm(behvariables=behvariables, task=t)
            for n, a in enumerate(activecells):
                # Full model
                endog = np.squeeze(self.F_norm[t][a, :])
                exog = sm.add_constant(beh_df)
                try:
                    res, res_coeffs, res_score = self.glm_function(endog, exog, task=t)
                except:
                    print('Coundnt run ', a)
                    continue

                # Leave oneout and test
                for split, (train_index, test_index) in enumerate(loo.split(behvariables)):
                    loo_beh = beh_df[beh_df.columns[train_index]]
                    exog = sm.add_constant(loo_beh)
                    # print(loo_beh)
                    try:
                        loo_res, loo_res_coeffs, loo_res_score = self.glm_function(endog, exog, task=t)
                    except:
                        print('LOO : Coundnt run ', a)
                        continue
                    ratio[t][split, n] = res_score - loo_res_score
                    score[t][split, n] = loo_res_score
                    coefficients = coefficients.append(loo_res_coeffs, ignore_index=True)
        return score, coefficients, ratio

    def run_ridge(self, behvariables, activecells):
        score = CommonFunctions.create_data_dict(self.taskstotest)
        coefficients = pd.DataFrame()
        for t in self.taskstotest:
            print(t)
            beh_df = self.behdataframe_for_glm(behvariables=behvariables, task=t)
            exog = np.asarray(beh_df)
            for a in activecells:
                endog = np.squeeze(self.F_norm[t][a, :])
                try:
                    res, res_score, res_coeffs = self.ridge_regression(endog, exog, task=t, behvariables=behvariables)
                except:
                    print('Coundnt run ', a)
                    continue
                score[t].append(res_score)
                coefficients = coefficients.append(res_coeffs, ignore_index=True)
        return score, coefficients

    def leave_one_out_test_ridge(self, behvariables, activecells):
        coefficients = pd.DataFrame()
        loo = LeaveOneOut()
        numsplits = loo.get_n_splits(behvariables)
        ratio = {k: np.zeros((numsplits, len(activecells))) for k in self.taskstotest}
        score = {k: np.zeros((numsplits, len(activecells))) for k in self.taskstotest}

        for t in self.taskstotest:
            print(t)
            beh_df = self.behdataframe_for_glm(behvariables=behvariables, task=t)
            for n, a in enumerate(activecells):
                # Full model
                endog = np.squeeze(self.F_norm[t][a, :])
                exog = np.asarray(beh_df)
                try:
                    res, res_score, res_coeffs = self.ridge_regression(endog, exog, task=t, behvariables=behvariables)
                except:
                    print('Coundnt run ', a)
                    continue

                # Leave oneout and test
                for split, (train_index, test_index) in enumerate(loo.split(behvariables)):
                    loo_beh = beh_df[beh_df.columns[train_index]]
                    exog = np.asarray(loo_beh)
                    try:
                        loo_res, loo_res_score, loo_res_coeffs = self.ridge_regression(endog, exog, task=t,
                                                                                       behvariables=beh_df.columns[
                                                                                           train_index])
                    except:
                        print('LOO : Coundnt run ', a)
                        continue
                    ratio[t][split, n] = (res_score - loo_res_score) / res_score
                    score[t][split, n] = loo_res_score
                    coefficients = coefficients.append(loo_res_coeffs, ignore_index=True)
        return score, coefficients, ratio

    def ridge_regression(self, endog, exog, task, behvariables):
        coeff = {}
        clf = Ridge(alpha=1.0)
        clf.fit(exog, endog)
        y_pred = clf.predict(exog)
        score = CommonFunctions.get_R2(endog, y_pred)
        # print(clf.coef_, score)
        for n, b in enumerate(behvariables):
            coeff[b] = clf.coef_[n]
        coeff['Task'] = task
        coeff['animalname'] = self.animalname
        coeff = pd.DataFrame([coeff])
        # print(coeff)
        return y_pred, score, coeff


class CommonFunctions(object):
    @staticmethod
    def create_data_dict(TaskDict):
        data_dict = {keys: [] for keys in TaskDict}
        return data_dict

    @staticmethod
    def len_iter(items):
        return sum(1 for _ in items)

    @staticmethod
    def consecutive_one(data):
        return [CommonFunctions.len_iter(run) for val, run in groupby(data) if val]

    @staticmethod
    def get_R2(y_actual, y_predicted):
        y_mean = np.mean(y_actual)
        R2 = 1 - np.sum((y_predicted - y_actual) ** 2) / np.sum((y_actual - y_mean) ** 2)
        return R2

    # @staticmethod
    # def find_log_likelihoodratio(y_pred1, y_pred2):
