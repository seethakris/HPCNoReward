""" Population analysis using SVMs"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import scipy.io
import os
from collections import OrderedDict
import scipy.stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import sys
from matplotlib.backends.backend_pdf import PdfPages

# For plotting styles
PlottingFormat_Folder = '/home/sheffieldlab/Desktop/NoReward/Scripts/PlottingTools/'
sys.path.append(PlottingFormat_Folder)
import plottingfunctions as pf

pf.set_style()

from get_data_for_bayes import LoadData
from get_data_for_bayes import SVMValidationPlots
from get_data_for_bayes import SVM
from get_data_for_bayes import PrepareBehaviorData
from get_data_for_bayes import CommonFunctions

# Some global parameters for easy change
plot_kfold = 0
plot_numcells = 0
kfold_splits = 10
numcell_iterations = 10
numcell_kfold_splits = 10


def runSVM(animalinfo):
    rb = RunBayeswithoutRewZone([10, 30])
    pdf = PdfPages(os.path.join(animalinfo['saveresults'], 'SVMResults.pdf'))
    # Load Fluorescence and Behavior data
    data = LoadData(FolderName=animalinfo['foldername'], Task_NumFrames=animalinfo['task_numframes'],
                    TaskDict=animalinfo['task_dict'], framestokeep=animalinfo['task_framestokeep'],
                    v73_flag=animalinfo['v73_flag'])
    bin_behdata = PrepareBehaviorData(BehaviorData=data.good_running_data, TaskDict=animalinfo['task_dict'],
                                      tracklength=animalinfo['tracklength'], trackbins=animalinfo['trackbins'],
                                      plotpdf=pdf,
                                      trackstart_index=animalinfo['trackstart_index'], figure_flag=0)

    # Get Xdata and Ydata for decoder
    Xdata = OrderedDict()
    Ydata = OrderedDict()
    for keys in animalinfo['task_dict'].keys():
        if 'Task2' in keys:
            Xdata[keys] = data.Fc3data_dict[keys][:, data.good_running_index[keys]].T[
                          data.lickstopframe:animalinfo['task_framestokeep'][keys], :]
            Ydata[keys] = bin_behdata.position_binary[keys][data.lickstopframe:animalinfo['task_framestokeep'][keys]]
        else:
            Xdata[keys] = data.Fc3data_dict[keys][:, data.good_running_index[keys]].T[
                          :animalinfo['task_framestokeep'][keys], :]
            Ydata[keys] = bin_behdata.position_binary[keys][:animalinfo['task_framestokeep'][keys]]

    # Remove end zones
    for keys in animalinfo['task_dict'].keys():
        Xdata[keys], Ydata[keys] = rb.remove_end_zone_frames(Xdata[keys], Ydata[keys])
        Xdata[keys] = Xdata[keys][:animalinfo['task_framestokeep_afterendzone'][keys], :]
        Ydata[keys] = Ydata[keys][:animalinfo['task_framestokeep_afterendzone'][keys]]
        print(f'Good Running Data Frames after endzone deleted: %s, %d' % (keys, np.size(Ydata[keys])))

    taskmodel = OrderedDict()
    taskmodel['Xdata'] = Xdata
    taskmodel['Ydata'] = Ydata
    for t in animalinfo['task_dict'].keys():
        taskmodel[t] = run_svm_on_task(xdata=Xdata[t], ydata=Ydata[t],
                                       task=t, plotpdf=pdf)

        # Plot some figures
        fs, ax = plt.subplots(1, 2, figsize=(20, 3))
        taskmodel[t]['cm'] = rb.plot_confusion_matrix_ofkfold(fighandle=fs, axis=ax[0],
                                                           cv_dataframe=taskmodel[t]['K-foldDataframe'],
                                                           tracklength=animalinfo['tracklength'],
                                                           trackbins=animalinfo['trackbins'])

        SVMValidationPlots.plotcrossvalidationresult(axis=ax[1], cv_dataframe=taskmodel[t]['K-foldDataframe'],
                                                     numsplits=kfold_splits,
                                                     trackbins=animalinfo['trackbins'])
        pdf.savefig(fs, bbox_inches='tight')
    pdf.close()

    # Save SVM models from each dataset
    np.save(os.path.join(animalinfo['saveresults'], 'modeloneachtask_withendzonerem'), taskmodel)
    return taskmodel


def run_svm_on_task(xdata, ydata, task, plotpdf):
    taskmodel = OrderedDict()

    taskmodel['x_train'], taskmodel['x_test'], taskmodel['y_train'], taskmodel['y_test'] = SVM.split_data(
        x=xdata, y=ydata)

    taskmodel['SVMmodel'] = SVM.fit_SVM(x_train=taskmodel['x_train'], y_train=taskmodel['y_train'])
    taskmodel['scores'], taskmodel['y_pred'], taskmodel['y_prob'] = SVM.validate_model(
        model=taskmodel['SVMmodel'],
        x_test=taskmodel['x_test'],
        y_test=taskmodel['y_test'],
        task=task, plotflag=1, plotpdf=plotpdf)

    taskmodel['K-foldDataframe'] = SVM().k_foldvalidation(x=xdata, y=ydata, task='K-fold validation',
                                                          split_size=kfold_splits)
    taskmodel['Numcells_Dataframe'] = SVM().decoderaccuracy_wtih_numcells(x=xdata, y=ydata,
                                                                          iterations=numcell_iterations,
                                                                          task='Accuracy by number of cells')

    return taskmodel


class RunBayeswithoutRewZone(object):
    def __init__(self, bounding_edges):
        self.bounding_edges = bounding_edges
        print('Removing end zones')

    def remove_end_zone_frames(self, xdata, ydata):
        indices = np.where((self.bounding_edges[0] < ydata) & (ydata < self.bounding_edges[1]))
        ydata_new = ydata[indices]
        xdata_new = xdata[indices]
        return xdata_new, ydata_new

    def plot_confusion_matrix_ofkfold(self, fighandle, axis, cv_dataframe, tracklength, trackbins):
        cm_all = np.zeros((int(tracklength / trackbins), int(tracklength / trackbins)))  # Bins by binss
        for i in np.arange(kfold_splits):
            y_actual = cv_dataframe['y_test'][i]
            y_predicted = cv_dataframe['y_predict'][i]
            cm = confusion_matrix(y_actual, y_predicted)
            print(np.shape(cm))
            if np.size(cm, 0) != int(tracklength / trackbins):
                cm_temp = np.zeros((int(tracklength / trackbins), int(tracklength / trackbins)))
                cm_temp[self.bounding_edges[0]:self.bounding_edges[1] - 1,
                self.bounding_edges[0]:self.bounding_edges[1] - 1] = cm
                print('Correcting', np.shape(cm_temp))
                cm_all += cm_temp
            else:
                cm_all += cm
        cm_all = cm_all.astype('float') / cm_all.sum(axis=1)[:, np.newaxis]
        img = axis.imshow(cm_all, cmap="Blues", vmin=0, vmax=0.5,
                          interpolation='bilinear')
        CommonFunctions.create_colorbar(fighandle=fighandle, axis=axis, imghandle=img, title='Probability')
        # convert axis ro track length
        CommonFunctions.convertaxis_to_tracklength(axis, tracklength, trackbins, convert_axis='both')
        axis.plot(axis.get_xlim()[::-1], axis.get_ylim(), ls="--", c=".3", lw=1)

        axis.set_ylabel('Actual')
        axis.set_xlabel('Predicted')
        pf.set_axes_style(axis, numticks=4)

        return cm_all
