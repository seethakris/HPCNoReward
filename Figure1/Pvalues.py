from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)
import scipy.stats
import numpy as np


class GetPValues:
    def get_shuffle_pvalue(self, accuracy_dataframe, taskstocompare):
        # Get two p-values. One with outlier and one without
        accuracy_dataframe = accuracy_dataframe[taskstocompare]
        outlier_rem_df, numoutliers = self.remove_outlier(accuracy_dataframe, taskstocompare)
        print(np.size(numoutliers))
        if np.size(numoutliers) > 0:
            print('\033[1mMultiple comparisons after removing Outliers\033[0m')
            test = self.get_normality(outlier_rem_df)
            self.get_multiplecomparisons(outlier_rem_df, test)
            print('\n\n\n')
        print('\033[1mMultiple comparisons without removing outliers\033[0m')
        test = self.get_normality(accuracy_dataframe)
        self.get_multiplecomparisons(accuracy_dataframe, test)
        print('\n\n\n')

    def get_normality(self, dataframe):
        dataframe = dataframe.dropna()
        test = 'ttest'
        for t in dataframe.columns:
            # print(t, np.asarray(dataframe[t], dtype=np.float64))
            d = np.asarray(dataframe[t], dtype=np.float64)
            ks, p = scipy.stats.shapiro(np.asarray(d))
            print('Normality test for %s p-value %0.3f' % (t, p))
            if p < 0.05:
                # Even if one isnt normal do kruskal
                test = 'kruskal'
                print('Performing Non Parametric test \n')
                return test
        print('Performing Parametric test \n')
        return test

    def get_multiplecomparisons(self, dataframe, test):
        # If distributions are different then do multiple comparisons
        dataframe = dataframe.dropna()
        print(dataframe)
        cleanbin = dataframe.melt(var_name='Bin', value_name='Value')
        MultiComp = MultiComparison(cleanbin['Value'],
                                    cleanbin['Bin'])
        if test == 'ttest':
            comp = MultiComp.allpairtest(scipy.stats.ttest_rel, method='Bonf')
        else:
            comp = MultiComp.allpairtest(scipy.stats.wilcoxon, method='Bonf')
        print(comp[0])

    def remove_outlier(self, dataframe, taskstocompare):
        index_outlier = []
        for t in taskstocompare:
            taskdata = dataframe[t]
            taskdata = taskdata.dropna()
            Q1 = taskdata.quantile(0.25)
            Q3 = taskdata.quantile(0.75)
            IQR = Q3 - Q1
            outlier = (taskdata < (Q1 - 1.5 * IQR)) | (taskdata > (Q3 + 1.5 * IQR))
            index_outlier.extend(taskdata.index[outlier])
        # Drop outliers and drop NaNs
        print('Removing ouliers %s' % index_outlier)
        nooutlier_df = dataframe.drop(index_outlier)
        nooutlier_df = nooutlier_df.dropna(how='all')

        return nooutlier_df, index_outlier
