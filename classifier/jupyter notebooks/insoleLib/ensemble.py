import os
import pandas as pd
import numpy as np
from scipy.stats import zscore
from collections import namedtuple

from columns import DataColumns
from dataHandler import DataHandler
from knnClassifiers import KnnClassifiers
from treeClassifiers import TreeClassifiers
from svmClassifiers import SvmClassifiers

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt
import seaborn as sn

class Ensemble:
    """
    Ensemble learning class.
    Uses multiple different classifiers and randomized datasets to classify the input data.

    """

    #some column combinations, can be used for filtering
    values_cols = DataColumns.getValuesCols()
    features_cols = DataColumns.getBasicFeaturesCols()
    force_cols = DataColumns.getForceCols()
    startT_cols = DataColumns.getStartTimeCols()
    maxT_cols = DataColumns.getMaxTimeCols()
    endT_cols = DataColumns.getEndTimeCols()
    phases_cols = DataColumns.getPhaseCols()
    phasesT_cols = DataColumns.getPhaseTimeCols()
    phasesF_cols = DataColumns.getPhaseForceCols()
    stepL_cols = DataColumns.getStepTimeCols()


    def testBagging(data):
        """
        Testing bagging results

        Arguments:
            data {array} -- Data
        """
        x_cols = DataColumns.getSelectedCols2()
        y_cols = ["label"]

        x = data.loc[:, x_cols]
        y = data.loc[:, y_cols]

        loo = LeaveOneOut()
        loo.get_n_splits(data)
        n=loo.split(data)

        parameters = namedtuple("parameters", ["class_weight", "criterion", "max_depth", "max_features", #namedTuple for tree parameters
            "max_leaf_nodes", "min_samples_leaf", "min_samples_split", "min_weight_fraction_leaf",
            "presort", "random_state", "splitter"])

        accuracy_a = []
        real_label = []
        pred_label = []
        
        #KNN parameters
        k = KnnClassifiers.findBestK(data, x_cols, y_cols) #best k

        for train_index, test_index in n: #Each row is test data once
            xtrain, xtest = x.iloc[train_index], x.iloc[test_index]
            ytrain, ytest = y.iloc[train_index], y.iloc[test_index]
            
            train = xtrain
            train.loc[:,"label"] = ytrain

            #generating datasets where data has been suffled and last random x rows has been dropped
            dataset_amount = 10
            drop_amount = 40
            datasets = DataHandler.genRandomDatasets(train, dataset_amount, drop_amount)

            

            #knn0_pred_label = KnnClassifiers.getKnnPredictions(train, xtest, k)
            knn1_pred_label = KnnClassifiers.getKnnPredictions(datasets[0], xtest, k)
            knn2_pred_label = KnnClassifiers.getKnnPredictions(datasets[1], xtest, k)
            knn3_pred_label = KnnClassifiers.getKnnPredictions(datasets[2], xtest, k)
            knn4_pred_label = KnnClassifiers.getKnnPredictions(datasets[3], xtest, k)
            knn5_pred_label = KnnClassifiers.getKnnPredictions(datasets[4], xtest, k)
            knn6_pred_label = KnnClassifiers.getKnnPredictions(datasets[5], xtest, k)
            knn7_pred_label = KnnClassifiers.getKnnPredictions(datasets[6], xtest, k)
            knn8_pred_label = KnnClassifiers.getKnnPredictions(datasets[7], xtest, k)
            knn9_pred_label = KnnClassifiers.getKnnPredictions(datasets[8], xtest, k)
            knn10_pred_label = KnnClassifiers.getKnnPredictions(datasets[9], xtest, k)

            #Tree1 parameters
            params1 = parameters(class_weight=None, criterion='gini', max_depth=5, max_features=None,
                max_leaf_nodes=None, min_samples_leaf=5, min_samples_split=2, min_weight_fraction_leaf=0.0,
                presort=False, random_state=123, splitter='best')

            #tree1_0_pred_label = TreeClassifiers.getTreePredictions(train, xtest, params1, x_cols, y_cols)
            tree1_1_pred_label = TreeClassifiers.getTreePredictions(datasets[0], xtest, params1, x_cols, y_cols)
            tree1_2_pred_label = TreeClassifiers.getTreePredictions(datasets[1], xtest, params1, x_cols, y_cols)
            tree1_3_pred_label = TreeClassifiers.getTreePredictions(datasets[2], xtest, params1, x_cols, y_cols)
            tree1_4_pred_label = TreeClassifiers.getTreePredictions(datasets[3], xtest, params1, x_cols, y_cols)
            tree1_5_pred_label = TreeClassifiers.getTreePredictions(datasets[4], xtest, params1, x_cols, y_cols)
            tree1_6_pred_label = TreeClassifiers.getTreePredictions(datasets[5], xtest, params1, x_cols, y_cols)
            tree1_7_pred_label = TreeClassifiers.getTreePredictions(datasets[6], xtest, params1, x_cols, y_cols)
            tree1_8_pred_label = TreeClassifiers.getTreePredictions(datasets[7], xtest, params1, x_cols, y_cols)
            tree1_9_pred_label = TreeClassifiers.getTreePredictions(datasets[8], xtest, params1, x_cols, y_cols)
            tree1_10_pred_label = TreeClassifiers.getTreePredictions(datasets[9], xtest, params1, x_cols, y_cols)

            #Tree2 parameters
            params2 = parameters(class_weight=None, criterion='entropy', max_depth=5, max_features=None,
                max_leaf_nodes=None, min_samples_leaf=5, min_samples_split=2, min_weight_fraction_leaf=0.0,
                presort=False, random_state=123, splitter='best')

            #tree2_0_pred_label = TreeClassifiers.getTreePredictions(train, xtest, params2, x_cols, y_cols)
            tree2_1_pred_label = TreeClassifiers.getTreePredictions(datasets[0], xtest, params2, x_cols, y_cols)
            tree2_2_pred_label = TreeClassifiers.getTreePredictions(datasets[1], xtest, params2, x_cols, y_cols)
            tree2_3_pred_label = TreeClassifiers.getTreePredictions(datasets[2], xtest, params2, x_cols, y_cols)
            tree2_4_pred_label = TreeClassifiers.getTreePredictions(datasets[3], xtest, params2, x_cols, y_cols)
            tree2_5_pred_label = TreeClassifiers.getTreePredictions(datasets[4], xtest, params2, x_cols, y_cols)
            tree2_6_pred_label = TreeClassifiers.getTreePredictions(datasets[5], xtest, params2, x_cols, y_cols)
            tree2_7_pred_label = TreeClassifiers.getTreePredictions(datasets[6], xtest, params2, x_cols, y_cols)
            tree2_8_pred_label = TreeClassifiers.getTreePredictions(datasets[7], xtest, params2, x_cols, y_cols)
            tree2_9_pred_label = TreeClassifiers.getTreePredictions(datasets[8], xtest, params2, x_cols, y_cols)
            tree2_10_pred_label = TreeClassifiers.getTreePredictions(datasets[9], xtest, params2, x_cols, y_cols)

            #xgboost0_pred_label = TreeClassifiers.getXGBoostPredictions(train, xtest, x_cols, y_cols)
            xgboost1_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[0], xtest, x_cols, y_cols)
            xgboost2_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[1], xtest, x_cols, y_cols)
            xgboost3_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[2], xtest, x_cols, y_cols)
            xgboost4_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[3], xtest, x_cols, y_cols)
            xgboost5_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[4], xtest, x_cols, y_cols)
            xgboost6_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[5], xtest, x_cols, y_cols)
            xgboost7_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[6], xtest, x_cols, y_cols)
            xgboost8_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[7], xtest, x_cols, y_cols)
            xgboost9_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[8], xtest, x_cols, y_cols)
            xgboost10_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[9], xtest, x_cols, y_cols)

            #SVM parameters
            kern = "rbf"

            #svm0_pred_label = SvmClassifiers.svmGetPredictions(train, xtest, kern)
            svm1_pred_label = SvmClassifiers.svmGetPredictions(datasets[0], xtest, kern)
            svm2_pred_label = SvmClassifiers.svmGetPredictions(datasets[1], xtest, kern)
            svm3_pred_label = SvmClassifiers.svmGetPredictions(datasets[2], xtest, kern)
            svm4_pred_label = SvmClassifiers.svmGetPredictions(datasets[3], xtest, kern)
            svm5_pred_label = SvmClassifiers.svmGetPredictions(datasets[4], xtest, kern)
            svm6_pred_label = SvmClassifiers.svmGetPredictions(datasets[5], xtest, kern)
            svm7_pred_label = SvmClassifiers.svmGetPredictions(datasets[6], xtest, kern)
            svm8_pred_label = SvmClassifiers.svmGetPredictions(datasets[7], xtest, kern)
            svm9_pred_label = SvmClassifiers.svmGetPredictions(datasets[8], xtest, kern)
            svm10_pred_label = SvmClassifiers.svmGetPredictions(datasets[9], xtest, kern)

            #tensorflow



            predictionSets = [
                knn1_pred_label, knn2_pred_label, knn3_pred_label, knn4_pred_label, knn5_pred_label,
                knn6_pred_label, knn7_pred_label, knn8_pred_label, knn9_pred_label, knn10_pred_label,
                tree1_1_pred_label, tree1_2_pred_label, tree1_3_pred_label, tree1_4_pred_label, tree1_5_pred_label,
                tree1_6_pred_label, tree1_7_pred_label, tree1_8_pred_label, tree1_9_pred_label, tree1_10_pred_label,
                tree2_1_pred_label, tree2_2_pred_label, tree2_3_pred_label, tree2_4_pred_label, tree2_5_pred_label,
                tree2_6_pred_label, tree2_7_pred_label, tree2_8_pred_label, tree2_9_pred_label, tree2_10_pred_label,
                xgboost1_pred_label, xgboost2_pred_label, xgboost3_pred_label, xgboost4_pred_label, xgboost5_pred_label,
                xgboost6_pred_label, xgboost7_pred_label, xgboost8_pred_label, xgboost9_pred_label, xgboost10_pred_label,
                svm1_pred_label, svm2_pred_label, svm3_pred_label, svm4_pred_label, svm5_pred_label,
                svm6_pred_label, svm7_pred_label, svm8_pred_label, svm9_pred_label, svm10_pred_label
            ]

            results = []

            #Voting process
            for index in range(0, len(xtest)): #going through every index one by one
                fall_votes = 0
                normal_votes = 0

                for set in predictionSets: #counting votes
                    if(set[index] == "Fall"):
                        fall_votes = fall_votes + 1
                    elif(set[index] == "Normal"):
                        normal_votes = normal_votes + 1
                    else:
                        print("Unknown label")

                if(fall_votes >= normal_votes): #appending result
                    results.append("Fall")
                else:
                    results.append("Normal")
            
            pred_label.append(results)
            real_label.append(ytest.values)

            acc = accuracy_score(ytest, results)
            accuracy_a.append(acc)

        avg_acc = np.mean(accuracy_a)
        
        pred_label_df = pd.DataFrame(columns=["label"])
        real_label_df = pd.DataFrame(columns=["label"])
        
        #Forming the dataframes
        for row in range(0,len(pred_label)):
            label_str = pred_label[row][0]
            pred_label_df.loc[row] = label_str
        
        for row in range(0,len(real_label)):
            label_str = real_label[row][0][0]
            real_label_df.loc[row] = label_str
        
        if(True):
            
            cm = confusion_matrix(real_label_df, pred_label_df)
            cm_df = pd.DataFrame(cm, ["Fall", "Normal"], ["Fall", "Normal"])
            
            sn.set(font_scale=1.5)
            sn.heatmap(cm_df, annot=True, annot_kws={"size": 32}, fmt='d')
            plt.savefig("../figs/bagging_heatmap.png", facecolor="w", bbox_inches="tight")
            plt.show()
        
        #Checking accuracy
        print("Ensemble average accuracy: ", round(avg_acc, 2)) #2 decimals
        
        #More detailed report
        print(classification_report(real_label_df, pred_label_df))

        return(avg_acc, real_label_df, pred_label_df)
    
    def testBaggingLearning(data, x_cols, y_cols, times, plots=False, orig_acc=0, orig_auc=0, file_name_prefix="Bagging"):
        """
        Suffling labels and fitting data many times.
        Verifying that classifier learns something from the data

        Arguments:
            data {array} -- Data
            x_cols {array} -- x columns
            y_cols {array} -- y columns
            times {int} -- how many times random accuracy is tested

        Keyword Arguments:
            plots {bool} -- Used for plotting (default: {False})
        """
        
        accs = []
        aucs = []
        
        for run in range(1,times+1):
            print("Run: ", run, " out of ", times)
            
            suffled_data = data
            suffled_data["label"] = np.random.permutation(suffled_data["label"].values)

            avg_acc, real_label, pred_label = Ensemble.testBagging(suffled_data)
            
            accs.append(avg_acc)
            
            pred_label_df = pred_label
            real_label_df = real_label

            pred_label_df = pred_label_df.replace("Normal", 0)
            pred_label_df = pred_label_df.replace("Fall", 1)

            real_label_df = real_label_df.replace("Normal", 0)
            real_label_df = real_label_df.replace("Fall", 1)
            
            avg_auc = roc_auc_score(real_label_df, pred_label_df)
            
            aucs.append(avg_auc)
        
        if(plots):
            
            plt.hist(accs, edgecolor='white', bins=14)
            plt.title("Permutations accuracy histogram")
            plt.xlabel("Accuracy")
            plt.ylabel("Count")
            plt.vlines(orig_acc, 0, (times/4.25), linestyle='-')
            plt.legend(["Classifier accuracy"])
            plt.savefig("../figs/"+file_name_prefix+"_acc_hist.png", facecolor="w", bbox_inches="tight")
            plt.show()

            plt.boxplot(accs)
            plt.title("Permutations accuracy boxplot")
            plt.savefig("../figs/"+file_name_prefix+"_acc_bplot.png", facecolor="w", bbox_inches="tight")
            plt.show()
            
            plt.hist(aucs, edgecolor='white', bins=14)
            plt.title("Permutations AUC histogram")
            plt.xlabel("AUC")
            plt.ylabel("Count")
            plt.vlines(orig_auc, 0, (times/4.25), linestyle='-')
            plt.legend(["Classifier AUC"])
            plt.savefig("../figs/"+file_name_prefix+"_auc_hist.png", facecolor="w", bbox_inches="tight")
            plt.show()

            plt.boxplot(aucs)
            plt.title("Permutations AUC boxplot")
            plt.savefig("../figs/"+file_name_prefix+"_auc_bplot.png", facecolor="w", bbox_inches="tight")
            plt.show()
        
        avg_acc = np.mean(accs)
        print("Average accuracy:", avg_acc)
        
        avg_auc = np.mean(aucs)
        print("Average AUC:", avg_auc)
        
        return(accs, aucs)


    def getBaggingPredictions(train, unlabeled):
        """
        Most popular label from multiple different classifiers wins. All votes are equal.
        Training with randomly drawn sets from training set

        Arguments:
            train {[type]} -- Training data for the classifiers
            unlabeled {[type]} -- Unlabeled data which needs label predictions
        """
        x_cols = DataColumns.getSelectedCols2()
        y_cols = ["label"]
        
        xtest = unlabeled

        parameters = namedtuple("parameters", ["class_weight", "criterion", "max_depth", "max_features", #namedTuple for tree parameters
            "max_leaf_nodes", "min_samples_leaf", "min_samples_split", "min_weight_fraction_leaf",
            "presort", "random_state", "splitter"])


        #generating datasets where data has been suffled and last random x rows has been dropped
        dataset_amount = 10
        drop_amount = 40
        datasets = DataHandler.genRandomDatasets(train, dataset_amount, drop_amount)

        #KNN parameters
        k = KnnClassifiers.findBestK(train, x_cols, y_cols) #best k

        #knn0_pred_label = KnnClassifiers.getKnnPredictions(train, xtest, k)
        knn1_pred_label = KnnClassifiers.getKnnPredictions(datasets[0], xtest, k)
        knn2_pred_label = KnnClassifiers.getKnnPredictions(datasets[1], xtest, k)
        knn3_pred_label = KnnClassifiers.getKnnPredictions(datasets[2], xtest, k)
        knn4_pred_label = KnnClassifiers.getKnnPredictions(datasets[3], xtest, k)
        knn5_pred_label = KnnClassifiers.getKnnPredictions(datasets[4], xtest, k)
        knn6_pred_label = KnnClassifiers.getKnnPredictions(datasets[5], xtest, k)
        knn7_pred_label = KnnClassifiers.getKnnPredictions(datasets[6], xtest, k)
        knn8_pred_label = KnnClassifiers.getKnnPredictions(datasets[7], xtest, k)
        knn9_pred_label = KnnClassifiers.getKnnPredictions(datasets[8], xtest, k)
        knn10_pred_label = KnnClassifiers.getKnnPredictions(datasets[9], xtest, k)

        #Tree1 parameters
        params1 = parameters(class_weight=None, criterion='gini', max_depth=5, max_features=None,
            max_leaf_nodes=None, min_samples_leaf=5, min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=123, splitter='best')

        #tree1_0_pred_label = TreeClassifiers.getTreePredictions(train, xtest, params1, x_cols, y_cols)
        tree1_1_pred_label = TreeClassifiers.getTreePredictions(datasets[0], xtest, params1, x_cols, y_cols)
        tree1_2_pred_label = TreeClassifiers.getTreePredictions(datasets[1], xtest, params1, x_cols, y_cols)
        tree1_3_pred_label = TreeClassifiers.getTreePredictions(datasets[2], xtest, params1, x_cols, y_cols)
        tree1_4_pred_label = TreeClassifiers.getTreePredictions(datasets[3], xtest, params1, x_cols, y_cols)
        tree1_5_pred_label = TreeClassifiers.getTreePredictions(datasets[4], xtest, params1, x_cols, y_cols)
        tree1_6_pred_label = TreeClassifiers.getTreePredictions(datasets[5], xtest, params1, x_cols, y_cols)
        tree1_7_pred_label = TreeClassifiers.getTreePredictions(datasets[6], xtest, params1, x_cols, y_cols)
        tree1_8_pred_label = TreeClassifiers.getTreePredictions(datasets[7], xtest, params1, x_cols, y_cols)
        tree1_9_pred_label = TreeClassifiers.getTreePredictions(datasets[8], xtest, params1, x_cols, y_cols)
        tree1_10_pred_label = TreeClassifiers.getTreePredictions(datasets[9], xtest, params1, x_cols, y_cols)

        #Tree2 parameters
        params2 = parameters(class_weight=None, criterion='entropy', max_depth=5, max_features=None,
            max_leaf_nodes=None, min_samples_leaf=5, min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=123, splitter='best')

        #tree2_0_pred_label = TreeClassifiers.getTreePredictions(train, xtest, params2, x_cols, y_cols)
        tree2_1_pred_label = TreeClassifiers.getTreePredictions(datasets[0], xtest, params2, x_cols, y_cols)
        tree2_2_pred_label = TreeClassifiers.getTreePredictions(datasets[1], xtest, params2, x_cols, y_cols)
        tree2_3_pred_label = TreeClassifiers.getTreePredictions(datasets[2], xtest, params2, x_cols, y_cols)
        tree2_4_pred_label = TreeClassifiers.getTreePredictions(datasets[3], xtest, params2, x_cols, y_cols)
        tree2_5_pred_label = TreeClassifiers.getTreePredictions(datasets[4], xtest, params2, x_cols, y_cols)
        tree2_6_pred_label = TreeClassifiers.getTreePredictions(datasets[5], xtest, params2, x_cols, y_cols)
        tree2_7_pred_label = TreeClassifiers.getTreePredictions(datasets[6], xtest, params2, x_cols, y_cols)
        tree2_8_pred_label = TreeClassifiers.getTreePredictions(datasets[7], xtest, params2, x_cols, y_cols)
        tree2_9_pred_label = TreeClassifiers.getTreePredictions(datasets[8], xtest, params2, x_cols, y_cols)
        tree2_10_pred_label = TreeClassifiers.getTreePredictions(datasets[9], xtest, params2, x_cols, y_cols)

        #xgboost0_pred_label = TreeClassifiers.getXGBoostPredictions(train, xtest, x_cols, y_cols)
        xgboost1_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[0], xtest, x_cols, y_cols)
        xgboost2_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[1], xtest, x_cols, y_cols)
        xgboost3_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[2], xtest, x_cols, y_cols)
        xgboost4_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[3], xtest, x_cols, y_cols)
        xgboost5_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[4], xtest, x_cols, y_cols)
        xgboost6_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[5], xtest, x_cols, y_cols)
        xgboost7_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[6], xtest, x_cols, y_cols)
        xgboost8_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[7], xtest, x_cols, y_cols)
        xgboost9_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[8], xtest, x_cols, y_cols)
        xgboost10_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[9], xtest, x_cols, y_cols)

        #SVM parameters
        kern = "rbf"

        #svm0_pred_label = SvmClassifiers.svmGetPredictions(train, xtest, kern)
        svm1_pred_label = SvmClassifiers.svmGetPredictions(datasets[0], xtest, kern)
        svm2_pred_label = SvmClassifiers.svmGetPredictions(datasets[1], xtest, kern)
        svm3_pred_label = SvmClassifiers.svmGetPredictions(datasets[2], xtest, kern)
        svm4_pred_label = SvmClassifiers.svmGetPredictions(datasets[3], xtest, kern)
        svm5_pred_label = SvmClassifiers.svmGetPredictions(datasets[4], xtest, kern)
        svm6_pred_label = SvmClassifiers.svmGetPredictions(datasets[5], xtest, kern)
        svm7_pred_label = SvmClassifiers.svmGetPredictions(datasets[6], xtest, kern)
        svm8_pred_label = SvmClassifiers.svmGetPredictions(datasets[7], xtest, kern)
        svm9_pred_label = SvmClassifiers.svmGetPredictions(datasets[8], xtest, kern)
        svm10_pred_label = SvmClassifiers.svmGetPredictions(datasets[9], xtest, kern)

        #tensorflow



        predictionSets = [
                knn1_pred_label, knn2_pred_label, knn3_pred_label, knn4_pred_label, knn5_pred_label,
                knn6_pred_label, knn7_pred_label, knn8_pred_label, knn9_pred_label, knn10_pred_label,
                tree1_1_pred_label, tree1_2_pred_label, tree1_3_pred_label, tree1_4_pred_label, tree1_5_pred_label,
                tree1_6_pred_label, tree1_7_pred_label, tree1_8_pred_label, tree1_9_pred_label, tree1_10_pred_label,
                tree2_1_pred_label, tree2_2_pred_label, tree2_3_pred_label, tree2_4_pred_label, tree2_5_pred_label,
                tree2_6_pred_label, tree2_7_pred_label, tree2_8_pred_label, tree2_9_pred_label, tree2_10_pred_label,
                xgboost1_pred_label, xgboost2_pred_label, xgboost3_pred_label, xgboost4_pred_label, xgboost5_pred_label,
                xgboost6_pred_label, xgboost7_pred_label, xgboost8_pred_label, xgboost9_pred_label, xgboost10_pred_label,
                svm1_pred_label, svm2_pred_label, svm3_pred_label, svm4_pred_label, svm5_pred_label,
                svm6_pred_label, svm7_pred_label, svm8_pred_label, svm9_pred_label, svm10_pred_label
            ]

        results = []

        #Voting process
        for index in range(0, len(unlabeled)): #going through every index one by one
            fall_votes = 0
            normal_votes = 0

            for set in predictionSets: #counting votes
                if(set[index] == "Fall"):
                    fall_votes = fall_votes + 1
                elif(set[index] == "Normal"):
                    normal_votes = normal_votes + 1
                else:
                    print("Unknown label")

            if(fall_votes >= normal_votes): #appending result
                results.append("Fall")
            else:
                results.append("Normal")

        return(results)

    #majority vote dataframe version
    def majorityVoteDF(labels):
        print("majorityVote input", labels)

        cols = labels.columns.values

        vote_df = pd.DataFrame(columns=["label"])

        for row in labels: #going through all rows one by one
            normal_counter = 0 #counters for votes
            fall_counter = 0

            for col in cols: #and all columns one by one
                value = labels.loc[row,col].values
                if(value == "Normal"):
                    normal_counter = normal_counter + 1
                elif(value == "Fall"):
                    fall_counter = fall_counter + 1
                else:
                    print("Unkown label")

            if(normal_counter > fall_counter):
                vote_df.loc[row, ["label"]] = "Normal"
            elif(fall_counter >= normal_counter): #if tie just labeling as fall
                vote_df.loc[row, ["label"]] = "Fall"


        print("vote_df result", vote_df)

        return(vote_df)


    def testBoosting(data):
        """
        Testing boosting results

        Arguments:
            data {array} -- Data
        """

        x_cols = DataColumns.getSelectedCols2()
        y_cols = ["label"]

        x = data.loc[:, x_cols]
        y = data.loc[:, y_cols]

        loo = LeaveOneOut()
        loo.get_n_splits(data)
        n=loo.split(data)

        parameters = namedtuple("parameters", ["class_weight", "criterion", "max_depth", "max_features", #namedTuple for tree parameters
            "max_leaf_nodes", "min_samples_leaf", "min_samples_split", "min_weight_fraction_leaf",
            "presort", "random_state", "splitter"])

        accuracy_a = []
        real_label = []
        pred_label = []
        
        #KNN parameters
        k = KnnClassifiers.findBestK(data, x_cols, y_cols) #best k

        for train_index, test_index in n: #Each row is test data once
            xtrain, xtest = x.iloc[train_index], x.iloc[test_index]
            ytrain, ytest = y.iloc[train_index], y.iloc[test_index]
            
            train = xtrain
            train.loc[:,"label"] = ytrain

            #generating datasets where data has been suffled and last random x rows has been dropped
            dataset_amount = 10
            drop_amount = 40
            datasets = DataHandler.genRandomDatasets(train, dataset_amount, drop_amount)

            
            #knn0_pred_label = KnnClassifiers.getKnnPredictions(train, xtest, k)
            knn1_pred_label = KnnClassifiers.getKnnPredictions(datasets[0], xtest, k)
            knn2_pred_label = KnnClassifiers.getKnnPredictions(datasets[1], xtest, k)
            knn3_pred_label = KnnClassifiers.getKnnPredictions(datasets[2], xtest, k)
            knn4_pred_label = KnnClassifiers.getKnnPredictions(datasets[3], xtest, k)
            knn5_pred_label = KnnClassifiers.getKnnPredictions(datasets[4], xtest, k)
            knn6_pred_label = KnnClassifiers.getKnnPredictions(datasets[5], xtest, k)
            knn7_pred_label = KnnClassifiers.getKnnPredictions(datasets[6], xtest, k)
            knn8_pred_label = KnnClassifiers.getKnnPredictions(datasets[7], xtest, k)
            knn9_pred_label = KnnClassifiers.getKnnPredictions(datasets[8], xtest, k)
            knn10_pred_label = KnnClassifiers.getKnnPredictions(datasets[9], xtest, k)

            #Tree1 parameters
            params1 = parameters(class_weight=None, criterion='gini', max_depth=5, max_features=None,
                max_leaf_nodes=None, min_samples_leaf=5, min_samples_split=2, min_weight_fraction_leaf=0.0,
                presort=False, random_state=123, splitter='best')

            #tree1_0_pred_label = TreeClassifiers.getTreePredictions(train, xtest, params1, x_cols, y_cols)
            tree1_1_pred_label = TreeClassifiers.getTreePredictions(datasets[0], xtest, params1, x_cols, y_cols)
            tree1_2_pred_label = TreeClassifiers.getTreePredictions(datasets[1], xtest, params1, x_cols, y_cols)
            tree1_3_pred_label = TreeClassifiers.getTreePredictions(datasets[2], xtest, params1, x_cols, y_cols)
            tree1_4_pred_label = TreeClassifiers.getTreePredictions(datasets[3], xtest, params1, x_cols, y_cols)
            tree1_5_pred_label = TreeClassifiers.getTreePredictions(datasets[4], xtest, params1, x_cols, y_cols)
            tree1_6_pred_label = TreeClassifiers.getTreePredictions(datasets[5], xtest, params1, x_cols, y_cols)
            tree1_7_pred_label = TreeClassifiers.getTreePredictions(datasets[6], xtest, params1, x_cols, y_cols)
            tree1_8_pred_label = TreeClassifiers.getTreePredictions(datasets[7], xtest, params1, x_cols, y_cols)
            tree1_9_pred_label = TreeClassifiers.getTreePredictions(datasets[8], xtest, params1, x_cols, y_cols)
            tree1_10_pred_label = TreeClassifiers.getTreePredictions(datasets[9], xtest, params1, x_cols, y_cols)

            #Tree2 parameters
            params2 = parameters(class_weight=None, criterion='entropy', max_depth=5, max_features=None,
                max_leaf_nodes=None, min_samples_leaf=5, min_samples_split=2, min_weight_fraction_leaf=0.0,
                presort=False, random_state=123, splitter='best')

            #tree2_0_pred_label = TreeClassifiers.getTreePredictions(train, xtest, params2, x_cols, y_cols)
            tree2_1_pred_label = TreeClassifiers.getTreePredictions(datasets[0], xtest, params2, x_cols, y_cols)
            tree2_2_pred_label = TreeClassifiers.getTreePredictions(datasets[1], xtest, params2, x_cols, y_cols)
            tree2_3_pred_label = TreeClassifiers.getTreePredictions(datasets[2], xtest, params2, x_cols, y_cols)
            tree2_4_pred_label = TreeClassifiers.getTreePredictions(datasets[3], xtest, params2, x_cols, y_cols)
            tree2_5_pred_label = TreeClassifiers.getTreePredictions(datasets[4], xtest, params2, x_cols, y_cols)
            tree2_6_pred_label = TreeClassifiers.getTreePredictions(datasets[5], xtest, params2, x_cols, y_cols)
            tree2_7_pred_label = TreeClassifiers.getTreePredictions(datasets[6], xtest, params2, x_cols, y_cols)
            tree2_8_pred_label = TreeClassifiers.getTreePredictions(datasets[7], xtest, params2, x_cols, y_cols)
            tree2_9_pred_label = TreeClassifiers.getTreePredictions(datasets[8], xtest, params2, x_cols, y_cols)
            tree2_10_pred_label = TreeClassifiers.getTreePredictions(datasets[9], xtest, params2, x_cols, y_cols)

            #xgboost0_pred_label = TreeClassifiers.getXGBoostPredictions(train, xtest, x_cols, y_cols)
            xgboost1_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[0], xtest, x_cols, y_cols)
            xgboost2_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[1], xtest, x_cols, y_cols)
            xgboost3_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[2], xtest, x_cols, y_cols)
            xgboost4_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[3], xtest, x_cols, y_cols)
            xgboost5_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[4], xtest, x_cols, y_cols)
            xgboost6_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[5], xtest, x_cols, y_cols)
            xgboost7_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[6], xtest, x_cols, y_cols)
            xgboost8_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[7], xtest, x_cols, y_cols)
            xgboost9_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[8], xtest, x_cols, y_cols)
            xgboost10_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[9], xtest, x_cols, y_cols)

            #SVM parameters
            kern = "rbf"

            #svm0_pred_label = SvmClassifiers.svmGetPredictions(train, xtest, kern)
            svm1_pred_label = SvmClassifiers.svmGetPredictions(datasets[0], xtest, kern)
            svm2_pred_label = SvmClassifiers.svmGetPredictions(datasets[1], xtest, kern)
            svm3_pred_label = SvmClassifiers.svmGetPredictions(datasets[2], xtest, kern)
            svm4_pred_label = SvmClassifiers.svmGetPredictions(datasets[3], xtest, kern)
            svm5_pred_label = SvmClassifiers.svmGetPredictions(datasets[4], xtest, kern)
            svm6_pred_label = SvmClassifiers.svmGetPredictions(datasets[5], xtest, kern)
            svm7_pred_label = SvmClassifiers.svmGetPredictions(datasets[6], xtest, kern)
            svm8_pred_label = SvmClassifiers.svmGetPredictions(datasets[7], xtest, kern)
            svm9_pred_label = SvmClassifiers.svmGetPredictions(datasets[8], xtest, kern)
            svm10_pred_label = SvmClassifiers.svmGetPredictions(datasets[9], xtest, kern)

            #tensorflow



            predictionSets = [
                knn1_pred_label, knn2_pred_label, knn3_pred_label, knn4_pred_label, knn5_pred_label,
                knn6_pred_label, knn7_pred_label, knn8_pred_label, knn9_pred_label, knn10_pred_label,
                tree1_1_pred_label, tree1_2_pred_label, tree1_3_pred_label, tree1_4_pred_label, tree1_5_pred_label,
                tree1_6_pred_label, tree1_7_pred_label, tree1_8_pred_label, tree1_9_pred_label, tree1_10_pred_label,
                tree2_1_pred_label, tree2_2_pred_label, tree2_3_pred_label, tree2_4_pred_label, tree2_5_pred_label,
                tree2_6_pred_label, tree2_7_pred_label, tree2_8_pred_label, tree2_9_pred_label, tree2_10_pred_label,
                xgboost1_pred_label, xgboost2_pred_label, xgboost3_pred_label, xgboost4_pred_label, xgboost5_pred_label,
                xgboost6_pred_label, xgboost7_pred_label, xgboost8_pred_label, xgboost9_pred_label, xgboost10_pred_label,
                svm1_pred_label, svm2_pred_label, svm3_pred_label, svm4_pred_label, svm5_pred_label,
                svm6_pred_label, svm7_pred_label, svm8_pred_label, svm9_pred_label, svm10_pred_label
                ]
        
            #prediction sources array (for weighting different classifiers)
            setTypes = [
                "knn", "knn", "knn", "knn", "knn",
                "knn", "knn", "knn", "knn", "knn",
                "tree1", "tree1", "tree1", "tree1", "tree1",
                "tree1", "tree1", "tree1", "tree1", "tree1",
                "tree2", "tree2", "tree2", "tree2", "tree2",
                "tree2", "tree2", "tree2", "tree2", "tree2",
                "xgboost", "xgboost", "xgboost", "xgboost", "xgboost",
                "xgboost", "xgboost", "xgboost", "xgboost", "xgboost",
                "svm", "svm", "svm", "svm", "svm",
                "svm", "svm", "svm", "svm", "svm"
                ]


            results = []

            #Weighted voting process
            for index in range(0, len(xtest)): #going through every index one by one
                fall_votes = 0
                normal_votes = 0

                for setIndex in range(0, len(predictionSets)): #counting votes
                    if(predictionSets[setIndex][index] == "Fall"):
                        if(setTypes[setIndex] == "knn"):
                            fall_votes = fall_votes + 1.1
                        elif(setTypes[setIndex] == "tree1"):
                            fall_votes = fall_votes + 1.1
                        elif(setTypes[setIndex] == "tree2"):
                            fall_votes = fall_votes + 1.1
                        elif(setTypes[setIndex] == "xgboost"):
                            fall_votes = fall_votes + 1.2
                        elif(setTypes[setIndex] == "svm"):
                            fall_votes = fall_votes + 1
                    elif(predictionSets[setIndex][index] == "Normal"):
                        if(setTypes[setIndex] == "knn"):
                            normal_votes = normal_votes + 1.1
                        elif(setTypes[setIndex] == "tree1"):
                            normal_votes = normal_votes + 1.1
                        elif(setTypes[setIndex] == "tree2"):
                            normal_votes = normal_votes + 1.1
                        elif(setTypes[setIndex] == "xgboost"):
                            normal_votes = normal_votes + 1.2
                        elif(setTypes[setIndex] == "svm"):
                            normal_votes = normal_votes + 1
                    else:
                        print("Unknown label")

                if(fall_votes >= normal_votes): #appending result
                    results.append("Fall")
                else:
                    results.append("Normal")
                    
                pred_label.append(results)
                real_label.append(ytest.values)

                acc = accuracy_score(ytest, results)
                accuracy_a.append(acc)

        avg_acc = np.mean(accuracy_a)
        
        pred_label_df = pd.DataFrame(columns=["label"])
        real_label_df = pd.DataFrame(columns=["label"])
        
        #Forming the dataframes
        for row in range(0,len(pred_label)):
            label_str = pred_label[row][0]
            pred_label_df.loc[row] = label_str
        
        for row in range(0,len(real_label)):
            label_str = real_label[row][0][0]
            real_label_df.loc[row] = label_str
        
        if(True):
            
            cm = confusion_matrix(real_label_df, pred_label_df)
            cm_df = pd.DataFrame(cm, ["Fall", "Normal"], ["Fall", "Normal"])
            
            sn.set(font_scale=1.5)
            sn.heatmap(cm_df, annot=True, annot_kws={"size": 32}, fmt='d')
            plt.savefig("../figs/boosting_heatmap.png", facecolor="w", bbox_inches="tight")
            plt.show()
        
        #Checking accuracy
        print("Ensemble average accuracy: ", round(avg_acc, 2)) #2 decimals
        
        #More detailed report
        print(classification_report(real_label_df, pred_label_df))

        return(avg_acc, real_label_df, pred_label_df)

        return(results)
    
    def testBoostingLearning(data, x_cols, y_cols, times, plots=False, orig_acc=0, orig_auc=0, file_name_prefix="Boosting"):
        """
        Suffling labels and fitting data many times.
        Verifying that classifier learns something from the data

        Arguments:
            data {array} -- Data
            x_cols {array} -- x columns
            y_cols {array} -- y columns
            times {int} -- how many times random accuracy is tested

        Keyword Arguments:
            plots {bool} -- Used for plotting (default: {False})
        """
        
        accs = []
        aucs = []
        
        for run in range(1,times+1):
            print("Run: ", run, " out of ", times)
            
            suffled_data = data
            suffled_data["label"] = np.random.permutation(suffled_data["label"].values)

            avg_acc, real_label, pred_label = Ensemble.testBoosting(suffled_data)
            
            accs.append(avg_acc)
            
            pred_label_df = pred_label
            real_label_df = real_label

            pred_label_df = pred_label_df.replace("Normal", 0)
            pred_label_df = pred_label_df.replace("Fall", 1)

            real_label_df = real_label_df.replace("Normal", 0)
            real_label_df = real_label_df.replace("Fall", 1)
            
            avg_auc = roc_auc_score(real_label_df, pred_label_df)
            
            aucs.append(avg_auc)
        
        if(plots):
            
            plt.hist(accs, edgecolor='white', bins=14)
            plt.title("Permutations accuracy histogram")
            plt.xlabel("Accuracy")
            plt.ylabel("Count")
            plt.vlines(orig_acc, 0, (times/4.25), linestyle='-')
            plt.legend(["Classifier accuracy"])
            plt.savefig("../figs/"+file_name_prefix+"_acc_hist.png", facecolor="w", bbox_inches="tight")
            plt.show()

            plt.boxplot(accs)
            plt.title("Permutations accuracy boxplot")
            plt.savefig("../figs/"+file_name_prefix+"_acc_bplot.png", facecolor="w", bbox_inches="tight")
            plt.show()
            
            plt.hist(aucs, edgecolor='white', bins=14)
            plt.title("Permutations AUC histogram")
            plt.xlabel("AUC")
            plt.ylabel("Count")
            plt.vlines(orig_auc, 0, (times/4.25), linestyle='-')
            plt.legend(["Classifier AUC"])
            plt.savefig("../figs/"+file_name_prefix+"_auc_hist.png", facecolor="w", bbox_inches="tight")
            plt.show()

            plt.boxplot(aucs)
            plt.title("Permutations AUC boxplot")
            plt.savefig("../figs/"+file_name_prefix+"_auc_bplot.png", facecolor="w", bbox_inches="tight")
            plt.show()
        
        avg_acc = np.mean(accs)
        print("Average accuracy:", avg_acc)
        
        avg_auc = np.mean(aucs)
        print("Average AUC:", avg_auc)
        
        return(accs, aucs)
    
    def testSkewedBoostingFall(data):
        """
        Testing boosting results (skewed towards falls)

        Arguments:
            data {array} -- Data
        """

        x_cols = DataColumns.getSelectedCols2()
        y_cols = ["label"]

        x = data.loc[:, x_cols]
        y = data.loc[:, y_cols]

        loo = LeaveOneOut()
        loo.get_n_splits(data)
        n=loo.split(data)

        parameters = namedtuple("parameters", ["class_weight", "criterion", "max_depth", "max_features", #namedTuple for tree parameters
            "max_leaf_nodes", "min_samples_leaf", "min_samples_split", "min_weight_fraction_leaf",
            "presort", "random_state", "splitter"])

        accuracy_a = []
        real_label = []
        pred_label = []
        
        #KNN parameters
        k = KnnClassifiers.findBestK(data, x_cols, y_cols) #best k

        for train_index, test_index in n: #Each row is test data once
            xtrain, xtest = x.iloc[train_index], x.iloc[test_index]
            ytrain, ytest = y.iloc[train_index], y.iloc[test_index]
            
            train = xtrain
            train.loc[:,"label"] = ytrain

            #generating datasets where data has been suffled and last random x rows has been dropped
            dataset_amount = 10
            drop_amount = 40
            datasets = DataHandler.genRandomDatasets(train, dataset_amount, drop_amount)

            
            #knn0_pred_label = KnnClassifiers.getKnnPredictions(train, xtest, k)
            knn1_pred_label = KnnClassifiers.getKnnPredictions(datasets[0], xtest, k)
            knn2_pred_label = KnnClassifiers.getKnnPredictions(datasets[1], xtest, k)
            knn3_pred_label = KnnClassifiers.getKnnPredictions(datasets[2], xtest, k)
            knn4_pred_label = KnnClassifiers.getKnnPredictions(datasets[3], xtest, k)
            knn5_pred_label = KnnClassifiers.getKnnPredictions(datasets[4], xtest, k)
            knn6_pred_label = KnnClassifiers.getKnnPredictions(datasets[5], xtest, k)
            knn7_pred_label = KnnClassifiers.getKnnPredictions(datasets[6], xtest, k)
            knn8_pred_label = KnnClassifiers.getKnnPredictions(datasets[7], xtest, k)
            knn9_pred_label = KnnClassifiers.getKnnPredictions(datasets[8], xtest, k)
            knn10_pred_label = KnnClassifiers.getKnnPredictions(datasets[9], xtest, k)

            #Tree1 parameters
            params1 = parameters(class_weight=None, criterion='gini', max_depth=5, max_features=None,
                max_leaf_nodes=None, min_samples_leaf=5, min_samples_split=2, min_weight_fraction_leaf=0.0,
                presort=False, random_state=123, splitter='best')

            #tree1_0_pred_label = TreeClassifiers.getTreePredictions(train, xtest, params1, x_cols, y_cols)
            tree1_1_pred_label = TreeClassifiers.getTreePredictions(datasets[0], xtest, params1, x_cols, y_cols)
            tree1_2_pred_label = TreeClassifiers.getTreePredictions(datasets[1], xtest, params1, x_cols, y_cols)
            tree1_3_pred_label = TreeClassifiers.getTreePredictions(datasets[2], xtest, params1, x_cols, y_cols)
            tree1_4_pred_label = TreeClassifiers.getTreePredictions(datasets[3], xtest, params1, x_cols, y_cols)
            tree1_5_pred_label = TreeClassifiers.getTreePredictions(datasets[4], xtest, params1, x_cols, y_cols)
            tree1_6_pred_label = TreeClassifiers.getTreePredictions(datasets[5], xtest, params1, x_cols, y_cols)
            tree1_7_pred_label = TreeClassifiers.getTreePredictions(datasets[6], xtest, params1, x_cols, y_cols)
            tree1_8_pred_label = TreeClassifiers.getTreePredictions(datasets[7], xtest, params1, x_cols, y_cols)
            tree1_9_pred_label = TreeClassifiers.getTreePredictions(datasets[8], xtest, params1, x_cols, y_cols)
            tree1_10_pred_label = TreeClassifiers.getTreePredictions(datasets[9], xtest, params1, x_cols, y_cols)

            #Tree2 parameters
            params2 = parameters(class_weight=None, criterion='entropy', max_depth=5, max_features=None,
                max_leaf_nodes=None, min_samples_leaf=5, min_samples_split=2, min_weight_fraction_leaf=0.0,
                presort=False, random_state=123, splitter='best')

            #tree2_0_pred_label = TreeClassifiers.getTreePredictions(train, xtest, params2, x_cols, y_cols)
            tree2_1_pred_label = TreeClassifiers.getTreePredictions(datasets[0], xtest, params2, x_cols, y_cols)
            tree2_2_pred_label = TreeClassifiers.getTreePredictions(datasets[1], xtest, params2, x_cols, y_cols)
            tree2_3_pred_label = TreeClassifiers.getTreePredictions(datasets[2], xtest, params2, x_cols, y_cols)
            tree2_4_pred_label = TreeClassifiers.getTreePredictions(datasets[3], xtest, params2, x_cols, y_cols)
            tree2_5_pred_label = TreeClassifiers.getTreePredictions(datasets[4], xtest, params2, x_cols, y_cols)
            tree2_6_pred_label = TreeClassifiers.getTreePredictions(datasets[5], xtest, params2, x_cols, y_cols)
            tree2_7_pred_label = TreeClassifiers.getTreePredictions(datasets[6], xtest, params2, x_cols, y_cols)
            tree2_8_pred_label = TreeClassifiers.getTreePredictions(datasets[7], xtest, params2, x_cols, y_cols)
            tree2_9_pred_label = TreeClassifiers.getTreePredictions(datasets[8], xtest, params2, x_cols, y_cols)
            tree2_10_pred_label = TreeClassifiers.getTreePredictions(datasets[9], xtest, params2, x_cols, y_cols)

            #xgboost0_pred_label = TreeClassifiers.getXGBoostPredictions(train, xtest, x_cols, y_cols)
            xgboost1_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[0], xtest, x_cols, y_cols)
            xgboost2_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[1], xtest, x_cols, y_cols)
            xgboost3_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[2], xtest, x_cols, y_cols)
            xgboost4_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[3], xtest, x_cols, y_cols)
            xgboost5_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[4], xtest, x_cols, y_cols)
            xgboost6_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[5], xtest, x_cols, y_cols)
            xgboost7_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[6], xtest, x_cols, y_cols)
            xgboost8_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[7], xtest, x_cols, y_cols)
            xgboost9_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[8], xtest, x_cols, y_cols)
            xgboost10_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[9], xtest, x_cols, y_cols)

            #SVM parameters
            kern = "rbf"

            #svm0_pred_label = SvmClassifiers.svmGetPredictions(train, xtest, kern)
            svm1_pred_label = SvmClassifiers.svmGetPredictions(datasets[0], xtest, kern)
            svm2_pred_label = SvmClassifiers.svmGetPredictions(datasets[1], xtest, kern)
            svm3_pred_label = SvmClassifiers.svmGetPredictions(datasets[2], xtest, kern)
            svm4_pred_label = SvmClassifiers.svmGetPredictions(datasets[3], xtest, kern)
            svm5_pred_label = SvmClassifiers.svmGetPredictions(datasets[4], xtest, kern)
            svm6_pred_label = SvmClassifiers.svmGetPredictions(datasets[5], xtest, kern)
            svm7_pred_label = SvmClassifiers.svmGetPredictions(datasets[6], xtest, kern)
            svm8_pred_label = SvmClassifiers.svmGetPredictions(datasets[7], xtest, kern)
            svm9_pred_label = SvmClassifiers.svmGetPredictions(datasets[8], xtest, kern)
            svm10_pred_label = SvmClassifiers.svmGetPredictions(datasets[9], xtest, kern)

            #tensorflow



            predictionSets = [
                knn1_pred_label, knn2_pred_label, knn3_pred_label, knn4_pred_label, knn5_pred_label,
                knn6_pred_label, knn7_pred_label, knn8_pred_label, knn9_pred_label, knn10_pred_label,
                tree1_1_pred_label, tree1_2_pred_label, tree1_3_pred_label, tree1_4_pred_label, tree1_5_pred_label,
                tree1_6_pred_label, tree1_7_pred_label, tree1_8_pred_label, tree1_9_pred_label, tree1_10_pred_label,
                tree2_1_pred_label, tree2_2_pred_label, tree2_3_pred_label, tree2_4_pred_label, tree2_5_pred_label,
                tree2_6_pred_label, tree2_7_pred_label, tree2_8_pred_label, tree2_9_pred_label, tree2_10_pred_label,
                xgboost1_pred_label, xgboost2_pred_label, xgboost3_pred_label, xgboost4_pred_label, xgboost5_pred_label,
                xgboost6_pred_label, xgboost7_pred_label, xgboost8_pred_label, xgboost9_pred_label, xgboost10_pred_label,
                svm1_pred_label, svm2_pred_label, svm3_pred_label, svm4_pred_label, svm5_pred_label,
                svm6_pred_label, svm7_pred_label, svm8_pred_label, svm9_pred_label, svm10_pred_label
                ]
        
            #prediction sources array (for weighting different classifiers)
            setTypes = [
                "knn", "knn", "knn", "knn", "knn",
                "knn", "knn", "knn", "knn", "knn",
                "tree1", "tree1", "tree1", "tree1", "tree1",
                "tree1", "tree1", "tree1", "tree1", "tree1",
                "tree2", "tree2", "tree2", "tree2", "tree2",
                "tree2", "tree2", "tree2", "tree2", "tree2",
                "xgboost", "xgboost", "xgboost", "xgboost", "xgboost",
                "xgboost", "xgboost", "xgboost", "xgboost", "xgboost",
                "svm", "svm", "svm", "svm", "svm",
                "svm", "svm", "svm", "svm", "svm"
                ]


            results = []

            #Weighted voting process
            for index in range(0, len(xtest)): #going through every index one by one
                fall_votes = 0
                normal_votes = 0

                for setIndex in range(0, len(predictionSets)): #counting votes
                    #print("setIndex: ", setIndex)
                    if(predictionSets[setIndex][index] == "Fall"):
                        if(setTypes[setIndex] == "knn"):
                            fall_votes = fall_votes + 6.5
                        elif(setTypes[setIndex] == "tree1"):
                            fall_votes = fall_votes + 5.0
                        elif(setTypes[setIndex] == "tree2"):
                            fall_votes = fall_votes + 1.5
                        elif(setTypes[setIndex] == "xgboost"):
                            fall_votes = fall_votes + 5.5
                        elif(setTypes[setIndex] == "svm"):
                            fall_votes = fall_votes + 1
                    elif(predictionSets[setIndex][index] == "Normal"):
                        if(setTypes[setIndex] == "knn"):
                            normal_votes = normal_votes + 2.5
                        elif(setTypes[setIndex] == "tree1"):
                            normal_votes = normal_votes + 2.5
                        elif(setTypes[setIndex] == "tree2"):
                            normal_votes = normal_votes + 1.5
                        elif(setTypes[setIndex] == "xgboost"):
                            normal_votes = normal_votes + 3.0
                        elif(setTypes[setIndex] == "svm"):
                            normal_votes = normal_votes + 1
                    else:
                        print("Unknown label")

                if(fall_votes >= normal_votes): #appending result
                    results.append("Fall")
                else:
                    results.append("Normal")
                    
                pred_label.append(results)
                real_label.append(ytest.values)

                acc = accuracy_score(ytest, results)
                accuracy_a.append(acc)

        avg_acc = np.mean(accuracy_a)
        
        pred_label_df = pd.DataFrame(columns=["label"])
        real_label_df = pd.DataFrame(columns=["label"])
        
        #Forming the dataframes
        for row in range(0,len(pred_label)):
            label_str = pred_label[row][0]
            pred_label_df.loc[row] = label_str
        
        for row in range(0,len(real_label)):
            label_str = real_label[row][0][0]
            real_label_df.loc[row] = label_str
        
        if(True):
            
            cm = confusion_matrix(real_label_df, pred_label_df)
            cm_df = pd.DataFrame(cm, ["Fall", "Normal"], ["Fall", "Normal"])
            
            sn.set(font_scale=1.5)
            sn.heatmap(cm_df, annot=True, annot_kws={"size": 32}, fmt='d')
            plt.savefig("../figs/boosting_skewed_to_fall_heatmap.png", facecolor="w", bbox_inches="tight")
            plt.show()
        
        #Checking accuracy
        print("Ensemble average accuracy: ", round(avg_acc, 2)) #2 decimals
        
        #More detailed report
        print(classification_report(real_label_df, pred_label_df))

        return(avg_acc, real_label_df, pred_label_df)

        return(results)
    
    def testFallBoostingLearning(data, x_cols, y_cols, times, plots=False, orig_acc=0, orig_auc=0, file_name_prefix="Fall_Boosting"):
        """
        Suffling labels and fitting data many times.
        Verifying that classifier learns something from the data

        Arguments:
            data {array} -- Data
            x_cols {array} -- x columns
            y_cols {array} -- y columns
            times {int} -- how many times random accuracy is tested

        Keyword Arguments:
            plots {bool} -- Used for plotting (default: {False})
        """
        
        accs = []
        aucs = []
        
        for run in range(1,times+1):
            print("Run: ", run, " out of ", times)
            
            suffled_data = data
            suffled_data["label"] = np.random.permutation(suffled_data["label"].values)

            avg_acc, real_label, pred_label = Ensemble.testSkewedBoostingFall(suffled_data)
            
            accs.append(avg_acc)
            
            pred_label_df = pred_label
            real_label_df = real_label

            pred_label_df = pred_label_df.replace("Normal", 0)
            pred_label_df = pred_label_df.replace("Fall", 1)

            real_label_df = real_label_df.replace("Normal", 0)
            real_label_df = real_label_df.replace("Fall", 1)
            
            avg_auc = roc_auc_score(real_label_df, pred_label_df)
            
            aucs.append(avg_auc)
        
        if(plots):
            
            plt.hist(accs, edgecolor='white', bins=14)
            plt.title("Permutations accuracy histogram")
            plt.xlabel("Accuracy")
            plt.ylabel("Count")
            plt.vlines(orig_acc, 0, (times/4.25), linestyle='-')
            plt.legend(["Classifier accuracy"])
            plt.savefig("../figs/"+file_name_prefix+"_acc_hist.png", facecolor="w", bbox_inches="tight")
            plt.show()

            plt.boxplot(accs)
            plt.title("Permutations accuracy boxplot")
            plt.savefig("../figs/"+file_name_prefix+"_acc_bplot.png", facecolor="w", bbox_inches="tight")
            plt.show()
            
            plt.hist(aucs, edgecolor='white', bins=14)
            plt.title("Permutations AUC histogram")
            plt.xlabel("AUC")
            plt.ylabel("Count")
            plt.vlines(orig_auc, 0, (times/4.25), linestyle='-')
            plt.legend(["Classifier AUC"])
            plt.savefig("../figs/"+file_name_prefix+"_auc_hist.png", facecolor="w", bbox_inches="tight")
            plt.show()

            plt.boxplot(aucs)
            plt.title("Permutations AUC boxplot")
            plt.savefig("../figs/"+file_name_prefix+"_auc_bplot.png", facecolor="w", bbox_inches="tight")
            plt.show()
        
        avg_acc = np.mean(accs)
        print("Average accuracy:", avg_acc)
        
        avg_auc = np.mean(aucs)
        print("Average AUC:", avg_auc)
        
        return(accs, aucs)
    
    
    def getBoostPredictions(train, unlabeled):
        """
        Different classifiers have different weights.
        Based on them, most popular label wins. Votes are not equal.
        Training with randomly drawn sets from training set

        Arguments:
            train {[type]} -- Training data for the classifiers
            unlabeled {[type]} -- Unlabeled data which needs label predictions
        """
        x_cols = DataColumns.getSelectedCols2()
        y_cols = ["label"]

        x = data.loc[:, x_cols]
        y = data.loc[:, y_cols]

        loo = LeaveOneOut()
        loo.get_n_splits(data)
        n=loo.split(data)

        parameters = namedtuple("parameters", ["class_weight", "criterion", "max_depth", "max_features", #namedTuple for tree parameters
            "max_leaf_nodes", "min_samples_leaf", "min_samples_split", "min_weight_fraction_leaf",
            "presort", "random_state", "splitter"])

        accuracy_a = []
        real_label = []
        pred_label = []
        
        #KNN parameters
        k = KnnClassifiers.findBestK(data, x_cols, y_cols) #best k

        for train_index, test_index in n: #Each row is test data once
            xtrain, xtest = x.iloc[train_index], x.iloc[test_index]
            ytrain, ytest = y.iloc[train_index], y.iloc[test_index]
            
            train = xtrain
            train.loc[:,"label"] = ytrain

            #generating datasets where data has been suffled and last random x rows has been dropped
            dataset_amount = 10
            drop_amount = 40
            datasets = DataHandler.genRandomDatasets(train, dataset_amount, drop_amount)

            

            #knn0_pred_label = KnnClassifiers.getKnnPredictions(train, xtest, k)
            knn1_pred_label = KnnClassifiers.getKnnPredictions(datasets[0], xtest, k)
            knn2_pred_label = KnnClassifiers.getKnnPredictions(datasets[1], xtest, k)
            knn3_pred_label = KnnClassifiers.getKnnPredictions(datasets[2], xtest, k)
            knn4_pred_label = KnnClassifiers.getKnnPredictions(datasets[3], xtest, k)
            knn5_pred_label = KnnClassifiers.getKnnPredictions(datasets[4], xtest, k)
            knn6_pred_label = KnnClassifiers.getKnnPredictions(datasets[5], xtest, k)
            knn7_pred_label = KnnClassifiers.getKnnPredictions(datasets[6], xtest, k)
            knn8_pred_label = KnnClassifiers.getKnnPredictions(datasets[7], xtest, k)
            knn9_pred_label = KnnClassifiers.getKnnPredictions(datasets[8], xtest, k)
            knn10_pred_label = KnnClassifiers.getKnnPredictions(datasets[9], xtest, k)

            #Tree1 parameters
            params1 = parameters(class_weight=None, criterion='gini', max_depth=5, max_features=None,
                max_leaf_nodes=None, min_samples_leaf=5, min_samples_split=2, min_weight_fraction_leaf=0.0,
                presort=False, random_state=123, splitter='best')

            #tree1_0_pred_label = TreeClassifiers.getTreePredictions(train, xtest, params1, x_cols, y_cols)
            tree1_1_pred_label = TreeClassifiers.getTreePredictions(datasets[0], xtest, params1, x_cols, y_cols)
            tree1_2_pred_label = TreeClassifiers.getTreePredictions(datasets[1], xtest, params1, x_cols, y_cols)
            tree1_3_pred_label = TreeClassifiers.getTreePredictions(datasets[2], xtest, params1, x_cols, y_cols)
            tree1_4_pred_label = TreeClassifiers.getTreePredictions(datasets[3], xtest, params1, x_cols, y_cols)
            tree1_5_pred_label = TreeClassifiers.getTreePredictions(datasets[4], xtest, params1, x_cols, y_cols)
            tree1_6_pred_label = TreeClassifiers.getTreePredictions(datasets[5], xtest, params1, x_cols, y_cols)
            tree1_7_pred_label = TreeClassifiers.getTreePredictions(datasets[6], xtest, params1, x_cols, y_cols)
            tree1_8_pred_label = TreeClassifiers.getTreePredictions(datasets[7], xtest, params1, x_cols, y_cols)
            tree1_9_pred_label = TreeClassifiers.getTreePredictions(datasets[8], xtest, params1, x_cols, y_cols)
            tree1_10_pred_label = TreeClassifiers.getTreePredictions(datasets[9], xtest, params1, x_cols, y_cols)

            #Tree2 parameters
            params2 = parameters(class_weight=None, criterion='entropy', max_depth=5, max_features=None,
                max_leaf_nodes=None, min_samples_leaf=5, min_samples_split=2, min_weight_fraction_leaf=0.0,
                presort=False, random_state=123, splitter='best')

            #tree2_0_pred_label = TreeClassifiers.getTreePredictions(train, xtest, params2, x_cols, y_cols)
            tree2_1_pred_label = TreeClassifiers.getTreePredictions(datasets[0], xtest, params2, x_cols, y_cols)
            tree2_2_pred_label = TreeClassifiers.getTreePredictions(datasets[1], xtest, params2, x_cols, y_cols)
            tree2_3_pred_label = TreeClassifiers.getTreePredictions(datasets[2], xtest, params2, x_cols, y_cols)
            tree2_4_pred_label = TreeClassifiers.getTreePredictions(datasets[3], xtest, params2, x_cols, y_cols)
            tree2_5_pred_label = TreeClassifiers.getTreePredictions(datasets[4], xtest, params2, x_cols, y_cols)
            tree2_6_pred_label = TreeClassifiers.getTreePredictions(datasets[5], xtest, params2, x_cols, y_cols)
            tree2_7_pred_label = TreeClassifiers.getTreePredictions(datasets[6], xtest, params2, x_cols, y_cols)
            tree2_8_pred_label = TreeClassifiers.getTreePredictions(datasets[7], xtest, params2, x_cols, y_cols)
            tree2_9_pred_label = TreeClassifiers.getTreePredictions(datasets[8], xtest, params2, x_cols, y_cols)
            tree2_10_pred_label = TreeClassifiers.getTreePredictions(datasets[9], xtest, params2, x_cols, y_cols)

            #xgboost0_pred_label = TreeClassifiers.getXGBoostPredictions(train, xtest, x_cols, y_cols)
            xgboost1_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[0], xtest, x_cols, y_cols)
            xgboost2_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[1], xtest, x_cols, y_cols)
            xgboost3_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[2], xtest, x_cols, y_cols)
            xgboost4_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[3], xtest, x_cols, y_cols)
            xgboost5_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[4], xtest, x_cols, y_cols)
            xgboost6_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[5], xtest, x_cols, y_cols)
            xgboost7_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[6], xtest, x_cols, y_cols)
            xgboost8_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[7], xtest, x_cols, y_cols)
            xgboost9_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[8], xtest, x_cols, y_cols)
            xgboost10_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[9], xtest, x_cols, y_cols)

            #SVM parameters
            kern = "rbf"

            #svm0_pred_label = SvmClassifiers.svmGetPredictions(train, xtest, kern)
            svm1_pred_label = SvmClassifiers.svmGetPredictions(datasets[0], xtest, kern)
            svm2_pred_label = SvmClassifiers.svmGetPredictions(datasets[1], xtest, kern)
            svm3_pred_label = SvmClassifiers.svmGetPredictions(datasets[2], xtest, kern)
            svm4_pred_label = SvmClassifiers.svmGetPredictions(datasets[3], xtest, kern)
            svm5_pred_label = SvmClassifiers.svmGetPredictions(datasets[4], xtest, kern)
            svm6_pred_label = SvmClassifiers.svmGetPredictions(datasets[5], xtest, kern)
            svm7_pred_label = SvmClassifiers.svmGetPredictions(datasets[6], xtest, kern)
            svm8_pred_label = SvmClassifiers.svmGetPredictions(datasets[7], xtest, kern)
            svm9_pred_label = SvmClassifiers.svmGetPredictions(datasets[8], xtest, kern)
            svm10_pred_label = SvmClassifiers.svmGetPredictions(datasets[9], xtest, kern)

            #tensorflow







            predictionSets = [
                knn1_pred_label, knn2_pred_label, knn3_pred_label, knn4_pred_label, knn5_pred_label,
                knn6_pred_label, knn7_pred_label, knn8_pred_label, knn9_pred_label, knn10_pred_label,
                tree1_1_pred_label, tree1_2_pred_label, tree1_3_pred_label, tree1_4_pred_label, tree1_5_pred_label,
                tree1_6_pred_label, tree1_7_pred_label, tree1_8_pred_label, tree1_9_pred_label, tree1_10_pred_label,
                tree2_1_pred_label, tree2_2_pred_label, tree2_3_pred_label, tree2_4_pred_label, tree2_5_pred_label,
                tree2_6_pred_label, tree2_7_pred_label, tree2_8_pred_label, tree2_9_pred_label, tree2_10_pred_label,
                xgboost1_pred_label, xgboost2_pred_label, xgboost3_pred_label, xgboost4_pred_label, xgboost5_pred_label,
                xgboost6_pred_label, xgboost7_pred_label, xgboost8_pred_label, xgboost9_pred_label, xgboost10_pred_label,
                svm1_pred_label, svm2_pred_label, svm3_pred_label, svm4_pred_label, svm5_pred_label,
                svm6_pred_label, svm7_pred_label, svm8_pred_label, svm9_pred_label, svm10_pred_label
                ]
        
            #prediction sources array (for weighting different classifiers)
            setTypes = [
                "knn", "knn", "knn", "knn", "knn",
                "knn", "knn", "knn", "knn", "knn",
                "tree1", "tree1", "tree1", "tree1", "tree1",
                "tree1", "tree1", "tree1", "tree1", "tree1",
                "tree2", "tree2", "tree2", "tree2", "tree2",
                "tree2", "tree2", "tree2", "tree2", "tree2",
                "xgboost", "xgboost", "xgboost", "xgboost", "xgboost",
                "xgboost", "xgboost", "xgboost", "xgboost", "xgboost",
                "svm", "svm", "svm", "svm", "svm",
                "svm", "svm", "svm", "svm", "svm"
                ]


        results = []

        #Voting process
        for index in range(0, len(unlabeled)): #going through every index one by one
            fall_votes = 0
            normal_votes = 0

            for setIndex in range(0, len(predictionSets)): #counting votes
                if(predictionSets[setIndex][index] == "Fall"):
                    if(setTypes[setIndex] == "knn"):
                        fall_votes = fall_votes + 1
                    elif(setTypes[setIndex] == "tree1"):
                        fall_votes = fall_votes + 1
                elif(predictionSets[setIndex][index] == "Normal"):
                    normal_votes = normal_votes + 1
                else:
                    print("Unknown label")

            if(fall_votes >= normal_votes): #appending result
                results.append("Fall")
            else:
                results.append("Normal")


        return(results)