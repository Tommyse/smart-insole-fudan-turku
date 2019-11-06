import os
import pandas as pd
import numpy as np
from scipy.stats import zscore
from collections import namedtuple

from .columns import DataColumns
from .dataHandler import DataHandler
from .knnClassifiers import KnnClassifiers
from .treeClassifiers import TreeClassifiers
from .svmClassifiers import SvmClassifiers
from .mlpClassifiers import MlpClassifiers

from .classifiers import ClassifierInterface, ClassifierAnalysisResult

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt
import seaborn as sn

# import common-training data
train_data = pd.read_csv('data/tommi+diego_test_data.csv', sep=";", header=0)

class Ensemble(ClassifierInterface):
    """
    Ensemble learning class.
    Uses multiple different classifiers and randomized datasets to classify the input data.
    s
    """

    @staticmethod
    def testBagging(data, x_cols=DataColumns.getSelectedCols2()):
        """
        Testing bagging results

        Arguments:
            data {array} -- Data
        """
        #x_cols = DataColumns.getSelectedCols2()
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
            dataset_amount = 5
            drop_amount = 500
            datasets = DataHandler.genRandomDatasets(train, dataset_amount, drop_amount)

            

            knn1_pred_label = KnnClassifiers.getKnnPredictions(datasets[0], xtest, k, x_cols)
            knn2_pred_label = KnnClassifiers.getKnnPredictions(datasets[1], xtest, k, x_cols)
            knn3_pred_label = KnnClassifiers.getKnnPredictions(datasets[2], xtest, k, x_cols)
            knn4_pred_label = KnnClassifiers.getKnnPredictions(datasets[3], xtest, k, x_cols)
            knn5_pred_label = KnnClassifiers.getKnnPredictions(datasets[4], xtest, k, x_cols)

            #Tree1 parameters
            params1 = parameters(class_weight=None, criterion='gini', max_depth=5, max_features=None,
                max_leaf_nodes=None, min_samples_leaf=5, min_samples_split=2, min_weight_fraction_leaf=0.0,
                presort=False, random_state=123, splitter='best')

            tree1_1_pred_label = TreeClassifiers.getTreePredictions(datasets[0], xtest, params1, x_cols, y_cols)
            tree1_2_pred_label = TreeClassifiers.getTreePredictions(datasets[1], xtest, params1, x_cols, y_cols)
            tree1_3_pred_label = TreeClassifiers.getTreePredictions(datasets[2], xtest, params1, x_cols, y_cols)
            tree1_4_pred_label = TreeClassifiers.getTreePredictions(datasets[3], xtest, params1, x_cols, y_cols)
            tree1_5_pred_label = TreeClassifiers.getTreePredictions(datasets[4], xtest, params1, x_cols, y_cols)

            #Tree2 parameters
            params2 = parameters(class_weight=None, criterion='entropy', max_depth=5, max_features=None,
                max_leaf_nodes=None, min_samples_leaf=5, min_samples_split=2, min_weight_fraction_leaf=0.0,
                presort=False, random_state=123, splitter='best')

            tree2_1_pred_label = TreeClassifiers.getTreePredictions(datasets[0], xtest, params2, x_cols, y_cols)
            tree2_2_pred_label = TreeClassifiers.getTreePredictions(datasets[1], xtest, params2, x_cols, y_cols)
            tree2_3_pred_label = TreeClassifiers.getTreePredictions(datasets[2], xtest, params2, x_cols, y_cols)
            tree2_4_pred_label = TreeClassifiers.getTreePredictions(datasets[3], xtest, params2, x_cols, y_cols)
            tree2_5_pred_label = TreeClassifiers.getTreePredictions(datasets[4], xtest, params2, x_cols, y_cols)

            xgboost1_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[0], xtest, x_cols, y_cols)
            xgboost2_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[1], xtest, x_cols, y_cols)
            xgboost3_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[2], xtest, x_cols, y_cols)
            xgboost4_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[3], xtest, x_cols, y_cols)
            xgboost5_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[4], xtest, x_cols, y_cols)

            #SVM parameters
            kern = "rbf"

            svm1_pred_label = SvmClassifiers.svmGetPredictions(datasets[0], xtest, kern, x_cols)
            svm2_pred_label = SvmClassifiers.svmGetPredictions(datasets[1], xtest, kern, x_cols)
            svm3_pred_label = SvmClassifiers.svmGetPredictions(datasets[2], xtest, kern, x_cols)
            svm4_pred_label = SvmClassifiers.svmGetPredictions(datasets[3], xtest, kern, x_cols)
            svm5_pred_label = SvmClassifiers.svmGetPredictions(datasets[4], xtest, kern, x_cols)

            #tensorflow



            predictionSets = [
                knn1_pred_label, knn2_pred_label, knn3_pred_label, knn4_pred_label, knn5_pred_label,
                tree1_1_pred_label, tree1_2_pred_label, tree1_3_pred_label, tree1_4_pred_label, tree1_5_pred_label,
                tree2_1_pred_label, tree2_2_pred_label, tree2_3_pred_label, tree2_4_pred_label, tree2_5_pred_label,
                xgboost1_pred_label, xgboost2_pred_label, xgboost3_pred_label, xgboost4_pred_label, xgboost5_pred_label,
                svm1_pred_label, svm2_pred_label, svm3_pred_label, svm4_pred_label, svm5_pred_label
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
    
    @staticmethod
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

    @staticmethod
    def getBaggingPredictions(train, unlabeled):
        """
        Most popular label from multiple different classifiers wins. All votes are equal.
        Training with randomly drawn sets from training set

        Arguments:
            train {array} -- Training data for the classifiers
            unlabeled {array} -- Unlabeled data which needs label predictions
        """
        x_cols = DataColumns.getSelectedCols2()
        y_cols = ["label"]
        
        xtest = unlabeled

        parameters = namedtuple("parameters", ["class_weight", "criterion", "max_depth", "max_features", #namedTuple for tree parameters
            "max_leaf_nodes", "min_samples_leaf", "min_samples_split", "min_weight_fraction_leaf",
            "presort", "random_state", "splitter"])


        #generating datasets where data has been suffled and last random x rows has been dropped
        dataset_amount = 5
        drop_amount = 500
        datasets = DataHandler.genRandomDatasets(train, dataset_amount, drop_amount)

        #KNN parameters
        k = KnnClassifiers.findBestK(train, x_cols, y_cols) #best k

        knn1_pred_label = KnnClassifiers.getKnnPredictions(datasets[0], xtest, k)
        knn2_pred_label = KnnClassifiers.getKnnPredictions(datasets[1], xtest, k)
        knn3_pred_label = KnnClassifiers.getKnnPredictions(datasets[2], xtest, k)
        knn4_pred_label = KnnClassifiers.getKnnPredictions(datasets[3], xtest, k)
        knn5_pred_label = KnnClassifiers.getKnnPredictions(datasets[4], xtest, k)

        #Tree1 parameters
        params1 = parameters(class_weight=None, criterion='gini', max_depth=5, max_features=None,
            max_leaf_nodes=None, min_samples_leaf=5, min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=123, splitter='best')

        tree1_1_pred_label = TreeClassifiers.getTreePredictions(datasets[0], xtest, params1, x_cols, y_cols)
        tree1_2_pred_label = TreeClassifiers.getTreePredictions(datasets[1], xtest, params1, x_cols, y_cols)
        tree1_3_pred_label = TreeClassifiers.getTreePredictions(datasets[2], xtest, params1, x_cols, y_cols)
        tree1_4_pred_label = TreeClassifiers.getTreePredictions(datasets[3], xtest, params1, x_cols, y_cols)
        tree1_5_pred_label = TreeClassifiers.getTreePredictions(datasets[4], xtest, params1, x_cols, y_cols)

        #Tree2 parameters
        params2 = parameters(class_weight=None, criterion='entropy', max_depth=5, max_features=None,
            max_leaf_nodes=None, min_samples_leaf=5, min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=123, splitter='best')

        tree2_1_pred_label = TreeClassifiers.getTreePredictions(datasets[0], xtest, params2, x_cols, y_cols)
        tree2_2_pred_label = TreeClassifiers.getTreePredictions(datasets[1], xtest, params2, x_cols, y_cols)
        tree2_3_pred_label = TreeClassifiers.getTreePredictions(datasets[2], xtest, params2, x_cols, y_cols)
        tree2_4_pred_label = TreeClassifiers.getTreePredictions(datasets[3], xtest, params2, x_cols, y_cols)
        tree2_5_pred_label = TreeClassifiers.getTreePredictions(datasets[4], xtest, params2, x_cols, y_cols)

        xgboost1_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[0], xtest, x_cols, y_cols)
        xgboost2_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[1], xtest, x_cols, y_cols)
        xgboost3_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[2], xtest, x_cols, y_cols)
        xgboost4_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[3], xtest, x_cols, y_cols)
        xgboost5_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[4], xtest, x_cols, y_cols)

        #SVM parameters
        kern = "rbf"

        svm1_pred_label = SvmClassifiers.svmGetPredictions(datasets[0], xtest, kern)
        svm2_pred_label = SvmClassifiers.svmGetPredictions(datasets[1], xtest, kern)
        svm3_pred_label = SvmClassifiers.svmGetPredictions(datasets[2], xtest, kern)
        svm4_pred_label = SvmClassifiers.svmGetPredictions(datasets[3], xtest, kern)
        svm5_pred_label = SvmClassifiers.svmGetPredictions(datasets[4], xtest, kern)



        predictionSets = [
                knn1_pred_label, knn2_pred_label, knn3_pred_label, knn4_pred_label, knn5_pred_label,
                tree1_1_pred_label, tree1_2_pred_label, tree1_3_pred_label, tree1_4_pred_label, tree1_5_pred_label,
                tree2_1_pred_label, tree2_2_pred_label, tree2_3_pred_label, tree2_4_pred_label, tree2_5_pred_label,
                xgboost1_pred_label, xgboost2_pred_label, xgboost3_pred_label, xgboost4_pred_label, xgboost5_pred_label,
                svm1_pred_label, svm2_pred_label, svm3_pred_label, svm4_pred_label, svm5_pred_label
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
    @staticmethod
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

    @staticmethod
    def testBoosting(data):
        """
        Testing boosting results

        Arguments:
            data {array} -- Data
        """

        x_cols = DataColumns.getSelectedCols3()
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
            dataset_amount = 3
            drop_amount = 500
            datasets = DataHandler.genRandomDatasets(train, dataset_amount, drop_amount)

            
            knn1_pred_label = KnnClassifiers.getKnnPredictions(datasets[0], xtest, k, x_cols)
            knn2_pred_label = KnnClassifiers.getKnnPredictions(datasets[1], xtest, k, x_cols)
            knn3_pred_label = KnnClassifiers.getKnnPredictions(datasets[2], xtest, k, x_cols)

            #Tree1 parameters
            params1 = parameters(class_weight=None, criterion='gini', max_depth=5, max_features=None,
                max_leaf_nodes=None, min_samples_leaf=5, min_samples_split=2, min_weight_fraction_leaf=0.0,
                presort=False, random_state=123, splitter='best')

            tree1_1_pred_label = TreeClassifiers.getTreePredictions(datasets[0], xtest, params1, x_cols, y_cols)
            tree1_2_pred_label = TreeClassifiers.getTreePredictions(datasets[1], xtest, params1, x_cols, y_cols)
            tree1_3_pred_label = TreeClassifiers.getTreePredictions(datasets[2], xtest, params1, x_cols, y_cols)

            #Tree2 parameters
            params2 = parameters(class_weight=None, criterion='entropy', max_depth=5, max_features=None,
                max_leaf_nodes=None, min_samples_leaf=5, min_samples_split=2, min_weight_fraction_leaf=0.0,
                presort=False, random_state=123, splitter='best')

            tree2_1_pred_label = TreeClassifiers.getTreePredictions(datasets[0], xtest, params2, x_cols, y_cols)
            tree2_2_pred_label = TreeClassifiers.getTreePredictions(datasets[1], xtest, params2, x_cols, y_cols)
            tree2_3_pred_label = TreeClassifiers.getTreePredictions(datasets[2], xtest, params2, x_cols, y_cols)

            xgboost1_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[0], xtest, x_cols, y_cols)
            xgboost2_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[1], xtest, x_cols, y_cols)
            xgboost3_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[2], xtest, x_cols, y_cols)

            #SVM parameters
            kern = "rbf"

            svm1_pred_label = SvmClassifiers.svmGetPredictions(datasets[0], xtest, kern, x_cols)
            svm2_pred_label = SvmClassifiers.svmGetPredictions(datasets[1], xtest, kern, x_cols)
            svm3_pred_label = SvmClassifiers.svmGetPredictions(datasets[2], xtest, kern, x_cols)

            #MLP, takes too much time
            #mlp_parameters = namedtuple("parameters", ["hidden_layer_sizes", "solver", "alpha", "batch_size", "learning_rate",
            #    "learning_rate_init", "max_iter", "random_state", "verbose", "early_stopping", "validation_fraction"])
            #
            ##Parameters
            #params = mlp_parameters(
            #		hidden_layer_sizes=[100, 100],
            #		solver="lbfgs",
            #		alpha=0.1,
            #		batch_size="auto",
            #		learning_rate="constant",
            #		learning_rate_init=0.001,
            #		max_iter=200,
            #		random_state=123,
            #		verbose=True,
            #		early_stopping=False,
            #		validation_fraction=0.1
            #	)
            #
            #mlp1_pred_label = MlpClassifiers.mlpGetPredictions(datasets[0], xtest, params, x_cols)
            #mlp2_pred_label = MlpClassifiers.mlpGetPredictions(datasets[1], xtest, params, x_cols)
            #mlp3_pred_label = MlpClassifiers.mlpGetPredictions(datasets[2], xtest, params, x_cols)




            predictionSets = [
                knn1_pred_label, knn2_pred_label, knn3_pred_label,
                tree1_1_pred_label, tree1_2_pred_label, tree1_3_pred_label,
                tree2_1_pred_label, tree2_2_pred_label, tree2_3_pred_label,
                xgboost1_pred_label, xgboost2_pred_label, xgboost3_pred_label,
                svm1_pred_label, svm2_pred_label, svm3_pred_label
                #mlp1_pred_label, mlp2_pred_label, mlp3_pred_label
                ]
        
            #prediction sources array (for weighting different classifiers)
            setTypes = [
                "knn", "knn", "knn",
                "tree1", "tree1", "tree1",
                "tree2", "tree2", "tree2",
                "xgboost", "xgboost", "xgboost",
                "svm", "svm", "svm"
                #"mlp", "mlp", "mlp"
                ]


            results = []

            #Weighted voting process
            for index in range(0, len(xtest)): #going through every index one by one
                fall_votes = 0
                normal_votes = 0

                for setIndex in range(0, len(predictionSets)): #counting votes
                    if(predictionSets[setIndex][index] == "Fall"):
                        if(setTypes[setIndex] == "knn"):
                            fall_votes = fall_votes + 1.3
                        elif(setTypes[setIndex] == "tree1"):
                            fall_votes = fall_votes + 1.1
                        elif(setTypes[setIndex] == "tree2"):
                            fall_votes = fall_votes + 1.1
                        elif(setTypes[setIndex] == "xgboost"):
                            fall_votes = fall_votes + 1.3
                        elif(setTypes[setIndex] == "svm"):
                            fall_votes = fall_votes + 1.3
                        elif(setTypes[setIndex] == "mlp"):
                            fall_votes = fall_votes + 1
                    elif(predictionSets[setIndex][index] == "Normal"):
                        if(setTypes[setIndex] == "knn"):
                            normal_votes = normal_votes + 1.3
                        elif(setTypes[setIndex] == "tree1"):
                            normal_votes = normal_votes + 1.1
                        elif(setTypes[setIndex] == "tree2"):
                            normal_votes = normal_votes + 1.1
                        elif(setTypes[setIndex] == "xgboost"):
                            normal_votes = normal_votes + 1.3
                        elif(setTypes[setIndex] == "svm"):
                            normal_votes = normal_votes + 1.3
                        elif(setTypes[setIndex] == "mlp"):
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
    
    @staticmethod
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
    
    @staticmethod
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
            dataset_amount = 5
            drop_amount = 500
            datasets = DataHandler.genRandomDatasets(train, dataset_amount, drop_amount)

            
            knn1_pred_label = KnnClassifiers.getKnnPredictions(datasets[0], xtest, k, x_cols)
            knn2_pred_label = KnnClassifiers.getKnnPredictions(datasets[1], xtest, k, x_cols)
            knn3_pred_label = KnnClassifiers.getKnnPredictions(datasets[2], xtest, k, x_cols)
            knn4_pred_label = KnnClassifiers.getKnnPredictions(datasets[3], xtest, k, x_cols)
            knn5_pred_label = KnnClassifiers.getKnnPredictions(datasets[4], xtest, k, x_cols)

            #Tree1 parameters
            params1 = parameters(class_weight=None, criterion='gini', max_depth=5, max_features=None,
                max_leaf_nodes=None, min_samples_leaf=5, min_samples_split=2, min_weight_fraction_leaf=0.0,
                presort=False, random_state=123, splitter='best')

            tree1_1_pred_label = TreeClassifiers.getTreePredictions(datasets[0], xtest, params1, x_cols, y_cols)
            tree1_2_pred_label = TreeClassifiers.getTreePredictions(datasets[1], xtest, params1, x_cols, y_cols)
            tree1_3_pred_label = TreeClassifiers.getTreePredictions(datasets[2], xtest, params1, x_cols, y_cols)
            tree1_4_pred_label = TreeClassifiers.getTreePredictions(datasets[3], xtest, params1, x_cols, y_cols)
            tree1_5_pred_label = TreeClassifiers.getTreePredictions(datasets[4], xtest, params1, x_cols, y_cols)

            #Tree2 parameters
            params2 = parameters(class_weight=None, criterion='entropy', max_depth=5, max_features=None,
                max_leaf_nodes=None, min_samples_leaf=5, min_samples_split=2, min_weight_fraction_leaf=0.0,
                presort=False, random_state=123, splitter='best')

            tree2_1_pred_label = TreeClassifiers.getTreePredictions(datasets[0], xtest, params2, x_cols, y_cols)
            tree2_2_pred_label = TreeClassifiers.getTreePredictions(datasets[1], xtest, params2, x_cols, y_cols)
            tree2_3_pred_label = TreeClassifiers.getTreePredictions(datasets[2], xtest, params2, x_cols, y_cols)
            tree2_4_pred_label = TreeClassifiers.getTreePredictions(datasets[3], xtest, params2, x_cols, y_cols)
            tree2_5_pred_label = TreeClassifiers.getTreePredictions(datasets[4], xtest, params2, x_cols, y_cols)

            xgboost1_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[0], xtest, x_cols, y_cols)
            xgboost2_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[1], xtest, x_cols, y_cols)
            xgboost3_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[2], xtest, x_cols, y_cols)
            xgboost4_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[3], xtest, x_cols, y_cols)
            xgboost5_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[4], xtest, x_cols, y_cols)

            #SVM parameters
            kern = "rbf"

            svm1_pred_label = SvmClassifiers.svmGetPredictions(datasets[0], xtest, kern, x_cols)
            svm2_pred_label = SvmClassifiers.svmGetPredictions(datasets[1], xtest, kern, x_cols)
            svm3_pred_label = SvmClassifiers.svmGetPredictions(datasets[2], xtest, kern, x_cols)
            svm4_pred_label = SvmClassifiers.svmGetPredictions(datasets[3], xtest, kern, x_cols)
            svm5_pred_label = SvmClassifiers.svmGetPredictions(datasets[4], xtest, kern, x_cols)

            #MLP
            mlp_parameters = namedtuple("parameters", ["hidden_layer_sizes", "solver", "alpha", "batch_size", "learning_rate",
            "learning_rate_init", "max_iter", "random_state", "verbose", "early_stopping", "validation_fraction"])

            #Parameters
            params = mlp_parameters(
            		hidden_layer_sizes=[100, 100],
            		solver="lbfgs",
            		alpha=0.1,
            		batch_size="auto",
            		learning_rate="constant",
            		learning_rate_init=0.001,
            		max_iter=200,
            		random_state=123,
            		verbose=True,
            		early_stopping=False,
            		validation_fraction=0.1
            	)
        
            mlp1_pred_label = MlpClassifiers.mlpGetPredictions(datasets[0], xtest, params, x_cols)
            mlp2_pred_label = MlpClassifiers.mlpGetPredictions(datasets[1], xtest, params, x_cols)
            mlp3_pred_label = MlpClassifiers.mlpGetPredictions(datasets[2], xtest, params, x_cols)
            mlp4_pred_label = MlpClassifiers.mlpGetPredictions(datasets[3], xtest, params, x_cols)
            mlp5_pred_label = MlpClassifiers.mlpGetPredictions(datasets[4], xtest, params, x_cols)


            predictionSets = [
                knn1_pred_label, knn2_pred_label, knn3_pred_label, knn4_pred_label, knn5_pred_label,
                tree1_1_pred_label, tree1_2_pred_label, tree1_3_pred_label, tree1_4_pred_label, tree1_5_pred_label,
                tree2_1_pred_label, tree2_2_pred_label, tree2_3_pred_label, tree2_4_pred_label, tree2_5_pred_label,
                xgboost1_pred_label, xgboost2_pred_label, xgboost3_pred_label, xgboost4_pred_label, xgboost5_pred_label,
                svm1_pred_label, svm2_pred_label, svm3_pred_label, svm4_pred_label, svm5_pred_label,
                mlp1_pred_label, mlp2_pred_label, mlp3_pred_label, mlp4_pred_label, mlp5_pred_label
                ]
        
            #prediction sources array (for weighting different classifiers)
            setTypes = [
                "knn", "knn", "knn", "knn", "knn",
                "tree1", "tree1", "tree1", "tree1", "tree1",
                "tree2", "tree2", "tree2", "tree2", "tree2",
                "xgboost", "xgboost", "xgboost", "xgboost", "xgboost",
                "svm", "svm", "svm", "svm", "svm",
                "mlp", "mlp", "mlp", "mlp", "mlp",
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
                            fall_votes = fall_votes + 1.3
                        elif(setTypes[setIndex] == "tree1"):
                            fall_votes = fall_votes + 1.1
                        elif(setTypes[setIndex] == "tree2"):
                            fall_votes = fall_votes + 1.1
                        elif(setTypes[setIndex] == "xgboost"):
                            fall_votes = fall_votes + 1.3
                        elif(setTypes[setIndex] == "svm"):
                            fall_votes = fall_votes + 1.3
                        elif(setTypes[setIndex] == "mlp"):
                            fall_votes = fall_votes + 1
                    elif(predictionSets[setIndex][index] == "Normal"):
                        if(setTypes[setIndex] == "knn"):
                            normal_votes = normal_votes + 1.3
                        elif(setTypes[setIndex] == "tree1"):
                            normal_votes = normal_votes + 1.1
                        elif(setTypes[setIndex] == "tree2"):
                            normal_votes = normal_votes + 1.1
                        elif(setTypes[setIndex] == "xgboost"):
                            normal_votes = normal_votes + 1.3
                        elif(setTypes[setIndex] == "svm"):
                            normal_votes = normal_votes + 1.3
                        elif(setTypes[setIndex] == "mlp"):
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
    
    @staticmethod
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
    
    @staticmethod
    def analyseImbalance(inputData):
        # print("LeN: ", len(inputData), len(inputData[0]))
        inputDataFrame = pd.DataFrame(data=np.array(inputData), columns=DataColumns.getAllCols())
        inputDataFrame = inputDataFrame.astype(DataColumns.getColTypes())
        results, normal_count, fall_count = Ensemble.getBoostingPredictions(train_data, inputDataFrame)

        riskFalling = False
        totalCount = (normal_count + fall_count)
        if totalCount > 0:
            riskFalling = (normal_count / totalCount) < 0.9

        return ClassifierAnalysisResult(goodSteps=normal_count, badSteps=fall_count, riskFalling=riskFalling)

    @staticmethod
    def getBoostingPredictions(train, unlabeled):
        """
        Different classifiers have different weights.
        Based on them, most popular label wins. Votes are not equal.
        Training with randomly drawn sets from training set

        Arguments:
            train {array} -- Training data for the classifiers
            unlabeled {array} -- Unlabeled data which needs label predictions
        """
        x_cols = DataColumns.getSelectedCols3() #Selected features
        y_cols = ["label"]
        
        #Adding some labels for filtering train and unlabeled data later
        train.loc[:, "data_type"] = "train"
        unlabeled.loc[:, "data_type"] = "unlabeled"
        unlabeled.loc[:, "label"] = "unknown" #placeholder
        #print(unlabeled)
        
        #Combining input data
        #data_df_row = 0
        data = train.append(unlabeled, ignore_index=True)
        #for row in range(0,len(train)):
        #    data.loc[data_df_row,:] = train.loc[row,:]
        #    data_df_row = data_df_row + 1
        #
        #for row in range(0,len(unlabeled)):
        #    data.loc[data_df_row,:] = unlabeled.loc[row,:]
        #    data_df_row = data_df_row + 1
        
        #Lots of pre-processing
        #data = data.reset_index(drop=True) #just in case
        #data = DataHandler.calculateTotalForce(data)
        #data = DataHandler.calculateStepTime(data)
        #data = DataHandler.calculateForceValues(data)
        #data = DataHandler.calculateStepStartValues(data)
        #data = DataHandler.calculateStepMaxTimeValues(data)
        #data = DataHandler.calculateStepEndTimeValues(data)
        #data = DataHandler.calculatePhaseForceValues(data)
        #data = DataHandler.calculatePressTimeValues(data)
        
        train = DataHandler.calculateTotalForce(train)
        train = DataHandler.calculateStepTime(train)
        train = DataHandler.calculateForceValues(train)
        train = DataHandler.calculatePhaseForceValues(train)
        
        unlabeled = DataHandler.calculateTotalForce(unlabeled)
        unlabeled = DataHandler.calculateStepTime(unlabeled)
        unlabeled = DataHandler.calculateForceValues(unlabeled)
        unlabeled = DataHandler.calculatePhaseForceValues(unlabeled)
        
        #Normalize features here

        

        #print(data.loc[:,"data_type"])



        #Back to separate dataframes
        #train = data.loc[data["data_type"] == "train"]
        #train = train.loc[train["Warning_code"] == 0] #filtering errors out of the training data
        #print("train", train)
        
        #unlabeled = data.loc[data["data_type"] == "unlabeled"]
        #print("unlabeled", unlabeled)

        xtrain = train.loc[:, x_cols]
        ytrain = train.loc[:, y_cols]

        xtest = unlabeled.loc[:, x_cols] #avoiding re-writing some code
        print(xtest)

        parameters = namedtuple("parameters", ["class_weight", "criterion", "max_depth", "max_features", #namedTuple for tree parameters
            "max_leaf_nodes", "min_samples_leaf", "min_samples_split", "min_weight_fraction_leaf",
            "presort", "random_state", "splitter"])

        #KNN parameters
        k = KnnClassifiers.findBestK(train, x_cols, y_cols) #best k

        #train = xtrain
        #train.loc[:,"label"] = ytrain
        

        #generating datasets where data has been suffled and last random x rows has been dropped
        dataset_amount = 5
        drop_amount = 500
        datasets = DataHandler.genRandomDatasets(train, dataset_amount, drop_amount)


        print(datasets[0])
        
        knn1_pred_label = KnnClassifiers.getKnnPredictions(datasets[0], xtest, k, x_cols)
        knn2_pred_label = KnnClassifiers.getKnnPredictions(datasets[1], xtest, k, x_cols)
        knn3_pred_label = KnnClassifiers.getKnnPredictions(datasets[2], xtest, k, x_cols)

        #Tree1 parameters
        params1 = parameters(class_weight=None, criterion='gini', max_depth=5, max_features=None,
            max_leaf_nodes=None, min_samples_leaf=5, min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=123, splitter='best')

        tree1_1_pred_label = TreeClassifiers.getTreePredictions(datasets[0], xtest, params1, x_cols, y_cols)
        tree1_2_pred_label = TreeClassifiers.getTreePredictions(datasets[1], xtest, params1, x_cols, y_cols)
        tree1_3_pred_label = TreeClassifiers.getTreePredictions(datasets[2], xtest, params1, x_cols, y_cols)

        #Tree2 parameters
        params2 = parameters(class_weight=None, criterion='entropy', max_depth=5, max_features=None,
            max_leaf_nodes=None, min_samples_leaf=5, min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=123, splitter='best')

        tree2_1_pred_label = TreeClassifiers.getTreePredictions(datasets[0], xtest, params2, x_cols, y_cols)
        tree2_2_pred_label = TreeClassifiers.getTreePredictions(datasets[1], xtest, params2, x_cols, y_cols)
        tree2_3_pred_label = TreeClassifiers.getTreePredictions(datasets[2], xtest, params2, x_cols, y_cols)

        xgboost1_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[0], xtest, x_cols, y_cols)
        xgboost2_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[1], xtest, x_cols, y_cols)
        xgboost3_pred_label = TreeClassifiers.getXGBoostPredictions(datasets[2], xtest, x_cols, y_cols)

        #SVM parameters
        kern = "rbf"

        svm1_pred_label = SvmClassifiers.svmGetPredictions(datasets[0], xtest, kern, x_cols)
        svm2_pred_label = SvmClassifiers.svmGetPredictions(datasets[1], xtest, kern, x_cols)
        svm3_pred_label = SvmClassifiers.svmGetPredictions(datasets[2], xtest, kern, x_cols)

        #MLP, takes a lot of time
        #mlp_parameters = namedtuple("parameters", ["hidden_layer_sizes", "solver", "alpha", "batch_size", "learning_rate",
        #    "learning_rate_init", "max_iter", "random_state", "verbose", "early_stopping", "validation_fraction"])
        #
        ##Parameters
        #params = mlp_parameters(
        #		hidden_layer_sizes=[100, 100],
        #		solver="lbfgs",
        #		alpha=0.1,
        #		batch_size="auto",
        #		learning_rate="constant",
        #		learning_rate_init=0.001,
        #		max_iter=200,
        #		random_state=123,
        #		verbose=True,
        #		early_stopping=False,
        #		validation_fraction=0.1
        #	)
        #
        #mlp1_pred_label = MlpClassifiers.mlpGetPredictions(datasets[0], xtest, params, x_cols)
        #mlp2_pred_label = MlpClassifiers.mlpGetPredictions(datasets[1], xtest, params, x_cols)
        #mlp3_pred_label = MlpClassifiers.mlpGetPredictions(datasets[2], xtest, params, x_cols)




        predictionSets = [
            knn1_pred_label, knn2_pred_label, knn3_pred_label,
            tree1_1_pred_label, tree1_2_pred_label, tree1_3_pred_label,
            tree2_1_pred_label, tree2_2_pred_label, tree2_3_pred_label,
            xgboost1_pred_label, xgboost2_pred_label, xgboost3_pred_label,
            svm1_pred_label, svm2_pred_label, svm3_pred_label
            #mlp1_pred_label, mlp2_pred_label, mlp3_pred_label
            ]
        
        #prediction sources array (for weighting different classifiers)
        setTypes = [
            "knn", "knn", "knn",
            "tree1", "tree1", "tree1",
            "tree2", "tree2", "tree2",
            "xgboost", "xgboost", "xgboost",
            "svm", "svm", "svm"
            #"mlp", "mlp", "mlp"
            ]


        results = []
        normal_count = 0
        fall_count = 0

        #Voting process
        for index in range(0, len(unlabeled)): #going through every index one by one
            fall_votes = 0
            normal_votes = 0

            for setIndex in range(0, len(predictionSets)): #counting votes
                if(predictionSets[setIndex][index] == "Fall"):
                    if(setTypes[setIndex] == "knn"):
                        fall_votes = fall_votes + 1.3
                    elif(setTypes[setIndex] == "tree1"):
                        fall_votes = fall_votes + 1.1
                    elif(setTypes[setIndex] == "tree2"):
                        fall_votes = fall_votes + 1.1
                    elif(setTypes[setIndex] == "xgboost"):
                        fall_votes = fall_votes + 1.3
                    elif(setTypes[setIndex] == "svm"):
                        fall_votes = fall_votes + 1.3
                    elif(setTypes[setIndex] == "mlp"):
                        fall_votes = fall_votes + 1
                elif(predictionSets[setIndex][index] == "Normal"):
                    if(setTypes[setIndex] == "knn"):
                        normal_votes = normal_votes + 1.3
                    elif(setTypes[setIndex] == "tree1"):
                        normal_votes = normal_votes + 1.1
                    elif(setTypes[setIndex] == "tree2"):
                        normal_votes = normal_votes + 1.1
                    elif(setTypes[setIndex] == "xgboost"):
                        normal_votes = normal_votes + 1.3
                    elif(setTypes[setIndex] == "svm"):
                        normal_votes = normal_votes + 1.3
                    elif(setTypes[setIndex] == "mlp"):
                        normal_votes = normal_votes + 1
                else:
                    print("Unknown label")

            if(fall_votes >= normal_votes): #appending results and updating counters
                results.append("Fall")
                fall_count = fall_count + 1
            else:
                results.append("Normal")
                normal_count = normal_count + 1


        return(results, normal_count, fall_count)