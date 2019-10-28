import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from collections import namedtuple
from columns import DataColumns
import io
import pydot

import xgboost as xgb

import matplotlib.pyplot as plt
import seaborn as sn

class TreeClassifiers:
    """
    Decision tree classifiers
    """
    
    #https://scikit-learn.org/stable/modules/tree.html
    #https://scikit-learn.org/stable/auto_examples/tree/plot_iris_dtc.html
    #http://dataaspirant.com/2017/02/01/decision-tree-algorithm-python-with-scikit-learn/
    #https://docs.python.org/dev/library/collections.html#collections.namedtuple
    
    @staticmethod
    def testTreePredictions(data, parameters, x_cols, y_cols, plots=False):
        """
        Testing tree prediction accuracies.
        
        Arguments:
            data {array} -- Labeled data for classifier testing.
            x_cols {array} -- x columns
            y_cols {array} -- y columns
            parameters {namedtuple} -- Parameters for the tree classifier. Using named tuple to keep things tidy.
            
        Keyword Arguments:
            plots {bool} -- Used for plotting (default: {False})
        """
        x = data.loc[:, x_cols]
        y = data.loc[:, y_cols]
    
        loo = LeaveOneOut()
        loo.get_n_splits(data)
        n=loo.split(data)
        
        #Creating the classifier with the input parameters
        treeClassifier = tree.DecisionTreeClassifier(
                class_weight = parameters.class_weight,
                criterion = parameters.criterion, 
                max_depth = parameters.max_depth, 
                max_features = parameters.max_features, 
                max_leaf_nodes = parameters.max_leaf_nodes, 
                min_samples_leaf = parameters.min_samples_leaf, 
                min_samples_split = parameters.min_samples_split, 
                min_weight_fraction_leaf = parameters.min_weight_fraction_leaf,
                presort = parameters.presort, 
                random_state = parameters.random_state, 
                splitter = parameters.splitter
            )
        
        accuracy_a = []
        real_label = []
        pred_label = []

        for train_index, test_index in n: #Each row is test data once
            xtrain, xtest = x.iloc[train_index], x.iloc[test_index]
            ytrain, ytest = y.iloc[train_index], y.iloc[test_index]

            #Fitting train data
            treeClassifier = treeClassifier.fit(xtrain, ytrain)
            #Predictions
            ypred=treeClassifier.predict(xtest)
            pred_label.append(ypred)
            real_label.append(ytest.values)
            #Accuracy
            acc = accuracy_score(ytest, ypred)
            accuracy_a.append(acc)
    
        pred_label_df = pd.DataFrame(columns=["label"])
        real_label_df = pd.DataFrame(columns=["label"])
        
        #Forming the dataframes
        for row in range(0,len(pred_label)):
            label_str = pred_label[row][0]
            pred_label_df.loc[row] = label_str
        
        for row in range(0,len(real_label)):
            label_str = real_label[row][0][0]
            real_label_df.loc[row] = label_str
        
        if(plots): #Plotting tree and accuracy heatmap
            
            #not found in the library for some reason, currently using old version?
            #plt.figure(figsize=[12, 12])
            #tree.plot_tree(treeClassifier, filled=True)
            #plt.show()
            
            #Workaround attempt for tree plotting
            dot = io.StringIO()
            tree.export_graphviz(treeClassifier, out_file=dot)
            (graph,)=pydot.graph_from_dot_data(dot.getvalue())
            graph.write_png("../figs/treeClassifier.png")
            
            cm = confusion_matrix(real_label_df, pred_label_df)
            cm_df = pd.DataFrame(cm, ["Fall", "Normal"], ["Fall", "Normal"])
            
            sn.set(font_scale=1.5)
            sn.heatmap(cm_df, annot=True, annot_kws={"size": 32}, fmt='d')
            plt.savefig("../figs/tree_heatmap.png", facecolor="w", bbox_inches="tight")
            plt.show()
            
        avg_acc = np.mean(accuracy_a)
        
        #Checking accuracy
        print("Tree average accuracy: ", round(avg_acc, 2)) #2 decimals
        
        #More detailed report
        print(classification_report(real_label_df, pred_label_df))
    
        return(avg_acc, real_label_df, pred_label_df)

    @staticmethod
    def testTreeLearning(data, parameters, x_cols, y_cols, times, plots=False, orig_acc=0, orig_auc=0, file_name_prefix="Tree"):
        """
        Suffling labels and fitting data many times.
        Verifying that classifier learns something from the data

        Arguments:
            data {array} -- Data
            parameters {namedTuple} -- parameters for classifier
            x_cols {array} -- x columns
            y_cols {array} -- y columns
            times {int} -- how many times random accuracy is tested

        Keyword Arguments:
            plots {bool} -- Used for plotting (default: {False})
            orig_acc {double} -- accuracy reference (default: {0})
            orig_auc {double} -- AUC reference (default: {0})
            file_name_prefix {str} -- file name prefix (default: {"Tree"})
        """
        
        accs = []
        aucs = []
        
        for run in range(1,times+1):
            print("Run: ", run, " out of ", times)
            
            suffled_data = data
            suffled_data["label"] = np.random.permutation(suffled_data["label"].values)

            avg_acc, real_label, pred_label = TreeClassifiers.testTreePredictions(suffled_data, parameters, x_cols, y_cols, False)
            
            pred_label_df = pred_label
            real_label_df = real_label

            pred_label_df = pred_label_df.replace("Normal", 0)
            pred_label_df = pred_label_df.replace("Fall", 1)

            real_label_df = real_label_df.replace("Normal", 0)
            real_label_df = real_label_df.replace("Fall", 1)
            
            avg_auc = roc_auc_score(real_label_df, pred_label_df)
            
            accs.append(avg_acc)
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
    def getTreePredictions(train, data, parameters, x_cols, y_cols, debug = False):
        """
        Using sklearn's decision tree classifier to classify the input data.
        
        Returns labeled test dataset.
        
        Arguments:
            train {array} -- Labeled data used to train the classifier.
            data {array} -- Data that needs to be labeled.
            parameters {namedtuple} -- Parameters for the tree classifier. Using named tuple to keep things tidy.
        
        Keyword Arguments:
            debug {bool} -- Used for debug printing (default: {False})
        """
        
        xtrain = train.loc[:, x_cols]
        ytrain = train.loc[:, y_cols]
    
        xdata = data.loc[:, x_cols]
    
        #Creating the classifier with the input parameters
        treeClassifier = tree.DecisionTreeClassifier(
                class_weight = parameters.class_weight,
                criterion = parameters.criterion, 
                max_depth = parameters.max_depth, 
                max_features = parameters.max_features, 
                max_leaf_nodes = parameters.max_leaf_nodes, 
                min_samples_leaf = parameters.min_samples_leaf, 
                min_samples_split = parameters.min_samples_split, 
                min_weight_fraction_leaf = parameters.min_weight_fraction_leaf,
                presort = parameters.presort, 
                random_state = parameters.random_state, 
                splitter = parameters.splitter
            )
    
        #Fitting train data
        treeClassifier = treeClassifier.fit(xtrain, ytrain)
    
        if(debug): #Plotting tree
            #Workaround attempt for tree plotting
            dot = io.StringIO()
            tree.export_graphviz(treeClassifier, out_file=dot)
            (graph,)=pydot.graph_from_dot_data(dot.getvalue())
            graph.write_png("../figs/treeClassifier.png")
    
        #Predictions
        ypred = treeClassifier.predict(xdata)
    
        #data["label"] = ypred
    
        return(ypred)
    
    @staticmethod
    def testXGBoostPredictions(data, parameters, x_cols, y_cols, plots=False):
        """
        Testing XGBoost prediction accuracies.

        Arguments:
            data {array} -- Labeled data for classifier testing.
            x_cols {array} -- x columns
            y_cols {array} -- y columns
            parameters {namedtuple} -- Parameters for the tree classifier. Using named tuple to keep things tidy.

        Keyword Arguments:
            plots {bool} -- Used for plotting (default: {False})
        """
        x = data.loc[:, x_cols]
        y = data.loc[:, y_cols]
    
        loo = LeaveOneOut()
        loo.get_n_splits(data)
        n=loo.split(data)
        
        xgbClassifier = xgb.XGBClassifier()
        
        accuracy_a = []
        real_label = []
        pred_label = []

        for train_index, test_index in n: #Each row is test data once
            xtrain, xtest = x.iloc[train_index], x.iloc[test_index]
            ytrain, ytest = y.iloc[train_index], y.iloc[test_index]

            #Fitting train data
            xgbClassifier = xgbClassifier.fit(xtrain, ytrain.values.ravel())
            #Predictions
            ypred=xgbClassifier.predict(xtest)
            pred_label.append(ypred)
            real_label.append(ytest.values)
            #Accuracy
            acc = accuracy_score(ytest, ypred)
            accuracy_a.append(acc)
    
    
        pred_label_df = pd.DataFrame(columns=["label"])
        real_label_df = pd.DataFrame(columns=["label"])
        
        #Forming the dataframes
        for row in range(0,len(pred_label)):
            label_str = pred_label[row][0]
            pred_label_df.loc[row] = label_str
        
        for row in range(0,len(real_label)):
            label_str = real_label[row][0][0]
            real_label_df.loc[row] = label_str
        
        if(plots): #Plotting tree and accuracy heatmap
            
            cm = confusion_matrix(real_label_df, pred_label_df)
            cm_df = pd.DataFrame(cm, ["Fall", "Normal"], ["Fall", "Normal"])
            
            sn.set(font_scale=1.5)
            sn.heatmap(cm_df, annot=True, annot_kws={"size": 32}, fmt='d')
            plt.savefig("../figs/xgboost_heatmap.png", facecolor="w", bbox_inches="tight")
            plt.show()
            
        avg_acc = np.mean(accuracy_a)
        
        #Checking accuracy
        print("Tree average accuracy: ", round(avg_acc, 2)) #2 decimals
        
        #More detailed report
        print(classification_report(real_label_df, pred_label_df))
    
        return(avg_acc, real_label_df, pred_label_df)

    @staticmethod
    def testXGBoostLearning(data, parameters, x_cols, y_cols, times, plots=False, orig_acc=0, orig_auc=0, file_name_prefix="XGBoost"):
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
            orig_acc {double} -- accuracy reference (default: {0})
            orig_auc {double} -- AUC reference (default: {0})
            file_name_prefix {str} -- file name prefix (default: {"XGBoost"})
        """
        
        accs = []
        aucs = []
        
        for run in range(1,times+1):
            print("Run: ", run, " out of ", times)
            
            suffled_data = data
            suffled_data["label"] = np.random.permutation(suffled_data["label"].values)

            avg_acc, real_label, pred_label = TreeClassifiers.testXGBoostPredictions(suffled_data, parameters, x_cols, y_cols, False)
            
            pred_label_df = pred_label
            real_label_df = real_label

            pred_label_df = pred_label_df.replace("Normal", 0)
            pred_label_df = pred_label_df.replace("Fall", 1)

            real_label_df = real_label_df.replace("Normal", 0)
            real_label_df = real_label_df.replace("Fall", 1)
            
            avg_auc = roc_auc_score(real_label_df, pred_label_df)
            
            accs.append(avg_acc)
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
    def getXGBoostPredictions(train, data, x_cols, y_cols):
        """
        XGBoost prediction for input data.

        Arguments:
            train {array} -- Labeled data used to train the classifier.
            data {array} -- Data that needs to be labeled. Unlabeled data can be used.
            x_cols {array} -- x columns
            y_cols {array} -- y columns
        """
        xtrain = train.loc[:, x_cols]
        ytrain = train.loc[:, y_cols]
    
        xdata = data.loc[:, x_cols]
        
        xgbClassifier = xgb.XGBClassifier()

        #Fitting train data
        xgbClassifier = xgbClassifier.fit(xtrain, ytrain.values.ravel())
        #Predictions
        ypred=xgbClassifier.predict(xdata)

        return(ypred)
        
    @staticmethod
    def testFigLayout(prefix="test_"):
        """
        Testing plot layout
        
        Keyword Arguments:
            prefix {str} -- file name prefix (default: {"test_"})
        """
        
        save_path = "../figs/"+prefix+"_plot.png"
        print("save_path: ", save_path)
        
        plt.hist([1,2,6,7,8,7,5,6,8,6,7,7,7,9,0,2], edgecolor='white', bins=14)
        plt.title("Permutations accuracy histogram")
        plt.xlabel("Accuracy")
        plt.ylabel("Count")
        plt.savefig(save_path, facecolor="w", bbox_inches="tight")
        plt.show()