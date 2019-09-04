import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from collections import namedtuple
from columns import DataColumns
import io
import pydot

import xgboost as xgb

import matplotlib.pyplot as plt
import seaborn as sn

class DecisionTreeClassifiers:
    """
    Decision tree classifiers
    """
    
    #https://scikit-learn.org/stable/modules/tree.html
    #https://scikit-learn.org/stable/auto_examples/tree/plot_iris_dtc.html
    #http://dataaspirant.com/2017/02/01/decision-tree-algorithm-python-with-scikit-learn/
    #https://docs.python.org/dev/library/collections.html#collections.namedtuple
    
    
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
            
            #print("pred_label", pred_label)
            #print("real_label", real_label)
            
            
            cm = confusion_matrix(real_label_df, pred_label_df)
            cm_df = pd.DataFrame(cm, ["Normal", "Fall"], ["Normal", "Fall"])
            
            sn.set(font_scale=1.5)
            sn.heatmap(cm_df, annot=True, annot_kws={"size": 20}) #font size 20
            plt.show() #TODO removing the exponent offset...
            
        avg_acc = np.mean(accuracy_a)
        
        #Checking accuracy
        print("Tree average accuracy: ", round(avg_acc, 2)) #2 decimals
        
        #More detailed report
        t_names = ["Normal", "Fall"]
        print(classification_report(real_label_df, pred_label_df, target_names=t_names))
    
        return(avg_acc, real_label_df, pred_label_df)
    
    
    def getTreePredictions(train, test, parameters, x_cols, y_cols, debug = False):
        """
        Using sklearn's decision tree classifier to classify the input data.
        
        Returns labeled test dataset.
        
        Arguments:
            train {array} -- Labeled data used to train the classifier.
            test {array} -- Data that needs to be labeled. Unlabeled data can be used.
            parameters {namedtuple} -- Parameters for the tree classifier. Using named tuple to keep things tidy.
        
        Keyword Arguments:
            debug {bool} -- Used for debug printing (default: {False})
        """
        
        xtrain = train.loc[:, x_cols]
        ytrain = train.loc[:, y_cols]
    
        xtest = test.loc[:, x_cols]
        ytest = pd.DataFrame()
    
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
        ypred = treeClassifier.predict(xtest)
    
        test["label"] = ypred
    
        return(test)
    
    
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
            cm_df = pd.DataFrame(cm, ["Normal", "Fall"], ["Normal", "Fall"])
            
            sn.set(font_scale=1.5)
            sn.heatmap(cm_df, annot=True, annot_kws={"size": 20}) #font size 20
            plt.show() #TODO removing the exponent offset...
            
        avg_acc = np.mean(accuracy_a)
        
        #Checking accuracy
        print("Tree average accuracy: ", round(avg_acc, 2)) #2 decimals
        
        #More detailed report
        t_names = ["Normal", "Fall"]
        print(classification_report(real_label_df, pred_label_df, target_names=t_names))
    
        return(avg_acc, real_label_df, pred_label_df)
    
    
    def getXGBoostPredictions(train, test, parameters, x_cols, y_cols):
        """
        XGBoost prediction for input data.

        Arguments:
            train {array} -- Labeled data used to train the classifier.
            test {array} -- Data that needs to be labeled. Unlabeled data can be used.
            x_cols {array} -- x columns
            y_cols {array} -- y columns
        """
        xtrain = train.loc[:, x_cols]
        ytrain = train.loc[:, y_cols]
    
        xtest = test.loc[:, x_cols]
        ytest = pd.DataFrame()
        
        xgbClassifier = xgb.XGBClassifier()

        #Fitting train data
        xgbClassifier = xgbClassifier.fit(xtrain, ytrain.values.ravel())
        #Predictions
        ypred=xgbClassifier.predict(xtest)
        
        test["label"] = ypred

        return(test)