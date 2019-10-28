import os
import pandas as pd
import numpy as np
from sklearn.semi_supervised import label_propagation
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

class UnlabeledClassifier:
    """
    Testing.
    Classifiers that labels unknown data for semi-supervised learning.
    
    """

    @staticmethod
    def findBestK(x, y, min_k, max_k, xcols, ycols):
        """
        Picking the best k for KNN, non-nested approach
        
        Arguments:
            x {array} -- x data
            y {array} -- y data
            min_k {int} -- min k to test
            max_k {int} -- max k to test
            xcols {array} -- x columns
            ycols {array} -- y columns
        """
        k_range = range(min_k,max_k) #k values to test
        best_k = 0
        best_acc = 0 #best accuracy value
    
        for k in k_range:
            #print("K =", k)
            knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
            ypred_a=[] #predictions array
            rlabel_a=[] #real labels array
    
            loo = LeaveOneOut()
    
            for train_i, test_i in loo.split(x): #train test splits
                xtrain, xtest = x.iloc[train_i], x.iloc[test_i]
                ytrain, ytest = y.iloc[train_i], y.iloc[test_i]
    
                knn.fit(xtrain, ytrain.values.ravel())
                ypred=knn.predict(xtest)
    
                ypred_a.append(ypred) #prediction
                rlabel_a.append(ytest.values) #real value
    
            rlabel_df = pd.DataFrame(columns=ycols) #Real labels df
            ypred_df = pd.DataFrame(columns=ycols) #Predicted labels df
    
            for i in range(0, len(ypred_a)): #Forming a dataframe from the list
                ypred_df.loc[i] = ypred_a[i][0]
    
            for i in range(0, len(rlabel_a)): #Forming a dataframe from the list
                rlabel_df.loc[i] = rlabel_a[i][0][0]
    
            acc = accuracy_score(rlabel_df.values.ravel(), ypred_df.values.ravel())
            if(acc > best_acc):
                best_k = k
                best_acc = acc
            
        return(best_k)
    
    @staticmethod
    def knnLabel(labeled, unlabeled, xcols, ycols, k):
        """
        Labeling the unlabeled data with KNN
        
        Arguments:
            labeled {array} -- labeled data
            unlabeled {array} -- unlabeled data
            xcols {array} -- x columns
            ycols {array} -- y columns
            k {int} -- amount of neighbors
        """
        labeled = pd.DataFrame(labeled)
        unlabeled = pd.DataFrame(unlabeled)
        cols = xcols + ycols
        labeled = labeled.loc[:, cols]
        for row in range(0, len(unlabeled)): #Going through all rows in unlabeled dataframe
            xtrain = labeled.loc[:, xcols]
            ytrain = labeled.loc[:, ycols]
            
            knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean") #resetting knn for every run, just in case
            knn.fit(xtrain, ytrain.values.ravel()) #Fitting currently labeled data
    
            unlabeled_row = pd.DataFrame(unlabeled.loc[row, :]).values
            unlabeled_np = np.array(unlabeled_row).reshape(1,-1) #reshaping to 2d since it is required
            ypred = knn.predict(unlabeled_np) #Predicting the label
    
            unlabeled_row = pd.DataFrame(unlabeled_np)
            unlabeled_row = unlabeled_row.reset_index(drop=True)
            unlabeled_row.loc[0, "label"] = ypred[0] #Adding prediction to row data
            unlabeled_row.columns = cols
    
            #print("unlabeled_row with prediction label: ")
            #print(unlabeled_row)
    
            labeled = labeled.append(unlabeled_row, sort=False).reset_index(drop=True) #Adding labeled row to data (using it in other predictions)
    
        return(labeled)
    
    @staticmethod
    def labelSpreading(labeled, unlabeled, xcols, ycols, alpha_v=0.8):
        """
        KNN label spreading testing
        
        Arguments:
            labeled {array} -- labeled data
            unlabeled {array} -- unlabeled data
            xcols {array} -- x columns
            ycols {array} -- y columns
        
        Keyword Arguments:
            alpha_v {double} -- alpha parameter (default: {0.8})
        """
    
        x = labeled.loc[:, xcols]
        y = labeled.loc[:, ycols]
    
        #Using LabelSpreading
        label_spread = label_propagation.LabelSpreading(kernel="knn", alpha=alpha_v)
        label_spread.fit(x, y.values.ravel())
    
        #output labels
        preds = label_spread.predict(unlabeled)
    
        unlabeled.loc[:, "label"] = preds
    
        labeled = pd.concat([labeled, unlabeled], sort=False).reset_index(drop=True) #Combining
    
        return(labeled)