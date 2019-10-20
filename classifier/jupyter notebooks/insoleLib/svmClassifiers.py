import numpy as np
import os
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import LeaveOneOut
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from columns import DataColumns

import matplotlib.pyplot as plt
import seaborn as sn


#rbf  and   poly    kernels

class SvmClassifiers:
    """
    SVM classifier stuff
    """

    @staticmethod
    def testSvm(data, kern, x_cols, y_cols, plots=False):
        """
        Non-nested approach to knn. Also for quick accuracy testing

        Arguments:
            data {array} -- Data
            x_cols {array} -- x columns
            y_cols {array} -- y columns

        Keyword Arguments:
            plots {bool} -- Used for plotting (default: {False})
        """

        x = data.loc[:, x_cols]
        y = data.loc[:, y_cols]

        loo = LeaveOneOut()
        loo.get_n_splits(data)
        n=loo.split(data)
        svmClassifier = svm.SVC(kernel=kern, gamma="scale", degree=4)

        accuracy_a = []
        real_label = []
        pred_label = []

        for train_index, test_index in n: #Each row is test data once
            xtrain, xtest = x.iloc[train_index], x.iloc[test_index]
            ytrain, ytest = y.iloc[train_index], y.iloc[test_index]

            svmClassifier.fit(xtrain, ytrain.values.ravel())
            ypred=svmClassifier.predict(xtest)
            pred_label.append(ypred)
            real_label.append(ytest.values)
            
            acc = accuracy_score(ytest, ypred)
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
        
        if(plots):
            cm = confusion_matrix(real_label_df, pred_label_df)
            cm_df = pd.DataFrame(cm, ["Fall", "Normal"], ["Fall", "Normal"])
            
            sn.set(font_scale=1.5)
            sn.heatmap(cm_df, annot=True, annot_kws={"size": 32}, fmt='d')
            plt.savefig("../figs/svm_heatmap.png", facecolor="w", bbox_inches="tight")
            plt.show()
        
        #Checking accuracy
        print("SVM average accuracy: ", round(avg_acc, 2)) #2 decimals
        
        #More detailed report
        print(classification_report(real_label_df, pred_label_df))

        return(avg_acc, real_label_df, pred_label_df)
    
    @staticmethod
    def testSvmLearning(data, kern, x_cols, y_cols, times, plots=False, orig_acc=0, orig_auc=0, file_name_prefix="KNN"):
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
            orig_acc {double} -- accuracy reference for plots (default: {0})
            orig_auc {double} -- AUC reference for plots (default: {0})
            file_name_prefix {str} -- filename prefix (default: {"KNN"})
        """
        
        accs = []
        aucs = []
        
        for run in range(1,times+1):
            print("Run: ", run, " out of ", times)
            
            suffled_data = data
            suffled_data["label"] = np.random.permutation(suffled_data["label"].values)

            avg_acc, real_label, pred_label = SvmClassifiers.testSvm(suffled_data, kern, x_cols, y_cols, False)
            
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
    def svmGetPredictions(train, data, kern, x_cols=DataColumns.getSelectedCols2()):
        """
        Classify input data
        
        Arguments:
            train {array} -- labeled pandas dataframe
            data {array} -- unlabeled pandas dataframe
            kern {str} -- kernel for classifier
        
        Keyword Arguments:
            x_cols {array} -- x columns (default: {DataColumns.getSelectedCols2()})
        """
        xtrain = train.loc[:, x_cols]
        ytrain = train.loc[:, "label"]
        xdata = data.loc[:, x_cols]

        knn = svm.SVC(kernel=kern, gamma="scale", degree=4)
        knn.fit(xtrain, ytrain.values.ravel())
        ypred=knn.predict(xdata)

        #data["label"] = ypred

        return(ypred)

