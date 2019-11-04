import numpy as np
import os
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import LeaveOneOut
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from columns import DataColumns

import matplotlib.pyplot as plt
import seaborn as sn

from collections import namedtuple

#rbf  and   poly    kernels

class MlpClassifiers:
    """
    Multi-layer perceptron classifier stuff
    """
    
    @staticmethod
    def findBestAlpha(data, x_cols, y_cols, parameters, alphas):
        """
        Best alpha value for MLP

        Arguments:
            data {array} -- Data
            x_cols {arrray} -- x columns
            y_cols {array} -- y columns
            parameters {namedTuple} -- parameters for the classifier
            alphas {array} -- array of alphas to test
        """

        best_alpha=0
        best_accu=0

        x = data.loc[:, x_cols]
        y = data.loc[:, y_cols]

        #Picking best k
        for a in alphas:
            loo = LeaveOneOut()
            loo.get_n_splits(data)
            n=loo.split(data)
            
            mlpClassifier = MLPClassifier(
            hidden_layer_sizes=parameters.hidden_layer_sizes,
			solver=parameters.solver,
			alpha=a,
   			batch_size=parameters.batch_size,
   			learning_rate=parameters.learning_rate,
    		learning_rate_init=parameters.learning_rate_init,
    		max_iter=parameters.max_iter,
    		random_state=parameters.random_state,
    		verbose=parameters.verbose,
    		early_stopping=parameters.early_stopping,
    		validation_fraction=parameters.validation_fraction
        )

            accuracy_a = []
            real_label = []
            pred_label = []

            for train_index, test_index in n: #Each row is test data once
                xtrain, xtest = x.iloc[train_index], x.iloc[test_index]
                ytrain, ytest = y.iloc[train_index], y.iloc[test_index]

                mlpClassifier.fit(xtrain, ytrain.values.ravel())
                
                ypred=mlpClassifier.predict(xtest)
                pred_label.append(ypred)
                real_label.append(ytest)
                
                acc = accuracy_score(ytest, ypred)
                accuracy_a.append(acc)
                
            avg_acc = np.mean(accuracy_a)
            print(a,": average accuracy ", avg_acc)

            if(avg_acc>best_accu): #Updating best_k if accuracy is better
                best_accu=avg_acc
                best_alpha=a

        print("Best alpha=",best_alpha)
        print("Best accuracy=",best_accu)

        return(best_alpha)

    @staticmethod
    def testMlp(data, parameters, x_cols, y_cols, plots=False):
        """
        testing MLP classifier

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
        
        mlpClassifier = MLPClassifier(
            hidden_layer_sizes=parameters.hidden_layer_sizes,
			solver=parameters.solver,
			alpha=parameters.alpha,
   			batch_size=parameters.batch_size,
   			learning_rate=parameters.learning_rate,
    		learning_rate_init=parameters.learning_rate_init,
    		max_iter=parameters.max_iter,
    		random_state=parameters.random_state,
    		verbose=parameters.verbose,
    		early_stopping=parameters.early_stopping,
    		validation_fraction=parameters.validation_fraction
        )

        accuracy_a = []
        real_label = []
        pred_label = []

        for train_index, test_index in n: #Each row is test data once
            xtrain, xtest = x.iloc[train_index], x.iloc[test_index]
            ytrain, ytest = y.iloc[train_index], y.iloc[test_index]

            mlpClassifier.fit(xtrain, ytrain.values.ravel())
            
            ypred=mlpClassifier.predict(xtest)
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
    def testMlpLearning(data, parameters, x_cols, y_cols, times, plots=False, orig_acc=0, orig_auc=0, file_name_prefix="MLP"):
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
            file_name_prefix {str} -- file name prefix (default: {"MLP"})
        """
        
        accs = []
        aucs = []
        
        for run in range(1,times+1):
            print("Run: ", run, " out of ", times)
            
            suffled_data = data
            suffled_data["label"] = np.random.permutation(suffled_data["label"].values)

            avg_acc, real_label, pred_label = MlpClassifiers.testMlp(suffled_data, parameters, x_cols, y_cols, False)
            
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
    def mlpGetPredictions(train, data, parameters, x_cols=DataColumns.getSelectedCols3()):
        """
        Classify input data
        
        Arguments:
            train {array} -- labeled pandas dataframe
            data {array} -- unlabeled pandas dataframe
            parameters {namedTuple} -- parameters for classifier
        """
        xtrain = train.loc[:, x_cols]
        ytrain = train.loc[:, "label"]
        xdata = data.loc[:, x_cols]

        mlpClassifier = MLPClassifier(
            hidden_layer_sizes=parameters.hidden_layer_sizes,
			solver=parameters.solver,
			alpha=parameters.alpha,
   			batch_size=parameters.batch_size,
   			learning_rate=parameters.learning_rate,
    		learning_rate_init=parameters.learning_rate_init,
    		max_iter=parameters.max_iter,
    		random_state=parameters.random_state,
    		verbose=parameters.verbose,
    		early_stopping=parameters.early_stopping,
    		validation_fraction=parameters.validation_fraction
        )
        
        mlpClassifier.fit(xtrain, ytrain.values.ravel())
        
        ypred=mlpClassifier.predict(xdata)

        return(ypred)

