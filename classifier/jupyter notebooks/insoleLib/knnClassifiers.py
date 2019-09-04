import numpy as np
import os
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from columns import DataColumns

import matplotlib.pyplot as plt
import seaborn as sn

class KnnClassifiers:
    """
    KNN classifier stuff
    """


    def findBestK(data, x_cols, y_cols):
        """
        Non-nested approach to knn. Also for quick accuracy testing

        Arguments:
            data {array} -- Data
        """

        best_k=0
        best_accu=0

        x = data.loc[:, x_cols]
        y = data.loc[:, y_cols]

        #Picking best k
        for k in range(2,11): #from 2 to 10
            loo = LeaveOneOut()
            loo.get_n_splits(data)
            n=loo.split(data)
            knn = KNeighborsClassifier(n_neighbors=k, weights="uniform", metric="euclidean")

            accuracy_a = []
            real_label = []
            pred_label = []

            for train_index, test_index in n: #Each row is test data once
                xtrain, xtest = x.iloc[train_index], x.iloc[test_index]
                ytrain, ytest = y.iloc[train_index], y.iloc[test_index]

                knn.fit(xtrain, ytrain.values.ravel())
                ypred=knn.predict(xtest)
                pred_label.append(ypred)
                real_label.append(ytest)
                
                acc = accuracy_score(ytest, ypred)
                accuracy_a.append(acc)
            avg_acc = np.mean(accuracy_a)
            print(k,": average accuracy ", avg_acc)

            if(avg_acc>best_accu): #Updating best_k if accuracy is better
                best_accu=avg_acc
                best_k=k

        print("Best k=",best_k)
        print("Best accuracy=",best_accu)

        return(best_k)

    def testKnn(data, k, x_cols, y_cols, plots=False):
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
        knn = KNeighborsClassifier(n_neighbors=k, weights="uniform", metric="euclidean")

        accuracy_a = []
        real_label = []
        pred_label = []

        for train_index, test_index in n: #Each row is test data once
            xtrain, xtest = x.iloc[train_index], x.iloc[test_index]
            ytrain, ytest = y.iloc[train_index], y.iloc[test_index]

            knn.fit(xtrain, ytrain.values.ravel())
            ypred=knn.predict(xtest)
            pred_label.append(ypred)
            real_label.append(ytest.values)
            
            acc = accuracy_score(ytest, ypred)
            accuracy_a.append(acc)
        
        avg_acc = np.mean(accuracy_a)
        print("KNN Average accuracy ", avg_acc)
        
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

        return(avg_acc, real_label, pred_label)

    def knnClassify(train, test, k):
        """
        Classify input data
        
        Arguments:
            train {array} -- labeled pandas dataframe
            test {array} -- unlabeled pandas dataframe
            k {int} -- number of nearest neighbors
        """
        xtrain = train[DataColumns.getSelectedCols()]
        ytrain = train["label"]
        xtest = test[DataColumns.getSelectedCols()]
        ytest = pd.DataFrame()

        knn = KNeighborsClassifier(n_neighbors=k, weights="uniform", metric="euclidean")
        knn.fit(xtrain, ytrain.values.ravel())
        ypred=knn.predict(xtest)

        test["label"] = ypred

        return(test)
















    ### Diegos stuff:
    #nested approach
#
#    STEPS_IN_SAMPLE = 5
#
#    all_features = ['App_time', 'Step_number', 'Insole_timer', 'Contact_time', 'S0_force', 'S0_start_time', 'S0_max_time',
#                    'S0_end_time', 'S1_force', 'S1_start_time', 'S1_max_time', 'S1_end_time', 'S2_force', 'S2_start_time',
#                    'S2_max_time', 'S2_end_time', 'S3_force', 'S3_start_time', 'S3_max_time', 'S3_end_time', 'S4_force',
#                    'S4_start_time', 'S4_max_time', 'S4_end_time', 'S5_force', 'S5_start_time', 'S5_max_time',
#                    'S5_end_time', 'S6_force', 'S6_start_time', 'S6_max_time', 'S6_end_time', 'F1_force', 'F1_time',
#                    'F2_force', 'F2_time', 'F3_force', 'F3_time', 'Warning_code', 'Left/Right', 'Size', 'Insole_id']
#
#    '''
#    num_features = ['Contact_time', 'S0_force', 'S0_start_time', 'S0_max_time', 'S0_end_time', 'S1_force', 'S1_start_time',
#                    'S1_max_time', 'S1_end_time', 'S2_force', 'S2_start_time', 'S2_max_time', 'S2_end_time', 'S3_force',
#                    'S3_start_time', 'S3_max_time', 'S3_end_time', 'S4_force', 'S4_start_time', 'S4_max_time',
#                    'S4_end_time',
#                    'S5_force', 'S5_start_time', 'S5_max_time', 'S5_end_time', 'S6_force', 'S6_start_time', 'S6_max_time',
#                    'S6_end_time', 'F1_force', 'F1_time', 'F2_force', 'F2_time', 'F3_force', 'F3_time']
#    '''
#    num_features = ['S0_force', 'S1_force', 'S2_force', 'S3_force',
#                    'S4_force', 'S5_force', 'S6_force']
#
#    label_column_name = 'label'
#
#    num_features_label = num_features + [label_column_name]
#    all_features_label = all_features + [label_column_name]
#
#    print(num_features_label)
#
#
#    def get_files_from_directory(directory, default_label=True):
#        files = []
#        for filename in os.listdir(directory):
#            if filename.endswith('.csv'):
#                files.append((directory + '/' + filename, default_label))
#        return files
#
#
#    def get_samples(files, delimiter=';'):
#        frames = []
#        for file, label in files:
#            frame = pd.read_csv(file, delimiter=delimiter, skiprows=[0, 1])
#            frame[label_column_name] = label
#            frames.append(frame[all_features_label])
#
#        return pd.concat(frames)
#
#
#    files = get_files_from_directory(r"C:\Users\agazor\PycharmProjects\untitled3\data\fall", False)
#    files += get_files_from_directory(r"C:\Users\agazor\PycharmProjects\untitled3\data\fast_walk", True)
#    files += get_files_from_directory(r"C:\Users\agazor\PycharmProjects\untitled3\data\normal_work", True)
#    files += get_files_from_directory(r"C:\Users\agazor\PycharmProjects\untitled3\data\run", True)
#
#    print(files)
#    raw_df = get_samples(files)
#
#    # remove errors
#    raw_df = raw_df[raw_df.Warning_code == '0']
#
#    '''
#
#    # combine the steps
#    def get_sample(rows):
#        n_fields = len(rows[0])
#        avg_values = [0 for i in range(n_fields)]
#        for row in rows:
#            field_index = 0
#            for field in row:
#                avg_values[field_index] += float(field)
#                field_index += 1
#
#        return np.array(avg_values) / n_fields
#
#
#    counter = 0
#    samples = []
#    sample_rows = []
#    numeric_df = raw_df[num_features]
#
#    for index, row in numeric_df.iterrows():
#        if counter == STEPS_IN_SAMPLE:
#            sample = get_sample(sample_rows)
#            samples.append(sample)
#            counter = 0
#            sample_rows = []
#        else:
#            sample_rows.append(row)
#            counter += 1
#
#    samples = np.array(samples)
#    '''
#    refined_df = raw_df[num_features_label]
#    print(refined_df)
#
#    ########
#
#
#    X = np.array(refined_df.as_matrix(columns=num_features))
#    y = np.array(refined_df.as_matrix(columns=[label_column_name]))
#    y = y.flatten()
#
#    print("dims = {} {}".format(X.shape, y.shape))
#
#    # TODO - STANDARDIZATION
#
#    k_range = range(1, 2)  # range(1, 21)
#    folds = range(X.shape[0])
#
#    loo = LeaveOneOut()
#
#    k_best = []
#    y_preds = np.zeros(X.shape[0])
#
#    for fold in folds:
#        # divide the data as test and train data for each Loo Loop
#        test_index = [fold]
#        train_index = np.delete(folds, fold)
#
#        X_train, X_test = X[train_index], X[test_index]
#        y_train, y_test = y[train_index], y[test_index]
#
#        perc = []
#
#        for k in k_range:  # Nested cross validation for hyperparameter selection
#            checkfit = []
#
#            for subtrain_index, validation_index in loo.split(X_train):
#                X_subtrain, X_validation = X[subtrain_index], X[validation_index]
#                y_subtrain, y_validation = y[subtrain_index], y[validation_index]
#
#                knn = KNeighborsClassifier(n_neighbors=k, weights='uniform')
#                knn.fit(X_subtrain, y_subtrain)
#                y_pred = knn.predict(X_validation)
#
#                checkfit.append((metrics.accuracy_score(y_validation, y_pred)))
#            perc.append(np.mean(checkfit))
#
#        k_best.append(k_range[perc.index(max(perc))])
#
#        knn = KNeighborsClassifier(n_neighbors=k_best[fold])
#        knn.fit(X_train, y_train)
#
#        y_preds[fold] = knn.predict(X_test)[0]
#        print("fold: {}/{}".format(fold, folds))
#    accuracy = np.sum(np.diagonal(confusion_matrix(y_preds, y)) / X.shape[0])
#
#    print("Accuracy: ", accuracy)
#