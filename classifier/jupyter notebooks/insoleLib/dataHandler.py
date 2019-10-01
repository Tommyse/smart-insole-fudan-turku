import os
import pandas as pd
import numpy as np
from scipy.stats import zscore
from columns import DataColumns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer

class DataHandler:
    """
    All data handling related functions.
    Selecting features and calculating needed values.

    """

    def calculateStepTime(data):
        """
        Calculating the step time and adding it to the dataframe (max - min time)
        
        Arguments:
            data {array} -- data dataframe
        """
        for row in range(0,len(data)):
            data.loc[row,"S0_press_time"] = (data.loc[row,"S0_end_time"] - data.loc[row,"S0_start_time"])
            data.loc[row,"S1_press_time"] = (data.loc[row,"S1_end_time"] - data.loc[row,"S1_start_time"])
            data.loc[row,"S2_press_time"] = (data.loc[row,"S2_end_time"] - data.loc[row,"S2_start_time"])
            data.loc[row,"S3_press_time"] = (data.loc[row,"S3_end_time"] - data.loc[row,"S3_start_time"])
            data.loc[row,"S4_press_time"] = (data.loc[row,"S4_end_time"] - data.loc[row,"S4_start_time"])
            data.loc[row,"S5_press_time"] = (data.loc[row,"S5_end_time"] - data.loc[row,"S5_start_time"])
            data.loc[row,"S6_press_time"] = (data.loc[row,"S6_end_time"] - data.loc[row,"S6_start_time"])
        return(data)

    def calculateTotalForce(data):
        """
        Calculating total forcce and adding the column to dataframe
        
        Arguments:
            data {array} -- data dataframe
        """
        for row in range(0,len(data)):
            totalF = 0
            totalF = totalF + data.loc[row,"S0_force"]
            totalF = totalF + data.loc[row,"S1_force"]
            totalF = totalF + data.loc[row,"S2_force"]
            totalF = totalF + data.loc[row,"S3_force"]
            totalF = totalF + data.loc[row,"S4_force"]
            totalF = totalF + data.loc[row,"S5_force"]
            totalF = totalF + data.loc[row,"S6_force"]
            data.loc[row,"force_total"] = totalF
        return(data)

    def calculateMedianDifference(data):
        """
        Trying to normalize the forces in input data. Calculating the difference to median
        Input should be data from one session from one person. Not an array with data from many persons
        
        Arguments:
            data {array} -- data dataframe
        """
        cols = DataColumns.getAllForceCols()
        for col in cols:
            col_median = data.loc[:,col].median(axis=1, skipna=True)
            for row in range(0,len(data)):
                data.loc[row,col] = (data.loc[row,col] - col_median)
        return(data)

    #step force differences
    stepFD_cols = DataColumns.getForceDiffCols()

    #
    def calculateForceDiff(data):
        """
        Comparing forces between steps.
        Pick same step number and calculate the difference in force to all sensors, (left - right force)
        
        Arguments:
            data {array} -- data dataframe
        """
        forceDF = pd.DataFrame(columns=["Step_number","S0_force_diff","S1_force_diff","S2_force_diff","S3_force_diff","S4_force_diff","S5_force_diff","S6_force_diff","total_force_diff",])
        unique_steps = data.loc[:,"Step_number"].unique()
        for stepNum in unique_steps:
            stepData = data.loc[data["Step_number"] == stepNum]
            #print(stepData)
            stepData = stepData.reset_index(drop=True)
            #print(stepData)

            stepDataL = stepData.loc[stepData["Left/Right"] == "L"]
            stepDataR = stepData.loc[stepData["Left/Right"] == "R"]
            row = stepNum - 1
            forceDF.loc[row,"Step_number"] = stepNum
            forceDF.loc[row,"S0_force_diff"] = (stepDataL.loc[:,"S0_force"].item() - stepDataR.loc[:,"S0_force"].item())
            forceDF.loc[row,"S1_force_diff"] = (stepDataL.loc[:,"S1_force"].item() - stepDataR.loc[:,"S1_force"].item())
            forceDF.loc[row,"S2_force_diff"] = (stepDataL.loc[:,"S2_force"].item() - stepDataR.loc[:,"S2_force"].item())
            forceDF.loc[row,"S3_force_diff"] = (stepDataL.loc[:,"S3_force"].item() - stepDataR.loc[:,"S3_force"].item())
            forceDF.loc[row,"S4_force_diff"] = (stepDataL.loc[:,"S4_force"].item() - stepDataR.loc[:,"S4_force"].item())
            forceDF.loc[row,"S5_force_diff"] = (stepDataL.loc[:,"S5_force"].item() - stepDataR.loc[:,"S5_force"].item())
            forceDF.loc[row,"S6_force_diff"] = (stepDataL.loc[:,"S6_force"].item() - stepDataR.loc[:,"S6_force"].item())
            forceDF.loc[row,"total_force_diff"] = ((stepDataL.loc[:,"S0_force"].item() - stepDataR.loc[:,"S0_force"].item()) + 
                (stepDataL.loc[:,"S1_force"].item() - stepDataR.loc[:,"S1_force"].item()) + 
                (stepDataL.loc[:,"S2_force"].item() - stepDataR.loc[:,"S2_force"].item()) + 
                (stepDataL.loc[:,"S3_force"].item() - stepDataR.loc[:,"S3_force"].item()) + 
                (stepDataL.loc[:,"S4_force"].item() - stepDataR.loc[:,"S4_force"].item()) + 
                (stepDataL.loc[:,"S5_force"].item() - stepDataR.loc[:,"S5_force"].item()) + 
                (stepDataL.loc[:,"S6_force"].item() - stepDataR.loc[:,"S6_force"].item()))

        forceDF_left = forceDF
        forceDF_left.loc[:,"Left/Right"] = "L"

        forceDF_right = forceDF #difference values should be inverted
        forceDF_right.loc[:,"S0_force_diff"] = -(forceDF_right.loc[:,"S0_force_diff"])
        forceDF_right.loc[:,"S1_force_diff"] = -(forceDF_right.loc[:,"S1_force_diff"])
        forceDF_right.loc[:,"S2_force_diff"] = -(forceDF_right.loc[:,"S2_force_diff"])
        forceDF_right.loc[:,"S3_force_diff"] = -(forceDF_right.loc[:,"S3_force_diff"])
        forceDF_right.loc[:,"S4_force_diff"] = -(forceDF_right.loc[:,"S4_force_diff"])
        forceDF_right.loc[:,"S5_force_diff"] = -(forceDF_right.loc[:,"S5_force_diff"])
        forceDF_right.loc[:,"S6_force_diff"] = -(forceDF_right.loc[:,"S6_force_diff"])
        forceDF_right.loc[:,"total_force_diff"] = -(forceDF_right.loc[:,"total_force_diff"])
        forceDF_right.loc[:,"Left/Right"] = "R"

        #forceDF_left_g = forceDF_left.groupby(["Step_number","Left/Right"])
        #forceDF_right_g = forceDF_right.groupby(["Step_number","Left/Right"])
        #data = pd.merge(data,forceDF_left_g,left_index=True,right_index=True)
        #data = pd.merge(data,forceDF_right_g,left_index=True,right_index=True)

        data = pd.merge(data, forceDF_left, on=["Step_number","Left/Right"], suffixes = ("","_x")) #left_index=True, right_index=False,
        data = pd.merge(data, forceDF_right, on=["Step_number","Left/Right"], suffixes = ("","_x"))
        #data = data.join(forceDF_left, how="right", on=["Step_number","Left/Right"])
        #data = data.join(forceDF_right, how="right", on=["Step_number","Left/Right"])
        #print(data.head)
        data=data.rename(columns = {"S0_force_diff_x":"S0_force_diff","S1_force_diff_x":"S1_force_diff","S2_force_diff_x":"S2_force_diff",
                                    "S3_force_diff_x":"S3_force_diff","S4_force_diff_x":"S4_force_diff","S5_force_diff_x":"S5_force_diff","S6_force_diff_x":"S6_force_diff"})
        #print(data.head)
        return(data)

    def getFeatures(data):
        """
        Getting all wanted fetures from the raw dataset
        
        Arguments:
            data {array} -- data dataframe
        """
        selected_cols = DataColumns.getSelectedCols() #features selected for learning
        data = calculateStepTime(data)
        data = calculateForceDiff(data)
        featDF = data.loc[:,selected_cols]
        return(featDF)

    def getAllFeatures(data):
        """
        Features for left, right and both separately, separate feet might be a better option
        
        Arguments:
            data {array} -- data dataframe
        """
        selected_cols = DataColumns.getSelectedCols() #features selected for learning
        data = calculateTotalForce(data)
        data = calculateStepTime(data)
        data = calculateForceDiff(data)

        dataL = data.loc[data["Left/Right"] == "L"] #separating to different feet
        dataR = data.loc[data["Left/Right"] == "R"]

        dataL = dataL.loc[:,selected_cols] #selecting only wanted columns
        dataR = dataR.loc[:,selected_cols]
        dataBoth = data.loc[:,selected_cols]

        return(dataL, dataR, dataBoth)

    #
    def getSplitData(data, x_cols, y_cols):
        """
        Splitting data to x and y
        
        Arguments:
            data {array} -- data dataframe
            x_cols {array} -- x column names
            y_cols {array} -- y column names
        """
        x = data.loc[:,x_cols]
        y = data.loc[:,y_cols]
        return(x,y)

    def zscoreStandardize(data):
        """
        Needs some fourther thinking and changes. Which columns need z-score normalization?
        
        Arguments:
            data {array} -- data dataframe
        """
        
        return(data.apply(zscore))
    
    def minmaxStandardizeForces(data):
        """
        Normalizing forces from data. Very sensitive to outliers though.
        
        Arguments:
            data {array} -- data dataframe
        """
        fCols = DataColumns.getAllForceCols()
        force_data = data.loc[:, fCols]
        other_data = data.drop(fCols, axis=1)
        
        minmax = MinMaxScaler()
        minmax.fit(force_data)
        force_normalized = pd.DataFrame(minmax.transform(force_data), columns=fCols)

        other_data[fCols] = force_normalized.loc[:, fCols]
        print("result ", other_data)
        
        return(other_data)
    
    def quantilesStandardizeForcesTest(data):
        """
        Needs proper testing
        
        Arguments:
            data {array} -- data dataframe
        """
        fCols = DataColumns.getAllForceCols()
        force_data = data.loc[:,fCols]
        other_data = data.drop(fCols, axis=1)
        
        quantile = QuantileTransformer(n_quantiles=10, random_state=0, output_distribution="uniform")
        quantile.fit_transform(force_data)
        force_normalized = quantile.transform(force_data)
        
        result = pd.concat([other_data,force_normalized], axis=0, ignore_index=True)
        
        return(result)
    
    def quantilesGaussianStandardizeForcesTest(data):
        """
        Needs proper testing
        
        Arguments:
            data {array} -- data dataframe
        """
        fCols = DataColumns.getAllForceCols()
        force_data = data.loc[:,fCols]
        other_data = data.drop(fCols, axis=1)
        
        quantile = QuantileTransformer(n_quantiles=10, random_state=0, output_distribution="normal")
        quantile.fit_transform(force_data)
        force_normalized = quantile.transform(force_data)
        
        result = pd.concat([other_data,force_normalized], axis=0, ignore_index=True)
        
        return(result)
    
    #scaling new data forces to same scale as the training data
    def scaleData(train, data):
        
        return(data)
    
    #generating random dataset
    def genRandomDatasets(data, amount, dropAmount):
        
        datasets = []
        
        for a in range(0,amount):
            dataset = pd.DataFrame(data)
            dataset = dataset.sample(frac=1).reset_index(drop=True) #suffling
            dataset = dataset.drop(dataset.tail(dropAmount).index)
            
            counts = dataset["label"].value_counts()
            
            normal_dataset = dataset.loc[dataset["label"] == "Normal"]
            fall_dataset = dataset.loc[dataset["label"] == "Fall"]
            
            #Fixing fall and normal data balance, counter indexes might change based on which is larger
            if(counts[0] > counts[1]): #normal
                #print(counts.index[0], " exceeds ", counts.index[1], " count. Balancing data...")
                difference = (counts[0] - counts[1])-130 #need more normal steps to train the classifiers
                normal_dataset = normal_dataset.sample(frac=1).reset_index(drop=True) #suffling
                normal_dataset = normal_dataset.drop(normal_dataset.tail(difference).index)
            elif(counts[1] > counts[0]): #fall
                #print(counts.index[1], " exceeds ", counts.index[0], " count. Balancing data...")
                difference = (counts[1] - counts[0])
                fall_dataset = fall_dataset.sample(frac=1).reset_index(drop=True) #suffling
                fall_dataset = fall_dataset.drop(fall_dataset.tail(difference).index)
                
            #Combining the normal and fall data again
            results_df = normal_dataset
            results_df = results_df.append(fall_dataset)
            results_df = results_df.sample(frac=1).reset_index(drop=True) #suffling again
            
            #print("results_df", results_df)
            res_counts = results_df["label"].value_counts()
            #print("res_counts", res_counts)
            
            datasets.append(results_df)
            
        return(datasets)
    
    #generating random datasets
    #for using same classifier multiple times
    # lots of "Normal" rows -> unbalanced data
    def genRandomDatasetsOld(data, amount, dropAmount):
        
        datasets = []
        
        for a in range(0,amount):
            dataset = pd.DataFrame(data)
            dataset = dataset.sample(frac=1).reset_index(drop=True) #suffling
            dataset.drop(dataset.tail(dropAmount).index,inplace=True)
            datasets.append(dataset)
            
        return(datasets)
    
    #Calculating "Fall" prediction accuracy. Ignoring normal labels
    def getFallAccuracy(real, pred):
        counter = 0
        correct = 0
        wrong = 0
        
        if(len(real) != len(pred)):
            raise Exception("Real labels and prediction labels length mismatch")
        
        for row in range(0, real):
            if(real[row] == "Fall" and pred[row] == "Fall"):
                correct = correct + 1
            else:
                wrong = wrong + 1
            counter = counter + 1
        
        return(correct/counter)
    
    #Calculating "Normal" prediction accuracy
    def getNormalAccuracy(real, pred):
        counter = 0
        correct = 0
        wrong = 0
        
        if(len(real) != len(pred)):
            raise Exception("Real labels and prediction labels length mismatch")
        
        for row in range(0, real):
            if(real[row] == "Normal" and pred[row] == "Normal"):
                correct = correct + 1
            else:
                wrong = wrong + 1
            counter = counter + 1
        
        return(correct/counter)