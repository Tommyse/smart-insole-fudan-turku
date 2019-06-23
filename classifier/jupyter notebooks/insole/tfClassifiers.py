"""
Classifiers that are implemented with tensorflow.


"""


#TODO properly test force difference as a feature


import os
import pandas as pd


#some column combinations, can be used for filtering
values_cols = ["Contact_time","S0_force","S0_start_time","S0_max_time","S0_end_time","S1_force",
               "S1_start_time","S1_max_time","S2_force","S2_start_time","S2_max_time","S2_end_time",
               "S3_force","S3_start_time","S3_max_time","S3_end_time","S4_force","S4_start_time",
               "S4_max_time","S4_end_time","S5_force","S5_start_time","S5_max_time","S5_end_time",
               "S6_force","S6_start_time","S6_max_time","S6_end_time","F1_force","F1_time","F2_force",
               "F2_time","F3_force","F3_time"]

features_cols = ["S0_force","S0_start_time","S0_max_time","S0_end_time","S1_force",
               "S1_start_time","S1_max_time","S2_force","S2_start_time","S2_max_time","S2_end_time",
               "S3_force","S3_start_time","S3_max_time","S3_end_time","S4_force","S4_start_time",
               "S4_max_time","S4_end_time","S5_force","S5_start_time","S5_max_time","S5_end_time",
               "S6_force","S6_start_time","S6_max_time","S6_end_time","F1_force","F1_time","F2_force",
               "F2_time","F3_force","F3_time"]

force_cols = ["S0_force","S1_force","S2_force","S3_force","S4_force","S5_force","S6_force"]
startT_cols = ["S0_start_time","S1_start_time","S2_start_time","S3_start_time","S4_start_time","S5_start_time","S6_start_time"]
maxT_cols = ["S0_max_time","S1_max_time","S2_max_time","S3_max_time","S4_max_time","S5_max_time","S6_max_time"]
endT_cols = ["S0_end_time","S1_end_time","S2_end_time","S3_end_time","S4_end_time","S5_end_time","S6_end_time"]
phases_cols = ["F1_force","F1_time","F2_force","F2_time","F3_force","F3_time"]
phasesT_cols = ["F1_time","F2_time","F3_time"]
phasesF_cols = ["F1_time","F2_time","F3_time"]



#step time length
stepL_cols = ["S0_press_time","S1_press_time","S2_press_time","S3_press_time","S4_press_time","S5_press_time","S6_press_time"]

#Calculating the step time and adding it to the dataframe     (max - min time)
def calculateStepTime(data):
    for row in range(0,len(data)):
        data.loc[row,"S0_press_time"] = (data.loc[row,"S0_end_time"] - data.loc[row,"S0_start_time"])
        data.loc[row,"S1_press_time"] = (data.loc[row,"S1_end_time"] - data.loc[row,"S1_start_time"])
        data.loc[row,"S2_press_time"] = (data.loc[row,"S2_end_time"] - data.loc[row,"S2_start_time"])
        data.loc[row,"S3_press_time"] = (data.loc[row,"S3_end_time"] - data.loc[row,"S3_start_time"])
        data.loc[row,"S4_press_time"] = (data.loc[row,"S4_end_time"] - data.loc[row,"S4_start_time"])
        data.loc[row,"S5_press_time"] = (data.loc[row,"S5_end_time"] - data.loc[row,"S5_start_time"])
        data.loc[row,"S6_press_time"] = (data.loc[row,"S6_end_time"] - data.loc[row,"S6_start_time"])
    return(data)


#Calculating total forcce and adding the column to dataframe
def calculateTotalForce(data):
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


#step force differences
stepFD_cols = ["S0_force_diff","S1_force_diff","S2_force_diff","S3_force_diff","S4_force_diff","S5_force_diff","S6_force_diff"]

#pick same step number and calculate the difference in force to all sensors, (left - right force)
def calculateForceDiff(data):
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



#features selected for learning
selected_cols = ["S0_force","S1_force","S2_force","S3_force","S4_force","S5_force","S6_force",
                "S0_press_time","S1_press_time","S2_press_time","S3_press_time","S4_press_time","S5_press_time","S6_press_time",
                "S0_force_diff","S1_force_diff","S2_force_diff","S3_force_diff","S4_force_diff","S5_force_diff","S6_force_diff","total_force_diff",
                "Contact_time","force_total"]

#Getting all wanted fetures from the raw dataset
def getFeatures(data):
    data = calculateStepTime(data)
    data = calculateForceDiff(data)
    featDF = data.loc[:,selected_cols]
    return(featDF)

#Features for left, right and both separately, separate feet might be a better option
def getAllFeatures(data):
    data = calculateTotalForce(data)
    data = calculateStepTime(data)
    data = calculateForceDiff(data)

    dataL = data.loc[data["Left/Right"] == "L"] #separating to different feet
    dataR = data.loc[data["Left/Right"] == "R"]

    dataL = dataL.loc[:,selected_cols] #selecting only wanted columns
    dataR = dataR.loc[:,selected_cols]
    dataBoth = data.loc[:,selected_cols]

    return(dataL, dataR, dataBoth)