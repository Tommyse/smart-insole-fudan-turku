class DataColumns:
    """
    Some column combinations, used for filtering data.

    Avoiding constantly rewriting these sets for every classifier.

    """
    all_columns = [
        "App_time","Step_number","Insole_timer","Contact_time","ESW_timer",
        "S0_force","S0_start_time","S0_max_time","S0_end_time",
        "S1_force","S1_start_time","S1_max_time","S1_end_time",
        "S2_force","S2_start_time","S2_max_time","S2_end_time",
        "S3_force","S3_start_time","S3_max_time","S3_end_time",
        "S4_force","S4_start_time","S4_max_time","S4_end_time",
        "S5_force","S5_start_time","S5_max_time","S5_end_time",
        "S6_force","S6_start_time","S6_max_time","S6_end_time",
        "F1_force","F1_time","F2_force","F2_time","F3_force","F3_time",
        "Warning_code","Lifetime_steps","Day","Month","Year",
        "Left/Right","Size","Insole_id","Battery","MAC"
    ]

    column_types = {
        "App_time" : 'str',
        "Step_number" : 'int32',
        "Insole_timer" : 'int32',
        "Contact_time" : 'int32',
        "ESW_timer" : 'int32',
        "S0_force" : 'int32',
        "S0_start_time" : 'int32',
        "S0_max_time" : 'int32',
        "S0_end_time" : 'int32',
        "S1_force" : 'int32',
        "S1_start_time" : 'int32',
        "S1_max_time" : 'int32',
        "S1_end_time" : 'int32',
        "S2_force" : 'int32',
        "S2_start_time" : 'int32',
        "S2_max_time" : 'int32',
        "S2_end_time" : 'int32',
        "S3_force" : 'int32',
        "S3_start_time" : 'int32',
        "S3_max_time" : 'int32',
        "S3_end_time" : 'int32',
        "S4_force" : 'int32',
        "S4_start_time" : 'int32',
        "S4_max_time" : 'int32',
        "S4_end_time" : 'int32',
        "S5_force" : 'int32',
        "S5_start_time" : 'int32',
        "S5_max_time" : 'int32',
        "S5_end_time" : 'int32',
        "S6_force" : 'int32',
        "S6_start_time" : 'int32',
        "S6_max_time" : 'int32',
        "S6_end_time" : 'int32',
        "F1_force" : 'int32',
        "F1_time" : 'int32',
        "F2_force" : 'int32',
        "F2_time" : 'int32',
        "F3_force" : 'int32',
        "F3_time" : 'int32',
        "Warning_code" : 'int32',
        "Lifetime_steps" : 'int32',
        "Day" : 'int32',
        "Month" : 'int32',
        "Year" : 'int32',
        "Left/Right" : 'str',
        "Size" : 'int32',
        "Insole_id" : 'int32',
        "Battery" : 'int32',
        "MAC" : 'str'
    }

    all_column_index = {
        "App_time": 0, "Step_number": 1, "Insole_timer": 2, "Contact_time": 3, "ESW_timer": 4,
        "S0_force": 5, "S0_start_time": 6, "S0_max_time": 7, "S0_end_time": 8,
        "S1_force": 9, "S1_start_time": 10, "S1_max_time": 11, "S1_end_time": 12,
        "S2_force": 13, "S2_start_time": 14, "S2_max_time": 15, "S2_end_time": 16,
        "S3_force": 17, "S3_start_time": 18, "S3_max_time": 19, "S3_end_time": 20,
        "S4_force": 21, "S4_start_time": 22, "S4_max_time": 23, "S4_end_time": 24,
        "S5_force": 25, "S5_start_time": 26, "S5_max_time": 27, "S5_end_time": 28,
        "S6_force": 29, "S6_start_time": 30, "S6_max_time": 31, "S6_end_time": 32,
        "F1_force": 33, "F1_time": 34, "F2_force": 35, "F2_time": 36, "F3_force": 37, "F3_time": 38,
        "Warning_code": 39, "Lifetime_steps": 40, "Day": 41, "Month": 42, "Year": 43,
        "Left/Right": 44, "Size": 45, "Insole_id": 46, "Battery": 47, "MAC": 48
     }

    @staticmethod
    def getAllCols():
        return DataColumns.all_columns

    @staticmethod
    def getColumnIndex(columnName):
        return DataColumns.all_column_index[columnName]

    @staticmethod
    def getColumnType(columnName):
        return DataColumns.column_types[columnName]

    @staticmethod
    def getColTypes():
        return DataColumns.column_types

    @staticmethod
    def getValuesCols():
        """
        All data columns with numeric values
        """
        values_cols = ["Contact_time","S0_force","S0_start_time","S0_max_time","S0_end_time","S1_force",
                   "S1_start_time","S1_max_time","S2_force","S2_start_time","S2_max_time","S2_end_time",
                   "S3_force","S3_start_time","S3_max_time","S3_end_time","S4_force","S4_start_time",
                   "S4_max_time","S4_end_time","S5_force","S5_start_time","S5_max_time","S5_end_time",
                   "S6_force","S6_start_time","S6_max_time","S6_end_time","F1_force","F1_time","F2_force",
                   "F2_time","F3_force","F3_time"]
        return(values_cols)
    
    @staticmethod
    def getBasicFeaturesCols():
        """
        Basic features
        """
        features_cols = ["S0_force","S0_start_time","S0_max_time","S0_end_time","S1_force",
                   "S1_start_time","S1_max_time","S2_force","S2_start_time","S2_max_time","S2_end_time",
                   "S3_force","S3_start_time","S3_max_time","S3_end_time","S4_force","S4_start_time",
                   "S4_max_time","S4_end_time","S5_force","S5_start_time","S5_max_time","S5_end_time",
                   "S6_force","S6_start_time","S6_max_time","S6_end_time","F1_force","F1_time","F2_force",
                   "F2_time","F3_force","F3_time"]
        return(features_cols)
    
    @staticmethod
    def getForceCols():
        """
        Force
        """
        force_cols = ["S0_force","S1_force","S2_force","S3_force","S4_force","S5_force","S6_force"]
        return(force_cols)
    
    @staticmethod
    def getStartTimeCols():
        """
        Start time
        """
        startT_cols = ["S0_start_time","S1_start_time","S2_start_time","S3_start_time","S4_start_time","S5_start_time","S6_start_time"]
        return(startT_cols)
    
    @staticmethod
    def getMaxTimeCols():
        """
        Time for max force during step
        """
        maxT_cols = ["S0_max_time","S1_max_time","S2_max_time","S3_max_time","S4_max_time","S5_max_time","S6_max_time"]
        return(maxT_cols)
    
    @staticmethod
    def getEndTimeCols():
        """
        End time
        """
        endT_cols = ["S0_end_time","S1_end_time","S2_end_time","S3_end_time","S4_end_time","S5_end_time","S6_end_time"]
        return(endT_cols)
    
    @staticmethod
    def getPhaseCols():
        """
        Step phase data columns
        """
        phases_cols = ["F1_force","F1_time","F2_force","F2_time","F3_force","F3_time"]
        return(phases_cols)
    
    @staticmethod
    def getPhaseTimeCols():
        """
        Step phase time columns
        """
        phasesT_cols = ["F1_time","F2_time","F3_time"]
        return(phasesT_cols)
    
    @staticmethod
    def getPhaseForceCols():
        """
        Step phase force columns
        """
        phasesF_cols = ["F1_force","F2_force","F3_force"]
        return(phasesF_cols)
    
    @staticmethod
    def getAllForceCols():
        """
        All force columns in data
        """
        force_cols = ["S0_force","S1_force","S2_force","S3_force","S4_force","S5_force","S6_force","F1_force","F2_force","F3_force"]
        return(force_cols)
    
    @staticmethod
    def getStepTimeCols():
        """
        Step time length
        """
        stepL_cols = ["S0_press_time","S1_press_time","S2_press_time","S3_press_time","S4_press_time","S5_press_time","S6_press_time"]
        return(stepL_cols)
    
    @staticmethod
    def getForceDiffCols():
        """
        Step force differences
        """
        stepFD_cols = ["S0_force_diff","S1_force_diff","S2_force_diff","S3_force_diff","S4_force_diff","S5_force_diff","S6_force_diff"]
        return(stepFD_cols)
    
    @staticmethod
    def getSelectedCols():
        """
        Features selected for classifiers
        """
        selected_cols = ["S0_force","S1_force","S2_force","S3_force","S4_force","S5_force","S6_force",
                    "S0_press_time","S1_press_time","S2_press_time","S3_press_time","S4_press_time","S5_press_time","S6_press_time",
                    "S0_force_diff","S1_force_diff","S2_force_diff","S3_force_diff","S4_force_diff","S5_force_diff","S6_force_diff","total_force_diff",
                    "Contact_time","force_total"]
        return(selected_cols)
    
    @staticmethod
    def getSelectedCols2():
        """
        Features selected for classifiers. No force differences
        """
        selected_cols = ["S0_force","S1_force","S2_force","S3_force","S4_force","S5_force","S6_force",
            "S0_press_time","S1_press_time","S2_press_time","S3_press_time","S4_press_time","S5_press_time","S6_press_time",
            "Contact_time","force_total"]
        return(selected_cols)
    
    @staticmethod
    def getSelectedCols2andY():
        """
        Features selected for classifiers. No force differences
        """
        selected_cols = ["S0_force","S1_force","S2_force","S3_force","S4_force","S5_force","S6_force",
            "S0_press_time","S1_press_time","S2_press_time","S3_press_time","S4_press_time","S5_press_time","S6_press_time",
            "Contact_time","force_total","label"]
        return(selected_cols)
    
    @staticmethod
    def getSelectedCols3():
        selected_cols = ["force_median", "phase_force_median", #"press_max_time_median", "press_start_time_median", "press_end_time_median",
            "S0_press_time","S1_press_time","S2_press_time","S3_press_time","S4_press_time","S5_press_time","S6_press_time",
            "Contact_time","force_total"]
        return(selected_cols)
    
    @staticmethod
    def getSelectedCols4():
        selected_cols = ["force_mean", "phase_force_mean", #"press_max_time_mean", "press_start_time_mean", "press_end_time_mean",
            "S0_press_time","S1_press_time","S2_press_time","S3_press_time","S4_press_time","S5_press_time","S6_press_time",
            "Contact_time","force_total"]
        return(selected_cols)
    
    @staticmethod
    def getSelectedCols5():
        selected_cols = ["force_variance", "phase_force_variance", #"press_max_time_variance", "press_start_time_variance", "press_end_time_variance",
            "S0_press_time","S1_press_time","S2_press_time","S3_press_time","S4_press_time","S5_press_time","S6_press_time",
            "Contact_time","force_total"]
        return(selected_cols)
    
    @staticmethod
    def getSelectedCols6():
        selected_cols = ["force_median_diff_avg", "phase_force_median_diff_avg", #"press_max_time_median_diff_avg", "press_start_time_median_diff_avg", "press_end_time_median_diff_avg",
            "S0_press_time","S1_press_time","S2_press_time","S3_press_time","S4_press_time","S5_press_time","S6_press_time",
            "Contact_time","force_total"]
        return(selected_cols)
    
    @staticmethod
    def getSelectedCols7():
        selected_cols = ["force_mean_diff_avg", "phase_force_mean_diff_avg", #"press_max_time_mean_diff_avg", "press_start_time_mean_diff_avg", "press_end_time_mean_diff_avg",
            "S0_press_time","S1_press_time","S2_press_time","S3_press_time","S4_press_time","S5_press_time","S6_press_time",
            "Contact_time","force_total"]
        return(selected_cols)
    
    """ old as backup
    def getSelectedCols3():

        selected_cols = ["force_median",
            "S0_press_time","S1_press_time","S2_press_time","S3_press_time","S4_press_time","S5_press_time","S6_press_time",
            "Contact_time","force_total"]
        return(selected_cols)
    
    def getSelectedCols4():

        selected_cols = ["force_mean",
            "S0_press_time","S1_press_time","S2_press_time","S3_press_time","S4_press_time","S5_press_time","S6_press_time",
            "Contact_time","force_total"]
        return(selected_cols)
    
    def getSelectedCols5():

        selected_cols = ["force_variance",
            "S0_press_time","S1_press_time","S2_press_time","S3_press_time","S4_press_time","S5_press_time","S6_press_time",
            "Contact_time","force_total"]
        return(selected_cols)
    
    def getSelectedCols6():

        selected_cols = ["force_median_diff_avg",
            "S0_press_time","S1_press_time","S2_press_time","S3_press_time","S4_press_time","S5_press_time","S6_press_time",
            "Contact_time","force_total"]
        return(selected_cols)
    
    def getSelectedCols7():

        selected_cols = ["force_mean_diff_avg",
            "S0_press_time","S1_press_time","S2_press_time","S3_press_time","S4_press_time","S5_press_time","S6_press_time",
            "Contact_time","force_total"]
        return(selected_cols)
    """