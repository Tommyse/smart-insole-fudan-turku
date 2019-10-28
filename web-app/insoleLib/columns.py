class DataColumns:
    """
    Some column combinations, used for filtering data.

    Avoiding constantly rewriting these sets for every classifier.

    """
    
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