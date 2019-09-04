class DataColumns:
    """
    Some column combinations, used for filtering data.

    Avoiding constantly rewriting these sets for every classifier.

    """
    
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
    
    def getForceCols():
        """
        Force
        """
        force_cols = ["S0_force","S1_force","S2_force","S3_force","S4_force","S5_force","S6_force"]
        return(force_cols)
    
    def getStartTimeCols():
        """
        Start time
        """
        startT_cols = ["S0_start_time","S1_start_time","S2_start_time","S3_start_time","S4_start_time","S5_start_time","S6_start_time"]
        return(startT_cols)
    
    def getMaxTimeCols():
        """
        Time for max force during step
        """
        maxT_cols = ["S0_max_time","S1_max_time","S2_max_time","S3_max_time","S4_max_time","S5_max_time","S6_max_time"]
        return(maxT_cols)
    
    def getEndTimeCols():
        """
        End time
        """
        endT_cols = ["S0_end_time","S1_end_time","S2_end_time","S3_end_time","S4_end_time","S5_end_time","S6_end_time"]
        return(endT_cols)
    
    def getPhaseCols():
        """
        Step phase data columns
        """
        phases_cols = ["F1_force","F1_time","F2_force","F2_time","F3_force","F3_time"]
        return(phases_cols)
    
    def getPhaseTimeCols():
        """
        Step phase time columns
        """
        phasesT_cols = ["F1_time","F2_time","F3_time"]
        return(phasesT_cols)
    
    def getPhaseForceCols():
        """
        Step phase force columns
        """
        phasesF_cols = ["F1_force","F2_force","F3_force"]
        return(phasesF_cols)
    
    def getAllForceCols():
        """
        All force columns in data
        """
        force_cols = ["S0_force","S1_force","S2_force","S3_force","S4_force","S5_force","S6_force","F1_force","F2_force","F3_force"]
        return(force_cols)
    
    def getStepTimeCols():
        """
        Step time length
        """
        stepL_cols = ["S0_press_time","S1_press_time","S2_press_time","S3_press_time","S4_press_time","S5_press_time","S6_press_time"]
        return(stepL_cols)
    
    def getForceDiffCols():
        """
        Step force differences
        """
        stepFD_cols = ["S0_force_diff","S1_force_diff","S2_force_diff","S3_force_diff","S4_force_diff","S5_force_diff","S6_force_diff"]
        return(stepFD_cols)
    
    def getSelectedCols():
        """
        Features selected for classifiers
        """
        selected_cols = ["S0_force","S1_force","S2_force","S3_force","S4_force","S5_force","S6_force",
                    "S0_press_time","S1_press_time","S2_press_time","S3_press_time","S4_press_time","S5_press_time","S6_press_time",
                    "S0_force_diff","S1_force_diff","S2_force_diff","S3_force_diff","S4_force_diff","S5_force_diff","S6_force_diff","total_force_diff",
                    "Contact_time","force_total"]
        return(selected_cols)
    
    def getSelectedCols2():
        """
        Features selected for classifiers. No force differences
        """
        selected_cols = ["S0_force","S1_force","S2_force","S3_force","S4_force","S5_force","S6_force",
            "S0_press_time","S1_press_time","S2_press_time","S3_press_time","S4_press_time","S5_press_time","S6_press_time",
            "Contact_time","force_total"]
        return(selected_cols)