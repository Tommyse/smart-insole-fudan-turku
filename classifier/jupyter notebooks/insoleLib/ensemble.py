import os
import pandas as pd
from scipy.stats import zscore
from columns import DataColumns

class Ensemble:
    """
    Ensemble learning class.
    Uses multiple different classifiers to classify the input data.

    """
#https://towardsdatascience.com/ensemble-methods-bagging-boosting-and-stacking-c9214a10a205

    #some column combinations, can be used for filtering
    values_cols = DataColumns.getValuesCols()
    features_cols = DataColumns.getBasicFeaturesCols()
    force_cols = DataColumns.getForceCols()
    startT_cols = DataColumns.getStartTimeCols()
    maxT_cols = DataColumns.getMaxTimeCols()
    endT_cols = DataColumns.getEndTimeCols()
    phases_cols = DataColumns.getPhaseCols()
    phasesT_cols = DataColumns.getPhaseTimeCols()
    phasesF_cols = DataColumns.getPhaseForceCols()
    stepL_cols = DataColumns.getStepTimeCols()

    #Bagging
    #Most popular label from multiple different classifiers wins
    #Each model has same weight for their vote
    #Training with randomly drawn sets from training set
    def bagging(train, test):

        return(test)

    #Boosting
    #TODO?
    def adaboost(train, test):

        return(test)