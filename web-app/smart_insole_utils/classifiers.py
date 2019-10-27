from enum import Enum
from abc import ABC, abstractclassmethod
from .utils import get_data_from_files, chunks, combine
from steplab.models import StepPrediction, StepGroup, StepGroupClassiffier
import json

import numpy as np
import pandas as pd
import random

class ClassifierType(Enum):
    '''
    http://www.blog.pythonlibrary.org/2018/03/20/python-3-an-intro-to-enumerations/
    '''
    MOCKED = -1 # for testing
    UNKNOWN = 0
    KNN = 1 # K-nearest neighbor
    DNN = 2 # Deep Neural Network
    
    '''
    TODO Add other classifiers.
    '''

class Classifier(ABC):

    '''
    Recives the input data and returns the predictions for each step.
    '''
    @abstractclassmethod
    def analyseImbalance(inputData): pass


class RandomClassifier(Classifier):

    @staticmethod
    def analyseImbalance(inputData):
        goodSteps = 0
        badSteps = 0
        for i in range(len(inputData)):
            r = random.choice([True, False])
            if r:
                goodSteps += 1
            else:
                badSteps += 1

        riskFalling = RandomClassifier.riskFalling(goodSteps, badSteps)

        return ClassifierAnalysisResult(goodSteps, badSteps, riskFalling)
    
    @staticmethod
    def riskFalling(goodSteps, badSteps):
        return random.choice([True, False])
       

class ClassifierAnalysisResult(object):

    @classmethod
    def __init__(self, goodSteps, badSteps, riskFalling):
        self.goodSteps = goodSteps
        self.badSteps = badSteps
        self.riskFalling = riskFalling

class ClassiffierFacade:

    @staticmethod
    def analyseImbalance(inputData, classifierType = ClassifierType.KNN):
        
        classifierResult = None
        
        if classifierType == ClassifierType.KNN:
            classifierResult = "" # TODO = KNN.analyseImbalance(predictSamples)
        elif classifierType == ClassifierType.DNN:
            classifierResult = "" # TODO = DNN.analyseImbalance(predictSamples)
        elif classifierType == ClassifierType.MOCKED:
            classifierResult = RandomClassifier.analyseImbalance(inputData)
        else:
            raise ValueError('{} is UNKNOW!'.format(classifierType.name))

        return classifierResult

    @staticmethod
    def analyseImbalances(currentUser, fileNames, filePaths, classifierTypes, groupSize):
        
        if len(classifierTypes) <= 0:
            return ValueError(f'No classification mehtod was given!')
        if groupSize <= 0:
            raise ValueError(f'A step group should have more than {groupSize}!')

        # Get Data from files
        fieldsList, samplesList = get_data_from_files(filePaths)
        samples = combine(samplesList)

        # Create a prediction
        stepPrediction = StepPrediction(user=currentUser, files=json.dumps(fileNames))
        stepPrediction.save()

        stepGroupIndex = 0
        origin = 0
        for stepGroupSamples in chunks(samples, groupSize):
            size = len(stepGroupSamples)
            end = origin + size
            # print(f'{origin} {end} {size} {stepGroupIndex}')
            
            # Create the Stepgroups of that predictions
            stepGroup = StepGroup(stepPrediction=stepPrediction, groupIndex=stepGroupIndex, originIndex=origin, endIndex=end, size=size)
            stepGroup.save()
            
            # For every group analyse using the selected classifier types.
            for classifierType in classifierTypes:
                classifierResult = ClassiffierFacade.analyseImbalance(stepGroupSamples, classifierType)
                stepGroupClassiffier = StepGroupClassiffier(
                    stepGroup= stepGroup, 
                    goodSteps= classifierResult.goodSteps, 
                    badSteps= classifierResult.badSteps, 
                    riskFalling= classifierResult.riskFalling,
                    classifierTypeStr= classifierType.name)
                stepGroupClassiffier.save()

            origin += size
            stepGroupIndex += 1
        
        return stepPrediction
