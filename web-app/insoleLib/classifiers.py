from enum import Enum
from abc import ABC, abstractclassmethod
import random

class ClassifierType(Enum):
    '''
    http://www.blog.pythonlibrary.org/2018/03/20/python-3-an-intro-to-enumerations/
    '''
    MOCKED = -1 # for testing
    UNKNOWN = 0
    KNN = 1 # K-nearest neighbor
    DNN = 2 # Deep Neural Network
    BOOSTING = 3 #This should be used only because it has many classifiers in it already with label voting
    
    '''
    TODO Add other classifiers.
    '''

class ClassifierInterface:

    '''
    Receives the input data and returns a ClassifierAnalysisResult.
    '''
    @abstractclassmethod
    def analyseImbalance(inputData): pass


class RandomClassifier(ClassifierInterface):

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

        return ClassifierAnalysisResult(goodSteps=goodSteps, badSteps=badSteps, riskFalling=riskFalling)
    
    @staticmethod
    def riskFalling(goodSteps, badSteps):
        return random.choice([True, False])
       

class ClassifierAnalysisResult(object):

    @classmethod
    def __init__(self, goodSteps, badSteps, riskFalling):
        self.goodSteps = goodSteps
        self.badSteps = badSteps
        self.riskFalling = riskFalling