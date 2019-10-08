from enum import Enum
from abc import ABC, abstractclassmethod

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
    This method should return an ClassifierAnalysisResult object.
    '''
    @abstractclassmethod
    def analyseInvalance(filePaths): pass

class ClassifierAnalysisResult(object):

    @classmethod
    def __init__(self, fallingRisk):
        self._fallingRisk = fallingRisk

    @classmethod
    def hasFallingRisk(self):
        return self._fallingRisk

class ClassiffierFacade:

    @staticmethod
    def analyseInvalance(filePaths, classifierType = ClassifierType.KNN):
        
        classifierResult = None
        
        if classifierType == ClassifierType.KNN:
            classifierResult = "" # TODO = KNN.analyseInvalance()
        elif classifierType == ClassifierType.DNN:
            classifierResult = "" # TODO = DNN.analyseInvalance()
        elif classifierType == ClassifierType.MOCKED:
            classifierResult = ClassifierAnalysisResult(True)
        else:
            raise ValueError('{} is UNKNOW!'.format(classifierType.name))

        return classifierResult

if __name__ == "__main__":
    classifierResult = ClassiffierFacade.analyseInvalance("", ClassifierType.MOCKED)
    print(classifierResult.hasFallingRisk())
