import json
from abc import ABC, abstractclassmethod

from .utils import get_data_from_files, chunks, combine
from .classifiers import ClassifierType
from steplab.models import StepPrediction, StepGroup, StepGroupClassiffier


# implementation of several classifiers
from .ensemble import Ensemble
from .classifiers import RandomClassifier

class ClassifierFacade:

    @staticmethod
    def analyseImbalance(inputData, classifierType = ClassifierType.BOOSTING):
        
        classifierResult = None
        
        if classifierType == ClassifierType.KNN:
            classifierResult = RandomClassifier.analyseImbalance(inputData)
        elif classifierType == ClassifierType.DNN:
            classifierResult = RandomClassifier.analyseImbalance(inputData)
        elif classifierType == ClassifierType.MOCKED:
            classifierResult = RandomClassifier.analyseImbalance(inputData)
        elif classifierType == ClassifierType.BOOSTING:
            classifierResult = Ensemble.analyseImbalance(inputData)
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

        goodSteps = 0
        badSteps = 0
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
                classifierResult = ClassifierFacade.analyseImbalance(stepGroupSamples, classifierType)
                stepGroupClassiffier = StepGroupClassiffier(
                    stepGroup= stepGroup, 
                    goodSteps= classifierResult.goodSteps, 
                    badSteps= classifierResult.badSteps, 
                    riskFalling= classifierResult.riskFalling,
                    classifierTypeStr= classifierType.name)
                stepGroupClassiffier.save()
                goodSteps += classifierResult.goodSteps
                badSteps += classifierResult.badSteps


            origin += size
            stepGroupIndex += 1
        
        totalSteps =  goodSteps + badSteps     

        stepPrediction.riskFalling = False
        if totalSteps > 0:
            stepPrediction.riskFalling = (goodSteps / totalSteps) < 0.9


        stepPrediction.save()

        return stepPrediction
