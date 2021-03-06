import os
import json

from django.shortcuts import render
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_protect

from .models import Post, StepSession, StepFile, StepPrediction, StepGroup, StepGroupClassiffier
from django.core.files.storage import default_storage
from insoleLib.utils import get_data, get_insole_property, get_number_steps
from insoleLib.classifiers import ClassifierType
from insoleLib.classifierFacade import ClassifierFacade
from insole_charts.utils import ChartContainerFactory, getNumberLabels

from collections import Iterable
from django.shortcuts import redirect


def getUserFile(user, fileName):
    return os.path.join(settings.MEDIA_ROOT, user.username, fileName).strip()
'''
url = "http://www.neo4j.com"
params = {'ref':"mark-blog"}
req.prepare_url(url, params)
'''
def _generateStepFiles(n, user):
    stepFiles = []
    n = 0
    for i in range(0, n):
        stepFiles.append(StepFile(title='____', author=user, footsize=42, productId='KKJFDUD58', steps=150, content=''))
    
    return stepFiles

@login_required(login_url='login')
def home(request):
    user = request.user

    n = 10
    stepFiles = _generateStepFiles(n, user)
    

    context = {
        'stepSessions'      : StepSession.objects.all(),
        'fields'            : ['field1'] * n,
        'stepRows'          : [['a'] * n] * n,
        'files'             : ['filename1'] * 10,
        'stepFiles'         : stepFiles,
        'title'             : 'StepLab'
    }
    return render(request, 'steplab/home.html', context) # request, template and context(arguments)

def getRecordingDetailsContext(fileName, path):
    fields, samples = get_data(path)
    forceLinearData = ChartContainerFactory.createForceLinearChartContainers(samples)
    startTimeData = ChartContainerFactory.createStartTimeLinearChartContainers(samples)
    maxTimeData = ChartContainerFactory.createMaxTimeLinearChartContainers(samples)
    endTimeData = ChartContainerFactory.createEndTimeLinearChartContainers(samples)

    fileContent = {'fields': fields, 'samples': samples}
    context = {
        'fileName'                      :   fileName,
        'fileContent'                   :   fileContent,
        'title'                         :   'Recordings/' + fileName,
        'linearForceChartData'          :   forceLinearData,
        'linearStartTimeChartData'      :   startTimeData,
        'linearMaxTimeChartData'        :   maxTimeData,
        'linearEndTimeChartData'        :   endTimeData,
        'steps'                         :   getNumberLabels(len(forceLinearData[0].data))
    }

    return context

@login_required(login_url='login')
def recordings(request):
    user = request.user
    context = {}
    if request.method == 'POST':
        #stepfiles = request.POST.getlist('new-stepfiles', None) # request.POST.get and request.POST.getlist
        stepfiles = request.FILES.getlist('new-stepfiles')
        print(stepfiles)
        for stepfile in stepfiles:
            #full_filename = os.path.join(settings.MEDIA_ROOT, usrFolder, stepfile.name)
            full_filename = getUserFile(user, stepfile.name)
            path = default_storage.save(full_filename, stepfile)

            fileName = os.path.basename(path)
            productId = get_insole_property(path, "Insole_id")
            size = get_insole_property(path, "Size")
            steps = get_number_steps(path)
            stepFileObj = StepFile(title=fileName, author=user, footsize=size, productId=productId, steps=steps, content='')
            stepFileObj.save()

    elif request.method == 'GET':
        fileName = request.GET.get('filename', '').strip()
        if fileName:
            path = getUserFile(user, fileName)

            if os.path.exists(path):
                context = getRecordingDetailsContext(fileName, path)
                return render(request, 'steplab/recordingDetail.html', context) # request, template and context(arguments)

    stepFiles = StepFile.objects.filter(author=user)

    context = {
                'stepFiles'         : stepFiles,
                'title'             : 'Recordings'
    }

    return render(request, 'steplab/recordings.html', context) # request, template and context(arguments)

@login_required(login_url='login')
def diagnosis(request):
    context = {'title': 'diagnosis'}
    return render(request, 'steplab/diagnosis.html', context)

@login_required(login_url='login')
@csrf_protect
def newDiagnose(request):
    url = "steplab/newDiagnose.html"
    stepFiles = StepFile.objects.filter(author=request.user)
    context = {
        'stepFiles'         : stepFiles,
        'title'             : 'diagnosis',
    }
    
    if request.method == 'GET':
        fileName = request.GET.get('filename', '').strip()
        if fileName:
            path = getUserFile(request.user, fileName)

            if os.path.exists(path):
                context = getRecordingDetailsContext(fileName, path)
                return render(request, 'steplab/recordingDetail.html', context) # request, template and context(arguments)

    if request.method == 'POST':
        postFilesJSON = request.POST.get("analyse", "")
        if postFilesJSON:
            postFiles = json.loads(postFilesJSON)
            classificationMethods = set([ClassifierType.BOOSTING, ClassifierType.KNN, ClassifierType.DNN])
            if postFiles and isinstance(postFiles, Iterable) and len(postFiles) > 0:
                filePaths = []
                for fileName in postFiles:
                    filePaths.append(getUserFile(request.user, fileName.strip()))

                stepPrediction = ClassifierFacade.analyseImbalances(request.user, postFiles, filePaths, classificationMethods, 100)
                if (stepPrediction):
                    return redirect(f'/diagnosis/result?stepPrediction={stepPrediction.id}')

    return render(request, url, context)


@login_required(login_url='login')
@csrf_protect
def diagnosisHistory(request):
    
    # Get the predicitons
    predictions = StepPrediction.objects.filter(user=request.user)

    context = {
        'title'             :   'history',
        'predictions'       :   predictions
    }

    return render(request, 'steplab/history.html', context)

@login_required(login_url='login')
@csrf_protect
def diagnosisResult(request):
    
    url = "steplab/result.html"

    stepPredictionID = request.GET.get("stepPrediction", "")

    context = {
        'title'             : 'result',
        'prediction'       :  {},
    }

    if request.method == 'GET':
        fileName = request.GET.get('filename', '').strip()
        if fileName:
            path = getUserFile(request.user, fileName)

            if os.path.exists(path):
                context = getRecordingDetailsContext(fileName, path)

                return render(request, 'steplab/recordingDetail.html', context) # request, template and context(arguments)


    if stepPredictionID and stepPredictionID.isdigit():
        #  StepPrediction
        stepPredictionID = int(stepPredictionID)
        stepPrediction = StepPrediction.objects.filter(id=stepPredictionID, user=request.user).first()

        # StepGroups
        stepGroups = []
        if (stepPrediction):
            stepGroups = StepGroup.objects.filter(stepPrediction=stepPrediction)

        goodSteps = 0
        badSteps = 0
        threshold = 0.9
        classifiers = set()
        # StepGroupClassiffiers
        groups = {}
        for stepGroup in stepGroups:
            stepGroupClassiffiers = StepGroupClassiffier.objects.filter(stepGroup=stepGroup)
            groups[stepGroup] = stepGroupClassiffiers
            

            for stepGroupClassiffier in stepGroupClassiffiers:
                goodSteps += stepGroupClassiffier.goodSteps
                badSteps += stepGroupClassiffier.badSteps

                # labels for the classification methods
                classifiers.add(stepGroupClassiffier.classifierTypeStr)

        totalSteps =  goodSteps + badSteps     

        predictions = {stepPrediction : groups}
        fileNames = {stepPrediction : json.loads(stepPrediction.files)}
        context ['predictions'] = predictions
        context ['classifiers'] = classifiers
        context ['files'] = fileNames
        context ['result'] = {
            'goodSteps' : goodSteps,
            'goodPercentageSteps' : goodSteps if totalSteps == 0 else goodSteps / totalSteps,
            'badSteps' : badSteps,
            'badPercentageSteps' :  badSteps if totalSteps == 0 else badSteps / totalSteps,
            'threshold' : threshold,
            'result' : stepPrediction.riskFalling,
        }

    return render(request, url, context)
    

def about(request):
    return render(request, 'steplab/about.html', {'title': 'About'})


'''
from django.contrib.auth.models import User
users = User.objects.all()
StepFile.objects.filter(footsize=42)

from steplab.models import Post, StepSession, StepFile, StepPrediction, StepGroup, StepGroupClassiffier
'''