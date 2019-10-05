import os

from django.shortcuts import render
from django.conf import settings
from django.contrib.auth.decorators import login_required
from .models import Post
from .models import StepSession
from .models import stepFile
from django.core.files.storage import default_storage

def _generateStepFiles(n):
    stepFiles = []

    for i in range(0, n):
        stepFiles.append(stepFile('University File', 'Diego', '43', '45as5df4', '500', None))
    
    return stepFiles

@login_required(login_url='login')
def home(request):
    n = 10
    stepFiles = _generateStepFiles(n)
    

    context = {
        'stepSessions'      : StepSession.objects.all(),
        'fields'            : ['field1'] * n,
        'stepRows'          : [['a'] * n] * n,
        'files'             : ['filename1'] * 10,
        'stepFiles'         : stepFiles,
        'title'             : 'StepLab'
    }
    return render(request, 'steplab/home.html', context) # request, template and context(arguments)


@login_required(login_url='login')
def recordings(request):
    user = request.user
    if request.method == 'POST':
        #stepfiles = request.POST.getlist('new-stepfiles', None) # request.POST.get and request.POST.getlist
        stepfiles = request.FILES.getlist('new-stepfiles')
        print(stepfiles)
        for stepfile in stepfiles:
            usrFolder = user.username # user.email + user.id
            full_filename = os.path.join(settings.MEDIA_ROOT, usrFolder, stepfile.name)
            # TODO: SAVE THE FILE INTO DB FOR LINDA USERS
            path = default_storage.save(full_filename, stepfile)
            
        print(stepfiles)
    stepFiles = _generateStepFiles(10)

    context = {
        'stepFiles'         : stepFiles,
        'title'             : 'Recordings'
    }

    return render(request, 'steplab/recordings.html', context) # request, template and context(arguments)

def about(request):
    return render(request, 'steplab/about.html', {'title': 'About'})


'''
from django.contrib.auth.models import User
users = User.objects.all()
'''