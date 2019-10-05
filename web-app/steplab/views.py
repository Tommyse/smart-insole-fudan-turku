from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from .models import Post
from .models import StepSession
from .models import stepFile


def _generateStepFiles(n):
    stepFiles = []

    for i in range(0, n):
        stepFiles.append(stepFile("University File", "Diego", "43", "45as5df4", "500", None))
    
    return stepFiles

@login_required(login_url='login')
def home(request):
    n = 10
    # (5 // 2) + (5%2!=0)

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
    stepFiles = _generateStepFiles(10)

    context = {
        'stepFiles'         : stepFiles,
        'title'             : 'Recordings'
    }

    return render(request, 'steplab/recordings.html', context) # request, template and context(arguments)

def about(request):
    return render(request, 'steplab/about.html', {'title': 'About'})
