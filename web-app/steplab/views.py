from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from .models import Post
from .models import StepSession


@login_required(login_url='login')
def home(request):
    n = 10
    # (5 // 2) + (5%2!=0)
    context = {
        'stepSessions'      : StepSession.objects.all(),
        'fields'            : ['field1'] * n,
        'stepRows'          : [['a'] * n] * n,
        'files'             : ['filename1'] * 10,
    }
    return render(request, 'steplab/home.html', context) # request, template and context(arguments)

def about(request):
    return render(request, 'steplab/about.html', {'title': 'About'})
