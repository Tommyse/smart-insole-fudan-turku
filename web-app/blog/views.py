from django.shortcuts import render
from .models import Post


def home(request):
    context = {
        'posts': Post.objects.all(),
    }
    return render(request, 'blog/home.html', context) # request, template and context(arguments)

def about(request):
    return render(request, 'blog/about.html', {'title': 'About'})
