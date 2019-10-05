from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name="steplab-home"),
    path('recordings/', views.recordings, name="steplab-recordings"),
    path('about/', views.about, name="steplab-about"),
]
