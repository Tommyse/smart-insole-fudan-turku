from django.urls import path
from . import views

urlpatterns = [
    path('', views.diagnosis, name="steplab-diagnosis"),
    path('recordings/', views.recordings, name="steplab-recordings"),
    path('diagnosis/new', views.newDiagnose, name="steplab-diagnosis-new"),
    path('diagnosis/result', views.diagnosisResult, name="steplab-diagnosis-result"),
    path('diagnosis/history', views.diagnosisHistory, name="steplab-diagnosis-history"),
    path('diagnosis/', views.diagnosis, name="steplab-diagnosis"),
    path('about/', views.about, name="steplab-about"),
]
