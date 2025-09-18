from django.urls import path
from . import views

urlpatterns = [
    path('mri-input/', views.mri_input_form, name='mri_input_form'),
    path('predict/', views.predict, name='predict'),
    path('download_pdf/<str:filename>', views.download_pdf, name='download_pdf'),
    path('mri-process/', views.api_mri_process, name='api_mri_process'),
]
