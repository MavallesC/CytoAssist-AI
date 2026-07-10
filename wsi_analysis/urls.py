from django.urls import path
from . import views

urlpatterns = [
    # Dashboard principal
    path('', views.dashboard, name='dashboard'),
    
    # Carga de muestras
    path('samples/upload/', views.upload_sample, name='upload_sample'),
    path('samples/<int:sample_id>/delete/', views.delete_sample, name='delete_sample'),
    
    # Detalle de corridas
    path('runs/<str:run_id>/', views.run_detail, name='run_detail'),
    path('runs/<str:run_id>/viewer/', views.run_viewer, name='run_viewer'),
    
    # Descargas
    path('runs/<str:run_id>/download/report/', views.download_report, name='download_report'),
    path('runs/<str:run_id>/download/predictions/', views.download_predictions, name='download_predictions'),
    path('runs/<str:run_id>/download/json/', views.download_json, name='download_json'),
    path('runs/<str:run_id>/export/', views.export_sample_crops, name='export_sample_crops'),
    
    # APIs AJAX
    path('api/runs/<str:run_id>/status/', views.api_run_status, name='api_run_status'),
    
    # Configuración de modelos
    path('model-config/', views.model_config, name='model_config'),
]
