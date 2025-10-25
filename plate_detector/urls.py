from django.urls import path
from . import views

urlpatterns = [
    path('', views.user_login, name='login'),
    path('logout/', views.user_logout, name='logout'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('detection/', views.detection_view, name='detection'),
    path('plates/', views.plate_list, name='plate_list'),
    path('vehicles/', views.vehicle_management, name='vehicle_management'),
    path('settings/', views.settings_view, name='settings'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('start_detection/', views.start_detection, name='start_detection'),
    path('stop_detection/', views.stop_detection, name='stop_detection'),
    path('image-analysis/', views.image_analysis_view, name='image_analysis'),
    path('upload-analyze/', views.upload_and_analyze_image, name='upload_analyze'),
    path('analysis/<int:analysis_id>/', views.analysis_detail_view, name='analysis_detail'),
    path('analysis/<int:analysis_id>/delete/', views.delete_analysis_view, name='delete_analysis'),
]