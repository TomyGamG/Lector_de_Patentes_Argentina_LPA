from django.contrib import admin
from .models import Client, Vehicle, PlateDetection, DetectionSettings

@admin.register(Client)
class ClientAdmin(admin.ModelAdmin):
    list_display = ['company_name', 'user', 'phone', 'created_at']
    list_filter = ['created_at']
    search_fields = ['company_name', 'user__username']

@admin.register(Vehicle)
class VehicleAdmin(admin.ModelAdmin):
    list_display = ['plate_number', 'brand', 'model', 'color', 'client', 'is_active', 'created_at']
    list_filter = ['is_active', 'brand', 'created_at']
    search_fields = ['plate_number', 'brand', 'model']
    list_editable = ['is_active']

@admin.register(PlateDetection)
class PlateDetectionAdmin(admin.ModelAdmin):
    list_display = ['plate_number', 'confidence', 'is_client', 'vehicle', 'detected_at']
    list_filter = ['is_client', 'detected_at', 'processed']
    search_fields = ['plate_number']
    readonly_fields = ['detected_at']
    date_hierarchy = 'detected_at'

@admin.register(DetectionSettings)
class DetectionSettingsAdmin(admin.ModelAdmin):
    list_display = ['camera_index', 'motion_threshold', 'detection_interval', 'is_active']
    list_editable = ['is_active']
