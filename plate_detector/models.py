from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
import os
from datetime import datetime

def plate_image_path(instance, filename):
    return f'plates/{instance.plate_number}/{filename}'

class Client(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    company_name = models.CharField(max_length=200)
    phone = models.CharField(max_length=20)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.company_name

class Vehicle(models.Model):
    client = models.ForeignKey(Client, on_delete=models.CASCADE)
    plate_number = models.CharField(max_length=20, unique=True)
    brand = models.CharField(max_length=100)
    model = models.CharField(max_length=100)
    color = models.CharField(max_length=50)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.plate_number} - {self.brand} {self.model}"

class PlateDetection(models.Model):
    plate_number = models.CharField(max_length=20)
    confidence = models.FloatField(default=0.0)
    image = models.ImageField(upload_to=plate_image_path)
    is_client = models.BooleanField(default=False)
    vehicle = models.ForeignKey(Vehicle, on_delete=models.SET_NULL, null=True, blank=True)
    detected_at = models.DateTimeField(auto_now_add=True)
    processed = models.BooleanField(default=False)

    class Meta:
        ordering = ['-detected_at']
    
    def __str__(self):
        return f"{self.plate_number} - {self.detected_at}"

class DetectionSettings(models.Model):
    motion_threshold = models.FloatField(default=0.5)
    camera_index = models.IntegerField(default=0)
    detection_interval = models.IntegerField(default=5)
    is_active = models.BooleanField(default=True)
    
    def __str__(self):
        return f"Settings - Camera {self.camera_index}"
    
# Agrega al final del archivo models.py

def analyzed_image_path(instance, filename):
    return f'analyzed/{datetime.now().strftime("%Y/%m/%d")}/{filename}'

class ImageAnalysis(models.Model):
    image = models.ImageField(upload_to=analyzed_image_path)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    analysis_result = models.JSONField(default=dict, blank=True)  # Para guardar resultados de YOLO
    plate_detected = models.BooleanField(default=False)
    detected_plate = models.CharField(max_length=20, blank=True, null=True)
    confidence = models.FloatField(default=0.0)
    
    class Meta:
        ordering = ['-uploaded_at']
        verbose_name_plural = "Image Analyses"
    
    def __str__(self):
        return f"Analysis {self.id} - {self.uploaded_at}"