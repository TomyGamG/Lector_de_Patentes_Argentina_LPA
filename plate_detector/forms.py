from django import forms
from django.contrib.auth.forms import AuthenticationForm
from .models import Vehicle, Client, DetectionSettings
import os

class LoginForm(AuthenticationForm):
    username = forms.CharField(
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Usuario'})
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': 'Contraseña'})
    )

class VehicleForm(forms.ModelForm):
    class Meta:
        model = Vehicle
        fields = ['plate_number', 'brand', 'model', 'color']
        widgets = {
            'plate_number': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'ABC123'}),
            'brand': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Toyota'}),
            'model': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Corolla'}),
            'color': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Rojo'}),
        }

class ClientForm(forms.ModelForm):
    class Meta:
        model = Client
        fields = ['company_name', 'phone']
        widgets = {
            'company_name': forms.TextInput(attrs={'class': 'form-control'}),
            'phone': forms.TextInput(attrs={'class': 'form-control'}),
        }

class SettingsForm(forms.ModelForm):
    class Meta:
        model = DetectionSettings
        fields = ['motion_threshold', 'camera_index', 'detection_interval', 'is_active']
        widgets = {
            'motion_threshold': forms.NumberInput(attrs={
                'class': 'form-control', 
                'step': '0.1', 
                'min': '0', 
                'max': '1'
            }),
            'camera_index': forms.NumberInput(attrs={'class': 'form-control'}),
            'detection_interval': forms.NumberInput(attrs={'class': 'form-control'}),
            'is_active': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }

# Agrega al final del archivo forms.py

from .models import ImageAnalysis

class ImageAnalysisForm(forms.ModelForm):
    class Meta:
        model = ImageAnalysis
        fields = ['image']
        widgets = {
            'image': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': 'image/*'
            })
        }
    
    def clean_image(self):
        image = self.cleaned_data.get('image')
        if image:
            # Validar tamaño del archivo (max 10MB)
            if image.size > 10 * 1024 * 1024:
                raise forms.ValidationError("La imagen no puede ser mayor a 10MB")
            
            # Validar tipo de archivo
            valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            ext = os.path.splitext(image.name)[1].lower()
            if ext not in valid_extensions:
                raise forms.ValidationError("Formato no soportado. Use JPG, PNG o BMP.")
        
        return image