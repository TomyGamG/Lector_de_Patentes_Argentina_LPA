from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import Client, Vehicle, DetectionSettings, ImageAnalysis

class CustomUserCreationForm(UserCreationForm):
    email = forms.EmailField(required=True)
    first_name = forms.CharField(max_length=30, required=True)
    last_name = forms.CharField(max_length=30, required=True)

    class Meta:
        model = User
        fields = ['username', 'first_name', 'last_name', 'email', 'password1', 'password2']

class LoginForm(forms.Form):
    username = forms.CharField()
    password = forms.CharField(widget=forms.PasswordInput)

class VehicleForm(forms.ModelForm):
    class Meta:
        model = Vehicle
        fields = ['plate_number', 'brand', 'model', 'color']
        widgets = {
            'plate_number': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Ej: ABC123'}),
            'brand': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Ej: Toyota'}),
            'model': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Ej: Corolla'}),
            'color': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Ej: Rojo'}),
        }
        labels = {
            'plate_number': 'Número de Placa',
            'brand': 'Marca',
            'model': 'Modelo',
            'color': 'Color',
        }

    def clean_plate_number(self):
        plate = self.cleaned_data['plate_number']
        return plate.replace(' ', '').upper()

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
            'motion_threshold': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'}),
            'camera_index': forms.NumberInput(attrs={'class': 'form-control'}),
            'detection_interval': forms.NumberInput(attrs={'class': 'form-control'}),
            'is_active': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }

class ImageAnalysisForm(forms.ModelForm):
    class Meta:
        model = ImageAnalysis
        fields = ['image']
        widgets = {
            'image': forms.FileInput(attrs={'class': 'form-control', 'accept': 'image/*'}),
        }

# Eliminar esta línea duplicada si existe
# class UserCreationForm(UserCreationForm):
#     ...
    

