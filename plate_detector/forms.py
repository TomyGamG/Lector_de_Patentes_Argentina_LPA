from django import forms
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib.auth.models import User
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
    

class CustomUserCreationForm(UserCreationForm):
    email = forms.EmailField(required=True, widget=forms.EmailInput(attrs={
        'class': 'form-control',
        'placeholder': 'Correo electrónico'
    }))
    first_name = forms.CharField(max_length=30, required=True, widget=forms.TextInput(attrs={
        'class': 'form-control',
        'placeholder': 'Nombre'
    }))
    last_name = forms.CharField(max_length=30, required=True, widget=forms.TextInput(attrs={
        'class': 'form-control',
        'placeholder': 'Apellido'
    }))

    class Meta:
        model = User
        fields = ['username', 'email', 'first_name', 'last_name', 'password1', 'password2']
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Aplicar clases Bootstrap a todos los campos
        for field_name in self.fields:
            self.fields[field_name].widget.attrs.update({'class': 'form-control'})
        
        # Placeholders personalizados
        self.fields['username'].widget.attrs['placeholder'] = 'Nombre de usuario'
        self.fields['password1'].widget.attrs['placeholder'] = 'Contraseña'
        self.fields['password2'].widget.attrs['placeholder'] = 'Confirmar contraseña'
    
    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        user.first_name = self.cleaned_data['first_name']
        user.last_name = self.cleaned_data['last_name']
        
        if commit:
            user.save()
            # Crear automáticamente el perfil de cliente
            Client.objects.get_or_create(user=user)
        
        return user