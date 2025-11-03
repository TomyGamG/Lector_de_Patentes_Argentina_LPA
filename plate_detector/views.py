from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators import gzip
from django.conf import settings
from django.contrib import messages
import cv2
import numpy
import threading
import time
import json
from .models import PlateDetection, Vehicle, Client, DetectionSettings, ImageAnalysis
from .forms import CustomUserCreationForm, LoginForm, VehicleForm, ClientForm, SettingsForm, ImageAnalysisForm, UserCreationForm
from .services import YOLOLicensePlateDetector, ImageAnalyzer, MotionDetector, save_detection, image_analyzer
import os
from django.utils import timezone
from datetime import timedelta

# Variables globales para la cámara
camera_running = False
camera = None

# En la clase VideoCamera, modifica el método get_frame:

class VideoCamera:
    def __init__(self, camera_index=0):
        self.video = cv2.VideoCapture(camera_index)
        self.detector = YOLOLicensePlateDetector()
        self.motion_detector = MotionDetector()
        self.last_detection = 0
        self.detection_interval = 5
        self.last_motion_check = 0
        self.debug_info = "Inicializando..."
        self.current_user = None  # Agregar para almacenar usuario
        
    def set_user(self, user):
        """Establecer el usuario actual para las detecciones"""
        self.current_user = user
        
    def get_frame(self):
        try:
            success, frame = self.video.read()
            if not success:
                self.debug_info = "Error leyendo cámara"
                return None
            
            # Redimensionar para performance
            frame = cv2.resize(frame, (640, 480))
            current_time = time.time()
            
            # Detectar movimiento
            if current_time - self.last_motion_check > 2:
                motion_detected = self.motion_detector.detect_motion(frame)
                self.last_motion_check = current_time
                
                if motion_detected and current_time - self.last_detection > self.detection_interval:
                    plates = self.detector.detect_plate(frame)
                    if plates:
                        self.last_detection = current_time
                        # Pasar el usuario a save_detection
                        threading.Thread(
                            target=save_detection, 
                            args=(plates[0], frame.copy(), self.current_user),  # Agregar usuario
                            daemon=True
                        ).start()
                        self.debug_info = f"Patente detectada: {plates[0]['text']}"
                    else:
                        self.debug_info = "Buscando patentes..."
            
            # Dibujar información de debug
            cv2.putText(frame, "SISTEMA ACTIVO", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, self.debug_info, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(frame, f"Movimientos: {self.motion_detector.motion_count}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(frame, f"Patentes: {self.detector.detection_count}", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Codificar frame
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes() if ret else None
            
        except Exception as e:
            self.debug_info = f"Error: {str(e)}"
            print(f"❌ Error en cámara: {e}")
            return None

def gen(camera):
    global camera_running
    while camera_running:
        frame = camera.get_frame()
        if frame:
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        time.sleep(0.1)


@login_required
def clear_old_data(request):
    """Elimina detecciones con más de 30 días"""
    if request.method == 'POST':
        try:
            # Calcular la fecha límite (30 días atrás)
            cutoff_date = timezone.now() - timedelta(days=30)
            
            # Filtrar detecciones antiguas del usuario actual
            old_detections = PlateDetection.objects.filter(
                user=request.user,
                detected_at__lt=cutoff_date
            )
            
            # Contar antes de eliminar
            count_before = old_detections.count()
            
            # Eliminar las detecciones
            old_detections.delete()
            
            # También eliminar análisis de imágenes antiguos
            old_analyses = ImageAnalysis.objects.filter(
                user=request.user,
                uploaded_at__lt=cutoff_date
            )
            analyses_count = old_analyses.count()
            old_analyses.delete()
            
            messages.success(
                request, 
                f'✅ Se eliminaron {count_before} detecciones y {analyses_count} análisis con más de 30 días.'
            )
            
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({
                    'success': True,
                    'message': f'Se eliminaron {count_before} detecciones y {analyses_count} análisis antiguos.',
                    'deleted_detections': count_before,
                    'deleted_analyses': analyses_count
                })
                
        except Exception as e:
            error_msg = f'❌ Error al limpiar datos: {str(e)}'
            messages.error(request, error_msg)
            
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({
                    'success': False,
                    'error': error_msg
                })
    
    return redirect('settings')

@gzip.gzip_page
def video_feed(request):
    global camera_running, camera
    try:
        if camera is None:
            try:
                settings_obj = DetectionSettings.objects.first()
                camera_index = settings_obj.camera_index if settings_obj else 0
                camera = VideoCamera(camera_index)
            except:
                camera = VideoCamera(0)
        
        # Configurar el usuario actual en la cámara
        if request.user.is_authenticated:
            camera.set_user(request.user)
            
        return StreamingHttpResponse(gen(camera), content_type="multipart/x-mixed-replace;boundary=frame")
    except Exception as e:
        print(f"Error en video feed: {e}")
        return JsonResponse({'error': str(e)})

def user_login(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
        
    if request.method == 'POST':
        form = LoginForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('dashboard')
    else:
        form = LoginForm()
    return render(request, 'login.html', {'form': form})

def register(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            
            # Autenticar y loguear al usuario automáticamente
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=password)
            
            if user is not None:
                login(request, user)
                messages.success(request, f'¡Cuenta creada exitosamente! Bienvenido, {user.first_name}.')
                return redirect('dashboard')
        else:
            messages.error(request, 'Por favor corrige los errores en el formulario.')
    else:
        form = CustomUserCreationForm()
    
    return render(request, 'register.html', {'form': form})

# Vista de login personalizada (opcional)
from django.contrib.auth.views import LoginView

class CustomLoginView(LoginView):
    template_name = 'login.html'
    
    def form_valid(self, form):
        messages.success(self.request, f'¡Bienvenido de nuevo, {form.get_user().first_name}!')
        return super().form_valid(form)

def user_logout(request):
    logout(request)
    return redirect('login')

@login_required
def dashboard(request):
    # Obtener o crear el cliente asociado al usuario
    client, created = Client.objects.get_or_create(user=request.user)
    
    # Filtrar detecciones solo del usuario actual
    total_detections = PlateDetection.objects.filter(user=request.user).count()
    client_detections = PlateDetection.objects.filter(user=request.user, is_client=True).count()
    recent_detections = PlateDetection.objects.filter(user=request.user).order_by('-detected_at')[:5]
    
    context = {
        'total_detections': total_detections,
        'client_detections': client_detections,
        'recent_detections': recent_detections,
        'client': client,
    }
    return render(request, 'dashboard.html', context)

@login_required
def detection_view(request):
    return render(request, 'detection.html')

@login_required
def plate_list(request):
    # Filtrar solo las detecciones del usuario actual
    detections = PlateDetection.objects.filter(user=request.user).order_by('-detected_at')
    
    context = {
        'detections': detections,
    }
    return render(request, 'plate_list.html', context)

@login_required
def start_detection(request):
    global camera_running, camera
    camera_running = True
    
    # Configurar el usuario en la cámara
    if camera is not None:
        camera.set_user(request.user)
    
    return JsonResponse({'status': 'Detección iniciada'})

@login_required
def stop_detection(request):
    global camera_running
    camera_running = False
    return JsonResponse({'status': 'Detección detenida'})

@login_required
def vehicle_management(request):
    client, created = Client.objects.get_or_create(user=request.user)
    
    if request.method == 'POST':
        form = VehicleForm(request.POST)
        if form.is_valid():
            vehicle = form.save(commit=False)
            vehicle.client = client
            vehicle.save()
            return redirect('vehicle_management')
    else:
        form = VehicleForm()
    
    vehicles = Vehicle.objects.filter(client=client)
    return render(request, 'vehicle_management.html', {
        'form': form,
        'vehicles': vehicles
    })

@login_required
def settings_view(request):
    settings_obj, created = DetectionSettings.objects.get_or_create(id=1)
    
    if request.method == 'POST':
        form = SettingsForm(request.POST, instance=settings_obj)
        if form.is_valid():
            form.save()
            return redirect('settings')
    else:
        form = SettingsForm(instance=settings_obj)
    
    # Agregar datos adicionales para el template
    total_detections = PlateDetection.objects.count()
    total_vehicles = Vehicle.objects.count()
    
    return render(request, 'settings.html', {
        'form': form,
        'total_detections': total_detections,
        'total_vehicles': total_vehicles,
        'storage_used': total_detections * 0.1  # Estimación simple
    })

@login_required
def image_analysis_view(request):
    """Vista para análisis de imágenes"""
    form = ImageAnalysisForm()
    analyses = ImageAnalysis.objects.filter(user=request.user).order_by('-uploaded_at')[:10]  # Filtrar por usuario
    
    context = {
        'form': form,
        'analyses': analyses
    }
    return render(request, 'image_analysis.html', context)

@login_required
def upload_and_analyze_image(request):
    """Procesa la imagen subida y realiza el análisis"""
    if request.method == 'POST':
        form = ImageAnalysisForm(request.POST, request.FILES)
        
        if form.is_valid():
            try:
                # Guardar la imagen primero para obtener la ruta
                image_analysis = form.save(commit=False)
                image_analysis.user = request.user  # Asignar usuario
                image_analysis.save()
                
                print(f"✅ Imagen guardada en: {image_analysis.image.path}")
                
                # Verificar que el archivo existe
                if not os.path.exists(image_analysis.image.path):
                    messages.error(request, '❌ Error: El archivo de imagen no se guardó correctamente')
                    image_analysis.delete()
                    return redirect('image_analysis')
                
                # Realizar análisis COMPLETO de la imagen
                analysis_result = image_analyzer.analyze_image(image_analysis.image.path)
                
                # Verificar si hubo error en el análisis
                if 'error' in analysis_result:
                    messages.error(request, f'❌ Error en análisis: {analysis_result["error"]}')
                    image_analysis.delete()
                    return redirect('image_analysis')
                
                # Actualizar el objeto con los resultados
                image_analysis.analysis_result = analysis_result
                
                # Verificar si se detectaron patentes
                license_plates = analysis_result.get('license_plates', [])
                if license_plates:
                    image_analysis.plate_detected = True
                    # Tomar la patente con mayor confianza
                    best_plate = max(license_plates, key=lambda x: x['combined_confidence'])
                    image_analysis.detected_plate = best_plate['text']
                    image_analysis.confidence = best_plate['combined_confidence']
                    
                    # Crear también una PlateDetection
                    PlateDetection.objects.create(
                        user=request.user,
                        plate_number=best_plate['text'],
                        confidence=best_plate['combined_confidence'],
                        image=image_analysis.image,
                        is_client=False  # Puedes ajustar esta lógica
                    )
                
                image_analysis.save()
                
                # Crear imagen anotada SIEMPRE (incluso si no hay patentes)
                annotated_image = image_analyzer.create_annotated_image(
                    image_analysis.image.path, 
                    analysis_result
                )
                
                if annotated_image is not None:
                    # Guardar imagen anotada
                    original_path = image_analysis.image.path
                    name, ext = os.path.splitext(original_path)
                    annotated_path = f"{name}_annotated{ext}"
                    cv2.imwrite(annotated_path, annotated_image)
                    print(f"✅ Imagen anotada guardada en: {annotated_path}")
                
                # Mensaje de éxito detallado
                total_objects = len(analysis_result.get('objects_detected', []))
                total_plates = len(license_plates)
                
                if total_plates > 0:
                    messages.success(request, 
                        f'✅ Análisis completado: {total_objects} objetos analizados, {total_plates} patente(s) detectada(s)')
                else:
                    messages.warning(request, 
                        f'⚠️ Análisis completado: {total_objects} objetos analizados, pero no se detectaron patentes')
                
                return redirect('image_analysis')
                
            except Exception as e:
                print(f"❌ Error analizando imagen: {e}")
                messages.error(request, f'❌ Error analizando imagen: {str(e)}')
                # Eliminar el objeto si hubo error
                if 'image_analysis' in locals() and image_analysis.pk:
                    image_analysis.delete()
        else:
            for error in form.errors.values():
                messages.error(request, error)
    
    return redirect('image_analysis')


# Filtros simples para templates
def replace_filter(value, old, new):
    """Filtro simple para reemplazar texto en templates"""
    return value.replace(old, new)

@login_required
def analysis_detail_view(request, analysis_id):
    """Vista detallada de un análisis específico"""
    try:
        # Filtrar solo análisis del usuario actual
        analysis = ImageAnalysis.objects.get(id=analysis_id, user=request.user)
        
        # Procesar la URL para la imagen anotada
        image_url = analysis.image.url
        annotated_url = image_url.replace('.jpg', '_annotated.jpg')
        annotated_url = annotated_url.replace('.jpeg', '_annotated.jpeg')
        annotated_url = annotated_url.replace('.png', '_annotated.png')
        
        # Verificar si la imagen anotada existe físicamente
        import os
        from django.conf import settings
        
        # Obtener la ruta física del archivo anotado
        original_path = analysis.image.path
        name, ext = os.path.splitext(original_path)
        annotated_path = f"{name}_annotated{ext}"
        annotated_exists = os.path.exists(annotated_path)
        
        context = {
            'analysis': analysis,
            'objects_detected': analysis.analysis_result.get('objects_detected', []),
            'license_plates': analysis.analysis_result.get('license_plates', []),
            'image_info': analysis.analysis_result.get('image_info', {}),
            'analysis_result': analysis.analysis_result,
            'annotated_image_url': annotated_url,
            'annotated_exists': annotated_exists,  # Nueva variable
        }
        return render(request, 'analysis_detail.html', context)
    except ImageAnalysis.DoesNotExist:
        messages.error(request, 'Análisis no encontrado')
        return redirect('image_analysis')
    


@login_required
def delete_analysis_view(request, analysis_id):
    """Elimina un análisis"""
    try:
        analysis = ImageAnalysis.objects.get(id=analysis_id)
        analysis.delete()
        messages.success(request, 'Análisis eliminado correctamente')
    except ImageAnalysis.DoesNotExist:
        messages.error(request, 'Análisis no encontrado')
    
    return redirect('image_analysis')