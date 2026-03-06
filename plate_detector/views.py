from django.shortcuts import render, redirect, get_object_or_404
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
from .forms import CustomUserCreationForm, LoginForm, VehicleForm, ClientForm, SettingsForm, ImageAnalysisForm
from .services import ImageAnalyzer, save_detection_to_django
import os
from django.utils import timezone
from datetime import timedelta
from datetime import datetime

# Variables globales para la cámara
camera_running = False
camera = None


# --- Analyzer singleton (evita reinstanciar el modelo en cada request) ---
_image_analyzer_singleton = None

def get_image_analyzer():
    global _image_analyzer_singleton
    if _image_analyzer_singleton is None:
        _image_analyzer_singleton = ImageAnalyzer(use_yolo=True, yolo_model="yolov10n")
    return _image_analyzer_singleton

def create_annotated_image(image_path: str, analysis_result: dict):
    """Dibuja bbox + texto en la imagen a partir del resultado del services nuevo."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    plates = analysis_result.get("license_plates") or []
    for p in plates:
        coords = p.get("coordinates")
        if not coords:
            continue

        x = int(coords.get("x", 0))
        y = int(coords.get("y", 0))
        w = int(coords.get("w", 0))
        h = int(coords.get("h", 0))

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{p.get('text','')} ({float(p.get('confidence') or 0.0):.2f})"
        cv2.putText(
            img,
            label,
            (x, max(0, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
        label = f"{p.get('text','')} ({float(p.get('confidence') or 0.0):.2f})"
        cv2.putText(img, label, (x, max(0, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return img
# En la clase VideoCamera, modifica el método get_frame:

class VideoCamera:
    def __init__(self, camera_index=0):
        self.video = cv2.VideoCapture(camera_index)
        self.last_detection = 0
        self.detection_interval = 5
        self.debug_info = "Inicializando..."
        self.current_user = None

        # Nuevo analyzer (services nuevo)
        self.analyzer = get_image_analyzer()

    def set_user(self, user):
        """Establecer el usuario actual para las detecciones"""
        self.current_user = user
        print(f"👤 Usuario establecido en cámara: {user.username if user else 'None'}")

    def get_frame(self):
        try:
            success, frame = self.video.read()
            if not success:
                self.debug_info = "Error leyendo cámara"
                return None

            frame = cv2.resize(frame, (640, 480))
            current_time = time.time()

            # Ejecutar análisis cada X segundos
            if current_time - self.last_detection > self.detection_interval:
                self.last_detection = current_time

                result = self.analyzer.pipeline.process_image(frame)
                plates = result.get("license_plates") or []

                if plates:
                    best = plates[0]
                    plate_text = best.get("text", "")
                    self.debug_info = f"Patente detectada: {plate_text}"

                    # Guardar detección en Django en un thread para no trabar el stream
                    threading.Thread(
                        target=save_detection_to_django,
                        kwargs={
                            "image_bgr": frame.copy(),
                            "plate_data": best,
                            "user": self.current_user,
                            "filename_prefix": "realtime",
                        },
                        daemon=True
                    ).start()
                else:
                    self.debug_info = "Buscando patentes..."

            # Overlay debug
            cv2.putText(frame, "SISTEMA ACTIVO", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, self.debug_info, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(frame,
                        f"Usuario: {self.current_user.username if self.current_user else 'None'}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

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
        # ✅ CORREGIDO: Pasar los datos correctamente al formulario
        form = LoginForm(data=request.POST)
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
    
    # Fecha actual
    today = timezone.now().date()
    now = timezone.now()
    
    # KPIs principales
    total_analyses = ImageAnalysis.objects.filter(user=request.user).count()
    
    # Patentes detectadas hoy
    today_plates = ImageAnalysis.objects.filter(
        user=request.user,
        uploaded_at__date=today,
        plate_detected=True
    ).count()
    
    # Tasa de éxito general
    successful_detections = ImageAnalysis.objects.filter(
        user=request.user,
        plate_detected=True
    ).count()
    failed_detections = total_analyses - successful_detections
    success_rate = round((successful_detections / total_analyses * 100) if total_analyses > 0 else 0, 1)
    
    # Tiempo promedio de procesamiento (simulado - ajustar según tu lógica)
    avg_processing_time = 1.8
    
    # Detecciones de clientes
    client_detections = PlateDetection.objects.filter(user=request.user, is_client=True).count()
    total_vehicles = Vehicle.objects.filter(client=client).count()
    
    # Análisis recientes con información de cliente
    recent_analyses = []
    analyses = ImageAnalysis.objects.filter(user=request.user).order_by('-uploaded_at')[:8]
    
    for analysis in analyses:
        is_client = False
        vehicle = None
        
        if analysis.detected_plate:
            try:
                vehicle = Vehicle.objects.filter(
                    client=client, 
                    plate_number=analysis.detected_plate
                ).first()
                if vehicle:
                    is_client = True
            except Vehicle.DoesNotExist:
                pass
        
        analysis.is_client = is_client
        analysis.vehicle = vehicle
        recent_analyses.append(analysis)
    
    # Métricas en tiempo real
    last_hour = now - timedelta(hours=1)
    last_hour_analyses = ImageAnalysis.objects.filter(
        user=request.user,
        uploaded_at__gte=last_hour
    ).count()
    
    # Esta semana
    start_of_week = today - timedelta(days=today.weekday())
    this_week_plates = ImageAnalysis.objects.filter(
        user=request.user,
        uploaded_at__date__gte=start_of_week,
        plate_detected=True
    ).count()
    
    # Tasa de éxito hoy
    today_successful = ImageAnalysis.objects.filter(
        user=request.user,
        uploaded_at__date=today,
        plate_detected=True
    ).count()
    today_total = ImageAnalysis.objects.filter(
        user=request.user,
        uploaded_at__date=today
    ).count()
    today_success_rate = round((today_successful / today_total * 100) if today_total > 0 else 0, 1)
    
    # Mejor hora para detección (simulado)
    best_hour = "14:00-15:00"
    
    # Datos para gráficos
    import json
    from collections import Counter
    
    # Gráfico de series de tiempo - Por hora (hoy)
    hourly_data = [0] * 24
    hourly_labels = [f"{h:02d}:00" for h in range(24)]
    
    today_analyses_hourly = ImageAnalysis.objects.filter(
        user=request.user,
        uploaded_at__date=today,
        plate_detected=True
    )
    
    for analysis in today_analyses_hourly:
        hour = analysis.uploaded_at.hour
        hourly_data[hour] += 1
    
    # Gráfico de series de tiempo - Por día (últimos 7 días)
    daily_data = []
    daily_labels = []
    
    for i in range(6, -1, -1):
        date = today - timedelta(days=i)
        daily_labels.append(date.strftime('%d/%m'))
        analyses_count = ImageAnalysis.objects.filter(
            user=request.user,
            uploaded_at__date=date,
            plate_detected=True
        ).count()
        daily_data.append(analyses_count)
    
    # Gráfico de series de tiempo - Por semana (últimas 8 semanas)
    weekly_data = []
    weekly_labels = []
    
    for i in range(7, -1, -1):
        week_start = today - timedelta(weeks=i+1, days=today.weekday())
        week_end = week_start + timedelta(days=6)
        weekly_labels.append(f"Sem {i+1}")
        analyses_count = ImageAnalysis.objects.filter(
            user=request.user,
            uploaded_at__date__range=[week_start, week_end],
            plate_detected=True
        ).count()
        weekly_data.append(analyses_count)
    
    # Gráfico circular - Distribución de resultados
    distribution_data = [successful_detections, failed_detections]
    
    # Gráfico de áreas - Tendencias semanales
    trend_labels = []
    trend_successful = []
    trend_failed = []
    
    for i in range(3, -1, -1):
        week_start = today - timedelta(weeks=i+1, days=today.weekday())
        week_end = week_start + timedelta(days=6)
        trend_labels.append(f"W{i+1}")
        
        successful = ImageAnalysis.objects.filter(
            user=request.user,
            uploaded_at__date__range=[week_start, week_end],
            plate_detected=True
        ).count()
        
        failed = ImageAnalysis.objects.filter(
            user=request.user,
            uploaded_at__date__range=[week_start, week_end],
            plate_detected=False
        ).count()
        
        trend_successful.append(successful)
        trend_failed.append(failed)
    
    context = {
        # KPIs principales
        'today_plates': today_plates,
        'success_rate': success_rate,
        'avg_processing_time': avg_processing_time,
        'total_analyses': total_analyses,
        
        # Estadísticas adicionales
        'successful_detections': successful_detections,
        'failed_detections': failed_detections,
        'client_detections': client_detections,
        'total_vehicles': total_vehicles,
        'last_hour_analyses': last_hour_analyses,
        'this_week_plates': this_week_plates,
        'today_success_rate': today_success_rate,
        'best_hour': best_hour,
        
        # Datos recientes
        'recent_analyses': recent_analyses,
        'client': client,
        
        # Datos para gráficos
        'hourly_labels': json.dumps(hourly_labels),
        'hourly_data': json.dumps(hourly_data),
        'daily_labels': json.dumps(daily_labels),
        'daily_data': json.dumps(daily_data),
        'weekly_labels': json.dumps(weekly_labels),
        'weekly_data': json.dumps(weekly_data),
        'distribution_data': json.dumps(distribution_data),
        'trend_labels': json.dumps(trend_labels),
        'trend_successful': json.dumps(trend_successful),
        'trend_failed': json.dumps(trend_failed),
    }
    
    return render(request, 'dashboard.html', context)

@login_required
def detection_view(request):
    return render(request, 'detection.html')

@login_required
def plate_list(request):
    # Obtener detecciones de PlateDetection
    detections = PlateDetection.objects.filter(user=request.user).order_by('-detected_at')
    
    # Obtener análisis de imágenes que tengan patentes detectadas
    analyses_with_plates = ImageAnalysis.objects.filter(
        user=request.user,
        plate_detected=True
    ).exclude(detected_plate__isnull=True).exclude(detected_plate='').order_by('-uploaded_at')
    
    # Combinar ambas listas evitando duplicados
    all_plates = []
    processed_plates = set()  # Para evitar duplicados
    
    # Primero agregar detecciones de PlateDetection
    for detection in detections:
        plate_key = f"{detection.plate_number}_{detection.detected_at.strftime('%Y%m%d%H%M')}"
        
        if plate_key not in processed_plates:
            all_plates.append({
                'type': 'detection',
                'object': detection,
                'plate_number': detection.plate_number,
                'date': detection.detected_at,
                'confidence': detection.confidence,
                'image': detection.image,
                'is_client': detection.is_client,
                'vehicle': detection.vehicle,
                'source': 'Detección en Tiempo Real',
                'unique_key': plate_key
            })
            processed_plates.add(plate_key)
    
    # Luego agregar análisis de ImageAnalysis que no estén duplicados
    for analysis in analyses_with_plates:
        plate_key = f"{analysis.detected_plate}_{analysis.uploaded_at.strftime('%Y%m%d%H%M')}"
        
        # Verificar si ya existe una detección similar (misma patente en mismo minuto)
        if plate_key not in processed_plates:
            # Verificar si es cliente
            is_client = False
            vehicle = None
            if analysis.detected_plate:
                try:
                    client_obj = Client.objects.get(user=request.user)
                    vehicle = Vehicle.objects.filter(
                        client=client_obj, 
                        plate_number=analysis.detected_plate
                    ).first()
                    if vehicle:
                        is_client = True
                except (Client.DoesNotExist, Vehicle.DoesNotExist):
                    pass
            
            all_plates.append({
                'type': 'analysis',
                'object': analysis,
                'plate_number': analysis.detected_plate,
                'date': analysis.uploaded_at,
                'confidence': analysis.confidence,
                'image': analysis.image,
                'is_client': is_client,
                'vehicle': vehicle,
                'source': 'Análisis de Imagen',
                'unique_key': plate_key
            })
            processed_plates.add(plate_key)
    
    # Ordenar por fecha (más reciente primero)
    all_plates.sort(key=lambda x: x['date'], reverse=True)
    
    # Contadores reales sin duplicados
    detections_count = len([p for p in all_plates if p['type'] == 'detection'])
    analyses_count = len([p for p in all_plates if p['type'] == 'analysis'])
    
    context = {
        'all_plates': all_plates,
        'detections_count': detections_count,
        'analyses_count': analyses_count,
        'total_count': len(all_plates)
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
def edit_vehicle(request, vehicle_id):
    client = Client.objects.get(user=request.user)
    vehicle = get_object_or_404(Vehicle, id=vehicle_id, client=client)

    if request.method == 'POST':
        form = VehicleForm(request.POST, instance=vehicle)
        if form.is_valid():
            form.save()
            messages.success(request, '🚗 Vehículo actualizado correctamente')
            return redirect('vehicle_management')
    else:
        form = VehicleForm(instance=vehicle)

    return render(request, 'vehicle_edit.html', {
        'form': form,
        'vehicle': vehicle
    })

@login_required
def delete_vehicle(request, vehicle_id):
    client = Client.objects.get(user=request.user)
    vehicle = get_object_or_404(Vehicle, id=vehicle_id, client=client)

    vehicle.is_active = False
    vehicle.save()

    messages.warning(request, f'🛑 Vehículo {vehicle.plate_number} desactivado')
    return redirect('vehicle_management')

@login_required
def reactivate_vehicle(request, vehicle_id):
    client = Client.objects.get(user=request.user)
    vehicle = get_object_or_404(Vehicle, id=vehicle_id, client=client)

    vehicle.is_active = True
    vehicle.save()

    messages.success(
        request,
        f'✅ Vehículo {vehicle.plate_number} reactivado correctamente'
    )
    return redirect('vehicle_management')


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

def make_json_safe_analysis_result(result: dict) -> dict:
    """
    Quita objetos no serializables (np.ndarray) del resultado del services,
    para poder guardarlo en un JSONField.
    """
    if not isinstance(result, dict):
        return {}

    safe = dict(result)

    plates = safe.get("license_plates") or []
    safe_plates = []
    for p in plates:
        if not isinstance(p, dict):
            continue
        p2 = dict(p)

        # Claves NO serializables (np.ndarray)
        p2.pop("region", None)
        p2.pop("warped", None)

        # Asegurar tipos simples
        if "coordinates" in p2 and p2["coordinates"] is not None:
            coords = p2["coordinates"]
            if isinstance(coords, dict):
                p2["coordinates"] = {
                    "x": int(coords.get("x", 0)),
                    "y": int(coords.get("y", 0)),
                    "w": int(coords.get("w", 0)),
                    "h": int(coords.get("h", 0)),
                }
            elif isinstance(coords, (list, tuple)) and len(coords) == 4:
                p2["coordinates"] = {
                    "x": int(coords[0]),
                    "y": int(coords[1]),
                    "w": int(coords[2]),
                    "h": int(coords[3]),
                }
            else:
                p2["coordinates"] = {"x": 0, "y": 0, "w": 0, "h": 0}

        for k in ["confidence", "ocr_confidence", "yolo_confidence", "prefilter_confidence"]:
            if k in p2 and p2[k] is not None:
                try:
                    p2[k] = float(p2[k])
                except Exception:
                    p2[k] = 0.0

        safe_plates.append(p2)

    safe["license_plates"] = safe_plates
    return safe

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
                analyzer = get_image_analyzer()
                analysis_result = analyzer.analyze_image(image_analysis.image.path)
                
                # Verificar si hubo error en el análisis
                if 'error' in analysis_result:
                    messages.error(request, f'❌ Error en análisis: {analysis_result["error"]}')
                    image_analysis.delete()
                    return redirect('image_analysis')
                
                # Actualizar el objeto con los resultados
                image_analysis.analysis_result = make_json_safe_analysis_result(analysis_result)
                
                # Verificar si se detectaron patentes
                license_plates = analysis_result.get('license_plates', [])
                if license_plates:
                    image_analysis.plate_detected = True
                    # Tomar la patente con mayor confianza
                    best_plate = max(license_plates, key=lambda x: float(x.get('confidence') or 0.0))
                    image_analysis.detected_plate = best_plate.get('text', '')
                    image_analysis.confidence = float(best_plate.get('confidence') or 0.0)
                    
                    # VERIFICAR SI ES CLIENTE antes de crear PlateDetection
                    is_client = False
                    vehicle = None
                    try:
                        client = Client.objects.get(user=request.user)
                        vehicle = Vehicle.objects.filter(
                            client=client, 
                            plate_number=best_plate.get('text','')
                        ).first()
                        if vehicle:
                            is_client = True
                            print(f"✅ PATENTE DE CLIENTE DETECTADA: {best_plate.get('text','')}")
                    except Client.DoesNotExist:
                        print(f"❌ Cliente no encontrado para usuario {request.user.username}")
                    
                    # Crear PlateDetection con la información correcta
                    '''PlateDetection.objects.create(
                        user=request.user,
                        plate_number=best_plate.get('text',''),
                        confidence=best_plate['confidence'],
                        image=image_analysis.image,
                        is_client=is_client,
                        vehicle=vehicle
                    )'''
                
                image_analysis.save()
                
                # Crear imagen anotada SIEMPRE (incluso si no hay patentes)
                annotated_image = create_annotated_image(
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
                    # Verificar si alguna patente es de cliente
                    client_plates = [p for p in license_plates if 
                                   Vehicle.objects.filter(
                                       client__user=request.user, 
                                       plate_number=p['text']
                                   ).exists()]
                    
                    if client_plates:
                        messages.success(request, 
                            f'✅ Análisis completado: {total_objects} objetos, {total_plates} patente(s), {len(client_plates)} CLIENTE(S) identificado(s)')
                    else:
                        messages.success(request, 
                            f'✅ Análisis completado: {total_objects} objetos, {total_plates} patente(s) detectada(s)')
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