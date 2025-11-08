from multiprocessing.connection import Client
import cv2
import numpy as np
import os
import datetime
import time
from django.conf import settings
from .models import PlateDetection, Vehicle
import threading
import requests
from PIL import Image
import json

print("🎯 Inicializando Sistema de Análisis con YOLOv8...")

class Timer:
    """Clase para medir y mostrar tiempos de ejecución"""
    
    def __init__(self, operation_name=""):
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        if self.operation_name:
            print(f"⏰ INICIANDO PROCESAMIENTO: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        execution_time = self.end_time - self.start_time
        if self.operation_name:
            status = "✅ COMPLETADO" if exc_type is None else "❌ FALLADO"
            print(f"⏰ {status}: {self.operation_name}")
            print(f"⏰ TIEMPO TOTAL: {execution_time:.2f} segundos")
        return False

class RealTimeVisualizer:
    """Clase para visualización en tiempo real del análisis"""
    
    def __init__(self, window_name="Sistema de Análisis - Tiempo Real"):
        self.window_name = window_name
        self.is_active = False
        self.current_frame = None
        self.detections = []
        self.processing_info = ""
        self.status_info = ""
        
    def start(self):
        """Iniciar la visualización en tiempo real"""
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 1200, 800)
            self.is_active = True
            print(f"🖥️  Visualización en tiempo real iniciada: {self.window_name}")
        except Exception as e:
            print(f"❌ Error iniciando visualización: {e}")
    
    def update_frame(self, frame, detections=None, processing_info="", status_info=""):
        """Actualizar el frame con las detecciones actuales"""
        self.current_frame = frame.copy()
        self.detections = detections if detections else []
        self.processing_info = processing_info
        self.status_info = status_info
        
        # Dibujar en el frame
        self._draw_detections()
        self._draw_info_panel()
        
    def _draw_detections(self):
        """Dibujar todas las detecciones en el frame"""
        if self.current_frame is None:
            return
            
        for detection in self.detections:
            try:
                # Obtener coordenadas
                if 'bbox' in detection:
                    x, y, w, h = detection['bbox']
                    
                    # Determinar color según el método de detección
                    color = self._get_detection_color(detection)
                    
                    # Dibujar bounding box
                    cv2.rectangle(self.current_frame, (x, y), (x + w, y + h), color, 3)
                    
                    # Preparar texto de etiqueta
                    label_parts = []
                    
                    # Agregar clase si existe
                    if 'class_name' in detection and detection['class_name'] != 'unknown':
                        label_parts.append(detection['class_name'])
                    
                    # Agregar método de detección
                    if 'method' in detection:
                        label_parts.append(f"Método: {detection['method']}")
                    
                    # Agregar confianza si existe
                    if 'confidence' in detection:
                        label_parts.append(f"Conf: {detection['confidence']:.2f}")
                    
                    # Agregar texto de patente si existe
                    if 'plate_text' in detection and detection['plate_text']:
                        plate_text = detection['plate_text']
                        label_parts.append(f"Patente: {plate_text}")
                        
                        # Dibujar fondo para texto de patente
                        text_size = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                        cv2.rectangle(self.current_frame, 
                                    (x, y - text_size[1] - 15), 
                                    (x + text_size[0] + 10, y), 
                                    color, -1)
                        
                        # Dibujar texto de patente
                        cv2.putText(self.current_frame, plate_text, 
                                  (x + 5, y - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    # Dibujar etiqueta informativa
                    if label_parts:
                        info_text = " | ".join(label_parts)
                        text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                        
                        # Fondo para la etiqueta
                        cv2.rectangle(self.current_frame, 
                                    (x, y + h), 
                                    (x + text_size[0] + 10, y + h + text_size[1] + 10), 
                                    color, -1)
                        
                        # Texto de la etiqueta
                        cv2.putText(self.current_frame, info_text, 
                                  (x + 5, y + h + text_size[1] + 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
            except Exception as e:
                print(f"❌ Error dibujando detección: {e}")
                continue
    
    def _get_detection_color(self, detection):
        """Obtener color según el tipo de detección"""
        method = detection.get('method', 'unknown')
        is_plate = detection.get('is_license_plate', False)
        
        if is_plate:
            return (0, 255, 0)  # Verde para patentes detectadas
        elif 'yolo' in method:
            return (255, 0, 0)   # Azul para YOLO
        elif 'color' in method:
            return (0, 255, 255) # Amarillo para color
        elif 'shape' in method:
            return (0, 0, 255)   # Rojo para forma
        elif 'grid' in method:
            return (255, 0, 255) # Magenta para cuadrícula
        else:
            return (128, 128, 128) # Gris para otros
    
    def _draw_info_panel(self):
        """Dibujar panel de información en tiempo real"""
        if self.current_frame is None:
            return
            
        height, width = self.current_frame.shape[:2]
        
        # Crear panel lateral de información
        panel_width = 400
        panel = np.zeros((height, panel_width, 3), dtype=np.uint8)
        
        # Título del sistema
        cv2.putText(panel, "SISTEMA DE ANALISIS", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Información de procesamiento
        y_offset = 70
        cv2.putText(panel, "INFORMACION EN TIEMPO REAL:", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        y_offset += 30
        
        # Mostrar información de procesamiento
        if self.processing_info:
            cv2.putText(panel, f"Procesando: {self.processing_info}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
        
        # Mostrar estado del sistema
        if self.status_info:
            cv2.putText(panel, f"Estado: {self.status_info}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
        
        # Estadísticas de detección
        total_detections = len(self.detections)
        plate_detections = len([d for d in self.detections if d.get('is_license_plate', False)])
        
        cv2.putText(panel, f"Total detecciones: {total_detections}", (10, y_offset + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(panel, f"Patentes detectadas: {plate_detections}", (10, y_offset + 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Leyenda de métodos
        y_offset += 120
        cv2.putText(panel, "LEYENDA DE COLORES:", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        y_offset += 30
        
        legend_items = [
            ("Patente Detectada", (0, 255, 0)),
            ("YOLO (Objetos)", (255, 0, 0)),
            ("Color (Patentes)", (0, 255, 255)),
            ("Forma (Rectangulos)", (0, 0, 255)),
            ("Cuadricula", (255, 0, 255)),
            ("Otros", (128, 128, 128))
        ]
        
        for i, (text, color) in enumerate(legend_items):
            # Cuadro de color
            cv2.rectangle(panel, (10, y_offset + i*25), (30, y_offset + i*25 + 15), color, -1)
            cv2.rectangle(panel, (10, y_offset + i*25), (30, y_offset + i*25 + 15), (255, 255, 255), 1)
            
            # Texto
            cv2.putText(panel, text, (40, y_offset + i*25 + 12), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Combinar panel con la imagen principal
        combined = np.hstack([self.current_frame, panel])
        self.current_frame = combined
    
    def show(self):
        """Mostrar el frame actual"""
        if self.current_frame is not None and self.is_active:
            cv2.imshow(self.window_name, self.current_frame)
    
    def stop(self):
        """Detener la visualización"""
        self.is_active = False
        cv2.destroyWindow(self.window_name)
        print("🖥️  Visualización en tiempo real detenida")

class YOLOLicensePlateDetector:
    def __init__(self, use_yolo=True, yolo_model='yolov10n'):
        self.model = None
        self.detection_count = 0
        self.last_processing_time = 0
        self.processing_interval = 2
        self.use_yolo = use_yolo
        self.yolo_model = yolo_model
        self.visualizer = RealTimeVisualizer()
        
        if self.use_yolo:
            self.load_yolo_model()
        else:
            print("⚠️  YOLO deshabilitado, usando métodos tradicionales")
    
    def load_yolo_model(self):
        """Carga el modelo YOLO"""
        try:
            import torch
            from ultralytics import YOLO
            
            # Usar YOLOv8s para mejor precisión
            self.model = YOLO('yolov10n.pt')
            
            # Verificar dispositivo
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"✅ YOLOv8 cargado en dispositivo: {self.device}")
            
            print("✅ Modelo YOLO inicializado y listo para detección")
            
        except Exception as e:
            print(f"❌ ERROR CRÍTICO: No se pudo cargar YOLO: {e}")
            self.model = None

    def start_real_time_analysis(self, video_source=0):
        """Iniciar análisis en tiempo real con visualización"""
        try:
            print("🎬 INICIANDO ANALISIS EN TIEMPO REAL...")
            self.visualizer.start()
            
            # Inicializar captura de video
            cap = cv2.VideoCapture(video_source)
            if not cap.isOpened():
                print(f"❌ No se pudo abrir la fuente de video: {video_source}")
                return
            
            print(f"✅ Cámara inicializada: {video_source}")
            
            frame_count = 0
            processing_frame = False
            
            while True:
                # Leer frame
                ret, frame = cap.read()
                if not ret:
                    print("❌ No se pudo leer el frame")
                    break
                
                frame_count += 1
                
                # Procesar cada 5 frames para mejor performance
                if frame_count % 5 == 0 and not processing_frame:
                    processing_frame = True
                    
                    # Procesar en un hilo separado para no bloquear la visualización
                    def process_frame():
                        try:
                            # Actualizar información de procesamiento
                            self.visualizer.processing_info = f"Frame {frame_count} - Analizando..."
                            
                            # Realizar detección
                            detections = self.scan_entire_image_for_plates(frame)
                            
                            # Actualizar visualización con resultados
                            self.visualizer.update_frame(
                                frame, 
                                detections,
                                processing_info=f"Frame {frame_count} - {len(detections)} objetos",
                                status_info="Analisis activo"
                            )
                            
                        except Exception as e:
                            print(f"❌ Error procesando frame: {e}")
                            self.visualizer.update_frame(
                                frame, 
                                [],
                                processing_info=f"Frame {frame_count} - Error",
                                status_info="Error en analisis"
                            )
                        finally:
                            processing_frame = False
                    
                    # Ejecutar procesamiento en hilo separado
                    threading.Thread(target=process_frame, daemon=True).start()
                
                # Mostrar frame actual
                self.visualizer.show()
                
                # Salir con 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Liberar recursos
            cap.release()
            self.visualizer.stop()
            cv2.destroyAllWindows()
            print("✅ Análisis en tiempo real finalizado")
            
        except Exception as e:
            print(f"❌ Error en análisis en tiempo real: {e}")
            self.visualizer.stop()

    def corregir_patente(self, texto):
        # ... (mantener el mismo código de corrección de patentes)
        import numpy as np

        digit_to_number = {
            'S': '5',
            'O': '0',
            'I': '1',
            'B': '8',
        }

        number_to_digit = {
            '5': 'S',
            '0': 'O',
            '1': 'I',
            '8': 'B',
            '6': 'G',
            '2': 'Z',
            '%': 'A',
            '@': 'A',
            '4': 'A',
            '7': 'T',
            '4': 'V',
        }
        
        if not texto:
            return ""
        
        if texto:
            array_texto = np.array(texto.upper().split())
            print(len(array_texto))
            print(array_texto)
            
            if len(array_texto) == 3:
                for i in range(len(array_texto[0])):
                    char = array_texto[0][i]
                    if char in number_to_digit:
                        array_texto[0] = array_texto[0][:i] + number_to_digit[char] + array_texto[0][i+1:]
                
                for i in range(len(array_texto[1])):
                    char = array_texto[1][i]
                    if char in digit_to_number:
                        array_texto[1] = array_texto[1][:i] + digit_to_number[char] + array_texto[1][i+1:]
                
                for i in range(len(array_texto[2])):
                    char = array_texto[2][i]
                    if char in number_to_digit:
                        array_texto[2] = array_texto[2][:i] + number_to_digit[char] + array_texto[2][i+1:]
            elif len(array_texto) == 2:
                for i in range(len(array_texto[0])):
                    char = array_texto[0][i]
                    if char in number_to_digit:
                        array_texto[0] = array_texto[0][:i] + number_to_digit[char] + array_texto[0][i+1:]
                
                for i in range(len(array_texto[1])):
                    char = array_texto[1][i]
                    if char in digit_to_number:
                        array_texto[1] = array_texto[1][:i] + digit_to_number[char] + array_texto[1][i+1:]
            
            corr = " ".join(array_texto)
            return corr
        else:
            corr = " ".join(array_texto)
            return corr

    def scan_entire_image_for_plates(self, image):
        """Escanea toda la imagen en busca de patentes usando múltiples métodos"""
        plate_candidates = []
        
        print("🔍 Escaneando toda la imagen para patentes...")
        
        # Método 1: Detección YOLO de objetos
        yolo_detections = self.detect_objects_yolo(image)
        plate_candidates.extend(yolo_detections)
        print(f"   ✅ YOLO encontró {len(yolo_detections)} objetos")
        
        # Método 2: Búsqueda por características de forma
        shape_based_candidates = self.find_plates_by_shape(image)
        plate_candidates.extend(shape_based_candidates)
        print(f"   ✅ Búsqueda por forma encontró {len(shape_based_candidates)} candidatos")
        
        # Método 3: Búsqueda por color
        color_based_candidates = self.find_plates_by_color(image)
        plate_candidates.extend(color_based_candidates)
        print(f"   ✅ Búsqueda por color encontró {len(color_based_candidates)} candidatos")
        
        # Método 4: División en cuadrícula
        grid_candidates = self.scan_image_grid(image)
        plate_candidates.extend(grid_candidates)
        print(f"   ✅ Escaneo en cuadrícula encontró {len(grid_candidates)} candidatos")
        
        # Procesar candidatos para identificar patentes
        final_detections = []
        for candidate in plate_candidates:
            # Asegurar que todas las claves necesarias existan
            detection_info = {
                'bbox': candidate.get('bbox', (0, 0, 0, 0)),
                'class_name': candidate.get('class_name', 'unknown'),
                'confidence': candidate.get('confidence', 0.0),
                'method': candidate.get('method', 'unknown'),
                'class_id': candidate.get('class_id', -1),  # Valor por defecto -1
                'region': candidate.get('region', None),
                'is_license_plate': False,
                'plate_text': None,
                'plate_confidence': 0.0
            }
            
            # Verificar si es patente potencial
            if self.is_potential_license_plate(candidate, image.shape):
                plate_text, plate_confidence = self.recognize_plate_text(candidate['region'])
                if plate_text:
                    detection_info.update({
                        'is_license_plate': True,
                        'plate_text': plate_text,
                        'plate_confidence': plate_confidence
                    })
                    print(f"✅ PATENTE ENCONTRADA: {plate_text}")
            
            final_detections.append(detection_info)
        
        return final_detections
    
    '''def scan_image_grid_comprehensive(self, image, grid_size=1):
        """Divide la imagen en cuadrícula y en cada celda busca por color y forma"""
        try:
            candidates = []
            height, width = image.shape[:2]
            
            cell_height = height // grid_size
            cell_width = width // grid_size
            
            print(f"🔍 Escaneando {grid_size}x{grid_size} cuadrícula ({grid_size*grid_size} celdas)...")
            
            for i in range(grid_size):
                for j in range(grid_size):
                    # Calcular coordenadas de la celda
                    y1 = i * cell_height
                    y2 = min((i + 1) * cell_height, height)
                    x1 = j * cell_width
                    x2 = min((j + 1) * cell_width, width)
                    
                    # Extraer celda
                    cell = image[y1:y2, x1:x2]
                    
                    if cell.size > 0:
                        # BUSCAR POR FORMA en esta celda
                        shape_candidates = self.find_plates_by_shape(cell)
                        
                        # BUSCAR POR COLOR en esta celda
                        color_candidates = self.find_plates_by_color(cell)
                        
                        # Procesar candidatos por forma
                        for candidate in shape_candidates:
                            # Convertir coordenadas relativas a absolutas
                            abs_x = x1 + candidate['bbox'][0]
                            abs_y = y1 + candidate['bbox'][1]
                            abs_w = candidate['bbox'][2]
                            abs_h = candidate['bbox'][3]
                            
                            candidates.append({
                                'region': candidate['region'],
                                'bbox': (abs_x, abs_y, abs_w, abs_h),
                                'confidence': candidate['confidence'] * 0.9,
                                'class_name': f"grid_shape_{i}_{j}",
                                'class_id': -3,
                                'method': f'grid_shape_{i}_{j}'
                            })
                        
                        # Procesar candidatos por color
                        for candidate in color_candidates:
                            # Convertir coordenadas relativas a absolutas
                            abs_x = x1 + candidate['bbox'][0]
                            abs_y = y1 + candidate['bbox'][1]
                            abs_w = candidate['bbox'][2]
                            abs_h = candidate['bbox'][3]
                            
                            candidates.append({
                                'region': candidate['region'],
                                'bbox': (abs_x, abs_y, abs_w, abs_h),
                                'confidence': candidate['confidence'] * 0.9,
                                'class_name': f"grid_color_{i}_{j}",
                                'class_id': -4,
                                'method': f'grid_color_{i}_{j}'
                            })
                        
                        # Mostrar progreso por celda
                        if shape_candidates or color_candidates:
                            print(f"   📍 Celda [{i},{j}]: {len(shape_candidates)} forma, {len(color_candidates)} color")
            
            return candidates
            
        except Exception as e:
            print(f"❌ Error en escaneo de cuadrícula comprehensivo: {e}")
            return []'''    
    def detect_objects_yolo(self, image):
        """Detección de objetos usando YOLO en toda la imagen"""
        try:
            if self.model is None:
                return []
            
            # Realizar detección con baja confianza para capturar más objetos
            results = self.model(image, verbose=False, conf=0.25, iou=0.4)
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        # Obtener coordenadas y confianza
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = result.names[class_id]
                        
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Extraer región
                        object_region = image[y1:y2, x1:x2]
                        
                        if object_region.size > 0:
                            detections.append({
                                'region': object_region,
                                'bbox': (x1, y1, x2-x1, y2-y1),
                                'confidence': float(confidence),
                                'class_id': class_id,
                                'class_name': class_name,
                                'method': 'yolo'
                            })
            
            return detections
            
        except Exception as e:
            print(f"❌ Error en detección YOLO: {e}")
            return []
    
    def find_plates_by_shape(self, image):
        """Busca patentes basándose en características de forma rectangular"""
        try:
            candidates = []
            
            # Múltiples métodos de detección de bordes
            edge_methods = [
                cv2.Canny(image, 30, 100),
                cv2.Canny(image, 50, 150),
                cv2.Canny(image, 70, 200)
            ]
            
            for i, edges in enumerate(edge_methods):
                # Operaciones morfológicas para mejorar los contornos
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                dilated = cv2.dilate(edges, kernel, iterations=1)
                closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
                
                # Encontrar contornos
                contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    # Filtrar por área
                    area = cv2.contourArea(contour)
                    if area < 200 or area > 50000:
                        continue
                    
                    # Obtener rectángulo del contorno
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Filtrar por relación de aspecto (patentes son rectangulares)
                    aspect_ratio = w / float(h)
                    if aspect_ratio < 1.5 or aspect_ratio > 6.0:
                        continue
                    
                    # Filtrar por tamaño razonable
                    if w < 40 or h < 15 or w > 400 or h > 150:
                        continue
                    
                    # Verificar solidez (qué tan rectangular es)
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    if hull_area > 0:
                        solidity = float(area) / hull_area
                        if solidity < 0.7:
                            continue
                    
                    # Extraer región candidata
                    plate_region = image[y:y+h, x:x+w]
                    
                    if plate_region.size > 0:
                        candidates.append({
                            'region': plate_region,
                            'bbox': (x, y, w, h),
                            'confidence': 0.3,
                            'class_name': 'shape_candidate',
                            'class_id': -1,  # Añadir class_id
                            'method': f'shape_method_{i}'
                        })
            
            return candidates
            
        except Exception as e:
            print(f"❌ Error en búsqueda por forma: {e}")
            return []

    def find_plates_by_color(self, image):
        """Busca patentes basándose en combinaciones de colores comunes"""
        try:
            candidates = []
            
            # Combinaciones de color comunes para patentes
            color_combinations = [
                # Patentes blancas (fondo blanco, texto oscuro)
                {
                    'name': 'white_plate',
                    'mask1': cv2.inRange(image, (200, 200, 200), (255, 255, 255)),
                    'confidence': 0.4
                },
                # Patentes negras (fondo negro, texto blanco)
                {
                    'name': 'black_plate',
                    'mask1': cv2.inRange(image, (0, 0, 0), (50, 50, 50)),
                    'confidence': 0.3
                }
            ]
            
            for color_combo in color_combinations:
                mask = color_combo['mask1']
                
                # Operaciones morfológicas para limpiar la máscara
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                
                # Encontrar contornos en la máscara
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < 500 or area > 20000:
                        continue
                    
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Filtrar por relación de aspecto
                    aspect_ratio = w / float(h)
                    if aspect_ratio < 1.8 or aspect_ratio > 5.5:
                        continue
                    
                    plate_region = image[y:y+h, x:x+w]
                    
                    if plate_region.size > 0:
                        candidates.append({
                            'region': plate_region,
                            'bbox': (x, y, w, h),
                            'confidence': color_combo['confidence'],
                            'class_name': f"color_{color_combo['name']}",
                            'class_id': -2,  # Añadir class_id
                            'method': 'color_based'
                        })
            
            return candidates
            
        except Exception as e:
            print(f"❌ Error en búsqueda por color: {e}")
            return []

    def scan_image_grid_comprehensive(self, image, grid_size=1):
        """Divide la imagen en cuadrícula y en cada celda busca por color y forma"""
        try:
            candidates = []
            height, width = image.shape[:2]
            
            cell_height = height // grid_size
            cell_width = width // grid_size
            
            print(f"🔍 Escaneando {grid_size}x{grid_size} cuadrícula ({grid_size*grid_size} celdas)...")
            
            for i in range(grid_size):
                for j in range(grid_size):
                    # Calcular coordenadas de la celda
                    y1 = i * cell_height
                    y2 = min((i + 1) * cell_height, height)
                    x1 = j * cell_width
                    x2 = min((j + 1) * cell_width, width)
                    
                    # Extraer celda
                    cell = image[y1:y2, x1:x2]
                    
                    if cell.size > 0:
                        # BUSCAR POR FORMA en esta celda
                        shape_candidates = self.find_plates_by_shape(cell)
                        
                        # BUSCAR POR COLOR en esta celda
                        color_candidates = self.find_plates_by_color(cell)
                        
                        # Procesar candidatos por forma
                        for candidate in shape_candidates:
                            # Convertir coordenadas relativas a absolutas
                            abs_x = x1 + candidate['bbox'][0]
                            abs_y = y1 + candidate['bbox'][1]
                            abs_w = candidate['bbox'][2]
                            abs_h = candidate['bbox'][3]
                            
                            candidates.append({
                                'region': candidate['region'],
                                'bbox': (abs_x, abs_y, abs_w, abs_h),
                                'confidence': candidate['confidence'] * 0.9,
                                'class_name': f"grid_shape_{i}_{j}",
                                'class_id': -3,  # Añadir class_id
                                'method': f'grid_shape_{i}_{j}'
                            })
                        
                        # Procesar candidatos por color
                        for candidate in color_candidates:
                            # Convertir coordenadas relativas a absolutas
                            abs_x = x1 + candidate['bbox'][0]
                            abs_y = y1 + candidate['bbox'][1]
                            abs_w = candidate['bbox'][2]
                            abs_h = candidate['bbox'][3]
                            
                            candidates.append({
                                'region': candidate['region'],
                                'bbox': (abs_x, abs_y, abs_w, abs_h),
                                'confidence': candidate['confidence'] * 0.9,
                                'class_name': f"grid_color_{i}_{j}",
                                'class_id': -4,  # Añadir class_id
                                'method': f'grid_color_{i}_{j}'
                            })
                        
                        # Mostrar progreso por celda
                        if shape_candidates or color_candidates:
                            print(f"   📍 Celda [{i},{j}]: {len(shape_candidates)} forma, {len(color_candidates)} color")
            
            return candidates
            
        except Exception as e:
            print(f"❌ Error en escaneo de cuadrícula comprehensivo: {e}")
            return []
    
    def scan_image_grid(self, image, grid_size=1):
        """Método original mantenido por compatibilidad"""
        return self.scan_image_grid_comprehensive(image, grid_size)
    
    def is_potential_license_plate(self, detection, image_shape):
        """Filtra candidatos para identificar patentes potenciales"""
        try:
            x, y, w, h = detection['bbox']
            
            # Filtros básicos
            aspect_ratio = w / float(h)
            area = w * h
            image_area = image_shape[0] * image_shape[1]
            
            # Patentes típicas tienen aspect ratio entre 2:1 y 5:1
            if aspect_ratio < 1.5 or aspect_ratio > 6.0:
                return False
            
            # Área razonable para una patente
            if area < 300 or area > (image_area * 0.2):
                return False
            
            # Tamaño mínimo y máximo
            if w < 30 or h < 10 or w > 500 or h > 200:
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Error filtrando patente: {e}")
            return False
    
    def preprocess_plate_image(self, plate_image):
        """Preprocesar imagen de patente para OCR"""
        try:
            if plate_image.size == 0:
                return None
            
            # Redimensionar para mejor OCR
            target_height = 100
            scale = target_height / plate_image.shape[0]
            target_width = int(plate_image.shape[1] * scale)
            
            resized = cv2.resize(plate_image, (target_width, target_height))
            
            # Convertir a escala de grises
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            # Mejorar contraste agresivamente
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(thresh)
            
            # Suavizado para reducir ruido
            smoothed = cv2.medianBlur(enhanced, 3)
            
            # Binarización adaptativa
            binary = cv2.adaptiveThreshold(smoothed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 8)
            
            # Operaciones morfológicas para conectar caracteres
            kernel = np.ones((2, 1), np.uint8)
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            return cleaned
            
        except Exception as e:
            print(f"❌ Error preprocesando imagen: {e}")
            return None
    
    def recognize_plate_text(self, plate_image):
        """Reconocer texto de patentes usando EasyOCR, Tesseract o PaddleOCR"""
        try:
            import easyocr
            
            # Inicializar EasyOCR una sola vez para mejor performance
            if not hasattr(self, 'easyocr_reader'):
                self.easyocr_reader = easyocr.Reader(['es'], gpu=False)
            
            processed_image = self.preprocess_for_easyocr(plate_image)
            if processed_image is None or processed_image.size == 0:
                return None, 0.0
            
            # EasyOCR como método principal
            results = self.easyocr_reader.readtext(
                processed_image,
                decoder='beamsearch',
                beamWidth=10,
                batch_size=1,
                contrast_ths=0.1,
                adjust_contrast=0.5,
                text_threshold=0.7,
                link_threshold=0.4,
                mag_ratio=1.5
            )
            
            best_text, best_conf = "", 0.0
            for _, text, conf in results:
                cleaned_text = self.corregir_patente(text)
                if self.is_valid_plate_format(cleaned_text) and conf > best_conf:
                    best_text, best_conf = cleaned_text, conf
            
            if best_text:
                return best_text, best_conf

            # Fallback a Tesseract
            text_t, conf_t = self.recognize_plate_text_tesseract(plate_image)
            if text_t:
                return text_t, conf_t

            return None, 0.0

        except Exception as e:
            print(f"❌ Error en reconocimiento de texto: {e}")
            # Intentar con métodos de fallback
            text_t, conf_t = self.recognize_plate_text_tesseract(plate_image)
            if text_t:
                return text_t, conf_t

    def preprocess_for_easyocr(self, plate_image):
        """Preprocesamiento optimizado para EasyOCR"""
        try:
            if plate_image.size == 0:
                return None
            
            # 1. Redimensionar para mejor reconocimiento
            height, width = plate_image.shape[:2]
            target_height = 100
            scale = target_height / height
            target_width = int(width * scale)
            resized = cv2.resize(plate_image, (target_width, target_height))
            
            # 2. Mejorar contraste y brillo
            lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # CLAHE para mejorar contraste
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            lab = cv2.merge((l, a, b))
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # 3. Enfocar la imagen ligeramente
            kernel = np.array([[-1, -1, -1], 
                            [-1, 9, -1], 
                            [-1, -1, -1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            
            return sharpened
            
        except Exception as e:
            print(f"❌ Error en preprocesamiento EasyOCR: {e}")
            return plate_image

    def recognize_plate_text_tesseract(self, plate_image):
        """Método de fallback con Tesseract (por si EasyOCR falla)"""
        try:
            import pytesseract
            
            enhanced = self.preprocess_plate_image(plate_image)
            if enhanced is None:
                return None, 0.0
            
            configs = [
                '--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            ]
            
            best_text = ""
            best_confidence = 0
            
            for config in configs:
                try:
                    data = pytesseract.image_to_data(enhanced, config=config, output_type=pytesseract.Output.DICT)
                    
                    text_parts = []
                    total_conf = 0
                    count = 0
                    
                    for i in range(len(data['text'])):
                        confidence = int(data['conf'][i])
                        text = data['text'][i].strip()
                        
                        if confidence > 25 and text:
                            text_parts.append(text)
                            total_conf += confidence
                            count += 1
                    
                    if count > 0:
                        avg_conf = total_conf / count
                        combined_text = ''.join(text_parts).upper()
                        clean_text = self.clean_ocr_text(combined_text)
                        
                        if self.is_valid_plate_format(clean_text) and avg_conf > best_confidence:
                            best_text = clean_text
                            best_confidence = avg_conf / 100.0
                            
                except Exception:
                    continue
            
            return best_text, best_confidence if best_text else (None, 0.0)
            
        except Exception as e:
            print(f"❌ Error en Tesseract fallback: {e}")
            return None, 0.0
    
    def is_valid_plate_format(self, text):
        """Valida formatos de patentes de manera flexible incluyendo espacios"""
        if not text or len(text) < 4 or len(text) > 12:
            return False
        
        # Limpiar texto pero mantener espacios temporalmente para validación
        clean_text = ''.join(c for c in text.upper() if c.isalnum() or c.isspace())
        
        # Remover espacios para contar letras y números
        text_without_spaces = clean_text.replace(' ', '')
        
        if len(text_without_spaces) < 4:
            return False
        
        # Contar letras y números
        letters = sum(1 for c in text_without_spaces if c.isalpha())
        digits = sum(1 for c in text_without_spaces if c.isdigit())
        
        # Debe tener mezcla de letras y números
        if letters < 1 or digits < 1:
            return False
        
        # ✅ NUEVOS FORMATOS CON ESPACIOS
        import re
        formats = [
            # Formatos sin espacios
            r'^[A-Z]{3}\d{3}$',                   # AAA111
            r'^[A-Z]{2}\d{3}[A-Z]{2}$',           # AA111BB  
            r'^\d{3}[A-Z]{3}$',                   # 111AAA
            
            # ✅ Formatos CON espacios
            r'^[A-Z]{3} \d{3}$',                  # AAA 111
            r'^[A-Z]{2} \d{3} [A-Z]{2}$',         # AA 111 BB
            r'^\d{3} [A-Z]{3}$',                  # 111 AAA
        ]
        
        for pattern in formats:
            if re.match(pattern, clean_text):
                return True
        
        # ✅ También aceptar cualquier combinación razonable con espacios
        return (letters >= 1 and digits >= 1 and 
                letters + digits >= 4 and
                len(clean_text.replace(' ', '')) == letters + digits)
    
    def detect_plate(self, image):
        """Detección principal - Analiza TODA la imagen"""
        try:
            if self.model is None:
                print("❌ Modelo YOLO no disponible")
                return []
            
            current_time = time.time()
            
            # Limitar frecuencia de procesamiento
            if current_time - self.last_processing_time < self.processing_interval:
                return []
            
            self.last_processing_time = current_time
            
            print(f"🔍 ANALIZANDO IMAGEN COMPLETA para patentes...")
            
            # Escanear toda la imagen usando múltiples métodos
            all_candidates = self.scan_entire_image_for_plates(image)
            
            print(f"🎯 Encontrados {len(all_candidates)} candidatos totales")
            
            plates_found = []
            
            # Procesar cada candidato
            for candidate in all_candidates:
                if self.is_potential_license_plate(candidate, image.shape):
                    # Intentar leer texto de la patente
                    plate_text, ocr_confidence = self.recognize_plate_text(candidate['region'])
                    
                    if plate_text:
                        # Calcular confianza combinada
                        combined_confidence = (candidate['confidence'] + ocr_confidence) / 2
                        
                        self.detection_count += 1
                        
                        plates_found.append({
                            'text': plate_text,
                            'region': candidate['region'],
                            'coordinates': candidate['bbox'],
                            'confidence': combined_confidence,
                            'yolo_confidence': candidate['confidence'],
                            'ocr_confidence': ocr_confidence,
                            'class_name': candidate['class_name'],
                            'method': candidate['method']
                        })
                        
                        print(f"✅ PATENTE DETECTADA: {plate_text}")
                        print(f"   📊 Método: {candidate['method']}")
                        print(f"   📊 Confianza: {combined_confidence:.2f}")
                        print(f"   📊 OCR: {ocr_confidence:.2f}")
                
            print(f"📈 Procesamiento completado: {len(plates_found)} patente(s) encontrada(s)")
            return plates_found
            
        except Exception as e:
            print(f"❌ Error en detección de patentes: {e}")
            return []
        
class ImageAnalyzer:
    def __init__(self, use_yolo=True, yolo_model='yolov10n'):
        self.yolo_detector = YOLOLicensePlateDetector(use_yolo=use_yolo, yolo_model=yolo_model)
        print("✅ Analizador de imágenes completo inicializado")

    def start_real_time_analysis(self, video_source=0):
        """Iniciar análisis en tiempo real"""
        self.yolo_detector.start_real_time_analysis(video_source)

    def analyze_image(self, image_path):
        """Analiza una imagen COMPLETA y detecta objetos y patentes"""
        with Timer(f"Procesamiento de imagen: {os.path.basename(image_path)}"):
            try:
                print(f"🔍 ANALIZANDO IMAGEN COMPLETA: {image_path}")
                
                # Cargar imagen
                image = cv2.imread(image_path)
                if image is None:
                    return {"error": "No se pudo cargar la imagen"}
                
                print(f"✅ Imagen cargada: {image.shape[1]}x{image.shape[0]}")
                
                # Realizar detección COMPLETA de la imagen
                detections = self.yolo_detector.scan_entire_image_for_plates(image)
                
                # Procesar resultados
                analysis_result = {
                    "objects_detected": [],
                    "license_plates": [],
                    "image_info": {
                        "width": image.shape[1],
                        "height": image.shape[0],
                        "channels": image.shape[2] if len(image.shape) > 2 else 1
                    },
                    "scan_methods_used": ["yolo", "shape", "color", "grid"],
                    "total_candidates": len(detections)
                }
                
                # Procesar cada detección
                for detection in detections:
                    # Validar y asegurar que todas las claves existan
                    object_info = {
                        "class_name": detection.get('class_name', 'unknown'),
                        "class_id": detection.get('class_id', -1),  # Valor por defecto
                        "confidence": detection.get('confidence', 0.0),
                        "bbox": detection.get('bbox', (0, 0, 0, 0)),
                        "method": detection.get('method', 'unknown'),
                        "is_license_plate": detection.get('is_license_plate', False),
                        "plate_text": detection.get('plate_text', None),
                        "plate_confidence": detection.get('plate_confidence', 0.0)
                    }
                    
                    # Verificar si es una patente detectada
                    if object_info["is_license_plate"] and object_info["plate_text"]:
                        # Agregar a la lista de patentes
                        analysis_result["license_plates"].append({
                            "text": object_info["plate_text"],
                            "confidence": object_info["plate_confidence"],
                            "yolo_confidence": object_info["confidence"],
                            "combined_confidence": (object_info["confidence"] + object_info["plate_confidence"]) / 2,
                            "bbox": object_info["bbox"],
                            "method": object_info["method"]
                        })
                    
                    # Siempre agregar a objetos detectados
                    analysis_result["objects_detected"].append(object_info)
                
                print(f"✅ Análisis COMPLETO: {len(analysis_result['objects_detected'])} objetos, {len(analysis_result['license_plates'])} patentes")
                return analysis_result
                
            except Exception as e:
                print(f"❌ Error analizando imagen: {e}")
                import traceback
                traceback.print_exc()
                return {"error": str(e)}
    
    def create_annotated_image(self, image_path, analysis_result):
        """Crea una imagen anotada con todas las detecciones"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            # Crear copia para anotaciones
            annotated_image = image.copy()
            
            # Dibujar todos los objetos detectados
            for obj in analysis_result.get("objects_detected", []):
                bbox = obj.get("bbox", (0, 0, 0, 0))
                if len(bbox) != 4:
                    continue
                    
                x, y, w, h = bbox
                
                # Elegir color según el tipo de objeto y método
                color_map = {
                    'yolo': (255, 0, 0),      # Azul
                    'color_based': (0, 255, 0), # Verde
                    'shape_method_0': (0, 0, 255), # Rojo
                    'shape_method_1': (0, 0, 255),
                    'shape_method_2': (0, 0, 255),
                    'grid': (255, 255, 0)     # Cian
                }
                
                color = color_map.get(obj.get('method', ''), (128, 128, 128))
                
                # Si es una patente, usar verde brillante
                if obj.get("is_license_plate", False):
                    color = (0, 255, 0)
                    thickness = 3
                    
                    # Dibujar texto de la patente
                    plate_text = obj.get("plate_text", "")
                    if plate_text:
                        # Fondo para el texto
                        text_size = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                        cv2.rectangle(annotated_image, 
                                    (x, y - text_size[1] - 10), 
                                    (x + text_size[0], y), 
                                    color, -1)
                        
                        # Texto
                        cv2.putText(annotated_image, plate_text, 
                                (x, y - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    thickness = 2
                
                # Dibujar bounding box
                cv2.rectangle(annotated_image, (x, y), (x + w, y + h), color, thickness)
                
                # Etiqueta del método y clase
                label = f"{obj.get('class_name', 'unknown')} ({obj.get('method', 'unknown')})"
                if not obj.get("is_license_plate", False):
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                    cv2.rectangle(annotated_image, 
                                (x, y + h), 
                                (x + label_size[0], y + h + label_size[1] + 5), 
                                color, -1)
                    cv2.putText(annotated_image, label, 
                            (x, y + h + label_size[1]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Agregar leyenda de métodos
            self.add_legend(annotated_image, color_map)
            
            print("✅ Imagen anotada creada correctamente")
            return annotated_image
            
        except Exception as e:
            print(f"❌ Error creando imagen anotada: {e}")
            return None
    
    def add_legend(self, image, color_map):
        """Agrega una leyenda con los colores de los métodos"""
        try:
            height, width = image.shape[:2]
            
            # Crear área de leyenda
            legend_height = 120
            legend_y = 10
            
            # Fondo semitransparente para la leyenda
            overlay = image.copy()
            cv2.rectangle(overlay, (10, legend_y), (300, legend_y + legend_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
            
            # Título
            cv2.putText(image, "LEYENDA - METODOS DE DETECCION", 
                      (20, legend_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Items de la leyenda
            legend_items = [
                ("YOLO (Objetos)", color_map['yolo']),
                ("Color (Patentes)", color_map['color_based']),
                ("Forma (Rectangulos)", color_map['shape_method_0']),
                ("Cuadricula (Busqueda)", color_map['grid']),
                ("Patente Detectada", (0, 255, 0))
            ]
            
            y_offset = legend_y + 40
            for i, (text, color) in enumerate(legend_items):
                # Cuadro de color
                cv2.rectangle(image, (20, y_offset + i*20), (40, y_offset + i*20 + 15), color, -1)
                cv2.rectangle(image, (20, y_offset + i*20), (40, y_offset + i*20 + 15), (255, 255, 255), 1)
                
                # Texto
                cv2.putText(image, text, (50, y_offset + i*20 + 12), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
        except Exception as e:
            print(f"❌ Error agregando leyenda: {e}")

class MotionDetector:
    def __init__(self):
        self.background = None
        self.motion_count = 0
        self.last_motion_time = 0
    
    def detect_motion(self, frame):
        """Detección simple de movimiento"""
        try:
            small_frame = cv2.resize(frame, (320, 240))
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            
            if self.background is None:
                self.background = gray
                return False
            
            frame_delta = cv2.absdiff(self.background, gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            motion_detected = False
            for contour in contours:
                if cv2.contourArea(contour) > 1000:
                    motion_detected = True
                    break
            
            if not motion_detected:
                cv2.accumulateWeighted(gray, self.background, 0.01)
            
            if motion_detected:
                current_time = time.time()
                if current_time - self.last_motion_time > 2:
                    self.last_motion_time = current_time
                    self.motion_count += 1
                    print(f"🎯 Movimiento detectado (#{self.motion_count})")
                    return True
            
            return False
            
        except Exception as e:
            print(f"❌ Error en detección de movimiento: {e}")
            return False

def save_detection(plate_data, image, user=None):
    """Guarda detección en base de datos con filtro por usuario"""
    try:
        plates_dir = os.path.join(settings.MEDIA_ROOT, 'plates')
        os.makedirs(plates_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"plate_{timestamp}.jpg"
        filepath = os.path.join(plates_dir, filename)
        
        debug_image = image.copy()
        x, y, w, h = plate_data['coordinates']
        
        cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(debug_image, plate_data['text'], (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imwrite(filepath, debug_image)
        
        # BUSCAR VEHÍCULO DEL USUARIO ACTUAL
        vehicle = None
        is_client = False
        
        if user and user.is_authenticated:
            try:
                # Obtener el cliente asociado al usuario
                client = Client.objects.get(user=user)
                # Buscar vehículo por patente Y que pertenezca al cliente
                vehicle = Vehicle.objects.filter(
                    client=client, 
                    plate_number=plate_data['text']
                ).first()
                
                if vehicle:
                    is_client = True
                    print(f"✅ VEHÍCULO ENCONTRADO: {plate_data['text']} para usuario {user.username}")
                else:
                    print(f"⚠️ Vehículo NO encontrado: {plate_data['text']} para usuario {user.username}")
                    
            except Client.DoesNotExist:
                print(f"❌ Cliente no encontrado para usuario {user.username}")
        
        detection = PlateDetection(
            user=user,  # Asignar usuario
            plate_number=plate_data['text'],
            confidence=plate_data.get('confidence', 0.8),
            image=f'plates/{filename}',
            is_client=is_client,
            vehicle=vehicle
        )
        detection.save()
        
        status = "✅ CLIENTE" if is_client else "⚠️ NO CLIENTE"
        print(f"💾 DETECCIÓN GUARDADA: {plate_data['text']} - {status} - Usuario: {user.username if user else 'Anónimo'}")
        
        return detection
        
    except Exception as e:
        print(f"❌ Error guardando detección: {e}")
        return None

# ✅ INSTANCIAS GLOBALES
image_analyzer = ImageAnalyzer(use_yolo=True, yolo_model='yolov10n')
motion_detector = MotionDetector()

print("🎯 Sistema de análisis con YOLO listo!")

def start_real_time_detection():
    """Función para iniciar detección en tiempo real desde Django"""
    try:
        print("🚀 INICIANDO DETECCIÓN EN TIEMPO REAL DESDE DJANGO...")
        image_analyzer.start_real_time_analysis(video_source=0)
        return True
    except Exception as e:
        print(f"❌ Error iniciando detección en tiempo real: {e}")
        return False