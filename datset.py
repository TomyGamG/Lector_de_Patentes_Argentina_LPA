# dataset_desde_nombres.py
import sys
import os
import csv
import shutil
import random
import re

print(f"🐍 Python: {sys.version}")

# Manejo robusto de imports
try:
    import numpy as np
    print("✅ numpy importado correctamente")
except ImportError as e:
    print(f"❌ Error importando numpy: {e}")
    sys.exit(1)

try:
    import cv2
    print("✅ opencv-python importado correctamente")
except ImportError as e:
    print(f"❌ Error importando opencv-python: {e}")
    sys.exit(1)

try:
    import yaml
    print("✅ PyYAML importado correctamente")
except ImportError as e:
    print(f"❌ Error importando PyYAML: {e}")
    sys.exit(1)

try:
    from ultralytics import YOLO
    print("✅ ultralytics importado correctamente")
except ImportError as e:
    print(f"❌ Error importando ultralytics: {e}")
    sys.exit(1)

# sklearn es opcional
try:
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
    print("✅ scikit-learn importado correctamente")
except ImportError as e:
    print(f"⚠️  scikit-learn no disponible, usando división manual: {e}")
    SKLEARN_AVAILABLE = False

class DatasetDesdeNombres:
    """Creador de dataset que extrae patentes desde nombres de archivo"""
    
    def __init__(self, images_dir, output_dir="dataset_patentes"):
        self.images_dir = images_dir
        self.output_dir = output_dir
        self.data = []
        
    def extraer_patente_desde_nombre(self, filename):
        """Extrae la patente desde el nombre del archivo"""
        # Ejemplo: "AA028MY_0.png" -> "AA028MY"
        # Ejemplo: "dataset\\AA028MY_0.png" -> "AA028MY"
        
        # Limpiar path y extensión
        base_name = os.path.basename(filename)
        name_without_ext = os.path.splitext(base_name)[0]
        
        # Extraer la parte de la patente (antes del _)
        if '_' in name_without_ext:
            patente = name_without_ext.split('_')[0]
        else:
            patente = name_without_ext
        
        # Validar formato de patente argentina
        if self.es_patente_valida(patente):
            return patente
        else:
            print(f"⚠️  Patente no válida en archivo: {filename} -> {patente}")
            return None
    
    def es_patente_valida(self, patente):
        """Valida si el texto tiene formato de patente argentina"""
        # Formatos comunes de patentes argentinas
        patrones = [
            r'^[A-Z]{2}\d{3}[A-Z]{2}$',    # AA123BB (nuevo formato)
            r'^[A-Z]{3}\d{3}$',            # AAA123 (viejo formato)
            r'^[A-Z]{2}\d{4}$',            # AA1234 (algunas motos)
            r'^[A-Z]\d{3}[A-Z]{3}$',       # A123BBB (Mercosur)
        ]
        
        for patron in patrones:
            if re.match(patron, patente):
                return True
        return False
    
    def buscar_imagenes(self):
        """Busca todas las imágenes en el directorio"""
        print(f"🔍 Buscando imágenes en: {self.images_dir}")
        
        extensiones = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
        imagenes = []
        
        for root, dirs, files in os.walk(self.images_dir):
            for file in files:
                if file.lower().endswith(extensiones):
                    ruta_completa = os.path.join(root, file)
                    imagenes.append(ruta_completa)
        
        print(f"✅ Encontradas {len(imagenes)} imágenes")
        return imagenes
    
    def procesar_imagenes(self):
        """Procesa todas las imágenes y extrae patentes"""
        imagenes = self.buscar_imagenes()
        
        for ruta_imagen in imagenes:
            patente = self.extraer_patente_desde_nombre(ruta_imagen)
            
            if patente:
                self.data.append({
                    'filename': os.path.basename(ruta_imagen),
                    'ruta_completa': ruta_imagen,
                    'patente': patente
                })
        
        print(f"📊 Patentes extraídas: {len(self.data)}/{len(imagenes)}")
        
        # Mostrar ejemplos
        if self.data:
            print("\n🔤 Ejemplos de patentes extraídas:")
            for i, item in enumerate(self.data[:5]):
                print(f"   {i+1}. {item['filename']} -> {item['patente']}")
    
    def crear_estructura_directorios(self):
        """Crea la estructura de directorios para YOLO"""
        directorios = [
            'train/images', 'train/labels',
            'val/images', 'val/labels',
            'test/images', 'test/labels'
        ]
        
        for directorio in directorios:
            path = os.path.join(self.output_dir, directorio)
            os.makedirs(path, exist_ok=True)
            print(f"📁 Creado: {path}")
    
    def estimar_coordenadas_automaticas(self, imagen_path):
        """Estima coordenadas automáticamente para la patente"""
        try:
            img = cv2.imread(imagen_path)
            if img is None:
                return 100, 100, 300, 200  # Coordenadas por defecto
            
            altura, ancho = img.shape[:2]
            
            # Diferentes estrategias de estimación
            
            # 1. Buscar áreas con texto usando detección de bordes
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Encontrar contornos
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 1000 < area < 50000:  # Área típica de patentes
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / float(h)
                    
                    # Las patentes suelen tener relación de aspecto entre 2:1 y 5:1
                    if 1.5 < aspect_ratio < 6.0:
                        return x, y, x + w, y + h
            
            # 2. Si no encuentra contornos adecuados, usar posición por defecto
            # Asumir que la patente está en la parte inferior central
            x1 = int(ancho * 0.1)
            y1 = int(altura * 0.7)
            x2 = int(ancho * 0.9)
            y2 = int(altura * 0.9)
            
            return x1, y1, x2, y2
            
        except Exception as e:
            print(f"⚠️  Error estimando coordenadas para {imagen_path}: {e}")
            return 100, 100, 300, 200  # Coordenadas por defecto
    
    def dividir_dataset(self):
        """Divide el dataset en train/val/test"""
        if not self.data:
            print("❌ No hay datos para dividir")
            return None, None, None
        
        # Si sklearn está disponible, usarlo
        if SKLEARN_AVAILABLE:
            train_data, temp_data = train_test_split(
                self.data, test_size=0.3, random_state=42
            )
            val_data, test_data = train_test_split(
                temp_data, test_size=0.5, random_state=42
            )
        else:
            # División manual
            random.shuffle(self.data)
            total = len(self.data)
            train_size = int(0.7 * total)
            val_size = int(0.15 * total)
            
            train_data = self.data[:train_size]
            val_data = self.data[train_size:train_size + val_size]
            test_data = self.data[train_size + val_size:]
        
        print(f"📊 División del dataset:")
        print(f"   🏋️  Entrenamiento: {len(train_data)} imágenes")
        print(f"   📊 Validación: {len(val_data)} imágenes")
        print(f"   🧪 Prueba: {len(test_data)} imágenes")
        
        return train_data, val_data, test_data
    
    def procesar_imagen(self, item, split_name):
        """Procesa una imagen y crea su anotación"""
        try:
            filename = item['filename']
            ruta_original = item['ruta_completa']
            
            if not os.path.exists(ruta_original):
                print(f"⚠️  Imagen no encontrada: {ruta_original}")
                return False
            
            # Ruta destino
            ruta_destino = os.path.join(self.output_dir, split_name, 'images', filename)
            
            # Copiar imagen
            shutil.copy2(ruta_original, ruta_destino)
            
            # Crear anotación YOLO
            if not self.crear_anotacion_yolo(item, split_name):
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Error procesando imagen {filename}: {e}")
            return False
    
    def crear_anotacion_yolo(self, item, split_name):
        """Crea archivo de anotación en formato YOLO"""
        try:
            filename = item['filename']
            ruta_imagen = item['ruta_completa']
            
            # Obtener coordenadas estimadas
            x1, y1, x2, y2 = self.estimar_coordenadas_automaticas(ruta_imagen)
            
            # Obtener dimensiones de la imagen
            try:
                img = cv2.imread(ruta_imagen)
                if img is None:
                    print(f"⚠️  No se pudo leer la imagen: {ruta_imagen}")
                    return False
                
                altura, ancho = img.shape[:2]
                
            except Exception as e:
                print(f"⚠️  Error obteniendo dimensiones de {filename}: {e}")
                return False
            
            # Convertir a formato YOLO (normalizado)
            x_centro = (x1 + x2) / 2 / ancho
            y_centro = (y1 + y2) / 2 / altura
            w = (x2 - x1) / ancho
            h = (y2 - y1) / altura
            
            # Validar coordenadas
            if not (0 <= x_centro <= 1 and 0 <= y_centro <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                print(f"⚠️  Coordenadas YOLO inválidas en: {filename}")
                print(f"    Coordenadas originales: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                print(f"    Dimensiones imagen: {ancho}x{altura}")
                return False
            
            # Crear contenido de anotación
            contenido = f"0 {x_centro:.6f} {y_centro:.6f} {w:.6f} {h:.6f}"
            
            # Nombre del archivo de anotación
            nombre_base = os.path.splitext(filename)[0]
            nombre_anotacion = f"{nombre_base}.txt"
            ruta_anotacion = os.path.join(self.output_dir, split_name, 'labels', nombre_anotacion)
            
            # Guardar anotación
            with open(ruta_anotacion, 'w') as f:
                f.write(contenido)
            
            return True
            
        except Exception as e:
            print(f"❌ Error creando anotación YOLO: {e}")
            return False
    
    def crear_configuracion_yaml(self):
        """Crea archivo de configuración YAML para YOLO"""
        config = {
            'path': os.path.abspath(self.output_dir),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': 1,
            'names': ['patente']
        }
        
        ruta_config = os.path.join(self.output_dir, 'dataset.yaml')
        with open(ruta_config, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"✅ Configuración YAML creada: {ruta_config}")
        return ruta_config
    
    def crear_dataset(self):
        """Crea el dataset completo"""
        print("🚀 INICIANDO CREACIÓN DE DATASET DESDE NOMBRES DE ARCHIVO")
        print("=" * 60)
        
        # 1. Procesar imágenes y extraer patentes
        self.procesar_imagenes()
        
        if not self.data:
            print("❌ No se pudieron extraer patentes de los nombres de archivo")
            return False
        
        # 2. Crear estructura
        self.crear_estructura_directorios()
        
        # 3. Dividir dataset
        train_data, val_data, test_data = self.dividir_dataset()
        if not train_data:
            return False
        
        # 4. Procesar cada split
        splits = [
            ('train', train_data),
            ('val', val_data),
            ('test', test_data)
        ]
        
        for split_name, data in splits:
            print(f"\n🔄 Procesando {split_name}...")
            exitos = 0
            
            for item in data:
                if self.procesar_imagen(item, split_name):
                    exitos += 1
            
            print(f"✅ {split_name}: {exitos}/{len(data)} imágenes procesadas")
        
        # 5. Crear configuración
        self.crear_configuracion_yaml()
        
        # 6. Generar reporte
        self.generar_reporte()
        
        print(f"\n🎉 DATASET CREADO EXITOSAMENTE en: {self.output_dir}")
        return True
    
    def generar_reporte(self):
        """Genera un reporte del dataset"""
        reporte_path = os.path.join(self.output_dir, 'reporte_dataset.txt')
        
        with open(reporte_path, 'w', encoding='utf-8') as f:
            f.write("=== REPORTE DEL DATASET DE PATENTES ===\n\n")
            f.write(f"Total de imágenes procesadas: {len(self.data)}\n")
            f.write(f"Directorio de imágenes: {self.images_dir}\n")
            f.write(f"Directorio de salida: {self.output_dir}\n\n")
            
            # Contar imágenes por split
            for split in ['train', 'val', 'test']:
                img_dir = os.path.join(self.output_dir, split, 'images')
                if os.path.exists(img_dir):
                    num_imagenes = len([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
                    f.write(f"{split}: {num_imagenes} imágenes\n")
            
            # Estadísticas de patentes
            f.write(f"\n📊 Estadísticas de patentes:\n")
            patrones = {}
            for item in self.data:
                patente = item['patente']
                if len(patente) not in patrones:
                    patrones[len(patente)] = 0
                patrones[len(patente)] += 1
            
            for longitud, cantidad in patrones.items():
                f.write(f"  Patentes de {longitud} caracteres: {cantidad}\n")
            
            # Mostrar algunas patentes de ejemplo
            f.write(f"\n🔤 Ejemplos de patentes extraídas:\n")
            for i, item in enumerate(self.data[:10]):
                f.write(f"  {i+1}. {item['filename']} -> {item['patente']}\n")

class EntrenadorSimple:
    """Entrenador simple para YOLO"""
    
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.model = None
    
    def entrenar(self, epochs=50):
        """Entrena el modelo YOLO"""
        print("🚀 INICIANDO ENTRENAMIENTO YOLO")
        
        config_path = os.path.join(self.dataset_path, 'dataset.yaml')
        if not os.path.exists(config_path):
            print(f"❌ No se encuentra el archivo de configuración: {config_path}")
            return False
        
        try:
            # Cargar modelo pre-entrenado
            self.model = YOLO('yolov8n.pt')
            
            # Entrenar
            resultados = self.model.train(
                data=config_path,
                epochs=epochs,
                imgsz=640,
                batch=8,
                save=True,
                device='cpu'
            )
            
            print("✅ ENTRENAMIENTO COMPLETADO")
            return True
            
        except Exception as e:
            print(f"❌ Error durante el entrenamiento: {e}")
            return False

def main():
    """Función principal"""
    print("🎯 CREADOR DE DATASET DESDE NOMBRES DE ARCHIVO")
    print("=" * 55)
    
    # Configura la ruta de tus imágenes aquí
    IMAGES_DIR = "dataset"  # Cambia por la ruta donde están tus imágenes
    OUTPUT_DIR = "dataset_patentes_yolo"
    
    # Verificar que el directorio de imágenes existe
    if not os.path.exists(IMAGES_DIR):
        print(f"❌ No se encuentra el directorio de imágenes: {IMAGES_DIR}")
        print("📁 Asegúrate de que el directorio con tus imágenes existe")
        return
    
    # Crear dataset
    creador = DatasetDesdeNombres(IMAGES_DIR, OUTPUT_DIR)
    
    if creador.crear_dataset():
        print("\n¿Quieres entrenar el modelo ahora? (s/n)")
        respuesta = input().strip().lower()
        
        if respuesta in ['s', 'si', 'sí', 'y', 'yes']:
            entrenador = EntrenadorSimple(OUTPUT_DIR)
            entrenador.entrenar(epochs=50)
    else:
        print("❌ Error creando el dataset")

if __name__ == "__main__":
    main()