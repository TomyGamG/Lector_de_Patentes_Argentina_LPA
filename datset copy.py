# entrenamiento_express.py
import os
import torch
from ultralytics import YOLO
import time

class EntrenamientoExpress:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.model = None
        self.tiempo_inicio = None
        
    def verificar_configuracion(self):
        """Verifica y optimiza la configuración"""
        print("🔍 Verificando configuración...")
        
        # Verificar dataset
        config_path = os.path.join(self.dataset_path, 'dataset.yaml')
        if not os.path.exists(config_path):
            print(f"❌ No se encuentra dataset.yaml en {self.dataset_path}")
            return False
        
        # Verificar imágenes
        splits = ['train', 'val']
        for split in splits:
            images_dir = os.path.join(self.dataset_path, split, 'images')
            if not os.path.exists(images_dir):
                print(f"❌ No se encuentra directorio de imágenes: {images_dir}")
                return False
            
            num_imagenes = len([f for f in os.listdir(images_dir) 
                              if f.endswith(('.jpg', '.png', '.jpeg'))])
            print(f"   📁 {split}: {num_imagenes} imágenes")
        
        # Verificar GPU
        if torch.cuda.is_available():
            device = 'cuda'
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🎮 GPU detectada: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            device = 'cpu'
            print("💻 Usando CPU - Más lento pero funcional")
        
        return True
    
    def optimizar_dataset_express(self):
        """Optimiza el dataset para entrenamiento rápido"""
        print("⚡ Optimizando dataset para velocidad...")
        
        # Reducir dataset si es muy grande (máximo 200 imágenes por split)
        max_imagenes = 200
        splits = ['train', 'val']
        
        for split in splits:
            images_dir = os.path.join(self.dataset_path, split, 'images')
            labels_dir = os.path.join(self.dataset_path, split, 'labels')
            
            if os.path.exists(images_dir):
                imagenes = [f for f in os.listdir(images_dir) 
                           if f.endswith(('.jpg', '.png', '.jpeg'))]
                
                if len(imagenes) > max_imagenes:
                    print(f"   📦 Reduciendo {split} de {len(imagenes)} a {max_imagenes} imágenes")
                    
                    # Mantener solo las primeras max_imagenes
                    imagenes_a_mantener = imagenes[:max_imagenes]
                    
                    # Eliminar excedentes
                    for img_file in os.listdir(images_dir):
                        if img_file not in imagenes_a_mantener:
                            os.remove(os.path.join(images_dir, img_file))
                            
                            # Eliminar anotación correspondiente
                            label_file = img_file.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')
                            label_path = os.path.join(labels_dir, label_file)
                            if os.path.exists(label_path):
                                os.remove(label_path)
    
    def entrenar_express(self, epochs=15):
        """Entrenamiento ultra rápido"""
        self.tiempo_inicio = time.time()
        
        print("🚀 INICIANDO ENTRENAMIENTO EXPRESS...")
        print("=" * 50)
        
        # Verificar configuración
        if not self.verificar_configuracion():
            return None
        
        # Optimizar dataset
        self.optimizar_dataset_express()
        
        # Configuración según hardware
        if torch.cuda.is_available():
            batch_size = 16
            workers = 8
            imgsz = 416
            device = 0
            amp = True  # Mixed precision
        else:
            batch_size = 4
            workers = 2  
            imgsz = 320
            device = 'cpu'
            amp = False
        
        print(f"⚙️  Configuración Express:")
        print(f"   📐 Tamaño imagen: {imgsz}px")
        print(f"   📦 Batch size: {batch_size}")
        print(f"   🔄 Workers: {workers}")
        print(f"   🎯 Épocas: {epochs}")
        print(f"   💡 Mixed Precision: {amp}")
        
        # Cargar modelo más rápido (nano)
        self.model = YOLO('yolov8n.pt')
        
        try:
            # ENTRENAMIENTO EXPRESS
            results = self.model.train(
                data=os.path.join(self.dataset_path, 'dataset.yaml'),
                
                # ⚡ CONFIGURACIÓN EXPRESS
                epochs=epochs,
                imgsz=imgsz,
                batch=batch_size,
                workers=workers,
                device=device,
                amp=amp,
                
                # 🎯 HIPERPARÁMETROS OPTIMIZADOS
                lr0=0.01,           # Learning rate alto
                lrf=0.01,
                momentum=0.9,
                weight_decay=0.0005,
                warmup_epochs=2.0,  # Poco warmup
                warmup_momentum=0.8,
                box=7.5,
                cls=0.5,
                dfl=1.5,
                
                # 🚀 OPTIMIZACIONES DE VELOCIDAD
                patience=10,        # Parada temprana
                save=True,
                save_period=5,      # Guardar cada 5 épocas
                cache=False,        # Sin cache (más rápido)
                verbose=False,      # Menos output
                
                # 📉 AUMENTO DE DATOS MÍNIMO
                hsv_h=0.01,        # Muy poco aumento de color
                hsv_s=0.5,
                hsv_v=0.3,
                degrees=0.0,       # Sin rotación
                translate=0.05,    # Muy poca traslación  
                scale=0.2,         # Poca escala
                shear=0.0,         # Sin shear
                perspective=0.0,   # Sin perspectiva
                flipud=0.0,        # Sin volteo vertical
                fliplr=0.3,        # Solo volteo horizontal básico
                mosaic=0.0,        # SIN MOSAIC (más rápido)
                mixup=0.0,         # SIN MIXUP (más rápido)
                copy_paste=0.0,    # SIN COPY-PASTE
                
                # 📁 PROYECTO
                project='entrenamiento_express',
                name=f'express_{int(time.time())}',
                exist_ok=True
            )
            
            tiempo_total = time.time() - self.tiempo_inicio
            print(f"\n✅ ENTRENAMIENTO EXPRESS COMPLETADO!")
            print(f"⏰ Tiempo total: {tiempo_total/60:.1f} minutos")
            
            return results
            
        except Exception as e:
            print(f"❌ Error en entrenamiento express: {e}")
            return None
    
    def evaluar_express(self):
        """Evaluación rápida del modelo entrenado"""
        print("\n📊 Evaluando modelo express...")
        
        # Buscar el mejor modelo
        model_path = self.buscar_mejor_modelo()
        if not model_path:
            print("❌ No se encontró modelo para evaluar")
            return None
        
        try:
            model = YOLO(model_path)
            
            # Evaluación rápida
            metrics = model.val(
                data=os.path.join(self.dataset_path, 'dataset.yaml'),
                split='val',
                conf=0.5,
                iou=0.5,
                verbose=False
            )
            
            print("🎯 RESULTADOS EXPRESS:")
            print(f"   🎯 mAP50: {metrics.box.map50:.3f}")
            print(f"   🎯 mAP50-95: {metrics.box.map:.3f}")
            print(f"   📈 Precisión: {metrics.box.precision:.3f}")
            print(f"   🔍 Recall: {metrics.box.recall:.3f}")
            
            return metrics
            
        except Exception as e:
            print(f"❌ Error en evaluación: {e}")
            return None
    
    def buscar_mejor_modelo(self):
        """Encuentra el mejor modelo entrenado"""
        posibles_rutas = [
            'entrenamiento_express/*/weights/best.pt',
            'runs/detect/train/weights/best.pt'
        ]
        
        import glob
        for patron in posibles_rutas:
            rutas = glob.glob(patron)
            if rutas:
                return rutas[0]
        
        return None
    
    def probar_modelo_rapido(self, imagen_prueba=None):
        """Prueba rápida del modelo con una imagen"""
        print("\n🔍 Probando modelo express...")
        
        model_path = self.buscar_mejor_modelo()
        if not model_path:
            print("❌ No se encontró modelo entrenado")
            return
        
        try:
            model = YOLO(model_path)
            
            # Si no hay imagen de prueba, usar una del dataset de validación
            if not imagen_prueba:
                val_dir = os.path.join(self.dataset_path, 'val', 'images')
                if os.path.exists(val_dir):
                    imagenes = [f for f in os.listdir(val_dir) 
                               if f.endswith(('.jpg', '.png', '.jpeg'))]
                    if imagenes:
                        imagen_prueba = os.path.join(val_dir, imagenes[0])
            
            if imagen_prueba and os.path.exists(imagen_prueba):
                print(f"   📸 Probando con: {os.path.basename(imagen_prueba)}")
                
                results = model(imagen_prueba, conf=0.5, verbose=False)
                
                for r in results:
                    if len(r.boxes) > 0:
                        print(f"   ✅ Detectadas {len(r.boxes)} patente(s)")
                        for i, box in enumerate(r.boxes):
                            conf = box.conf[0].cpu().numpy()
                            print(f"      {i+1}. Confianza: {conf:.2f}")
                    else:
                        print("   ❌ No se detectaron patentes")
            else:
                print("   ⚠️  No hay imagen para probar")
                
        except Exception as e:
            print(f"❌ Error en prueba: {e}")

def main():
    """Función principal - Ejecuta todo el pipeline express"""
    print("🎯 ENTRENAMIENTO EXPRESS - PATENTES ARGENTINAS")
    print("=" * 55)
    
    # Configuración
    DATASET_PATH = "dataset_patentes_yolo"  # Ajusta esta ruta
    
    # Verificar que existe el dataset
    if not os.path.exists(DATASET_PATH):
        print(f"❌ No se encuentra el dataset: {DATASET_PATH}")
        print("📁 Asegúrate de que el directorio del dataset existe")
        return
    
    # Crear y ejecutar entrenamiento express
    entrenador = EntrenamientoExpress(DATASET_PATH)
    
    # 🚀 ENTRENAMIENTO EXPRESS (15 épocas)
    resultados = entrenador.entrenar_express(epochs=15)
    
    if resultados:
        # 📊 EVALUACIÓN EXPRESS
        entrenador.evaluar_express()
        
        # 🔍 PRUEBA RÁPIDA
        entrenador.probar_modelo_rapido()
        
        print(f"\n🎉 PIPELINE EXPRESS COMPLETADO!")
        print("💡 El modelo está listo en: entrenamiento_express/")
        
    else:
        print("❌ El entrenamiento express falló")

if __name__ == "__main__":
    main()