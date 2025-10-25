import cv2
import os
import sys
import django
import numpy as np

# Configurar Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'lpd_project.settings')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
django.setup()

from plate_detector.services import LicensePlateDetector

def test_with_sample_images():
    """Probar detección con imágenes de ejemplo"""
    detector = LicensePlateDetector()
    
    # Crear imágenes de prueba simples
    print("🧪 Creando imágenes de prueba...")
    
    # Imagen 1: Patente simulada
    img1 = np.ones((200, 400, 3), dtype=np.uint8) * 255
    cv2.putText(img1, "ABC123", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    
    # Imagen 2: Rectángulo similar a patente
    img2 = np.ones((150, 300, 3), dtype=np.uint8) * 200
    cv2.rectangle(img2, (20, 20), (280, 130), (0, 0, 0), 2)
    
    test_images = [img1, img2]
    
    for i, img in enumerate(test_images):
        print(f"\n📸 Procesando imagen {i+1}...")
        plates = detector.detect_plate(img)
        
        if plates:
            for plate in plates:
                print(f"✅ Patente detectada: {plate['text']}")
        else:
            print("❌ No se detectaron patentes")
        
        # Guardar imagen de prueba
        cv2.imwrite(f'test_image_{i+1}.jpg', img)

def test_with_camera():
    """Probar detección en tiempo real con cámara"""
    print("\n🎥 Probando con cámara web...")
    
    cap = cv2.VideoCapture(0)
    detector = LicensePlateDetector()
    
    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara")
        return
    
    print("✅ Cámara abierta. Presiona 'q' para salir, 's' para guardar frame")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Procesar cada 30 frames
        if frame_count % 30 == 0:
            plates = detector.detect_plate(frame)
            
            if plates:
                for plate in plates:
                    print(f"🎯 Patente detectada: {plate['text']}")
                    
                    # Dibujar rectángulo
                    x, y, w, h = plate['coordinates']
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, plate['text'], (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Mostrar frame
        cv2.imshow('Prueba de Detección', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Guardar frame actual
            cv2.imwrite('debug_frame.jpg', frame)
            print("💾 Frame guardado como 'debug_frame.jpg'")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("=== PRUEBA DE DETECCIÓN DE PATENTES ===")
    
    # Probar con imágenes de ejemplo
    test_with_sample_images()
    
    # Probar con cámara (opcional)
    respuesta = input("\n¿Probar con cámara web? (s/n): ")
    if respuesta.lower() == 's':
        test_with_camera()
    
    print("\n=== PRUEBA COMPLETADA ===")