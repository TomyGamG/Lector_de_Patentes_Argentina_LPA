import os
import sys
import django

# Configurar Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'lpd_project.settings')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
django.setup()

def test_tesseract():
    """Prueba la configuración de Tesseract"""
    print("🧪 Probando configuración de Tesseract...")
    
    try:
        import pytesseract
        
        # Rutas posibles
        paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        ]
        
        for path in paths:
            if os.path.exists(path):
                print(f"✅ Tesseract encontrado en: {path}")
                pytesseract.pytesseract.tesseract_cmd = path
                try:
                    version = pytesseract.get_tesseract_version()
                    print(f"✅ Versión: {version}")
                    return True
                except Exception as e:
                    print(f"❌ Error con esta ruta: {e}")
        
        # Probar si está en PATH
        try:
            version = pytesseract.get_tesseract_version()
            print(f"✅ Tesseract en PATH: versión {version}")
            return True
        except:
            print("❌ Tesseract no encontrado en ninguna ruta")
            return False
            
    except ImportError:
        print("❌ pytesseract no instalado en el virtualenv")
        return False

def test_opencv():
    """Prueba OpenCV"""
    print("\n🧪 Probando OpenCV...")
    try:
        import cv2
        print(f"✅ OpenCV versión: {cv2.__version__}")
        
        # Probar cámara
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✅ Cámara disponible")
            cap.release()
        else:
            print("⚠️  Cámara no disponible")
            
        return True
    except Exception as e:
        print(f"❌ Error con OpenCV: {e}")
        return False

if __name__ == "__main__":
    print("=== PRUEBA DE CONFIGURACIÓN VIRTUALENV ===\n")
    test_tesseract()
    test_opencv()
    print("\n=== PRUEBA COMPLETADA ===")