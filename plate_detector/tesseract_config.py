import os
import sys
import subprocess

def find_tesseract_windows():
    """Busca Tesseract en rutas comunes de Windows"""
    possible_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        r'C:\Users\{}\AppData\Local\Tesseract-OCR\tesseract.exe'.format(os.getenv('USERNAME')),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

def is_tesseract_in_path():
    """Verifica si tesseract está en el PATH"""
    try:
        result = subprocess.run(['tesseract', '--version'], 
                              capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except:
        return False

def configure_tesseract():
    """Configura Tesseract automáticamente"""
    print("🔍 Buscando Tesseract OCR...")
    
    # Verificar si está en PATH
    if is_tesseract_in_path():
        print("✅ Tesseract encontrado en PATH del sistema")
        return True
    
    # Buscar en rutas comunes de Windows
    tesseract_path = find_tesseract_windows()
    if tesseract_path:
        print(f"✅ Tesseract encontrado en: {tesseract_path}")
        # Configurar pytesseract
        try:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            # Verificar que funciona
            version = pytesseract.get_tesseract_version()
            print(f"✅ Tesseract versión: {version}")
            return True
        except Exception as e:
            print(f"❌ Error configurando pytesseract: {e}")
            return False
    
    print("❌ Tesseract no encontrado. Usando modo simulado.")
    return False

# Configurar al importar
TESSERACT_AVAILABLE = configure_tesseract()

# Si Tesseract está disponible, configurar pytesseract
if TESSERACT_AVAILABLE:
    try:
        import pytesseract
        # Configurar la ruta si se encontró
        tesseract_path = find_tesseract_windows()
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
    except ImportError:
        print("⚠️  pytesseract no está instalado en el virtualenv")
        TESSERACT_AVAILABLE = False