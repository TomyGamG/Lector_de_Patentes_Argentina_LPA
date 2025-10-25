import sys
import subprocess
import importlib

def check_dependency(package_name, import_name=None):
    """Verifica si un paquete está instalado"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name} está instalado")
        return True
    except ImportError:
        print(f"❌ {package_name} NO está instalado")
        return False

def install_dependency(package_name):
    """Instala un paquete faltante"""
    try:
        print(f"📦 Instalando {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"✅ {package_name} instalado correctamente")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Error instalando {package_name}")
        return False

def main():
    print("🔍 Verificando dependencias para YOLO...")
    print("=" * 50)
    
    dependencies = [
        ("ultralytics", "ultralytics"),
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("opencv-python", "cv2"),
        ("pytesseract", "pytesseract"),
        ("Pillow", "PIL"),
        ("numpy", "numpy"),
    ]
    
    missing_deps = []
    
    for package, import_name in dependencies:
        if not check_dependency(package, import_name):
            missing_deps.append(package)
    
    print("=" * 50)
    
    if missing_deps:
        print(f"❌ Faltan {len(missing_deps)} dependencias: {', '.join(missing_deps)}")
        response = input("¿Instalar automáticamente? (s/n): ")
        if response.lower() == 's':
            for package in missing_deps:
                install_dependency(package)
            print("🎯 Todas las dependencias instaladas. Reinicia el sistema.")
        else:
            print("💡 Instala manualmente: pip install " + " ".join(missing_deps))
    else:
        print("🎯 Todas las dependencias están instaladas correctamente!")
        
        # Verificar CUDA
        try:
            import torch
            if torch.cuda.is_available():
                print(f"✅ CUDA disponible: {torch.cuda.get_device_name(0)}")
            else:
                print("ℹ️  CUDA no disponible - Usando CPU")
        except:
            print("❌ No se pudo verificar CUDA")

if __name__ == "__main__":
    main()