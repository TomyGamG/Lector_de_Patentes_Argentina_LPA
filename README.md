# Sistema de Detección y Reconocimiento de Patentes con YOLOv8

## 📄 Descripción

Este proyecto implementa un sistema avanzado de reconocimiento automático de patentes de vehículos, utilizando visión por computadora y OCR. Permite:

- Detectar vehículos y patentes en imágenes o video streams.
- Extraer texto de patentes usando EasyOCR con fallback a Tesseract.
- Validar y normalizar patentes de formatos antiguos y nuevos de Argentina.
- Guardar registros de entrada/salida en base de datos Django.
- Visualizar resultados en un dashboard web.

El sistema está pensado para control de accesos, auditoría de flotas y monitoreo vehicular.

---

## ⚙️ Características principales

- Detección de vehículos y patentes con **YOLOv8**.
- OCR híbrido: **EasyOCR** + fallback **Tesseract**.
- Corrección automática de caracteres comúnmente confundidos (B↔8, O↔0, I↔1, etc.).
- Preprocesamiento de imagen optimizado (rescale, grises, CLAHE, morphología).
- Validación flexible de formatos de patentes argentinas.
- Registro de detecciones en base de datos con Django.
- Generación de imágenes anotadas con bounding boxes y leyendas.
- Detección de movimiento simple para optimizar el análisis.
- Logging y métricas de rendimiento.

---

## 📦 Componentes

| Componente | Tecnología | Función |
|------------|-----------|---------|
| Backend | Django + Python | API, Dashboard, BD |
| Computer Vision | YOLOv8, OpenCV | Detección vehículo/patente |
| OCR | EasyOCR + Tesseract | Reconocimiento de caracteres |
| Base de datos | PostgreSQL / SQLite | Patentes y accesos |
| Frontend | Django Templates | UI administración y dashboard |

---

## 🧠 Flujo de reconocimiento

1. Captura de imagen o frame de cámara.
2. Detección de vehículo y matrícula con YOLO.
3. Recorte de la región de la patente.
4. Preprocesamiento: grises, threshold, resize, filtros.
5. OCR con EasyOCR; fallback Tesseract si falla.
6. Corrección de caracteres confundidos.
7. Validación del formato de patente argentino.
8. Registro en base de datos.
9. Visualización en el dashboard.

---

## 📊 Dashboard / Panel Web

- Visualización en tiempo real de detecciones.
- Tabla de entradas/salidas con búsqueda y filtros.
- Gestión de vehículos autorizados.
- Exportación de registros.
- Visualización de capturas asociadas a detecciones.

---

## 🧪 Precisión del sistema

| Condición | Precisión aproximada |
|-----------|-------------------|
| Buena iluminación | 95%+ |
| Noche / lluvia / ruido | 80–90% |
| EasyOCR solo | ~80% |
| OCR híbrido + normalización | +15–20% mejora |


---

## 🚀 Roadmap

- Tracking multi-vehículos.
- Integración con cámara RTSP.
- Métricas en tiempo real: FPS, latency, accuracy.
- API para barreras automáticas.
- Notificaciones vía Telegram / WhatsApp.
- Version Dockerizada para despliegue rápido.
- Optimización para Edge Devices (Jetson Nano / Raspberry Pi).

---

## 💻 Requisitos

- Python 3.10+
- Django 4.x
- OpenCV
- NumPy
- Pillow
- EasyOCR
- pytesseract
- ultralytics (YOLOv8)
- torch (CPU/GPU)
- PostgreSQL o SQLite

---

## ⚡ Instalación rápida

```bash
# Clonar repositorio
git clone https://github.com/TomyGamG/Proyecto.git
cd vehicle-plate-detection

# Crear entorno virtual
pip install virtualenv
cd entorno ... Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Migrar base de datos Django
python manage.py migrate

# Correr servidor
python manage.py runserver


