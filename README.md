# SDXL + AnimateDiff + Frame-to-Frame Hybrid Video Generation

Generación de **video corto animado realista** combinando:

- **SDXL** para crear un keyframe de alta calidad (768×768)
- **AnimateDiff** (sobre base Realistic Vision v5.1) para generar movimiento sutil
- Exportación a video con movimiento realista (respiración, parpadeo, iluminación cinematográfica)

Proyecto híbrido pensado para tarjetas gráficas de gama media-alta con memoria limitada (~6 GB VRAM).

## Características principales

- Genera keyframe con **Stable Diffusion XL** si no existe
- Usa **AnimateDiff v1-5** con modelo base realista
- Scheduler **DPM++ 2M Karras** optimizado
- Técnicas de bajo consumo de VRAM:
  - `enable_model_cpu_offload()`
  - `enable_vae_slicing()`
  - `enable_forward_chunking()`
- Resultado: video ~8–16 frames (0.6–1.3 segundos aprox. a 12 fps)

## Hardware de referencia (pruebas)

| Componente     | Especificación                     |
|----------------|-------------------------------------|
| GPU            | NVIDIA GeForce GTX 1060 6 GB       |
| RAM del sistema| 96 GB                              |
| CPU            | Intel Xeon E5-2650 v4 (12c/24t)    |
| Sistema        | Linux / Windows + PyTorch + CUDA   |

Funciona (con paciencia) en tarjetas de **6 GB VRAM**. Mejora notable con 8–12 GB.<br>
Espacio de disco ocupado **<20GB**.

## Requisitos

- Python 3.9–3.11 recomendado
- CUDA 11.8 o 12.x (12.1 muy estable en 2024–2025)
- ~10–12 GB de espacio en disco (modelos + caché)

### Dependencias principales

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install diffusers==0.29.0  # o la versión más reciente compatible
pip install accelerate transformers
pip install opencv-python pillow  # para manejo de imágenes/video
```
## Instalación rápida
1. Clona el repositorio
```
git clone https://github.com/aayes89/HIVCreator.git
cd HIVCreator
```
2. Crea y activa entorno virtual (recomendado)
```
python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows
```
3. Instala las dependencias
4. Modifica el prompt en el script
- Si quieres generación completa pasar al paso 4 directamente
- Si quieres generación de video a partir de imagen previa, cambiar nombre a **keyframe.png**
5. Ejecuta el script
```
python SDAD.py
```
6. Imagen generada con nombre **keyframe.png** y Video generado con nombre **hybrid_output.mp4**

# Licencias y créditos

* Modelos de Hugging Face (stabilityai, guoyww, SG161222, stable-diffusion)
* Licencia de cada checkpoint → revisar en su página oficial

### Ejemplos de imágenes

<img width="1024" height="1024" alt="keyframe_good" src="https://github.com/user-attachments/assets/ed1e09c6-1ae1-4792-a8d5-87ea80908991" />

<img width="768" height="768" alt="keyframe" src="https://github.com/user-attachments/assets/93ede75f-7557-4d48-823d-ecbffc79f327" />

