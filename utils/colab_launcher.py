# utils/colab_launcher.py
# ============================================================
# GENERADOR DE NOTEBOOK PARA GOOGLE COLAB
# ============================================================
# Genera un archivo .ipynb completo y autocontenido que:
#   1. Instala todas las dependencias
#   2. Monta Google Drive
#   3. Descomprime el proyecto desde Drive
#   4. Detecta y activa GPU en Colab
#   5. Ejecuta el pipeline completo (modo offline dentro de Colab)
#   6. Descarga el frame PNG de resultados
#   7. Detiene la ejecución automáticamente al terminar
#
# El notebook se guarda localmente Y en Drive.
# Se abre automáticamente en el navegador.
# ============================================================

import json
import webbrowser
from pathlib import Path

from utils.logger import Logger


def generar_notebook(
    cfg_colab: dict,
    cfg_drive: dict,
    cfg_general: dict,
    ruta_zip_drive: Path | None,
) -> Path:
    """
    Genera el notebook .ipynb y opcionalmente lo abre en el navegador.

    Parámetros
    ----------
    cfg_colab      : sección colab del config.yaml
    cfg_drive      : sección drive del config.yaml
    cfg_general    : sección general del config.yaml
    ruta_zip_drive : ruta local del ZIP en Drive (para info en el notebook)

    Retorna
    -------
    Path al notebook generado localmente.
    """
    Logger.seccion("Generando Notebook para Google Colab")

    nombre_nb      = cfg_colab.get("nombre_notebook", "fraud_detection_colab.ipynb")
    carpeta_drive  = cfg_drive.get("carpeta_drive", "fraud_detection_ml")
    nombre_zip     = cfg_drive.get("nombre_zip", "fraud_detection.zip")
    cuenta         = cfg_drive.get("cuenta_principal", "")
    instalar_rapids= cfg_colab.get("instalar_rapids", False)
    abrir_nav      = cfg_colab.get("abrir_navegador", True)

    # ── Construir celdas del notebook ─────────────────────────
    celdas = []

    # Celda 0 — Título y contexto
    celdas.append(_celda_markdown(f"""# 🔍 Detección de Fraude Financiero
**Tecnológico Superior de Jalisco · Aprendizaje Automático · ISIC 7A**

Este notebook ejecuta el pipeline completo de detección de fraude financiero.
Corre cada celda en orden. La última celda genera el frame de resultados y lo descarga.

> **Cuenta Drive utilizada:** `{cuenta}`
> **Carpeta en Drive:** `{carpeta_drive}/{nombre_zip}`
"""))

    # Celda 1 — Verificar GPU
    celdas.append(_celda_codigo("""\
# ── Verificar GPU disponible en Colab ──────────────────────
import subprocess
result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
if result.returncode == 0:
    print("✅ GPU detectada:")
    print(result.stdout)
else:
    print("⚠️  No se detectó GPU. Ve a: Entorno de ejecución → Cambiar tipo de entorno → GPU")
    print("   El pipeline seguirá funcionando en CPU, pero será más lento.")
"""))

    # Celda 2 — Instalar dependencias
    pip_rapids = ""
    if instalar_rapids:
        pip_rapids = """\
# RAPIDS cuML para Random Forest en GPU (solo funciona con GPU T4/A100)
!pip install cuml-cu12 --extra-index-url=https://pypi.nvidia.com -q
"""

    celdas.append(_celda_codigo(f"""\
# ── Instalar dependencias ──────────────────────────────────
print("Instalando dependencias... (puede tardar 1-2 minutos)")
!pip install scikit-learn xgboost openml pandas numpy matplotlib \\
             seaborn pyyaml joblib anthropic tqdm colorama tabulate -q
{pip_rapids}
print("✅ Dependencias instaladas")
"""))

    # Celda 3 — Montar Drive y descomprimir
    celdas.append(_celda_codigo(f"""\
# ── Montar Google Drive ────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

import os, zipfile, shutil
from pathlib import Path

# Ruta del ZIP en Drive (ajusta si cambiaste carpeta_drive en config.yaml)
ruta_zip = Path('/content/drive/MyDrive/{carpeta_drive}/{nombre_zip}')

if not ruta_zip.exists():
    raise FileNotFoundError(
        f"No se encontró el ZIP en Drive: {{ruta_zip}}\\n"
        f"Asegúrate de que el ZIP fue subido correctamente desde tu PC."
    )

# Descomprimir en /content/
print(f"Descomprimiendo {{ruta_zip.name}}...")
with zipfile.ZipFile(ruta_zip, 'r') as zf:
    zf.extractall('/content/')

# El proyecto queda en /content/fraud_detection/
os.chdir('/content/fraud_detection')
print(f"✅ Proyecto listo en: {{os.getcwd()}}")
print("Contenido:")
!ls -la
"""))

    # Celda 4 — Forzar modo offline dentro de Colab
    celdas.append(_celda_codigo("""\
# ── Ajustar config para Colab (modo offline dentro de Colab) ──
# En Colab no tiene sentido volver a subir a Drive desde dentro.
# Cambiamos modo a "offline" para que corra directamente.
import yaml
from pathlib import Path

cfg_path = Path('config.yaml')
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

cfg['general']['modo'] = 'offline'

with open(cfg_path, 'w') as f:
    yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)

print("✅ Config ajustado para Colab (modo: offline)")
"""))

    # Celda 5 — Ejecutar pipeline
    celdas.append(_celda_codigo("""\
# ── Ejecutar pipeline completo ─────────────────────────────
# Esto entrena los modelos, evalúa y genera el frame de resultados.
# Con GPU activa, XGBoost usará CUDA automáticamente.
print("Iniciando pipeline...")
print("=" * 60)
%run main.py
"""))

    # Celda 6 — Descargar resultado
    celdas.append(_celda_codigo("""\
# ── Descargar el frame de resultados ──────────────────────
import glob
from google.colab import files
from pathlib import Path

resultados = sorted(glob.glob('results/reporte_*.png'))
if resultados:
    ruta_frame = resultados[-1]  # el más reciente
    print(f"✅ Frame generado: {ruta_frame}")
    
    # Copiar también a Drive para no perderlo
    import shutil
    ruta_drive_out = Path('/content/drive/MyDrive/{carpeta_drive}') / Path(ruta_frame).name
    shutil.copy2(ruta_frame, ruta_drive_out)
    print(f"✅ Copia guardada en Drive: {{ruta_drive_out}}")
    
    # Descargar al equipo local
    files.download(ruta_frame)
    print("⬇️  Descarga iniciada en tu navegador.")
else:
    print("⚠️  No se encontró el frame. Revisa si hubo errores en la celda anterior.")
"""))

    # Celda 7 — Detener ejecución
    celdas.append(_celda_codigo("""\
# ── Detener sesión de Colab ────────────────────────────────
# El análisis terminó. Detenemos el runtime para liberar recursos.
print("Pipeline completado. Deteniendo sesión de Colab...")
from google.colab import runtime
runtime.unassign()
"""))

    # ── Armar estructura del notebook ─────────────────────────
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            },
            "accelerator": "GPU",
            "colab": {
                "provenance": [],
                "gpuType": "T4"
            }
        },
        "cells": celdas,
    }

    # ── Guardar localmente ────────────────────────────────────
    carpeta_results = Path(cfg_general.get("carpeta_resultados", "results"))
    carpeta_results.mkdir(parents=True, exist_ok=True)
    ruta_nb = carpeta_results / nombre_nb
    ruta_nb_raiz = Path(nombre_nb)   # también en raíz para fácil acceso

    with open(ruta_nb, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)

    with open(ruta_nb_raiz, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)

    Logger.ok(f"Notebook generado: {ruta_nb_raiz}")

    # ── Copiar notebook a Drive también ───────────────────────
    if ruta_zip_drive is not None:
        try:
            carpeta_drive_path = ruta_zip_drive.parent
            ruta_nb_drive = carpeta_drive_path / nombre_nb
            import shutil
            shutil.copy2(ruta_nb_raiz, ruta_nb_drive)
            Logger.ok(f"Notebook copiado a Drive: {ruta_nb_drive}")
        except Exception as e:
            Logger.warn(f"No se pudo copiar el notebook a Drive: {e}")

    # ── Abrir en navegador ────────────────────────────────────
    if abrir_nav:
        _abrir_colab(ruta_nb_drive if ruta_zip_drive else ruta_nb_raiz, cuenta)

    return ruta_nb_raiz


# ── Helpers ───────────────────────────────────────────────────────────────────

def _celda_codigo(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
        "id": _id_unico(),
    }


def _celda_markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source,
        "id": _id_unico(),
    }


def _id_unico() -> str:
    import uuid
    return uuid.uuid4().hex[:8]


def _abrir_colab(ruta_nb: Path, cuenta: str):
    """
    Abre el link de Colab en el navegador.
    Colab puede abrir notebooks directamente desde Drive.
    """
    Logger.info("Abriendo Google Colab en el navegador...")
    Logger.info(
        "INSTRUCCIONES:\n"
        f"  1. Inicia sesión en Colab con: {cuenta}\n"
        "  2. Ve a: Archivo → Abrir cuaderno → Google Drive\n"
        f"  3. Busca el archivo: fraud_detection_colab.ipynb\n"
        "  4. Activa GPU: Entorno de ejecución → Cambiar tipo de entorno → GPU T4\n"
        "  5. Haz clic en: Entorno de ejecución → Ejecutar todo\n"
        "  6. Al terminar, el frame se descarga automáticamente."
    )

    # Abrir colab.research.google.com para que el usuario navegue al notebook
    url = "https://colab.research.google.com/"
    try:
        webbrowser.open(url)
        Logger.ok(f"Navegador abierto en: {url}")
    except Exception:
        Logger.warn(f"Abre manualmente: {url}")
