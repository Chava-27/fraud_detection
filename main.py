#!/usr/bin/env python3
# ============================================================
# DETECCIÓN DE FRAUDE FINANCIERO — MAIN
# ============================================================
# MODOS:
#   offline  → ejecuta todo localmente
#   online   → comprime, sube a Drive, genera notebook, abre Colab
#   preguntar→ pregunta al usuario
#
# Configuración en: config.yaml
# Ejecutar: python main.py
# ============================================================

import sys
import yaml
import traceback
from pathlib import Path

from utils.logger import Logger
from utils.gpu_manager import imprimir_info_gpu, configurar_gpu_modelos
from pipeline.loader import cargar_dataset
from pipeline.cleaner import limpiar
from pipeline.preprocessor import preprocesar
from pipeline.trainer import entrenar_modelos
from pipeline.evaluator import evaluar_modelos
from pipeline.reporter import generar_reporte


# ── Configuración ─────────────────────────────────────────────────────────────

def cargar_config(ruta: str = "config.yaml") -> dict:
    ruta_p = Path(ruta)
    if not ruta_p.exists():
        Logger.error(f"No se encontró: {ruta}")
        sys.exit(1)
    with open(ruta_p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    Logger.ok(f"Configuración cargada: {ruta}")
    return cfg


def resolver_modo(cfg_general: dict) -> str:
    """Determina el modo de ejecución: online / offline."""
    modo = cfg_general.get("modo", "offline").lower()
    if modo == "preguntar":
        Logger.info("Modos disponibles:")
        Logger.info("  offline → ejecuta todo en esta PC (sin internet)")
        Logger.info("  online  → sube a Drive y ejecuta en Google Colab (GPU)")
        resp = Logger.preguntar("¿Qué modo usar? (offline/online):").lower()
        if resp in ("online", "o", "on"):
            return "online"
        return "offline"
    if modo not in ("online", "offline"):
        Logger.warn(f"Modo desconocido '{modo}' en config.yaml. Se usará 'offline'.")
        return "offline"
    return modo


# ── Flujo ONLINE ──────────────────────────────────────────────────────────────

def flujo_online(cfg: dict):
    """
    Modo online:
      1. Comprime el proyecto
      2. Sube el ZIP a Google Drive local
      3. Genera el notebook .ipynb para Colab
      4. Abre Colab en el navegador
      5. Termina — Colab hace el resto
    """
    Logger.seccion("Modo ONLINE — Preparando para Google Colab")

    from utils.drive_manager import DriveManager
    from utils.colab_launcher import generar_notebook

    cfg_drive   = cfg.get("drive", {})
    cfg_colab   = cfg.get("colab", {})
    cfg_general = cfg.get("general", {})

    raiz = Path(__file__).parent.resolve()

    # 1. Subir a Drive
    drive_mgr    = DriveManager(cfg_drive, cfg_general)
    ruta_zip     = drive_mgr.subir(raiz)

    if ruta_zip is None:
        Logger.warn(
            "No se pudo subir a Drive automáticamente.\n"
            "Sube manualmente el ZIP a tu Google Drive y luego abre el notebook generado."
        )

    # 2. Generar notebook
    ruta_nb = generar_notebook(cfg_colab, cfg_drive, cfg_general, ruta_zip)

    # 3. Fin — Colab ejecutará el pipeline
    Logger.seccion("Listo para Colab")
    Logger.ok(f"Notebook generado: {ruta_nb}")
    Logger.ok("Sigue las instrucciones que aparecen arriba para abrir en Colab.")
    Logger.info(
        "Una vez en Colab:\n"
        "  1. Activa GPU: Entorno de ejecución → Cambiar tipo → GPU T4\n"
        "  2. Ejecuta todo: Entorno de ejecución → Ejecutar todo\n"
        "  3. El frame PNG se descargará automáticamente al terminar."
    )


# ── Flujo OFFLINE ─────────────────────────────────────────────────────────────

def ejecutar_pipeline_dataset(cfg_dataset, cfg_modelos, cfg_general):
    nombre = cfg_dataset["nombre"]
    try:
        df_crudo   = cargar_dataset(cfg_dataset, cfg_general)
        df_limpio  = limpiar(df_crudo, cfg_dataset)
        datos      = preprocesar(df_limpio, cfg_dataset, cfg_general)
        modelos_e  = entrenar_modelos(datos, cfg_modelos, cfg_general, nombre)
        resultados = evaluar_modelos(modelos_e, datos, nombre)
        return resultados
    except ConnectionAbortedError as e:
        Logger.warn(f"'{nombre}' omitido: {e}")
    except FileNotFoundError as e:
        Logger.warn(f"'{nombre}' no disponible: {e}")
    except Exception as e:
        Logger.error(f"Error en '{nombre}': {e}")
        Logger.error(traceback.format_exc())
    return None


def flujo_offline(cfg: dict):
    """Ejecuta el pipeline completo localmente."""
    cfg_general    = cfg.get("general", {})
    cfg_datasets   = cfg.get("datasets", [])
    cfg_modelos    = cfg.get("modelos", [])
    cfg_conclusion = cfg.get("conclusion_ia", {})
    cfg_gpu        = cfg.get("gpu", {})

    # GPU
    imprimir_info_gpu()
    cfg_modelos = configurar_gpu_modelos(cfg_modelos, cfg_gpu)

    # Filtrar activos
    datasets_activos = [d for d in cfg_datasets if d.get("enabled", True)]
    modelos_activos  = [m for m in cfg_modelos  if m.get("enabled", True)]

    Logger.info(f"Datasets : {[d['nombre'] for d in datasets_activos]}")
    Logger.info(f"Modelos  : {[m['nombre'] for m in modelos_activos]}")

    if not datasets_activos:
        Logger.error("No hay datasets activos en config.yaml.")
        sys.exit(1)
    if not modelos_activos:
        Logger.error("No hay modelos activos en config.yaml.")
        sys.exit(1)

    todos_resultados = {}
    for cfg_dataset in datasets_activos:
        nombre = cfg_dataset["nombre"]
        Logger.seccion(f"Dataset: {nombre}")
        res = ejecutar_pipeline_dataset(cfg_dataset, modelos_activos, cfg_general)
        if res is not None:
            todos_resultados[nombre] = res
            Logger.ok(f"'{nombre}' procesado correctamente.")
        else:
            Logger.warn(f"'{nombre}' omitido del reporte.")

    if not todos_resultados:
        Logger.error("Ningún dataset pudo procesarse.")
        sys.exit(1)

    ruta_frame = generar_reporte(todos_resultados, cfg_general, cfg_conclusion)

    Logger.seccion("Proceso Terminado")
    Logger.ok(f"Frame guardado en: {ruta_frame}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    Logger.seccion("Detección de Fraude Financiero")

    cfg  = cargar_config("config.yaml")
    modo = resolver_modo(cfg.get("general", {}))

    Logger.info(f"Modo de ejecución: {modo.upper()}")

    if modo == "online":
        flujo_online(cfg)
    else:
        flujo_offline(cfg)


if __name__ == "__main__":
    main()
