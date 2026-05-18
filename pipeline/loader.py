# pipeline/loader.py
# ============================================================
# ETAPA 1 — CARGA DE DATOS
# Responsabilidad: obtener el DataFrame crudo (sin modificar).
# Toda la lógica de online/offline está en DatasetManager.
# ============================================================

import pandas as pd
from utils.dataset_manager import DatasetManager
from utils.logger import Logger


def cargar_dataset(cfg_dataset: dict, cfg_general: dict) -> pd.DataFrame:
    """
    Carga un dataset según su configuración.

    Parámetros
    ----------
    cfg_dataset : dict
        Sección del dataset desde config.yaml (nombre, fuente, target_col, etc.)
    cfg_general : dict
        Sección general desde config.yaml (modo, carpeta_datasets, etc.)

    Retorna
    -------
    pd.DataFrame
        DataFrame crudo con todas las columnas, incluyendo la columna objetivo.
    """
    Logger.seccion(f"Etapa 1 · Carga — {cfg_dataset['nombre']}")

    manager = DatasetManager(cfg_general)
    df = manager.cargar(cfg_dataset)

    # Informe rápido
    target = cfg_dataset["target_col"]
    Logger.info(f"Forma del dataset: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    Logger.info(f"Columna objetivo : '{target}'")

    if target in df.columns:
        conteo = df[target].value_counts()
        Logger.info(f"Distribución de clases:\n{conteo.to_string()}")
        pct = (df[target].value_counts(normalize=True) * 100).round(2)
        Logger.info(f"Porcentaje:\n{pct.to_string()}")
    else:
        Logger.warn(f"La columna objetivo '{target}' no se encontró en el DataFrame.")

    return df
