# utils/dataset_manager.py
# ============================================================
# Gestión de datasets: verificar caché local, descargar online,
# preguntar al usuario según el modo configurado.
# ============================================================

import os
import pandas as pd
from pathlib import Path
from .logger import Logger


class DatasetManager:
    """
    Carga datasets desde caché local o descarga desde OpenML/CSV.

    Flujo:
      1. Verificar si existe el CSV en utils/datasets/
      2. Si existe  → cargar localmente (rápido)
      3. Si no existe → según config.modo:
           - "offline"   → error, no descarga
           - "online"    → descarga directa
           - "preguntar" → pregunta al usuario
      4. Si descarga → guarda CSV localmente para siguientes corridas
    """

    def __init__(self, cfg_general: dict):
        self.carpeta = Path(cfg_general.get("carpeta_datasets", "utils/datasets"))
        self.modo = cfg_general.get("modo", "preguntar")
        self.carpeta.mkdir(parents=True, exist_ok=True)

    # ── API pública ───────────────────────────────────────────
    def cargar(self, cfg_dataset: dict) -> pd.DataFrame:
        """
        Retorna un DataFrame con los datos crudos del dataset.
        La columna objetivo se llama como indica cfg_dataset['target_col'].
        """
        nombre    = cfg_dataset["nombre"]
        fuente    = cfg_dataset.get("fuente", "openml")
        archivo   = cfg_dataset.get("archivo_local", "")
        ruta_csv  = self.carpeta / archivo if archivo else None
        frac      = cfg_dataset.get("sample_frac", 1.0)

        Logger.subseccion(f"Dataset: {nombre}")

        # ── Caso 1: CSV local ─────────────────────────────────
        if fuente == "csv_local":
            return self._cargar_csv(ruta_csv, cfg_dataset, frac)

        # ── Caso 2: OpenML con caché ──────────────────────────
        if fuente == "openml":
            return self._cargar_openml(ruta_csv, cfg_dataset, frac)

        raise ValueError(f"Fuente desconocida: '{fuente}'. Usa 'openml' o 'csv_local'.")

    # ── Métodos internos ──────────────────────────────────────
    def _cargar_csv(self, ruta: Path, cfg: dict, frac: float) -> pd.DataFrame:
        if ruta is None or not ruta.exists():
            raise FileNotFoundError(
                f"Archivo no encontrado: {ruta}\n"
                f"Coloca el CSV en la carpeta '{self.carpeta}' y ajusta 'archivo_local' en config.yaml."
            )
        Logger.info(f"Cargando CSV local: {ruta}")
        df = pd.read_csv(ruta)
        return self._sample(df, frac, cfg["nombre"])

    def _cargar_openml(self, ruta_csv: Path, cfg: dict, frac: float) -> pd.DataFrame:
        # ¿Ya está en caché?
        if ruta_csv and ruta_csv.exists():
            Logger.ok(f"Caché local encontrado: {ruta_csv.name}")
            df = pd.read_csv(ruta_csv)
            return self._sample(df, frac, cfg["nombre"])

        # Decidir si descargar
        if not self._debe_descargar(cfg["nombre"]):
            raise ConnectionAbortedError(
                f"Dataset '{cfg['nombre']}' no disponible localmente y descarga cancelada.\n"
                f"Cambia 'modo: online' en config.yaml o descarga el CSV manualmente."
            )

        # Descargar desde OpenML
        df = self._descargar_openml(cfg)

        # Guardar en caché
        if ruta_csv:
            Logger.info(f"Guardando caché en: {ruta_csv}")
            df.to_csv(ruta_csv, index=False)
            Logger.ok("Guardado correctamente para futuras ejecuciones.")

        return self._sample(df, frac, cfg["nombre"])

    def _debe_descargar(self, nombre: str) -> bool:
        """
        Decide si se permite descargar según el modo configurado.

        En modo 'offline', si el CSV no está en caché, se pregunta al usuario
        si desea descargarlo una vez (quedará guardado para futuras ejecuciones).
        En modo 'online' se descarga automáticamente sin preguntar.
        """
        if self.modo == "online":
            return True

        # Tanto en "offline" como en "preguntar": avisar y preguntar.
        Logger.warn(
            f"No se encontró '{nombre}' localmente.\n"
            f"  Descargar ahora guardará el CSV en '{self.carpeta}' para futuras ejecuciones.\n"
            f"  La primera descarga puede tardar varios minutos (requiere internet)."
        )
        resp = Logger.preguntar(f"¿Descargar '{nombre}' ahora? (s/n):")
        if resp in ("s", "si", "sí", "y", "yes"):
            return True
        Logger.warn(
            f"Descarga cancelada. Para ejecutar '{nombre}' sin internet:\n"
            f"  1. Descarga el CSV manualmente.\n"
            f"  2. Colócalo en '{self.carpeta}' con el nombre indicado en config.yaml.\n"
            f"  3. Vuelve a ejecutar main.py."
        )
        return False

    @staticmethod
    def _descargar_openml(cfg: dict) -> pd.DataFrame:
        """Descarga un dataset de OpenML y lo devuelve como DataFrame."""
        from sklearn.datasets import fetch_openml

        oml_name    = cfg.get("openml_name", cfg["nombre"])
        oml_version = cfg.get("openml_version", 1)
        target_col  = cfg["target_col"]

        Logger.info(f"Descargando desde OpenML: '{oml_name}' v{oml_version} ...")

        try:
            X, y = fetch_openml(
                name=oml_name,
                version=oml_version,
                as_frame=True,
                return_X_y=True,
                parser="auto"
            )
        except Exception as e:
            raise RuntimeError(f"Error descargando '{oml_name}' desde OpenML: {e}")

        df = X.copy()
        df[target_col] = y
        Logger.ok(f"Descarga completada. Filas: {len(df):,}  Columnas: {df.shape[1]}")
        return df

    @staticmethod
    def _sample(df: pd.DataFrame, frac: float, nombre: str) -> pd.DataFrame:
        """Aplica muestreo estratificado si frac < 1.0."""
        if frac < 1.0:
            df = df.sample(frac=frac, random_state=42).reset_index(drop=True)
            Logger.warn(f"'{nombre}' muestreado al {frac*100:.0f}%: {len(df):,} filas")
        else:
            Logger.info(f"'{nombre}' cargado completo: {len(df):,} filas")
        return df
