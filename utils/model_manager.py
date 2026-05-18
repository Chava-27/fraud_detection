# utils/model_manager.py
# ============================================================
# Serialización y carga de modelos entrenados con joblib.
# Permite no reentrenar si el modelo ya existe en disco.
# ============================================================

import re
import joblib
from pathlib import Path
from .logger import Logger


class ModelManager:
    """
    Guarda y carga modelos entrenados en utils/modelos/.

    Nomenclatura de archivo:
      {nombre_modelo}__{nombre_dataset}.pkl
      Ejemplo: random_forest__creditcard.pkl
    """

    def __init__(self, cfg_general: dict):
        self.carpeta = Path(cfg_general.get("carpeta_modelos", "utils/modelos"))
        self.activo  = cfg_general.get("guardar_modelos", True)
        self.carpeta.mkdir(parents=True, exist_ok=True)

    # ── Helpers ───────────────────────────────────────────────
    @staticmethod
    def _slug(texto: str) -> str:
        """Convierte un nombre a slug seguro para nombres de archivo."""
        return re.sub(r"[^a-z0-9]+", "_", texto.lower()).strip("_")

    def _ruta(self, nombre_modelo: str, nombre_dataset: str) -> Path:
        slug_m = self._slug(nombre_modelo)
        slug_d = self._slug(nombre_dataset)
        return self.carpeta / f"{slug_m}__{slug_d}.pkl"

    # ── API pública ───────────────────────────────────────────
    def existe(self, nombre_modelo: str, nombre_dataset: str) -> bool:
        return self._ruta(nombre_modelo, nombre_dataset).exists()

    def guardar(self, modelo, nombre_modelo: str, nombre_dataset: str):
        if not self.activo:
            return
        ruta = self._ruta(nombre_modelo, nombre_dataset)
        joblib.dump(modelo, ruta)
        Logger.ok(f"Modelo guardado: {ruta.name}")

    def cargar(self, nombre_modelo: str, nombre_dataset: str):
        ruta = self._ruta(nombre_modelo, nombre_dataset)
        if not ruta.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {ruta}")
        Logger.info(f"Cargando modelo desde caché: {ruta.name}")
        return joblib.load(ruta)
