# utils/drive_manager.py
# ============================================================
# GESTOR DE GOOGLE DRIVE LOCAL
# ============================================================
# Comprime el proyecto en un ZIP y lo copia a la carpeta
# de Google Drive sincronizada en tu PC.
#
# NO requiere API keys ni OAuth — usa la carpeta local de
# Drive for Desktop (el cliente oficial de Google).
#
# Rutas típicas de Drive for Desktop:
#   Windows : C:\Users\<usuario>\Google Drive\
#             G:\Mi unidad\
#   Mac     : /Users/<usuario>/Google Drive/
#   Linux   : ~/Google Drive/  (si está montado)
# ============================================================

import os
import shutil
import zipfile
from pathlib import Path
from datetime import datetime
import tempfile

from utils.logger import Logger


# Rutas donde Drive for Desktop suele estar montado
_RUTAS_DRIVE_WINDOWS = [
    Path.home() / "Google Drive",
    Path("G:/") / "Mi unidad",
    Path("G:/"),
    Path.home() / "OneDrive",  # fallback por si acaso
]

_RUTAS_DRIVE_UNIX = [
    Path.home() / "Google Drive",
    Path.home() / "GoogleDrive",
    Path("/mnt/gdrive"),
]


class DriveManager:
    """
    Sube el proyecto a Google Drive local (Drive for Desktop).

    Flujo:
      1. Busca la carpeta raíz de Google Drive en el sistema
      2. Crea la subcarpeta configurada (drive.carpeta_drive)
      3. Comprime el proyecto en ZIP (excluye __pycache__, .git, venv)
      4. Copia el ZIP a la carpeta de Drive
      5. Drive for Desktop sincroniza automáticamente a la nube
    """

    def __init__(self, cfg_drive: dict, cfg_general: dict):
        self.cfg = cfg_drive
        self.carpeta_dest = cfg_drive.get("carpeta_drive", "fraud_detection_ml")
        self.nombre_zip = cfg_drive.get("nombre_zip", "fraud_detection.zip")
        self.cuenta = cfg_drive.get("cuenta_principal", "")
        self.cuenta_sec = cfg_drive.get("cuenta_secundaria", "")

    # ── API pública ───────────────────────────────────────────
    def subir(self, raiz_proyecto: Path) -> Path | None:
        """
        Comprime el proyecto y lo copia a Drive.

        Parámetros
        ----------
        raiz_proyecto : Path al directorio raíz del proyecto

        Retorna
        -------
        Path al archivo ZIP dentro de Drive, o None si falló.
        """
        Logger.seccion("Subida a Google Drive")

        # 1. Encontrar raíz de Drive
        ruta_drive = self._encontrar_drive()
        if ruta_drive is None:
            Logger.error(
                "No se encontró la carpeta de Google Drive en tu PC.\n"
                "  Soluciones:\n"
                "  1. Instala 'Drive for Desktop': https://drive.google.com/drive/download\n"
                "  2. Inicia sesión con: " + self.cuenta + "\n"
                "  3. Espera a que sincronice y vuelve a ejecutar."
            )
            return None

        Logger.ok(f"Google Drive encontrado en: {ruta_drive}")

        # 2. Crear subcarpeta destino
        carpeta_destino = ruta_drive / self.carpeta_dest
        carpeta_destino.mkdir(parents=True, exist_ok=True)
        Logger.info(f"Carpeta destino: {carpeta_destino}")

        # 3. Comprimir proyecto
        ruta_zip_tmp = Path(tempfile.gettempdir()) / self.nombre_zip
        Logger.info(f"Comprimiendo proyecto...")
        self._comprimir(raiz_proyecto, ruta_zip_tmp)
        tam_mb = ruta_zip_tmp.stat().st_size / 1e6
        Logger.ok(f"ZIP creado: {ruta_zip_tmp.name} ({tam_mb:.1f} MB)")

        # 4. Copiar a Drive
        ruta_zip_drive = carpeta_destino / self.nombre_zip
        shutil.copy2(ruta_zip_tmp, ruta_zip_drive)
        Logger.ok(f"ZIP copiado a Drive: {ruta_zip_drive}")
        Logger.info("Drive for Desktop sincronizará automáticamente a la nube.")

        # 5. Limpiar tmp
        ruta_zip_tmp.unlink(missing_ok=True)

        return ruta_zip_drive

    # ── Métodos internos ──────────────────────────────────────
    def _encontrar_drive(self) -> Path | None:
        # Si hay ruta manual en config, usarla directamente
        ruta_manual = self.cfg.get("ruta_manual")
        if ruta_manual:
            p = Path(ruta_manual)
            if p.exists():
                return p

        """Busca la carpeta raíz de Google Drive en el sistema."""
        import platform

        sistema = platform.system()

        if sistema == "Windows":
            rutas = _RUTAS_DRIVE_WINDOWS
        else:
            rutas = _RUTAS_DRIVE_UNIX

        for ruta in rutas:
            if ruta.exists() and ruta.is_dir():
                return ruta

        # Último recurso: buscar en variables de entorno
        gdrive_env = os.environ.get("GDRIVE_PATH")
        if gdrive_env:
            p = Path(gdrive_env)
            if p.exists():
                return p

        return None

    @staticmethod
    def _comprimir(raiz: Path, destino: Path):
        """
        Crea un ZIP del proyecto excluyendo archivos innecesarios.
        """
        EXCLUIR_DIRS = {
            "__pycache__",
            ".git",
            "venv",
            ".venv",
            "env",
            ".env",
            "node_modules",
            ".mypy_cache",
        }
        EXCLUIR_EXTS = {".pyc", ".pyo", ".log", ".tmp"}
        EXCLUIR_ARCHIVOS = {".DS_Store", "Thumbs.db"}

        with zipfile.ZipFile(destino, "w", zipfile.ZIP_DEFLATED) as zf:
            for archivo in raiz.rglob("*"):
                # Saltar carpetas excluidas
                if any(parte in EXCLUIR_DIRS for parte in archivo.parts):
                    continue
                # Saltar extensiones excluidas
                if archivo.suffix in EXCLUIR_EXTS:
                    continue
                # Saltar archivos específicos
                if archivo.name in EXCLUIR_ARCHIVOS:
                    continue
                # Solo archivos (no directorios vacíos)
                if archivo.is_file():
                    arcname = archivo.relative_to(raiz.parent)
                    zf.write(archivo, arcname)
