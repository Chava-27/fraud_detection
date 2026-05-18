#!/usr/bin/env python3
# ============================================================
# DESCARGA DE DATASETS ADICIONALES — Modo Offline Helper
# ============================================================
# Ejecuta este script UNA VEZ con internet para descargar
# los datasets que quieras usar. Después puedes trabajar
# completamente offline.
#
# Uso:
#   python utils/descargar_datasets.py
#
# Los CSV se guardan en utils/datasets/ y el pipeline los
# detecta automáticamente en futuras ejecuciones.
# ============================================================

import sys
from pathlib import Path

# Asegurar que el path raíz del proyecto esté en sys.path
raiz = Path(__file__).parent.parent
sys.path.insert(0, str(raiz))

from utils.logger import Logger

CARPETA_DATASETS = raiz / "utils" / "datasets"
CARPETA_DATASETS.mkdir(parents=True, exist_ok=True)


# ── Datasets disponibles ──────────────────────────────────────────────────────

DATASETS = [
    {
        "nombre"    : "CreditCard Fraud (OpenML)",
        "archivo"   : "creditcard.csv",
        "fuente"    : "openml",
        "openml_name": "creditcard",
        "openml_version": 1,
        "target_col": "Class",
        "descripcion": "284,807 transacciones, 0.17% fraudes. Dataset estándar de benchmark.",
        "tamano_aprox": "~145 MB",
    },
    {
        "nombre"    : "BankSim (OpenML)",
        "archivo"   : "banksim.csv",
        "fuente"    : "openml",
        "openml_name": "BankSim",
        "openml_version": 1,
        "target_col": "fraud",
        "descripcion": "594,643 transacciones simuladas de un banco español.",
        "tamano_aprox": "~50 MB",
    },
    {
        "nombre"    : "PaySim (Kaggle — instrucciones manuales)",
        "archivo"   : "paysim.csv",
        "fuente"    : "manual",
        "url"       : "https://www.kaggle.com/datasets/ealaxi/paysim1",
        "descripcion": "6.3 M transacciones de dinero móvil. Muy desbalanceado.",
        "tamano_aprox": "~470 MB",
        "instrucciones": (
            "1. Crea una cuenta en Kaggle (gratis).\n"
            "2. Entra a: https://www.kaggle.com/datasets/ealaxi/paysim1\n"
            "3. Descarga 'PS_20174392719_1491204439457_log.csv'.\n"
            "4. Renámbralo a 'paysim.csv'.\n"
            "5. Colócalo en: utils/datasets/paysim.csv\n"
            "6. En config.yaml, activa el dataset PaySim (enabled: true)."
        ),
    },
    {
        "nombre"    : "IEEE-CIS Fraud Detection (Kaggle — instrucciones manuales)",
        "archivo"   : "ieee_cis.csv",
        "fuente"    : "manual",
        "url"       : "https://www.kaggle.com/competitions/ieee-fraud-detection/data",
        "descripcion": "590,540 transacciones con 394 features. Competencia de referencia.",
        "tamano_aprox": "~1.1 GB total",
        "instrucciones": (
            "1. Acepta las reglas de la competencia en Kaggle.\n"
            "2. Descarga: train_transaction.csv y train_identity.csv.\n"
            "3. Colócalos en utils/datasets/.\n"
            "4. Ejecuta: python utils/ieee_merge.py  (fusiona ambos archivos).\n"
            "5. Esto genera utils/datasets/ieee_cis.csv listo para usar.\n"
            "6. En config.yaml, activa el dataset IEEE-CIS (enabled: true)."
        ),
    },
]


def descargar_openml(cfg: dict) -> bool:
    """Descarga un dataset desde OpenML y lo guarda como CSV."""
    ruta = CARPETA_DATASETS / cfg["archivo"]

    if ruta.exists():
        Logger.ok(f"'{cfg['nombre']}' ya existe en caché: {ruta.name}")
        return True

    Logger.info(f"Descargando '{cfg['nombre']}' desde OpenML ...")
    Logger.info(f"Tamaño aproximado: {cfg.get('tamano_aprox', 'desconocido')}")

    try:
        from sklearn.datasets import fetch_openml
        import pandas as pd

        X, y = fetch_openml(
            name=cfg["openml_name"],
            version=cfg["openml_version"],
            as_frame=True,
            return_X_y=True,
            parser="auto",
        )
        df = X.copy()
        df[cfg["target_col"]] = y

        Logger.info(f"Guardando en: {ruta} ...")
        df.to_csv(ruta, index=False)
        Logger.ok(f"'{cfg['nombre']}' guardado correctamente ({len(df):,} filas).")
        return True

    except Exception as e:
        Logger.error(f"Error descargando '{cfg['nombre']}': {e}")
        return False


def mostrar_instrucciones_manuales(cfg: dict):
    """Muestra instrucciones de descarga manual para datasets de Kaggle."""
    Logger.warn(f"'{cfg['nombre']}' requiere descarga manual desde Kaggle.")
    Logger.info(f"URL: {cfg.get('url', 'N/A')}")
    Logger.info(f"Descripción: {cfg.get('descripcion', '')}")
    Logger.info(f"Tamaño aproximado: {cfg.get('tamano_aprox', 'N/A')}")
    print()
    for linea in cfg.get("instrucciones", "").split("\n"):
        print(f"    {linea}")
    print()


def main():
    Logger.seccion("Descarga de Datasets — Modo Offline Helper")
    Logger.info(f"Carpeta de destino: {CARPETA_DATASETS}")
    print()

    # Mostrar lista numerada
    Logger.info("Datasets disponibles:")
    for i, ds in enumerate(DATASETS, 1):
        estado = "✔ ya descargado" if (CARPETA_DATASETS / ds["archivo"]).exists() else "─ no descargado"
        print(f"  [{i}] {ds['nombre']}  [{estado}]")
        print(f"      {ds['descripcion']}")
        print(f"      Tamaño: {ds.get('tamano_aprox', 'N/A')}")
        print()

    Logger.info("Opciones: ingresa número(s) separados por coma, 'todos', o 'salir'")
    resp = Logger.preguntar("¿Qué deseas descargar?:").strip().lower()

    if resp in ("salir", "s", "exit", "q"):
        Logger.info("Saliendo.")
        return

    if resp == "todos":
        indices = list(range(len(DATASETS)))
    else:
        try:
            indices = [int(x.strip()) - 1 for x in resp.split(",") if x.strip()]
            indices = [i for i in indices if 0 <= i < len(DATASETS)]
        except ValueError:
            Logger.error("Entrada inválida. Usa números separados por coma.")
            return

    if not indices:
        Logger.warn("No se seleccionó ningún dataset válido.")
        return

    print()
    exitosos = 0
    for i in indices:
        ds = DATASETS[i]
        Logger.seccion(f"Dataset {i+1}: {ds['nombre']}")

        if ds["fuente"] == "openml":
            if descargar_openml(ds):
                exitosos += 1
        else:
            mostrar_instrucciones_manuales(ds)

    print()
    Logger.seccion("Descarga completada")
    Logger.ok(f"Datasets descargados automáticamente: {exitosos}")
    Logger.info("Revisa config.yaml para activar/desactivar los datasets que quieras usar.")
    Logger.info(f"CSVs en: {CARPETA_DATASETS}")


if __name__ == "__main__":
    main()
