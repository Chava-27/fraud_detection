#!/usr/bin/env python3
# ============================================================
# IEEE-CIS Fraud Detection — Fusión de archivos Kaggle
# ============================================================
# Uso DESPUÉS de descargar train_transaction.csv y train_identity.csv
# desde: https://www.kaggle.com/competitions/ieee-fraud-detection/data
#
# Coloca ambos archivos en utils/datasets/ y luego ejecuta:
#   python utils/ieee_merge.py
#
# Genera: utils/datasets/ieee_cis.csv
# ============================================================

import sys
from pathlib import Path

raiz = Path(__file__).parent.parent
sys.path.insert(0, str(raiz))

from utils.logger import Logger

CARPETA = raiz / "utils" / "datasets"
TRANS   = CARPETA / "train_transaction.csv"
IDENT   = CARPETA / "train_identity.csv"
SALIDA  = CARPETA / "ieee_cis.csv"


def main():
    Logger.seccion("IEEE-CIS — Fusión de archivos")

    if not TRANS.exists():
        Logger.error(f"No se encontró: {TRANS}")
        Logger.info("Descarga train_transaction.csv de Kaggle y colócalo en utils/datasets/")
        sys.exit(1)

    if not IDENT.exists():
        Logger.error(f"No se encontró: {IDENT}")
        Logger.info("Descarga train_identity.csv de Kaggle y colócalo en utils/datasets/")
        sys.exit(1)

    if SALIDA.exists():
        Logger.ok(f"ieee_cis.csv ya existe en: {SALIDA}")
        resp = Logger.preguntar("¿Sobreescribir? (s/n):")
        if resp not in ("s", "si", "sí", "y", "yes"):
            Logger.info("Operación cancelada.")
            return

    try:
        import pandas as pd

        Logger.info("Cargando train_transaction.csv ...")
        df_trans = pd.read_csv(TRANS)
        Logger.ok(f"Transacciones: {df_trans.shape[0]:,} filas × {df_trans.shape[1]} columnas")

        Logger.info("Cargando train_identity.csv ...")
        df_ident = pd.read_csv(IDENT)
        Logger.ok(f"Identidades: {df_ident.shape[0]:,} filas × {df_ident.shape[1]} columnas")

        Logger.info("Fusionando (left join por TransactionID) ...")
        df = df_trans.merge(df_ident, on="TransactionID", how="left")
        Logger.ok(f"Dataset fusionado: {df.shape[0]:,} filas × {df.shape[1]} columnas")

        fraude_pct = df["isFraud"].mean() * 100
        Logger.info(f"Tasa de fraude: {fraude_pct:.3f}%")

        Logger.info(f"Guardando en: {SALIDA} ...")
        df.to_csv(SALIDA, index=False)
        Logger.ok("ieee_cis.csv generado correctamente.")
        Logger.info("Activa el dataset en config.yaml (enabled: true) para usarlo.")

    except Exception as e:
        Logger.error(f"Error durante la fusión: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
