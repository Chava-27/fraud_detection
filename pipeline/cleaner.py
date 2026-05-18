# pipeline/cleaner.py
# ============================================================
# ETAPA 2 — LIMPIEZA Y VALIDACIÓN
# Responsabilidad: producir un DataFrame limpio y confiable.
# No transforma (eso es preprocessor), solo elimina ruido.
# ============================================================

import pandas as pd
from utils.logger import Logger


def limpiar(df: pd.DataFrame, cfg_dataset: dict) -> pd.DataFrame:
    """
    Limpia el DataFrame crudo.

    Operaciones realizadas:
      - Reporta y elimina valores faltantes (NaN)
      - Reporta y elimina duplicados exactos
      - Convierte la columna objetivo a entero (0/1)
      - Mapea el valor de fraude a 1 si no lo es ya
      - Convierte columnas object/category a numérico donde sea posible
      - Descarta columnas no numéricas que no se puedan convertir

    Parámetros
    ----------
    df          : DataFrame crudo de la etapa de carga
    cfg_dataset : sección del dataset desde config.yaml

    Retorna
    -------
    pd.DataFrame limpio
    """
    Logger.seccion(f"Etapa 2 · Limpieza — {cfg_dataset['nombre']}")

    target      = cfg_dataset["target_col"]
    fraud_value = cfg_dataset.get("fraud_value", 1)
    filas_ini   = len(df)

    # ── 1. Valores faltantes ──────────────────────────────────
    nan_total = df.isna().sum().sum()
    if nan_total > 0:
        Logger.warn(f"Valores faltantes encontrados: {nan_total:,} → eliminando filas")
        df = df.dropna()
    else:
        Logger.ok("Sin valores faltantes.")

    # ── 2. Duplicados ─────────────────────────────────────────
    dups = df.duplicated().sum()
    if dups > 0:
        Logger.warn(f"Duplicados encontrados: {dups:,} → eliminando")
        df = df.drop_duplicates()
    else:
        Logger.ok("Sin filas duplicadas.")

    # ── 3. Columna objetivo → 0/1 entero ─────────────────────
    if target not in df.columns:
        raise KeyError(f"Columna objetivo '{target}' no existe en el DataFrame.")

    df[target] = pd.to_numeric(df[target], errors="coerce")

    # Si el valor de fraude no es 1, recodificar
    if fraud_value != 1:
        Logger.info(f"Recodificando fraude: '{fraud_value}' → 1, resto → 0")
        df[target] = (df[target] == fraud_value).astype(int)
    else:
        df[target] = df[target].astype(int)

    # ── 4. Columnas no numéricas ──────────────────────────────
        categorical_cols = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    categorical_cols = [col for col in categorical_cols if col != target]

    if categorical_cols:
        Logger.info(
            "Columnas categóricas conservadas para encoding posterior: "
            f"{categorical_cols}"
        )
    else:
        Logger.ok("No se detectaron columnas categóricas.")

    # ==========================================================
    # cols_no_num = df.select_dtypes(include=["object", "category"]).columns.tolist()
    # if cols_no_num:
    #     Logger.warn(f"Columnas no numéricas: {cols_no_num}")
    #     convertidas = []
    #     descartadas = []
    #     for col in cols_no_num:
    #         if col == target:
    #             continue
    #         convertido = pd.to_numeric(df[col], errors="coerce")
    #         if convertido.notna().sum() > 0.8 * len(df):
    #             df[col] = convertido
    #             convertidas.append(col)
    #         else:
    #             descartadas.append(col)

    #     if convertidas:
    #         Logger.ok(f"Convertidas a numérico: {convertidas}")
    #     if descartadas:
    #         Logger.warn(f"Descartadas (no convertibles): {descartadas}")
    #         df = df.drop(columns=descartadas)


    # ── 5. Resumen final ──────────────────────────────────────
    filas_fin = len(df)
    Logger.ok(
        f"Limpieza completada. "
        f"Filas: {filas_ini:,} → {filas_fin:,} "
        f"(eliminadas: {filas_ini - filas_fin:,})"
    )
    Logger.info(f"Shape final: {df.shape}")

    dist = df[target].value_counts()
    pct  = (df[target].value_counts(normalize=True) * 100).round(2)
    
    Logger.info(
        f"Distribución final:\n"
        f"  Legítimas: {dist.get(0, 0):,} ({pct.get(0, 0):.2f}%)\n"
        f"  Fraudes:   {dist.get(1, 0):,} ({pct.get(1, 0):.2f}%)"
    )

    return df
