# pipeline/evaluator.py
# ============================================================
# ETAPA 5 — EVALUACIÓN
# Responsabilidad: calcular todas las métricas en el conjunto
# de prueba para cada modelo entrenado. Devuelve datos
# estructurados listos para el reporter.
# ============================================================

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from utils.logger import Logger


def evaluar_modelos(modelos_entrenados: dict, datos: dict, nombre_dataset: str) -> dict:
    """
    Evalúa todos los modelos entrenados sobre el conjunto de prueba.

    Parámetros
    ----------
    modelos_entrenados : dict devuelto por trainer.entrenar_modelos()
    datos              : dict con X_test, y_test
    nombre_dataset     : nombre del dataset (para logging)

    Retorna
    -------
    dict {nombre_modelo: {métricas, reporte, matriz_confusion, y_pred}}
    """
    Logger.seccion(f"Etapa 5 · Evaluación — {nombre_dataset}")

    X_test = datos["X_test"]
    y_test = datos["y_test"]

    resultados = {}

    for nombre, info in modelos_entrenados.items():
        Logger.subseccion(f"Evaluando: {nombre}")

        estimador = info["estimador"]
        y_pred    = estimador.predict(X_test)

        # ── Métricas escalar ──────────────────────────────────
        metricas = {
            "accuracy" : accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall"   : recall_score(y_test, y_pred, zero_division=0),
            "f1"       : f1_score(y_test, y_pred, zero_division=0),
        }

        # ── Reporte de clasificación (texto) ──────────────────
        reporte = classification_report(
            y_test, y_pred,
            target_names=["Legítima", "Fraude"],
            zero_division=0
        )

        # ── Matriz de confusión ───────────────────────────────
        cm = confusion_matrix(y_test, y_pred)

        # ── Imprimir en consola ───────────────────────────────
        Logger.result(
            f"Accuracy={metricas['accuracy']:.4f} | "
            f"Precision={metricas['precision']:.4f} | "
            f"Recall={metricas['recall']:.4f} | "
            f"F1={metricas['f1']:.4f}"
        )
        Logger.info(f"Reporte completo:\n{reporte}")

        # Interpretar la matriz
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
        Logger.info(
            f"Matriz de confusión:\n"
            f"  TN (legítimas correctas)  = {tn:,}\n"
            f"  FP (falsas alarmas)       = {fp:,}\n"
            f"  FN (fraudes no detectados)= {fn:,}  ← crítico\n"
            f"  TP (fraudes detectados)   = {tp:,}"
        )

        resultados[nombre] = {
            "metricas"        : metricas,
            "reporte"         : reporte,
            "confusion_matrix": cm,
            "y_pred"          : y_pred,
            "y_test"          : y_test,
        }

    # ── Tabla resumen en consola ──────────────────────────────
    _imprimir_tabla_resumen(resultados, nombre_dataset)

    return resultados


def _imprimir_tabla_resumen(resultados: dict, nombre_dataset: str):
    """Imprime tabla comparativa de todos los modelos en consola."""
    try:
        from tabulate import tabulate
        filas = []
        for nombre, info in resultados.items():
            m = info["metricas"]
            filas.append([
                nombre,
                f"{m['accuracy']:.4f}",
                f"{m['precision']:.4f}",
                f"{m['recall']:.4f}",
                f"{m['f1']:.4f}",
            ])

        headers = ["Modelo", "Accuracy", "Precision", "Recall", "F1"]
        tabla   = tabulate(filas, headers=headers, tablefmt="fancy_grid")
        Logger.seccion(f"Resumen — {nombre_dataset}")
        print(tabla)
    except ImportError:
        # Fallback sin tabulate
        Logger.seccion(f"Resumen — {nombre_dataset}")
        df = pd.DataFrame(
            {n: info["metricas"] for n, info in resultados.items()}
        ).T.round(4)
        print(df.to_string())
