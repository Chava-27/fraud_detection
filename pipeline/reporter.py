# pipeline/reporter.py
# ============================================================
# ETAPA 6 — REPORTE VISUAL + CONCLUSIÓN IA
# Responsabilidad: generar un único frame matplotlib con:
#   - Barras comparativas de métricas por modelo y dataset
#   - Matrices de confusión de cada modelo
#   - Tabla resumen de todas las métricas
#   - Conclusión generada por IA (Anthropic API o rule-based)
# Guarda la imagen en results/
# ============================================================

import os
import textwrap
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib

matplotlib.use("Agg")  # sin display; compatible con servidores
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

from utils.logger import Logger


# ── Paleta de colores ─────────────────────────────────────────────────────────
COLORES_MODELO = [
    "#4C72B0",
    "#DD8452",
    "#55A868",
    "#C44E52",
    "#8172B3",
    "#937860",
    "#DA8BC3",
    "#8C8C8C",
    "#CCB974",
    "#64B5CD",
]
CMAP_CM = LinearSegmentedColormap.from_list("fraud_cm", ["#EFF3FF", "#084594"])


def generar_reporte(
    todos_resultados: dict,  # {nombre_dataset: {nombre_modelo: {metricas, cm, ...}}}
    cfg_general: dict,
    cfg_conclusion: dict,
) -> Path:
    """
    Genera el frame visual completo y lo guarda en results/.

    Parámetros
    ----------
    todos_resultados : resultados agrupados por dataset y modelo
    cfg_general      : sección general del config
    cfg_conclusion   : sección conclusion_ia del config

    Retorna
    -------
    Path al archivo PNG guardado
    """
    Logger.seccion("Etapa 6 · Reporte Visual")

    datasets = list(todos_resultados.keys())
    n_ds = len(datasets)

    # Recolectar todos los nombres de modelos
    modelos = []
    for ds_res in todos_resultados.values():
        for m in ds_res:
            if m not in modelos:
                modelos.append(m)
    n_mod = len(modelos)

    Logger.info(f"Datasets: {datasets}")
    Logger.info(f"Modelos : {modelos}")

    # ── Obtener conclusión IA ─────────────────────────────────
    conclusion_texto = _generar_conclusion(todos_resultados, cfg_conclusion)

    # ── Layout del frame ──────────────────────────────────────
    # Filas:
    #   0     → Título principal
    #   1     → Gráficas de barras (4 métricas × n_datasets)
    #   2     → Matrices de confusión (n_mod × n_ds)
    #   3     → Tabla resumen
    #   4     → Conclusión IA

    alto_titulo = 0.6
    alto_barras = 3.5 * max(n_ds, 1)
    alto_cms = 2.8 * max(n_ds, 1)
    alto_tabla = 1.2 + 0.35 * n_mod * n_ds
    alto_concl = max(2.5, len(conclusion_texto) / 120)
    alto_total = alto_titulo + alto_barras + alto_cms + alto_tabla + alto_concl + 1.5

    ancho = max(16, 4 * n_mod)

    fig = plt.figure(figsize=(ancho, alto_total), facecolor="#F7F9FC")

    gs = gridspec.GridSpec(
        5,
        1,
        figure=fig,
        height_ratios=[alto_titulo, alto_barras, alto_cms, alto_tabla, alto_concl],
        hspace=0.55,
        left=0.05,
        right=0.97,
        top=0.97,
        bottom=0.02,
    )

    # ── Fila 0: Título ────────────────────────────────────────
    ax_titulo = fig.add_subplot(gs[0])
    ax_titulo.axis("off")
    ax_titulo.text(
        0.5,
        0.7,
        "Reporte de Detección de Fraude — Comparación de Modelos y Datasets",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color="#1A1A2E",
        transform=ax_titulo.transAxes,
    )
    ax_titulo.text(
        0.5,
        0.1,
        f"Generado: {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}",
        ha="center",
        va="center",
        fontsize=9,
        color="#555555",
        transform=ax_titulo.transAxes,
    )

    # ── Fila 1: Gráficas de barras agrupadas ──────────────────
    _plot_barras(fig, gs[1], todos_resultados, modelos, datasets)

    # ── Fila 2: Matrices de confusión ─────────────────────────
    _plot_confusion_matrices(fig, gs[2], todos_resultados, modelos, datasets)

    # ── Fila 3: Tabla resumen ─────────────────────────────────
    _plot_tabla(fig, gs[3], todos_resultados, modelos, datasets)

    # ── Fila 4: Conclusión IA ─────────────────────────────────
    _plot_conclusion(fig, gs[4], conclusion_texto)

    # ── Guardar ───────────────────────────────────────────────
    carpeta = Path(cfg_general.get("carpeta_resultados", "results"))
    carpeta.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta = carpeta / f"reporte_{ts}.png"

    fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    Logger.ok(f"Frame guardado en: {ruta}")
    return ruta


# ── Sub-plots ─────────────────────────────────────────────────────────────────


def _plot_barras(fig, gs_slot, todos_resultados, modelos, datasets):
    """Barras agrupadas por métrica, una subfigura por dataset."""
    metricas = ["accuracy", "precision", "recall", "f1"]
    etiquetas = ["Accuracy", "Precision", "Recall", "F1-Score"]
    n_ds = len(datasets)

    gs_inner = gridspec.GridSpecFromSubplotSpec(
        1, n_ds, subplot_spec=gs_slot, wspace=0.35
    )

    for di, ds_nombre in enumerate(datasets):
        ax = fig.add_subplot(gs_inner[0, di])
        ds_res = todos_resultados[ds_nombre]
        x = np.arange(len(metricas))
        ancho_b = 0.8 / max(len(modelos), 1)

        for mi, modelo in enumerate(modelos):
            if modelo not in ds_res:
                continue
            vals = [ds_res[modelo]["metricas"].get(m, 0) for m in metricas]
            offset = (mi - len(modelos) / 2 + 0.5) * ancho_b
            bars = ax.bar(
                x + offset,
                vals,
                ancho_b * 0.9,
                label=modelo,
                color=COLORES_MODELO[mi % len(COLORES_MODELO)],
                alpha=0.87,
                edgecolor="white",
                linewidth=0.5,
            )
            # Valor encima de cada barra
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.008,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=5.5,
                    color="#333333",
                )

        ax.set_xticks(x)
        ax.set_xticklabels(etiquetas, fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.set_title(_truncar(ds_nombre, 35), fontsize=9, fontweight="bold", pad=6)
        ax.set_ylabel("Score", fontsize=8)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

        if di == 0:
            ax.legend(
                fontsize=6.5,
                loc="upper right",
                framealpha=0.7,
                ncol=1,
            )


def _plot_confusion_matrices(fig, gs_slot, todos_resultados, modelos, datasets):
    """Cuadrícula de matrices de confusión: filas=datasets, cols=modelos."""
    n_ds = len(datasets)
    n_mod = len(modelos)

    gs_inner = gridspec.GridSpecFromSubplotSpec(
        n_ds, n_mod, subplot_spec=gs_slot, hspace=0.6, wspace=0.4
    )

    for di, ds_nombre in enumerate(datasets):
        ds_res = todos_resultados[ds_nombre]
        for mi, modelo in enumerate(modelos):
            ax = fig.add_subplot(gs_inner[di, mi])

            if modelo not in ds_res:
                ax.axis("off")
                ax.text(
                    0.5,
                    0.5,
                    "N/A",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    color="#999999",
                )
                continue

            cm = ds_res[modelo]["confusion_matrix"]
            im = ax.imshow(cm, cmap=CMAP_CM, aspect="auto")

            # Valores en celdas
            vmax = cm.max()
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    color = "white" if cm[i, j] > vmax * 0.5 else "#1A1A2E"
                    ax.text(
                        j,
                        i,
                        f"{cm[i, j]:,}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color=color,
                        fontweight="bold",
                    )

            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Leg.", "Fraude"], fontsize=6)
            ax.set_yticklabels(["Leg.", "Fraude"], fontsize=6, rotation=90, va="center")
            ax.set_xlabel("Pred.", fontsize=6)

            titulo_mod = _truncar(modelo, 18)
            titulo_ds = _truncar(ds_nombre, 18)
            ax.set_title(
                f"{titulo_mod}\n{titulo_ds}", fontsize=6.5, fontweight="bold", pad=3
            )


def _plot_tabla(fig, gs_slot, todos_resultados, modelos, datasets):
    """Tabla completa de métricas para todos los modelos y datasets."""
    ax = fig.add_subplot(gs_slot)
    ax.axis("off")

    metricas = ["accuracy", "precision", "recall", "f1"]
    headers = ["Modelo", "Dataset", "Accuracy", "Precision", "Recall", "F1"]
    filas = []

    for ds_nombre in datasets:
        ds_res = todos_resultados[ds_nombre]
        for modelo in modelos:
            if modelo not in ds_res:
                continue
            m = ds_res[modelo]["metricas"]
            row = [
                _truncar(modelo, 22),
                _truncar(ds_nombre, 28),
                f"{m['accuracy']:.4f}",
                f"{m['precision']:.4f}",
                f"{m['recall']:.4f}",
                f"{m['f1']:.4f}",
            ]
            filas.append(row)

    if not filas:
        ax.text(0.5, 0.5, "Sin datos", ha="center", va="center", transform=ax.transAxes)
        return

    tabla = ax.table(
        cellText=filas,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(8)
    tabla.scale(1, 1.4)

    # Estilo de cabecera
    for j in range(len(headers)):
        tabla[0, j].set_facecolor("#1A3A5C")
        tabla[0, j].set_text_props(color="white", fontweight="bold")

    # Alternar filas
    for i in range(1, len(filas) + 1):
        color = "#EAF1FB" if i % 2 == 0 else "#FFFFFF"
        for j in range(len(headers)):
            tabla[i, j].set_facecolor(color)

    ax.set_title(
        "Tabla Comparativa Completa",
        fontsize=10,
        fontweight="bold",
        pad=8,
        color="#1A3A5C",
    )


def _plot_conclusion(fig, gs_slot, conclusion_texto: str):
    """Caja de texto con la conclusión generada por IA."""
    ax = fig.add_subplot(gs_slot)
    ax.axis("off")

    # Fondo de la caja
    ax.add_patch(
        plt.Rectangle(
            (0, 0),
            1,
            1,
            transform=ax.transAxes,
            facecolor="#EFF6FF",
            edgecolor="#2563EB",
            linewidth=1.5,
            clip_on=False,
        )
    )

    ax.text(
        0.5,
        0.97,
        "📊 Conclusión — Análisis IA",
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
        color="#1E40AF",
        transform=ax.transAxes,
    )

    # Envolver el texto
    ancho_wrap = 130
    texto_wrap = "\n".join(
        textwrap.fill(linea, ancho_wrap) for linea in conclusion_texto.split("\n")
    )

    ax.text(
        0.5,
        0.82,
        texto_wrap,
        ha="center",
        va="top",
        fontsize=8,
        color="#1F2937",
        transform=ax.transAxes,
        wrap=True,
        family="monospace",
    )


# ── Conclusión IA ─────────────────────────────────────────────────────────────


def _generar_conclusion(todos_resultados: dict, cfg_conclusion: dict) -> str:
    """
    Genera la conclusión final usando Anthropic API o rule-based fallback.
    """
    if not cfg_conclusion.get("enabled", True):
        return "Conclusión IA desactivada en config.yaml."

    motor = cfg_conclusion.get("motor", "rule_based")

    if motor == "anthropic_api":
        try:
            return _conclusion_anthropic(todos_resultados, cfg_conclusion)
        except Exception as e:
            Logger.warn(f"Error con Anthropic API: {e}. Usando rule-based.")
            return _conclusion_rule_based(todos_resultados)
    else:
        return _conclusion_rule_based(todos_resultados)


def _conclusion_anthropic(todos_resultados: dict, cfg: dict) -> str:
    """Llama a la API de Anthropic con los resultados y pide una conclusión."""
    import os

    # Verificar que la API key esté disponible antes de intentar la llamada
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "Variable de entorno ANTHROPIC_API_KEY no configurada.\n"
            "  Para habilitar conclusiones IA, agrega tu clave:\n"
            "    Windows: set ANTHROPIC_API_KEY=sk-ant-...\n"
            "    Linux/Mac: export ANTHROPIC_API_KEY=sk-ant-...\n"
            "  O cambia conclusion_ia.motor a 'rule_based' en config.yaml."
        )

    import anthropic

    Logger.info("Generando conclusión con Anthropic API...")

    # Preparar resumen de resultados para el prompt
    resumen_lineas = []
    for ds, modelos_res in todos_resultados.items():
        resumen_lineas.append(f"\nDataset: {ds}")
        for modelo, info in modelos_res.items():
            m = info["metricas"]
            resumen_lineas.append(
                f"  {modelo}: Accuracy={m['accuracy']:.4f} | "
                f"Precision={m['precision']:.4f} | "
                f"Recall={m['recall']:.4f} | "
                f"F1={m['f1']:.4f}"
            )
    resumen = "\n".join(resumen_lineas)

    prompt = (
        "Eres un experto en machine learning aplicado a detección de fraude financiero. "
        "Analiza los siguientes resultados de modelos de clasificación evaluados en "
        "múltiples datasets de fraude con tarjeta de crédito:\n\n"
        f"{resumen}\n\n"
        "En español, escribe una conclusión técnica de 4-6 oraciones que incluya:\n"
        "1. Qué modelo tuvo mejor desempeño global y por qué el recall es la métrica más importante en este contexto.\n"
        "2. Observaciones sobre el desbalance de clases y cómo afecta los resultados.\n"
        "3. Qué modelo recomendarías en producción y por qué.\n"
        "4. Una advertencia breve sobre posibles mejoras (SMOTE, threshold tuning, etc.).\n"
        "Sé directo y técnico. No uses listas, escribe en párrafo."
    )

    client = anthropic.Anthropic()
    mensaje = client.messages.create(
        model=cfg.get("modelo_api", "claude-sonnet-4-20250514"),
        max_tokens=cfg.get("max_tokens", 600),
        messages=[{"role": "user", "content": prompt}],
    )

    texto = mensaje.content[0].text.strip()
    Logger.ok("Conclusión generada por Anthropic API.")
    return texto


def _conclusion_rule_based(todos_resultados: dict) -> str:
    """Genera una conclusión determinista basada en las métricas."""
    mejor_global = None
    mejor_recall = -1
    mejor_dataset = None
    peor_recall = 2
    peor_modelo = None

    for ds, modelos_res in todos_resultados.items():
        for modelo, info in modelos_res.items():
            recall = info["metricas"]["recall"]
            if recall > mejor_recall:
                mejor_recall = recall
                mejor_global = modelo
                mejor_dataset = ds
            if recall < peor_recall:
                peor_recall = recall
                peor_modelo = modelo

    lineas = [
        f"Conclusión automática (modo rule-based):",
        f"",
        f"El modelo '{mejor_global}' obtuvo el mejor Recall ({mejor_recall:.4f}) "
        f"en el dataset '{mejor_dataset}'. En detección de fraude, el Recall es la "
        f"métrica más crítica porque mide cuántos fraudes reales fueron detectados; "
        f"un falso negativo (fraude no detectado) tiene mayor costo que una falsa alarma.",
        f"",
        f"El dataset de fraude con tarjeta de crédito presenta un fuerte desbalance de "
        f"clases (usualmente <0.2% de fraudes), lo que puede inflar el Accuracy "
        f"artificialmente. Por ello, Precision, Recall y F1 son métricas más informativas.",
        f"",
        f"Se recomienda '{mejor_global}' como modelo base para producción, complementado "
        f"con técnicas de rebalanceo (SMOTE, class_weight) y ajuste del umbral de "
        f"clasificación para optimizar la relación Precision-Recall según el costo "
        f"de negocio aceptable.",
    ]

    return "\n".join(lineas)


# ── Utilidad ──────────────────────────────────────────────────────────────────


def _truncar(texto: str, max_len: int) -> str:
    return texto if len(texto) <= max_len else texto[: max_len - 1] + "…"
