# utils/gpu_manager.py
# ============================================================
# DETECCIÓN Y CONFIGURACIÓN DE GPU
# ============================================================
# Detecta si hay GPU CUDA disponible y configura los modelos
# que la soportan para usarla automáticamente.
#
# Modelos con soporte GPU real:
#   XGBoost    → tree_method="hist" + device="cuda"
#   LightGBM   → device="gpu"  (si está instalado)
#   cuML RF    → requiere RAPIDS (solo Linux/Colab con GPU)
#
# sklearn (LR, DT, RF, SVM) NO tienen soporte GPU nativo.
# Para ellos se maximizan los n_jobs en CPU.
# ============================================================

from utils.logger import Logger


def detectar_gpu() -> dict:
    """
    Detecta la GPU disponible en el sistema.

    Retorna
    -------
    dict con:
      - disponible : bool
      - nombre     : str  (nombre de la GPU o "N/A")
      - memoria_gb : float
      - cuda_version: str
      - backend    : "cuda" | "none"
    """
    info = {
        "disponible"   : False,
        "nombre"       : "N/A",
        "memoria_gb"   : 0.0,
        "cuda_version" : "N/A",
        "backend"      : "none",
    }

    # ── Intento 1: torch (más común, detecta cualquier GPU CUDA) ──
    try:
        import torch
        if torch.cuda.is_available():
            idx  = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            info.update({
                "disponible"   : True,
                "nombre"       : props.name,
                "memoria_gb"   : round(props.total_memory / 1e9, 2),
                "cuda_version" : torch.version.cuda or "N/A",
                "backend"      : "cuda",
            })
            return info
    except ImportError:
        pass

    # ── Intento 2: cupy ───────────────────────────────────────────
    try:
        import cupy as cp
        dev  = cp.cuda.Device(0)
        mem  = dev.mem_info
        info.update({
            "disponible"  : True,
            "nombre"      : "CUDA GPU (cupy)",
            "memoria_gb"  : round(mem[1] / 1e9, 2),
            "backend"     : "cuda",
        })
        return info
    except Exception:
        pass

    # ── Intento 3: pynvml (NVIDIA Management Library) ─────────────
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        nombre = pynvml.nvmlDeviceGetName(handle)
        mem    = pynvml.nvmlDeviceGetMemoryInfo(handle)
        info.update({
            "disponible"  : True,
            "nombre"      : nombre if isinstance(nombre, str) else nombre.decode(),
            "memoria_gb"  : round(mem.total / 1e9, 2),
            "backend"     : "cuda",
        })
        return info
    except Exception:
        pass

    return info


def configurar_gpu_modelos(cfg_modelos: list, cfg_gpu: dict) -> list:
    """
    Ajusta los parámetros de cada modelo para usar GPU si está disponible.

    Parámetros
    ----------
    cfg_modelos : lista de configs de modelos desde config.yaml
    cfg_gpu     : sección gpu del config.yaml

    Retorna
    -------
    Lista de configs de modelos con parámetros GPU ajustados.
    """
    if not cfg_gpu.get("enabled", True) or cfg_gpu.get("forzar_cpu", False):
        Logger.warn("GPU desactivada por configuración. Usando CPU.")
        return cfg_modelos

    gpu_info = detectar_gpu()

    if not gpu_info["disponible"]:
        Logger.warn("No se detectó GPU CUDA. Todos los modelos usarán CPU.")
        return cfg_modelos

    Logger.ok(
        f"GPU detectada: {gpu_info['nombre']} "
        f"| VRAM: {gpu_info['memoria_gb']} GB "
        f"| CUDA: {gpu_info['cuda_version']}"
    )

    modelos_ajustados = []
    for cfg in cfg_modelos:
        cfg_mod = dict(cfg)  # copia para no mutar el original

        if cfg_mod.get("gpu_support", False):
            clase = cfg_mod.get("clase", "")

            # ── XGBoost con CUDA ──────────────────────────────
            if clase == "XGBClassifier":
                params = dict(cfg_mod.get("params_fijos", {}))
                params["device"]      = "cuda"
                params["tree_method"] = "hist"
                # n_jobs no aplica en GPU
                params.pop("n_jobs", None)
                cfg_mod["params_fijos"] = params
                Logger.ok(f"  {cfg_mod['nombre']} → GPU (CUDA, tree_method=hist)")

        modelos_ajustados.append(cfg_mod)

    return modelos_ajustados


def imprimir_info_gpu():
    """Imprime un resumen del estado de GPU al iniciar."""
    info = detectar_gpu()
    Logger.seccion("Detección de Hardware")
    if info["disponible"]:
        Logger.ok(f"GPU : {info['nombre']}")
        Logger.ok(f"VRAM: {info['memoria_gb']} GB")
        Logger.ok(f"CUDA: {info['cuda_version']}")
        Logger.info("XGBoost usará GPU automáticamente.")
    else:
        Logger.warn("GPU CUDA no disponible. Procesamiento en CPU.")
        Logger.info(
            "Para habilitar GPU instala: pip install torch  (con soporte CUDA)\n"
            "O ejecuta en Google Colab con runtime GPU activado."
        )
