# pipeline/trainer.py
# ============================================================
# ETAPA 4 — ENTRENAMIENTO
# Responsabilidad: construir pipelines sklearn, ejecutar
# GridSearchCV con CV estratificada, y devolver los mejores
# estimadores. Usa ModelManager para cachear en disco.
#
# Ahora el Pipeline incluye:
#   1. preprocessor (ColumnTransformer)
#   2. model
#
# El preprocessor aplica:
#   - StandardScaler a variables numéricas
#   - OneHotEncoder a variables categóricas
# ============================================================

import importlib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from utils.logger import Logger
from utils.model_manager import ModelManager

# Mapa de clase string → módulo sklearn/xgboost
_CLASE_MAP = {
    "LogisticRegression": ("sklearn.linear_model", "LogisticRegression"),
    "DecisionTreeClassifier": ("sklearn.tree", "DecisionTreeClassifier"),
    "RandomForestClassifier": ("sklearn.ensemble", "RandomForestClassifier"),
    "SVC": ("sklearn.svm", "SVC"),
    "XGBClassifier": ("xgboost", "XGBClassifier"),
}


def _importar_clase(nombre_clase: str):
    """Importa dinámicamente una clase de sklearn o xgboost."""
    if nombre_clase not in _CLASE_MAP:
        raise ValueError(
            f"Clase desconocida: '{nombre_clase}'. "
            f"Opciones válidas: {list(_CLASE_MAP.keys())}"
        )

    modulo_str, clase_str = _CLASE_MAP[nombre_clase]
    modulo = importlib.import_module(modulo_str)

    return getattr(modulo, clase_str)


def _construir_pipeline(cfg_modelo: dict, preprocessor, random_state: int) -> Pipeline:
    """
        Construye un Pipeline sklearn.

    Pipeline final:
        preprocessor -> model

    Nota:
    El preprocessor ya contiene:
      - StandardScaler
      - OneHotEncoder

    Por lo tanto, ya no se usa el scaler definido por
    cfg_modelo['usa_scaler'].
    """
    ClaseModelo = _importar_clase(cfg_modelo["clase"])
    params_fijos = cfg_modelo.get("params_fijos", {})

    # Solo pasar random_state si el modelo lo acepta
    try:
        modelo = ClaseModelo(**params_fijos, random_state=random_state)
    except TypeError:
        modelo = ClaseModelo(**params_fijos)

    pasos = [
        ("preprocessor", preprocessor),
        ("model", modelo),
    ]

    # if cfg_modelo.get("usa_scaler", False):
    #     pasos.append(("scaler", StandardScaler()))
    # pasos.append(("model", modelo))

    return Pipeline(pasos)


def entrenar_modelos(
    datos: dict,
    cfg_modelos: list,
    cfg_general: dict,
    nombre_dataset: str,
) -> dict:
    """
    Entrena (o carga desde caché) todos los modelos activos.

    Parámetros
    ----------
    datos          : dict con X_train, y_train, cv, scoring
    cfg_modelos    : lista de configuraciones de modelos (config.yaml)
    cfg_general    : sección general de config.yaml
    nombre_dataset : nombre del dataset actual (para slug de caché)

    Retorna
    -------
    dict  {nombre_modelo: {"estimador": ..., "best_params": ..., "cv_results": ...}}


    Entrena (o carga desde caché) todos los modelos activos.

    Parámetros
    ----------
    datos:
        dict con:
        - X_train
        - y_train
        - preprocessor
        - cv
        - scoring

    cfg_modelos:
        Lista de modelos definidos en config.yaml

    cfg_general:
        Configuración general

    nombre_dataset:
        Nombre del dataset actual

    Retorna
    -------
    dict
        {
            nombre_modelo: {
                "estimador": ...,
                "best_params": ...,
                "cv_results": ...
            }
        }

    """
    Logger.seccion(f"Etapa 4 · Entrenamiento — {nombre_dataset}")

    X_train = datos["X_train"]
    y_train = datos["y_train"]
    preprocessor = datos["preprocessor"]
    cv = datos["cv"]
    scoring = datos["scoring"]
    random_state = cfg_general.get("random_state", 42)
    refit_metric = cfg_general.get("refit_metric", "recall")
    n_jobs = cfg_general.get("n_jobs", -1)

    model_mgr = ModelManager(cfg_general)
    resultados = {}

    modelos_activos = [m for m in cfg_modelos if m.get("enabled", True)]

    Logger.info(f"Modelos a entrenar: " f"{[m['nombre'] for m in modelos_activos]}")

    for cfg_modelo in modelos_activos:
        nombre = cfg_modelo["nombre"]
        Logger.subseccion(f"Modelo: {nombre}")

        # ── ¿Existe en caché? ─────────────────────────────────
        if model_mgr.existe(nombre, nombre_dataset):
            Logger.ok(f"Cargando desde caché (omite entrenamiento)")
            estimador = model_mgr.cargar(nombre, nombre_dataset)
            resultados[nombre] = {
                "estimador": estimador,
                "best_params": getattr(estimador, "best_params_", {}),
                "cv_results": None,
                "desde_cache": True,
            }
            continue

        # ── Construir pipeline ────────────────────────────────
        pipe = _construir_pipeline(
            cfg_modelo=cfg_modelo,
            preprocessor=preprocessor,
            random_state=random_state,
        )
        param_grid = cfg_modelo.get("grid", {})

        Logger.info(
            f"Grid search: {len(param_grid)} parámetros | " f"refit='{refit_metric}'"
        )

        # ── GridSearchCV ──────────────────────────────────────
        grid = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring=scoring,
            refit=refit_metric,
            cv=cv,
            n_jobs=n_jobs,
            return_train_score=True,
            verbose=0,
        )

        # Entrenamiento
        grid.fit(X_train, y_train)

        Logger.ok(f"Mejores hiperparámetros: {grid.best_params_}")

        # Comparación train vs val del mejor conjunto
        # ── Métricas del mejor modelo ────────────────────────
        idx = grid.best_index_
        cr = grid.cv_results_
        Logger.result(
            f"Train recall={cr['mean_train_recall'][idx]:.4f} | "
            f"Val recall={cr['mean_test_recall'][idx]:.4f} | "
            f"Val precision={cr['mean_test_precision'][idx]:.4f}"
        )

        # ── Guardar en caché ──────────────────────────────────
        model_mgr.guardar(grid.best_estimator_, nombre, nombre_dataset)

        resultados[nombre] = {
            "estimador": grid.best_estimator_,
            "best_params": grid.best_params_,
            "cv_results": cr,
            "best_index": idx,
            "desde_cache": False,
        }

    return resultados
