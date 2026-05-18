# pipeline/preprocessor.py
# ============================================================
# ETAPA 3 — PREPROCESAMIENTO
# Responsabilidad: separar X/y, hacer train/test split,
# preparar el objeto CV y el scoring dict.
# El escalado se aplica DENTRO del pipeline de cada modelo
# (StandardScaler como step) para evitar data leakage.
# ============================================================

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from utils.logger import Logger
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from utils.logger import Logger


def preprocesar(df: pd.DataFrame, cfg_dataset: dict, cfg_general: dict) -> dict:
    """
    Prepara los datos para el entrenamiento.

    Retorna un diccionario con:
      - X_train, X_test : DataFrames de features
      - y_train, y_test : Series con etiquetas
      - cv              : objeto StratifiedKFold listo
      - scoring         : dict de métricas para GridSearchCV
    """
    Logger.seccion(f"Etapa 3 · Preprocesamiento — {cfg_dataset['nombre']}")

    target = cfg_dataset["target_col"]
    random_state = cfg_general.get("random_state", 42)
    test_size = cfg_general.get("test_size", 0.20)
    cv_folds = cfg_general.get("cv_folds", 5)
    max_levels = cfg_general.get("max_categorical_levels", 100)

    # ── Validar test_size ─────────────────────────────────────
    if not (0.0 < test_size < 1.0):
        Logger.warn(
            f"test_size={test_size} inválido (debe ser entre 0 y 1). "
            f"Se usará 0.20 por defecto."
        )
        test_size = 0.20

    # ── Separar features y etiqueta ───────────────────────────
    X = df.drop(columns=[target])
    y = df[target]

    Logger.info(f"Features : {X.shape[1]} columnas")
    Logger.info(f"Muestras : {len(y):,} filas")

    # ── Detectar columnas numéricas y categóricas ─────────────
    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    Logger.info(f"Columnas numéricas: {len(numeric_features)}")
    Logger.info(f"Columnas categóricas detectadas: {len(categorical_features)}")


    # ── Filtrar categóricas con alta cardinalidad ─────────────
    categorical_features_filtered = []
    excluded_high_cardinality = []

    for col in categorical_features:
        unique_count = X[col].nunique(dropna=True)

        if unique_count <= max_levels:
            categorical_features_filtered.append(col)
        else:
            excluded_high_cardinality.append((col, unique_count))

    categorical_features = categorical_features_filtered

    if categorical_features:
        Logger.info(
            f"Columnas categóricas usadas para encoding: " f"{categorical_features}"
        )
    else:
        Logger.warn("No se utilizarán columnas categóricas.")

    if excluded_high_cardinality:
        Logger.warn(
            "Columnas excluidas por alta cardinalidad " f"(>{max_levels} categorías):"
        )
        for col, unique_count in excluded_high_cardinality:
            Logger.warn(f"  - {col}: {unique_count:,} valores únicos")

    # ── Construir ColumnTransformer ───────────────────────────
    transformers = []

    if numeric_features:
        transformers.append(
            (
                "num",
                StandardScaler(),
                numeric_features,
            )
        )

    if categorical_features:
        # Compatibilidad con distintas versiones de scikit-learn
        try:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

        transformers.append(
            (
                "cat",
                encoder,
                categorical_features,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    # ── Train/Test split estratificado ────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    Logger.ok(
        f"Train: {X_train.shape[0]:,} filas  |  " f"Test: {X_test.shape[0]:,} filas"
    )

    # ── Validación cruzada ────────────────────────────────────
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    Logger.info(f"CV estratificado: {cv_folds} folds")
    # ── Métricas ──────────────────────────────────────────────
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }

    # ── Resumen del preprocesador ─────────────────────────────
    Logger.ok("Preprocesador creado correctamente.")
    Logger.info(
        f"Transformadores activos: "
        f"{len(transformers)} "
        f"(numéricas={len(numeric_features)}, "
        f"categóricas={len(categorical_features)})"
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "cv": cv,
        "scoring": scoring,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    }
