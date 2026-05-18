# 🔍 Detección de Fraude Financiero con Machine Learning

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?logo=scikitlearn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow)

**Tecnológico Superior de Jalisco · Ingeniería en Sistemas Computacionales · ISIC 7A**  
**Materia:** Aprendizaje Automático | **Mtra:** Rebeca Anaya González | **Autor:** Vargas Pelayo Salvador

> ⚠️ **Aviso:** Proyecto educativo y de investigación. No usar en producción sin validación adicional.

---

## 📌 Descripción

Pipeline completo de Machine Learning para detectar transacciones fraudulentas en datasets reales. Compara múltiples modelos, genera reportes visuales y conclusiones con IA. Todo se controla desde `config.yaml`.

**Objetivos clave:**

- Alta precisión en detección de fraudes
- Reducción de falsos positivos
- Comparación de algoritmos
- Base escalable para APIs o dashboards

---

## 🧠 Algoritmos utilizados

|       Modelo        |     Tipo      | GPU soportada |
| :-----------------: | :-----------: | :-----------: |
| Logistic Regression | Clasificación |      ❌       |
|    Decision Tree    | Clasificación |      ❌       |
|    Random Forest    | Clasificación |      ❌       |
|  Gradient Boosting  | Clasificación |      ❌       |
|       XGBoost       | Clasificación | ✅ (opcional) |
|      LightGBM       | Clasificación | ✅ (opcional) |

---

## 📊 Dataset

Transacciones financieras con etiqueta binaria: `0` = legítimo, `1` = fraude.

|     Característica     |              Descripción               |
| :--------------------: | :------------------------------------: |
|         Monto          |        Valor de la transacción         |
|         Tiempo         |      Timestamp o tiempo relativo       |
|   Categoría comercio   |            Tipo de negocio             |
| Variables anonimizadas |      PCA u otras transformaciones      |
|   Etiqueta de fraude   | Variable objetivo (`isFraud`, `Class`) |

---

## ⚙️ Pipeline

```text
Carga → Limpieza → EDA → Ingeniería → Split → Balanceo (SMOTE) → Entrenamiento → Evaluación → Mejor modelo → Serialización → Predicción
```

---

## Estructura del proyecto

```text
fraud_detection/
│
├── main.py                        ← Punto de entrada
├── config.yaml                    ← Configuración central
├── requirements.txt               ← Dependencias Python
├── .gitignore
│
├── pipeline/                      ← Etapas del pipeline (una por archivo)
│   ├── loader.py                  ← Etapa 1: Carga del dataset
│   ├── cleaner.py                 ← Etapa 2: Limpieza y validación
│   ├── preprocessor.py            ← Etapa 3: Encoding + split + CV
│   ├── trainer.py                 ← Etapa 4: Entrenamiento con GridSearchCV
│   ├── evaluator.py               ← Etapa 5: Evaluación en test
│   └── reporter.py                ← Etapa 6: Frame PNG + conclusión IA
│
├── utils/
│   ├── logger.py                  ← Logger con colores para toda la app
│   ├── dataset_manager.py         ← Gestión de datasets (local / OpenML)
│   ├── model_manager.py           ← Caché de modelos entrenados (.pkl)
│   ├── gpu_manager.py             ← Detección de GPU CUDA
│   ├── drive_manager.py           ← Subida a Google Drive (modo online)
│   ├── colab_launcher.py          ← Generación de notebook para Colab
│   ├── descargar_datasets.py      ← Helper interactivo para descargar datasets
│   ├── ieee_merge.py              ← Fusión de archivos del dataset IEEE-CIS
│   ├── datasets/                  ← CSVs descargados (ignorados por git)
│   │   └── .gitkeep
│   └── modelos/                   ← Modelos entrenados .pkl (ignorados por git)
│
└── results/                       ← Frames PNG generados automáticamente
```

---

## 🚀 Instalación

```bash
git clone https://github.com/tu-usuario/fraud_detection.git
cd fraud_detection
python -m venv venv
source venv/bin/activate      # Linux/Mac, o .\venv\Scripts\Activate.ps1 en Windows
pip install -r requirements.txt
```

> Python 3.10 o superior

---

## 📥 Descarga de datasets

Los datasets no están incluidos (archivos grandes). Usa el script interactivo:

```bash
python utils/descargar_datasets.py
```

O descarga manualmente (ver tabla siguiente) y coloca los CSVs en utils/datasets/. Activa cada dataset en config.yaml (enabled: true).

---

## Datasets disponibles

| Dataset | Filas | % Fraude | Fuente | Archivo esperado |
|:---:|:---:|:---:|:---:|:---:|
| CreditCard Fraud | 284,807 | 0.17% | OpenML/Kaggle | `creditcard.csv` |
| BankSim | 594,643 | 1.20% | Kaggle | `banksim.csv` |
| PaySim | 6,362,620 | 0.13% | Kaggle | `paysim.csv` (usa `sample_frac:0.3`) |
| Banking Fraud Risk Analytics | 10,000 | 12.51% | Kaggle | `banking_transactions.csv` |
| Indian Banking Transactions | 550,000 | 0.89% | Kaggle | `indian_banking_transactions.csv` |
| IEEE-CIS Fraud Detection | 590,540 | 3.50% | Kaggle competition | `ieee_cis.csv`¹ |

¹ Ejecuta python utils/ieee_merge.py para fusionar train_transaction.csv y train_identity.csv.

---

## ▶️ Ejecución

```bash
python main.py
```

El pipeline procesa todos los datasets activos en config.yaml y genera frames PNG en results/.

### Modos de ejecución (general.modo en config.yaml)
|Modo	|Comportamiento|
|:---:|:---:|
|offline|	Solo archivos locales (por defecto). Si falta un CSV, pregunta descargar.|
|online	|Sube proyecto a Google Drive y genera notebook para Colab (GPU).|
|preguntar|	Pregunta al iniciar.|
---

## 🔁 Flujo detallado (por dataset)

1. loader.py – Carga CSV desde disco o OpenML.
2. cleaner.py – Elimina NaNs, duplicados; normaliza objetivo (0/1); detecta categóricas.
3. preprocessor.py – Escala numéricas (StandardScaler), codifica categóricas (OneHotEncoder), descarta columnas > max_categorical_levels, split estratificado (80/20), prepara StratifiedKFold.
4. trainer.py – Para cada modelo activo: si existe .pkl en caché lo carga; si no, ejecuta GridSearchCV (optimiza recall) y guarda el mejor modelo.
5. evaluator.py – Predice en test, calcula accuracy, precision, recall, F1, matrices de confusión.
6. reporter.py – Genera PNG con gráficas comparativas, tablas y conclusión (IA o rule-based).

---

## ⚙️ Configuración (config.yaml)

### Parámetros generales

```yaml
general:
  random_state: 42
  test_size: 0.2
  cv_folds: 5
  refit_metric: "recall"
  n_jobs: -1
  modo: "offline"
  guardar_modelos: true
  max_categorical_levels: 100
```

### Activar datasets / modelos

```yaml
datasets:
  - nombre: "PaySim (Kaggle)"
    enabled: true
    fuente: "csv_local"
    archivo_local: "paysim.csv"
    target_col: "isFraud"
    fraud_value: 1
    sample_frac: 0.3

modelos:
  - nombre: "Random Forest"
    enabled: true
  - nombre: "SVM"
    enabled: false
```

### Conclusión IA

```yaml
conclusion_ia:
  enabled: true
  motor: "anthropic_api" # o "rule_based"
  modelo_api: "claude-sonnet-4-20250514"
  max_tokens: 600
  idioma: "español"
```

Requiere variable de entorno ANTHROPIC_API_KEY. Si no, usa rule_based.

### 📊 Modelos disponibles (detalle)

|       Modelo        |    Habilitado     | Escalado |              GPU Notas              |
| :-----------------: | :---------------: | :------: | :---------------------------------: |
| Logistic Regression | ✅ StandardScaler |    ❌    |         Rápido, línea base          |
|    Decision Tree    |       ✅ No       |    ❌    |            Interpretable            |
|    Random Forest    |       ✅ No       |    ❌    |   Mejor balance precisión/recall    |
|         SVM         | ❌ StandardScaler |    ❌    | O(n²) – desactivado por rendimiento |
|       XGBoost       |       ❌ No       |    ✅    |   Requiere GPU para ventaja real    |

---

## 💾 Caché de modelos

Los modelos entrenados se guardan en utils/modelos/\*.pkl. En ejecuciones posteriores se cargan en segundos. Para forzar reentrenamiento, elimina los .pkl correspondientes.

```bash
# Windows
del utils\modelos\*.pkl
# Linux/Mac
rm utils/modelos/*.pkl
```

---

## 📈 Métricas evaluadas

|  Métrica  | Importancia |                                 Por qué                                  |
| :-------: | :---------: | :----------------------------------------------------------------------: |
|  Recall   |   ⭐⭐⭐    |       Crítica Porcentaje de fraudes detectados (evitar pérdidas).        |
| Precision |    ⭐⭐     | Importante Porcentaje de alertas que son fraude real (evitar molestias). |
| F1-Score  |    ⭐⭐     |                    Balance entre recall y precision.                     |
| Accuracy  |     ⭐      |        Referencia Puede ser engañosa en datasets desbalanceados.         |

> GridSearchCV optimiza recall y todos los modelos usan class_weight="balanced".

---

## ➕ Agregar nuevo dataset

CSV local: coloca archivo en utils/datasets/ y añade entrada en config.yaml:

```bash
- nombre: "Mi Dataset"
  enabled: true
  fuente: "csv_local"
  archivo_local: "mi_dataset.csv"
  target_col: "fraud_label"
  fraud_value: 1
  sample_frac: 1.0
```

OpenML: similar, usa fuente: "openml", openml_name, openml_version.

---

## 🧩 Agregar nuevo modelo

Añade entrada en config.yaml (ejemplo con Gradient Boosting).

Registra la clase en pipeline/trainer.py dentro de \_CLASE_MAP.

```bash
- nombre: "Gradient Boosting"
  enabled: true
  clase: "GradientBoostingClassifier"
  usa_scaler: false
  grid:
    model__max_depth: [3,5]
    model__learning_rate: [0.05,0.1]
```

---

## 🖥️ GPU (opcional)

Detección automática (PyTorch → CuPy → pynvml). Solo modelos con gpu_support: true usan CUDA (ej. XGBoost). Para forzar CPU:

```bash
gpu:
  enabled: true
  forzar_cpu: true
```

---

## ☁️ Modo online (Google Colab)

1. Cambia modo: "online" en config.yaml.
2. Ejecuta python main.py.
3. Se comprime el proyecto, sube a Google Drive y genera un notebook .ipynb.
4. Abre el notebook en Colab, activa GPU (T4) y ejecuta.

> Asegúrate de tener los datasets descargados localmente antes de cambiar a modo online.

---

## ❓ Preguntas frecuentes

¿Por qué tarda la primera vez?
GridSearchCV completo ~20 min. Siguientes ejecuciones usan caché (segundos).

¿Sin internet?
En modo offline, si falta un CSV pregunta si descargar; si respondes n, omite ese dataset.

¿Columnas de texto?
Se aplica OneHotEncoder automáticamente a categóricas con ≤ max_categorical_levels únicos. Columnas de alta cardinalidad (IDs) se descartan.

¿Cómo conocer la columna objetivo?

```bash
python -c "import pandas as pd; print(pd.read_csv('utils/datasets/mi_archivo.csv', nrows=1).columns.tolist())"
```

---

## 📜 Licencia

Este proyecto se distribuye bajo la licencia MIT.
