# 🔍 Detección de Fraude Financiero con Machine Learning
# 🔍 Detección de Fraude Financiero con Machine Learning

**Tecnológico Superior de Jalisco · Ingeniería en Sistemas Computacionales · ISIC 7A**  
**Materia:** Aprendizaje Automático  
**Mtra:** Rebeca Anaya González  |  **Autor:** Vargas Pelayo Salvador

---

## ¿Qué hace este proyecto?

Pipeline completo de Machine Learning para detectar transacciones fraudulentas en datasets reales de fraude financiero. Compara múltiples modelos de clasificación supervisada y al finalizar genera un **frame visual** con métricas comparativas, matrices de confusión y una conclusión generada por IA.
Pipeline completo de Machine Learning para detectar transacciones fraudulentas en datasets reales de fraude financiero. Compara múltiples modelos de clasificación supervisada y al finalizar genera un **frame visual** con métricas comparativas, matrices de confusión y una conclusión generada por IA.

Todo se controla desde un único archivo: `config.yaml`. Ahí se define qué datasets procesar, qué modelos entrenar, si usar GPU, y cómo generar la conclusión final.
Todo se controla desde un único archivo: `config.yaml`. Ahí se define qué datasets procesar, qué modelos entrenar, si usar GPU, y cómo generar la conclusión final.

---

## Estructura del proyecto
## Estructura del proyecto

```
fraud_detection/
│
├── main.py                        ← Punto de entrada
├── config.yaml                    ← Configuración central
├── requirements.txt               ← Dependencias Python
├── .gitignore
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

## Instalación

### 1. Clonar el repositorio
### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/fraud_detection.git
git clone https://github.com/tu-usuario/fraud_detection.git
cd fraud_detection
```

### 2. Crear entorno virtual
### 2. Crear entorno virtual

```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / Mac
python -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

> **Python requerido: 3.10 o superior**

---

## Descarga de datasets

Los datasets **no se incluyen en el repositorio** porque son archivos grandes (hasta 470 MB). Debes descargarlos antes de ejecutar el pipeline.

### Opción A — Script interactivo (recomendado)

```bash
python utils/descargar_datasets.py
```

El script muestra los datasets disponibles y te guía paso a paso. Los que se pueden descargar automáticamente (OpenML) lo hacen sin salir del script. Los de Kaggle muestran instrucciones para descarga manual.

### Opción B — Descarga manual

Descarga cada CSV de los enlaces de abajo, colócalo en `utils/datasets/` con el nombre exacto indicado y actívalo en `config.yaml` cambiando `enabled: false` a `enabled: true`.

---

## Datasets disponibles

| Dataset | Filas | Fraudes | Fuente | Archivo esperado |
|---------|------:|--------:|--------|-----------------|
| CreditCard Fraud | 284,807 | 0.17 % | [OpenML / Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | `creditcard.csv` |
| BankSim | 594,643 | 1.20 % | [Kaggle](https://www.kaggle.com/datasets/ealaxi/banksim1) | `banksim.csv` |
| PaySim | 6,362,620 | 0.13 % | [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1) | `paysim.csv` |
| Banking Fraud Risk Analytics | 10,000 | 12.51 % | [Kaggle](https://www.kaggle.com/datasets/deepeshkansotia/banking-fraud-detection-and-risk-analytics-dataset) | `banking_transactions.csv` |
| Indian Banking Transactions 2019–2024 | 550,000 | 0.89 % | [Kaggle](https://www.kaggle.com/datasets/belbino/indian-banking-transactions-20192024) | `indian_banking_transactions.csv` |
| IEEE-CIS Fraud Detection | 590,540 | 3.50 % | [Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection/data) | `ieee_cis.csv` ¹ |

> ¹ IEEE-CIS requiere combinar dos archivos. Ver instrucciones más abajo.

### Instrucciones de descarga por dataset

#### CreditCard Fraud
1. Entra a: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Descarga `creditcard.csv`
3. Colócalo en `utils/datasets/creditcard.csv`

> También se puede descargar automáticamente desde OpenML con el script interactivo.

#### BankSim
1. Entra a: https://www.kaggle.com/datasets/ealaxi/banksim1
2. Descarga `bs140513_032310.csv`
3. Renómbralo a `banksim.csv`
4. Colócalo en `utils/datasets/banksim.csv`

#### PaySim
1. Entra a: https://www.kaggle.com/datasets/ealaxi/paysim1
2. Descarga `PS_20174392719_1491204439457_log.csv`
3. Renómbralo a `paysim.csv`
4. Colócalo en `utils/datasets/paysim.csv`

> PaySim tiene ~6.3 M filas. En `config.yaml` está configurado con `sample_frac: 0.30` (usa solo el 30 %).

#### Banking Fraud Risk Analytics
1. Entra a: https://www.kaggle.com/datasets/deepeshkansotia/banking-fraud-detection-and-risk-analytics-dataset
2. Descarga el CSV principal
3. Guárdalo como `utils/datasets/banking_transactions.csv`

#### Indian Banking Transactions 2019–2024
1. Entra a: https://www.kaggle.com/datasets/belbino/indian-banking-transactions-20192024
2. Descarga el CSV
3. Guárdalo como `utils/datasets/indian_banking_transactions.csv`

> Este dataset incluye columnas de texto categóricas (`account_type`, `transaction_type`, `merchant_category`, etc.). El pipeline las codifica automáticamente con **OneHotEncoder** antes de entrenar.

#### IEEE-CIS Fraud Detection
1. Acepta las reglas de la competencia en: https://www.kaggle.com/competitions/ieee-fraud-detection/data
2. Descarga `train_transaction.csv` y `train_identity.csv`
3. Coloca ambos en `utils/datasets/`
4. Ejecuta el script de fusión:
   ```bash
   python utils/ieee_merge.py
   ```
5. Esto genera `utils/datasets/ieee_cis.csv` listo para usar
6. Actívalo en `config.yaml` con `enabled: true`
> **Python requerido: 3.10 o superior**

---

## Descarga de datasets

Los datasets **no se incluyen en el repositorio** porque son archivos grandes (hasta 470 MB). Debes descargarlos antes de ejecutar el pipeline.

### Opción A — Script interactivo (recomendado)

```bash
python utils/descargar_datasets.py
```

El script muestra los datasets disponibles y te guía paso a paso. Los que se pueden descargar automáticamente (OpenML) lo hacen sin salir del script. Los de Kaggle muestran instrucciones para descarga manual.

### Opción B — Descarga manual

Descarga cada CSV de los enlaces de abajo, colócalo en `utils/datasets/` con el nombre exacto indicado y actívalo en `config.yaml` cambiando `enabled: false` a `enabled: true`.

---

## Datasets disponibles

| Dataset | Filas | Fraudes | Fuente | Archivo esperado |
|---------|------:|--------:|--------|-----------------|
| CreditCard Fraud | 284,807 | 0.17 % | [OpenML / Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | `creditcard.csv` |
| BankSim | 594,643 | 1.20 % | [Kaggle](https://www.kaggle.com/datasets/ealaxi/banksim1) | `banksim.csv` |
| PaySim | 6,362,620 | 0.13 % | [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1) | `paysim.csv` |
| Banking Fraud Risk Analytics | 10,000 | 12.51 % | [Kaggle](https://www.kaggle.com/datasets/deepeshkansotia/banking-fraud-detection-and-risk-analytics-dataset) | `banking_transactions.csv` |
| Indian Banking Transactions 2019–2024 | 550,000 | 0.89 % | [Kaggle](https://www.kaggle.com/datasets/belbino/indian-banking-transactions-20192024) | `indian_banking_transactions.csv` |
| IEEE-CIS Fraud Detection | 590,540 | 3.50 % | [Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection/data) | `ieee_cis.csv` ¹ |

> ¹ IEEE-CIS requiere combinar dos archivos. Ver instrucciones más abajo.

### Instrucciones de descarga por dataset

#### CreditCard Fraud
1. Entra a: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Descarga `creditcard.csv`
3. Colócalo en `utils/datasets/creditcard.csv`

> También se puede descargar automáticamente desde OpenML con el script interactivo.

#### BankSim
1. Entra a: https://www.kaggle.com/datasets/ealaxi/banksim1
2. Descarga `bs140513_032310.csv`
3. Renómbralo a `banksim.csv`
4. Colócalo en `utils/datasets/banksim.csv`

#### PaySim
1. Entra a: https://www.kaggle.com/datasets/ealaxi/paysim1
2. Descarga `PS_20174392719_1491204439457_log.csv`
3. Renómbralo a `paysim.csv`
4. Colócalo en `utils/datasets/paysim.csv`

> PaySim tiene ~6.3 M filas. En `config.yaml` está configurado con `sample_frac: 0.30` (usa solo el 30 %).

#### Banking Fraud Risk Analytics
1. Entra a: https://www.kaggle.com/datasets/deepeshkansotia/banking-fraud-detection-and-risk-analytics-dataset
2. Descarga el CSV principal
3. Guárdalo como `utils/datasets/banking_transactions.csv`

#### Indian Banking Transactions 2019–2024
1. Entra a: https://www.kaggle.com/datasets/belbino/indian-banking-transactions-20192024
2. Descarga el CSV
3. Guárdalo como `utils/datasets/indian_banking_transactions.csv`

> Este dataset incluye columnas de texto categóricas (`account_type`, `transaction_type`, `merchant_category`, etc.). El pipeline las codifica automáticamente con **OneHotEncoder** antes de entrenar.

#### IEEE-CIS Fraud Detection
1. Acepta las reglas de la competencia en: https://www.kaggle.com/competitions/ieee-fraud-detection/data
2. Descarga `train_transaction.csv` y `train_identity.csv`
3. Coloca ambos en `utils/datasets/`
4. Ejecuta el script de fusión:
   ```bash
   python utils/ieee_merge.py
   ```
5. Esto genera `utils/datasets/ieee_cis.csv` listo para usar
6. Actívalo en `config.yaml` con `enabled: true`

---

## Ejecutar

```bash
python main.py
```

Al arrancar, el pipeline lee `config.yaml`, procesa cada dataset activo y genera un frame PNG en `results/`.
Al arrancar, el pipeline lee `config.yaml`, procesa cada dataset activo y genera un frame PNG en `results/`.

### Modos de ejecución

| Modo | Comportamiento |
|------|---------------|
| `offline` | Solo usa archivos locales. Si falta un CSV, pregunta si quieres descargarlo. **Modo por defecto.** |
| `online` | Comprime el proyecto, lo sube a Google Drive y genera un notebook para ejecutar en Google Colab con GPU. |
| `preguntar` | Pregunta en terminal qué modo usar al iniciar. |
| Modo | Comportamiento |
|------|---------------|
| `offline` | Solo usa archivos locales. Si falta un CSV, pregunta si quieres descargarlo. **Modo por defecto.** |
| `online` | Comprime el proyecto, lo sube a Google Drive y genera un notebook para ejecutar en Google Colab con GPU. |
| `preguntar` | Pregunta en terminal qué modo usar al iniciar. |

Se configura en `config.yaml`:

```yaml
general:
  modo: "offline"   # "online" | "offline" | "preguntar"
```

---

## Flujo del pipeline

Cada dataset activo pasa por 6 etapas en orden:

```
config.yaml
    │
    ▼
[Etapa 1] loader.py
    Carga el CSV desde disco o descarga desde OpenML si no existe.
    │
    ▼
[Etapa 2] cleaner.py
    Elimina NaN y duplicados exactos.
    Normaliza la columna objetivo a 0/1.
    Detecta columnas categóricas para encoding posterior.
    │
    ▼
[Etapa 3] preprocessor.py
    Separa features (X) y etiqueta (y).
    Aplica StandardScaler a columnas numéricas.
    Aplica OneHotEncoder a columnas categóricas (≤ max_categorical_levels únicos).
    Excluye columnas de alta cardinalidad (IDs, hashes, etc.).
    Hace split estratificado train/test (80/20 por defecto).
    Prepara StratifiedKFold para validación cruzada.
    │
    ▼
[Etapa 4] trainer.py
    Por cada modelo activo:
      - Si existe .pkl en caché → lo carga directamente (sin reentrenar).
      - Si no → construye sklearn Pipeline, ejecuta GridSearchCV
                y guarda el mejor modelo en utils/modelos/.
    │
    ▼
[Etapa 5] evaluator.py
    Predice sobre X_test con cada modelo.
    Calcula accuracy, precision, recall, F1 y matrices de confusión.
    Imprime tabla comparativa en consola.
    │
    ▼
[Etapa 6] reporter.py
    Genera frame PNG en results/ con:
      - Barras comparativas de métricas por dataset
      - Matrices de confusión por modelo
      - Tabla resumen completa
      - Conclusión generada por IA (Anthropic API o rule-based)
## Flujo del pipeline

Cada dataset activo pasa por 6 etapas en orden:

```
config.yaml
    │
    ▼
[Etapa 1] loader.py
    Carga el CSV desde disco o descarga desde OpenML si no existe.
    │
    ▼
[Etapa 2] cleaner.py
    Elimina NaN y duplicados exactos.
    Normaliza la columna objetivo a 0/1.
    Detecta columnas categóricas para encoding posterior.
    │
    ▼
[Etapa 3] preprocessor.py
    Separa features (X) y etiqueta (y).
    Aplica StandardScaler a columnas numéricas.
    Aplica OneHotEncoder a columnas categóricas (≤ max_categorical_levels únicos).
    Excluye columnas de alta cardinalidad (IDs, hashes, etc.).
    Hace split estratificado train/test (80/20 por defecto).
    Prepara StratifiedKFold para validación cruzada.
    │
    ▼
[Etapa 4] trainer.py
    Por cada modelo activo:
      - Si existe .pkl en caché → lo carga directamente (sin reentrenar).
      - Si no → construye sklearn Pipeline, ejecuta GridSearchCV
                y guarda el mejor modelo en utils/modelos/.
    │
    ▼
[Etapa 5] evaluator.py
    Predice sobre X_test con cada modelo.
    Calcula accuracy, precision, recall, F1 y matrices de confusión.
    Imprime tabla comparativa en consola.
    │
    ▼
[Etapa 6] reporter.py
    Genera frame PNG en results/ con:
      - Barras comparativas de métricas por dataset
      - Matrices de confusión por modelo
      - Tabla resumen completa
      - Conclusión generada por IA (Anthropic API o rule-based)
```

---

## Configuración (`config.yaml`)

### Parámetros generales
### Parámetros generales

```yaml
general:
  random_state: 42
  test_size: 0.20              # 20% para test, 80% para entrenamiento
  cv_folds: 5                  # folds para validación cruzada
  refit_metric: "recall"       # métrica que usa GridSearchCV para elegir el mejor modelo
  n_jobs: -1                   # -1 = usar todos los núcleos del CPU
  test_size: 0.20              # 20% para test, 80% para entrenamiento
  cv_folds: 5                  # folds para validación cruzada
  refit_metric: "recall"       # métrica que usa GridSearchCV para elegir el mejor modelo
  n_jobs: -1                   # -1 = usar todos los núcleos del CPU
  modo: "offline"
  guardar_modelos: true        # guarda .pkl en utils/modelos/ para reusar
  max_categorical_levels: 100  # columnas con más valores únicos se descartan
  guardar_modelos: true        # guarda .pkl en utils/modelos/ para reusar
  max_categorical_levels: 100  # columnas con más valores únicos se descartan
```

### Activar / desactivar un dataset
### Activar / desactivar un dataset

```yaml
datasets:
  - nombre: "PaySim (Kaggle)"
    enabled: true        # ← true = se procesa, false = se omite
    fuente: "csv_local"
    archivo_local: "paysim.csv"
    target_col: "isFraud"
    fraud_value: 1
    sample_frac: 0.30    # usar solo el 30% (útil para datasets grandes)
  - nombre: "PaySim (Kaggle)"
    enabled: true        # ← true = se procesa, false = se omite
    fuente: "csv_local"
    archivo_local: "paysim.csv"
    target_col: "isFraud"
    fraud_value: 1
    sample_frac: 0.30    # usar solo el 30% (útil para datasets grandes)
```

### Activar / desactivar un modelo
### Activar / desactivar un modelo

```yaml
modelos:
  - nombre: "Random Forest"
    enabled: true        # ← true = se entrena, false = se omite

  - nombre: "SVM"
    enabled: false       # deshabilitado — muy lento en datasets grandes

  - nombre: "XGBoost"
    enabled: false       # deshabilitado — requiere GPU configurada
```
    enabled: true        # ← true = se entrena, false = se omite

  - nombre: "SVM"
    enabled: false       # deshabilitado — muy lento en datasets grandes

  - nombre: "XGBoost"
    enabled: false       # deshabilitado — requiere GPU configurada
```

### Conclusión IA

```yaml
conclusion_ia:
  enabled: true
  motor: "anthropic_api"          # "anthropic_api" | "rule_based"
  motor: "anthropic_api"          # "anthropic_api" | "rule_based"
  modelo_api: "claude-sonnet-4-20250514"
  max_tokens: 600
  idioma: "español"
```

Para usar `anthropic_api` configura tu API key como variable de entorno:
Para usar `anthropic_api` configura tu API key como variable de entorno:

```bash
# Windows PowerShell
$env:ANTHROPIC_API_KEY = "sk-ant-..."

# Linux / Mac
export ANTHROPIC_API_KEY="sk-ant-..."
```

Si no tienes API key, cambia a `motor: "rule_based"` para generar la conclusión automáticamente sin llamadas externas.

# Linux / Mac
export ANTHROPIC_API_KEY="sk-ant-..."
```

Si no tienes API key, cambia a `motor: "rule_based"` para generar la conclusión automáticamente sin llamadas externas.

---

## Modelos disponibles

| Modelo | Habilitado | Escalado | GPU | Notas |
|--------|:----------:|:--------:|:---:|-------|
| Logistic Regression | ✅ | StandardScaler | ❌ | Rápido, buena línea base |
| Decision Tree | ✅ | No | ❌ | Interpretable |
| Random Forest | ✅ | No | ❌ | Mejor balance precision/recall |
| SVM | ❌ | StandardScaler | ❌ | Deshabilitado — O(n²) en filas |
| XGBoost | ❌ | No | ✅ | Deshabilitado — requiere GPU |
| Modelo | Habilitado | Escalado | GPU | Notas |
|--------|:----------:|:--------:|:---:|-------|
| Logistic Regression | ✅ | StandardScaler | ❌ | Rápido, buena línea base |
| Decision Tree | ✅ | No | ❌ | Interpretable |
| Random Forest | ✅ | No | ❌ | Mejor balance precision/recall |
| SVM | ❌ | StandardScaler | ❌ | Deshabilitado — O(n²) en filas |
| XGBoost | ❌ | No | ✅ | Deshabilitado — requiere GPU |

---

## Caché de modelos

Los modelos entrenados se guardan en `utils/modelos/` como archivos `.pkl` (ignorados por git). En ejecuciones posteriores se cargan directamente sin reentrenar, reduciendo el tiempo de ~20 minutos a segundos.
Los modelos entrenados se guardan en `utils/modelos/` como archivos `.pkl` (ignorados por git). En ejecuciones posteriores se cargan directamente sin reentrenar, reduciendo el tiempo de ~20 minutos a segundos.

Para **forzar reentrenamiento**, elimina el `.pkl` correspondiente:
Para **forzar reentrenamiento**, elimina el `.pkl` correspondiente:

```bash
# Windows
del utils\modelos\random_forest__creditcard_fraud_openml.pkl

# Linux / Mac
rm utils/modelos/random_forest__creditcard_fraud_openml.pkl
rm utils/modelos/random_forest__creditcard_fraud_openml.pkl

# Reentrenar todo desde cero
del utils\modelos\*.pkl      # Windows
rm utils/modelos/*.pkl       # Linux / Mac
# Reentrenar todo desde cero
del utils\modelos\*.pkl      # Windows
rm utils/modelos/*.pkl       # Linux / Mac
```

---

## Métricas evaluadas

En fraude financiero la exactitud (accuracy) sola es engañosa porque los datasets están muy desbalanceados. Las métricas más importantes son:
En fraude financiero la exactitud (accuracy) sola es engañosa porque los datasets están muy desbalanceados. Las métricas más importantes son:

| Métrica | Importancia | Por qué |
|---------|:-----------:|---------|
| **Recall** | ⭐⭐⭐ Crítica | % de fraudes reales detectados (falso negativo = pérdida económica) |
| **Precision** | ⭐⭐ Importante | % de alertas que son fraude real (falso positivo = molestia al cliente) |
| **F1-Score** | ⭐⭐ | Balance entre precision y recall |
| Accuracy | ⭐ Referencia | Puede ser alta aunque no se detecte ningún fraude |
| Métrica | Importancia | Por qué |
|---------|:-----------:|---------|
| **Recall** | ⭐⭐⭐ Crítica | % de fraudes reales detectados (falso negativo = pérdida económica) |
| **Precision** | ⭐⭐ Importante | % de alertas que son fraude real (falso positivo = molestia al cliente) |
| **F1-Score** | ⭐⭐ | Balance entre precision y recall |
| Accuracy | ⭐ Referencia | Puede ser alta aunque no se detecte ningún fraude |

> `refit_metric: "recall"` hace que GridSearchCV optimice hiperparámetros por recall. Todos los modelos usan `class_weight: "balanced"` para compensar el desbalance de clases.
> `refit_metric: "recall"` hace que GridSearchCV optimice hiperparámetros por recall. Todos los modelos usan `class_weight: "balanced"` para compensar el desbalance de clases.

---

## Agregar un nuevo dataset

**CSV local:**

1. Coloca el archivo en `utils/datasets/tu_archivo.csv`
2. Agrega en `config.yaml`:

```yaml
datasets:
  - nombre: "Mi Dataset"
    enabled: true
    fuente: "csv_local"
    archivo_local: "tu_archivo.csv"
    target_col: "columna_fraude"   # nombre exacto de la columna objetivo
    fraud_value: 1                  # valor que representa fraude en esa columna
    sample_frac: 1.0
```

**Desde OpenML:**

```yaml
datasets:
  - nombre: "Mi Dataset OpenML"
    enabled: true
    fuente: "openml"
    openml_name: "nombre_en_openml"
    openml_version: 1
    archivo_local: "mi_cache.csv"   # nombre con que se guarda localmente
    target_col: "Class"
    fraud_value: 1
    sample_frac: 1.0
```
## Agregar un nuevo dataset

**CSV local:**

1. Coloca el archivo en `utils/datasets/tu_archivo.csv`
2. Agrega en `config.yaml`:

```yaml
datasets:
  - nombre: "Mi Dataset"
    enabled: true
    fuente: "csv_local"
    archivo_local: "tu_archivo.csv"
    target_col: "columna_fraude"   # nombre exacto de la columna objetivo
    fraud_value: 1                  # valor que representa fraude en esa columna
    sample_frac: 1.0
```

**Desde OpenML:**

```yaml
datasets:
  - nombre: "Mi Dataset OpenML"
    enabled: true
    fuente: "openml"
    openml_name: "nombre_en_openml"
    openml_version: 1
    archivo_local: "mi_cache.csv"   # nombre con que se guarda localmente
    target_col: "Class"
    fraud_value: 1
    sample_frac: 1.0
```

---

## Agregar un nuevo modelo

1. Agrega en `config.yaml`:
1. Agrega en `config.yaml`:

```yaml
modelos:
  - nombre: "Gradient Boosting"
    enabled: true
    clase: "GradientBoostingClassifier"
    usa_scaler: false
    gpu_support: false
    params_fijos:
      n_estimators: 100
    grid:
      model__max_depth: [3, 5]
      model__learning_rate: [0.05, 0.1]
modelos:
  - nombre: "Gradient Boosting"
    enabled: true
    clase: "GradientBoostingClassifier"
    usa_scaler: false
    gpu_support: false
    params_fijos:
      n_estimators: 100
    grid:
      model__max_depth: [3, 5]
      model__learning_rate: [0.05, 0.1]
```

2. Registra la clase en `_CLASE_MAP` dentro de `pipeline/trainer.py`:
2. Registra la clase en `_CLASE_MAP` dentro de `pipeline/trainer.py`:

```python
_CLASE_MAP = {
    ...
    "GradientBoostingClassifier": ("sklearn.ensemble", "GradientBoostingClassifier"),
}
```

---

## GPU (opcional)

El proyecto detecta GPU automáticamente en este orden: PyTorch → CuPy → pynvml. Solo los modelos con `gpu_support: true` en `config.yaml` se configuran para CUDA (actualmente XGBoost). Los modelos de scikit-learn siempre usan CPU.
## GPU (opcional)

El proyecto detecta GPU automáticamente en este orden: PyTorch → CuPy → pynvml. Solo los modelos con `gpu_support: true` en `config.yaml` se configuran para CUDA (actualmente XGBoost). Los modelos de scikit-learn siempre usan CPU.

Para deshabilitar la GPU aunque esté disponible:
Para deshabilitar la GPU aunque esté disponible:

```yaml
gpu:
gpu:
  enabled: true
  forzar_cpu: true
```

---

## Modo online (Google Colab)

Para entrenar con GPU en la nube:

1. Cambia `modo: "online"` en `config.yaml`
2. Ejecuta `python main.py`
3. El script comprime el proyecto, lo sube a Google Drive y genera un notebook `.ipynb`
4. Abre el notebook en Colab, activa GPU (Entorno → Cambiar tipo → T4) y ejecuta todo

> Ten los datasets descargados en `utils/datasets/` **antes** de cambiar a modo online.
  forzar_cpu: true
```

---

## Modo online (Google Colab)

Para entrenar con GPU en la nube:

1. Cambia `modo: "online"` en `config.yaml`
2. Ejecuta `python main.py`
3. El script comprime el proyecto, lo sube a Google Drive y genera un notebook `.ipynb`
4. Abre el notebook en Colab, activa GPU (Entorno → Cambiar tipo → T4) y ejecuta todo

> Ten los datasets descargados en `utils/datasets/` **antes** de cambiar a modo online.

---

## Preguntas frecuentes

**¿Por qué tarda tanto la primera vez?**  
El primer run ejecuta GridSearchCV completo (~20 min con 3 modelos y CreditCard). Las siguientes ejecuciones cargan los `.pkl` en segundos.
El primer run ejecuta GridSearchCV completo (~20 min con 3 modelos y CreditCard). Las siguientes ejecuciones cargan los `.pkl` en segundos.

**¿Qué pasa si no tengo internet al ejecutar?**  
En modo `offline`, si un CSV no está en disco el pipeline pregunta si quieres descargarlo. Si dices `n`, omite ese dataset y continúa con los demás.
**¿Qué pasa si no tengo internet al ejecutar?**  
En modo `offline`, si un CSV no está en disco el pipeline pregunta si quieres descargarlo. Si dices `n`, omite ese dataset y continúa con los demás.

**¿Por qué SVM y XGBoost están desactivados?**  
SVM escala cuadráticamente — en CreditCard (284k filas) puede tardar horas. XGBoost requiere GPU para tener ventaja real sobre Random Forest. Puedes activarlos en `config.yaml`.

**¿Qué pasa con las columnas de texto?**  
El `preprocessor.py` aplica `OneHotEncoder` automáticamente a columnas categóricas con hasta `max_categorical_levels` valores únicos. Las columnas con más valores (IDs, hashes) se descartan. El umbral se controla con `max_categorical_levels` en `config.yaml`.
**¿Por qué SVM y XGBoost están desactivados?**  
SVM escala cuadráticamente — en CreditCard (284k filas) puede tardar horas. XGBoost requiere GPU para tener ventaja real sobre Random Forest. Puedes activarlos en `config.yaml`.

**¿Qué pasa con las columnas de texto?**  
El `preprocessor.py` aplica `OneHotEncoder` automáticamente a columnas categóricas con hasta `max_categorical_levels` valores únicos. Las columnas con más valores (IDs, hashes) se descartan. El umbral se controla con `max_categorical_levels` en `config.yaml`.

**¿Cómo sé el nombre exacto de la columna objetivo en mi dataset?**
```bash
python -c "import pandas as pd; print(pd.read_csv('utils/datasets/mi_archivo.csv', nrows=1).columns.tolist())"
```

---

## Licencia

Proyecto académico — Tecnológico Superior de Jalisco · 2026