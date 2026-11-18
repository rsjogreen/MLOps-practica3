# MLOps-practica3: predicción de fuga de empleados

Repositorio del proyecto MLOps para predecir la fuga de empleados (`Target_Fuga`). Incluye preparación de datos, entrenamiento y registro de modelos, inferencia por lotes y API, orquestación con Prefect y monitoreo de drift.

**Repositorio en GitHub:** [rsjogreen/MLOps-practica3](https://github.com/rsjogreen/MLOps-practica3)

**Clonar:** `git clone https://github.com/rsjogreen/MLOps-practica3.git`

Los comandos de esta guía se ejecutan desde la raíz del repositorio. La carpeta `actividad-3/` contiene material de guía; el proyecto activo y su configuración están en la raíz.

## Componentes

- **Datos y configuración:** el dataset original y sus particiones están en `data/`; `configs/config.yaml` define sus rutas, variables, candidatos, métricas y parámetros de inferencia y monitoreo.
- **Modelo:** `domain/modelo.py` implementa el preprocesamiento, los clasificadores y las métricas. `scripts/load_data.py` valida y carga las columnas esperadas.
- **Entrenamiento:** `infraestructure/flows/entrenamiento_flow.py` compara candidatos y conjuntos de variables, registra resultados en MLflow y promueve el modelo con mejor AUC al alias `champion` si cumple el umbral configurado.
- **Inferencia:** `infraestructure/flows/inferencia_flow.py` carga el `champion`, genera predicciones y registra la ejecución en MLflow.
- **Monitoreo:** `infraestructure/flows/drift_features_flow.py` genera un reporte de drift de variables con Evidently; `infraestructure/flows/drift_evento_flow.py` compara la distribución de `Target_Fuga`.
- **API:** `infraestructure/api/main.py` expone predicciones individuales con FastAPI.
- **Servicios:** `services/` contiene la configuración de MLflow y Prefect para Docker Compose y la guía para ejecutarlos sin Docker.
- **Pruebas:** `tests/` cubre datos, modelo, monitoreo y API.

```text
.
├── configs/config.yaml
├── data/
├── domain/
├── infraestructure/
│   ├── api/
│   └── flows/
├── scripts/
├── services/
└── tests/
```

## Requisitos

- Python `>=3.12,<3.15` (CI utiliza Python 3.12).
- Poetry 2.x.
- Docker Desktop para ejecutar MLflow y Prefect con Docker Compose, o Python 3.12 para instalarlos localmente.

Instala las dependencias desde la raíz:

```powershell
poetry install --no-interaction
```

## Preparar los datos

El dataset fuente es `data/pipelyne_DatasetFuga.csv`. El script divide los datos cronológicamente y genera `data/train_fuga.csv` y `data/test_fuga.csv`, usando los meses definidos en `scripts/split_data.py`:

```powershell
poetry run python scripts/split_data.py
```

Las rutas de datos se definen en `configs/config.yaml`. El objetivo de clasificación es `Target_Fuga`; la columna `Mes` se utiliza para la separación temporal y no como variable predictora.

## Iniciar MLflow y Prefect

Con Docker Desktop en ejecución, desde la raíz:

```powershell
docker compose -f services/docker-compose.yml up -d
```

- MLflow: [http://localhost:5000](http://localhost:5000)
- Prefect: [http://localhost:4200](http://localhost:4200)

Para detener los servicios:

```powershell
docker compose -f services/docker-compose.yml down
```

Las instrucciones para ejecutar los servicios sin Docker están en [`services/README.md`](services/README.md). Define estas variables en la terminal desde la que ejecutarás los flows o la API:

```powershell
$env:PREFECT_API_URL = "http://localhost:4200/api"
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
$env:PYTHONUTF8 = "1"
```

## Flujo de trabajo

Ejecuta los flows desde la raíz y con los servicios disponibles.

### 1. Entrenar y registrar modelos

```powershell
poetry run python infraestructure/flows/entrenamiento_flow.py
```

El flow compara los modelos y variables definidos en `configs/config.yaml`, calcula AUC, accuracy, precision, recall y F1, y registra cada ejecución en MLflow. El mejor modelo se registra como `fuga-modelo-v2` con el alias `champion` si supera `registro.umbral_auc` y, cuando ya existe un champion, también supera su AUC.

### 2. Ejecutar inferencia por lote

Requiere que exista un modelo `champion`:

```powershell
poetry run python infraestructure/flows/inferencia_flow.py
```

El flow toma un lote de `data/test_fuga.csv`, aplica el nivel de drift configurado y guarda la salida en `data/prediccion_<fecha>.csv`. El lote utilizado para el monitoreo se guarda en la ruta `monitoreo.lote_actual`. La inferencia registra sus métricas y la versión del modelo usada en MLflow.

### 3. Revisar drift

Ejecuta los flows después de generar o seleccionar un lote:

```powershell
poetry run python -m infraestructure.flows.drift_features_flow
poetry run python -m infraestructure.flows.drift_evento_flow
```

El primero compara las variables del lote con los datos de entrenamiento, guarda un reporte HTML en `reports/` y alerta según `monitoreo.umbral_drift`. El segundo compara la distribución de `Target_Fuga` y alerta según `monitoreo.alfa_evento`; necesita que el lote tenga etiquetas.

### 4. Registrar despliegues en Prefect

```powershell
poetry run python infraestructure/flows/desplegar.py
```

Registra el despliegue de entrenamiento manual y el de inferencia diaria (06:00 UTC). Mantén abierta la terminal que ejecuta este comando para que Prefect pueda lanzar las ejecuciones.

### 5. Iniciar la API

La API carga el alias `champion` al arrancar. Si se promueve una nueva versión, reinicia la API para que la cargue:

```powershell
poetry run uvicorn infraestructure.api.main:app --port 8000
```

- Documentación interactiva: [http://localhost:8000/docs](http://localhost:8000/docs)
- Estado y versión del modelo: [http://localhost:8000/health](http://localhost:8000/health)
- Predicción: `POST http://localhost:8000/predict`

El endpoint `/predict` recibe los datos de un empleado según el esquema documentado en `/docs` y devuelve `probabilidad_fuga`, `prediccion_fuga` y `version_modelo`.

## Configuración

Edita `configs/config.yaml` para cambiar:

- Rutas del dataset, entrenamiento y prueba.
- Conjuntos de variables y candidatos de clasificación (regresión logística, gradient boosting, random forest, extra trees, hist gradient boosting y LightGBM).
- Grid search, métricas, umbral de clasificación y umbral mínimo de AUC para promover un champion.
- URI y nombre del experimento MLflow, y nombre registrado del modelo.
- Tamaño y nivel de drift del lote de inferencia; ruta del lote y carpeta de reportes.

El valor de `MLFLOW_TRACKING_URI`, cuando está definido, tiene prioridad sobre `mlflow.tracking_uri` en el YAML.

## Pruebas y CI

Ejecuta las pruebas desde la raíz:

```powershell
poetry run pytest
```

Las pruebas no necesitan los servidores de MLflow ni Prefect; la API utiliza un modelo de prueba. El workflow de GitHub Actions en `.github/workflows/ci.yml` instala las dependencias y ejecuta `pytest` desde la raíz al abrir o actualizar un pull request y al hacer push a `main`.

## Contribuidores

- Mario Alejandro Mendoza Suarez
- Jose Esteban Arcos Muñoz
- Robert Douglas Sjogreen Escorcia
