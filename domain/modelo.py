"""Lógica de negocio del modelo de fuga de empleados: variables, pipeline y métricas.

No usa Prefect ni MLflow: se puede probar sin servicios levantados.
"""
from joblib import parallel_backend
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ESTIMADORES = {
    "logistic_regression": LogisticRegression,
    "gradient_boosting": GradientBoostingClassifier,
    "random_forest": RandomForestClassifier,
    "extra_trees": ExtraTreesClassifier,
    "hist_gradient_boosting": HistGradientBoostingClassifier,
    "lightgbm": LGBMClassifier,
}

# Variables del modelo adaptadas al dataset de Fuga de Empleados. 
# La columna 'Mes' se excluye aquí porque se usa para separar los datos, no para entrenar.
NUMERICAS = [
    "Tenure_Meses",
    "Edad",
    "Meses_En_Cargo",
    "Meses_Con_Jefe",
    "Salario_Mes",
    "Variacion_Absoluta_Salario",
    "Porcentaje_Variacion_Salario",
    "Ausentismo_Mes",
    "Ausentismo_Trimestre",
    "Dias_Vacaciones_Mes"
]

CATEGORICAS = [
    "tip_cont", 
    "sex_empl", 
    "est_civi", 
    "cod_area", 
    "cod_carg", 
    "ID_Jefe_Directo"
]

VARIABLES = NUMERICAS + CATEGORICAS
OBJETIVO = "Target_Fuga"


def resolver_features(columnas):
    """Convierte `columnas` de config.yaml ('ALL' o una lista de nombres) en (numericas, categoricas)."""
    if columnas == "ALL":
        elegidas = VARIABLES
    else:
        desconocidas = [v for v in columnas if v not in VARIABLES]
        if desconocidas:
            raise ValueError(f"variables desconocidas en 'columnas': {desconocidas}")
        elegidas = columnas

    if not elegidas:
        raise ValueError("la selección de features no tiene ninguna variable")
    numericas = [v for v in NUMERICAS if v in elegidas]
    categoricas = [v for v in CATEGORICAS if v in elegidas]
    return numericas, categoricas


def construir_modelo(tipo, parametros, numericas, categoricas):
    """Preprocesado (escalar y one-hot) más el estimador elegido en la configuración."""
    if tipo not in ESTIMADORES:
        raise ValueError(f"modelo '{tipo}' no soportado")
    estimador = ESTIMADORES[tipo](**parametros)

    transformadores = []
    if numericas:
        transformadores.append(("num", StandardScaler(), numericas))
    if categoricas:
        # HistGradientBoostingClassifier requiere matriz densa; el resto admite dispersa.
        sparse_output = tipo != "hist_gradient_boosting"
        # handle_unknown="ignore" es clave porque en Test pueden aparecer jefes o cargos nuevos
        transformadores.append(
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=sparse_output), categoricas)
        )

    preprocesado = ColumnTransformer(transformadores)
    return Pipeline([("preprocesado", preprocesado), ("modelo", estimador)])


def ajustar_modelo(
    modelo,
    X_train,
    y_train,
    param_grid=None,
    cv=3,
    scoring="roc_auc",
    random_state=42,
    n_jobs=-1,
):
    """Entrena el pipeline directamente o usando GridSearchCV si se indica `param_grid`.

    Devuelve `(modelo_entrenado, mejores_parametros, mejor_score_cv)`.
    """
    if not param_grid:
        modelo.fit(X_train, y_train)
        return modelo, {}, None

    grid = {
        k if k.startswith("modelo__") else f"modelo__{k}": v
        for k, v in param_grid.items()
    }
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    search = GridSearchCV(
        estimator=modelo,
        param_grid=grid,
        cv=splitter,
        scoring=scoring,
        n_jobs=n_jobs,
    )
    # En Windows dentro de tareas de Prefect, el backend loky por defecto puede fallar con WinError 6.
    # threading evita subprocesos y mantiene paralelismo porque sklearn/LightGBM liberan el GIL.
    with parallel_backend("threading", n_jobs=n_jobs):
        search.fit(X_train, y_train)
    mejores_parametros = {
        k.removeprefix("modelo__"): v for k, v in search.best_params_.items()
    }
    return search.best_estimator_, mejores_parametros, float(search.best_score_)


def calcular_metricas(y_true, proba, umbral):
    """Métricas de clasificación binaria a partir de la probabilidad de fuga."""
    pred = (proba >= umbral).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, proba)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }