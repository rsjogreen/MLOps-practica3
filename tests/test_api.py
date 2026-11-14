"""Tests de la API con un modelo falso (sin MLflow)."""
import sys
from types import SimpleNamespace

import mlflow.sklearn
import mlflow.tracking
import numpy as np
import pytest
from fastapi.testclient import TestClient

from domain.modelo import VARIABLES

EMPLEADO = {
    "tip_cont": "I",
    "Tenure_Meses": 24,
    "Edad": 35,
    "sex_empl": 0,
    "est_civi": "C",
    "cod_area": 332,
    "cod_carg": "130",
    "Meses_En_Cargo": 12,
    "ID_Jefe_Directo": "EMP_01369",
    "Meses_Con_Jefe": 10,
    "Salario_Mes": 3500000.0,
    "Variacion_Absoluta_Salario": 150000.0,
    "Porcentaje_Variacion_Salario": 0.04,
    "Ausentismo_Mes": 0,
    "Ausentismo_Trimestre": 2,
    "Dias_Vacaciones_Mes": 0,
}


class ModeloFalso:
    feature_names_in_ = np.array(VARIABLES)

    def predict_proba(self, X):
        return np.tile([0.3, 0.7], (len(X), 1))


@pytest.fixture
def client(monkeypatch):
    # Se parchean los objetos de módulo (no rutas en texto) y después se importa la API:
    # al importarse, main.py carga el modelo desde MLflow.
    monkeypatch.setattr(mlflow.sklearn, "load_model", lambda uri: ModeloFalso())
    monkeypatch.setattr(mlflow.tracking, "MlflowClient", lambda: SimpleNamespace(
        get_model_version_by_alias=lambda nombre, alias: SimpleNamespace(version="7")))
    sys.modules.pop("infraestructure.api.main", None)

    import infraestructure.api.main as main

    return TestClient(main.app)


def test_health_y_predict(client):
    assert client.get("/health").json()["version"] == "7"
    assert client.post("/predict", json=EMPLEADO).json() == {
        "probabilidad_fuga": 0.7,
        "prediccion_fuga": 1,
        "version_modelo": "7",
    }
