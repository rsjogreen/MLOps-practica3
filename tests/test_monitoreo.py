"""Tests de drift. No necesitan servicios."""
import numpy as np
import pandas as pd
import pytest

from domain.modelo import NUMERICAS, OBJETIVO
from domain.monitoreo import (
    SUBIDA_TASA_MAXIMA,
    asegurar_lote,
    aplicar_drift,
    drift_de_target,
    fraccion_desde_evidently,
    validar_lote,
)


@pytest.fixture
def datos():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({columna: rng.normal(0, 1, 500) for columna in NUMERICAS})
    df[OBJETIVO] = rng.integers(0, 2, 500)
    return df


def test_drift_cero_no_cambia_los_datos(datos):
    assert aplicar_drift(datos, 0.0).equals(datos)


def test_drift_sube_la_tasa_del_objetivo_en_proporcion_al_nivel(datos):
    tasa = datos[OBJETIVO].mean()
    lote = aplicar_drift(datos, 0.6)
    assert lote[OBJETIVO].mean() == pytest.approx(tasa + 0.6 * SUBIDA_TASA_MAXIMA, abs=0.01)


def test_drift_cambia_la_fraccion_indicada_de_variables(datos):
    lote = aplicar_drift(datos, 0.6)
    cambiadas = [c for c in NUMERICAS if not np.allclose(lote[c], datos[c])]
    assert len(cambiadas) == round(0.6 * len(NUMERICAS))


def test_fraccion_desde_evidently():
    resultado = {"metrics": [
        {"metric_name": "ValueDrift(column=Edad)", "value": 0.5},
        {"metric_name": "DriftedColumnsCount(drift_share=0.5)", "value": {"count": 7, "share": 0.5}},
    ]}
    assert fraccion_desde_evidently(resultado) == 0.5


def test_validar_lote_exige_objetivo(datos):
    with pytest.raises(ValueError, match=OBJETIVO):
        validar_lote(datos.drop(columns=[OBJETIVO]))


def test_drift_de_target_detecta_cambio_de_tasa():
    referencia = pd.Series([0] * 800 + [1] * 200)
    lote = pd.Series([0] * 400 + [1] * 600)
    assert drift_de_target(referencia, lote, alfa=0.05)["drift"] is True


def test_drift_de_target_no_alerta_con_la_misma_tasa():
    referencia = pd.Series([0] * 800 + [1] * 200)
    lote = pd.Series([0] * 80 + [1] * 20)
    assert drift_de_target(referencia, lote, alfa=0.05)["drift"] is False


def test_asegurar_lote_crea_lote_con_drift_si_no_existe(datos, tmp_path, monkeypatch):
    import domain.monitoreo as monitoreo

    monkeypatch.setattr(monitoreo, "cargar_datos", lambda ruta: datos)
    ruta_lote = tmp_path / "lotes" / "lote_actual.csv"
    cfg = {
        "monitoreo": {"lote_actual": str(ruta_lote)},
        "datos": {"ruta_test": "test.csv"},
        "inferencia": {"tamano_lote": 50, "nivel_drift": 0.6},
        "division": {"random_state": 42},
    }

    assert asegurar_lote(cfg) == str(ruta_lote)
    lote = pd.read_csv(ruta_lote)
    assert len(lote) == 50
    assert OBJETIVO in lote.columns
    assert asegurar_lote(cfg) == str(ruta_lote)
