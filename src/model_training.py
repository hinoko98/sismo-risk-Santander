"""Misión 2 (Nivel Medio) y reto supervisado: feature engineering + RandomForest.

Se predice la categoría de riesgo (Bajo/Medio/Alto) que tendrá un municipio el AÑO SIGUIENTE a partir
de lo ocurrido hasta el año t. La evaluación es temporal (últimos años como test) para no mirar el futuro.
"""
import json
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import (
    COL_CATEGORICA, ETIQUETAS_RIESGO, OBJETIVO, RUTA_METRICAS, RUTA_MODELO, SEMILLA, TIPOS_SUELO,
    UMBRAL_ALTO, UMBRAL_MEDIO, VARIABLES,
)

log = logging.getLogger(__name__)


def construir_pipeline() -> Pipeline:
    """One-Hot de suelo_predominante + RandomForest con pesos de clase balanceados.

    El encoder va dentro del Pipeline para evitar fuga de datos en la validación cruzada y para
    que la app infiera con un único objeto serializado.
    """
    pre = ColumnTransformer(
        [("suelo", OneHotEncoder(categories=[TIPOS_SUELO], handle_unknown="ignore", sparse_output=False),
          [COL_CATEGORICA])],
        remainder="passthrough",
    )
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced",
                                n_jobs=-1, random_state=SEMILLA)
    return Pipeline([("pre", pre), ("rf", rf)])


def dividir_temporal(df: pd.DataFrame, frac_test: float = 0.20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train = años más antiguos (≈80 %), test = años más recientes (≈20 %)."""
    anios = np.sort(df["anio"].unique())
    n_test = max(1, round(len(anios) * frac_test))
    corte = anios[-n_test]
    return df[df["anio"] < corte], df[df["anio"] >= corte]


def _clase_por_persistencia(mag_max_actual: pd.Series) -> np.ndarray:
    """Línea base ingenua: asumir que el próximo año repetirá la magnitud máxima de este año."""
    return np.digitize(mag_max_actual.fillna(0), [UMBRAL_MEDIO, UMBRAL_ALTO]) + 1


def entrenar(panel: pd.DataFrame, guardar: bool = True) -> dict:
    """Entrena, evalúa (hold-out temporal ≈20 % + CV de series de tiempo) y guarda el modelo.

    Devuelve el diccionario de métricas.
    """
    datos = panel.dropna(subset=[OBJETIVO]).sort_values(["anio", "municipio"]).copy()
    datos[OBJETIVO] = datos[OBJETIVO].astype(int)
    train, test = dividir_temporal(datos)
    X_tr, y_tr, X_te, y_te = train[VARIABLES], train[OBJETIVO], test[VARIABLES], test[OBJETIVO]

    modelo = construir_pipeline()
    cv = TimeSeriesSplit(n_splits=4)  # cada pliegue valida con años posteriores a los de entrenamiento
    f1_cv = cross_val_score(modelo, X_tr, y_tr, cv=cv, scoring="f1_weighted", n_jobs=1)

    modelo.fit(X_tr, y_tr)
    pred = modelo.predict(X_te)
    prec, rec, f1, _ = precision_recall_fscore_support(y_te, pred, average="weighted", zero_division=0)

    base_pers = _clase_por_persistencia(test["mag_max"])
    nombres = [n.split("__", 1)[-1] for n in modelo.named_steps["pre"].get_feature_names_out()]
    importancias = dict(sorted(zip(nombres, modelo.named_steps["rf"].feature_importances_.round(4).tolist()),
                               key=lambda kv: kv[1], reverse=True))
    metricas = {
        "accuracy": round(float(accuracy_score(y_te, pred)), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "f1_macro": round(float(f1_score(y_te, pred, average="macro", zero_division=0)), 4),
        "cv_f1_media": round(float(np.mean(f1_cv)), 4),
        "cv_f1_desv": round(float(np.std(f1_cv)), 4),
        "baseline_mayoritaria_accuracy": round(float((y_te == y_tr.mode().iloc[0]).mean()), 4),
        "baseline_persistencia_accuracy": round(float(accuracy_score(y_te, base_pers)), 4),
        "baseline_persistencia_f1": round(float(f1_score(y_te, base_pers, average="weighted", zero_division=0)), 4),
        "anios_train": f"{int(train['anio'].min())}–{int(train['anio'].max())}",
        "anios_test": f"{int(test['anio'].min())}–{int(test['anio'].max())}",
        "n_train": int(len(train)), "n_test": int(len(test)),
        "importancias": importancias,
        "matriz_confusion": pd.crosstab(y_te.map(ETIQUETAS_RIESGO), pd.Series(pred, index=y_te.index).map(ETIQUETAS_RIESGO)).to_dict(),
    }
    if guardar:
        # Para producción se reajusta con todos los años etiquetados (el test ya sirvió para medir)
        modelo.fit(datos[VARIABLES], datos[OBJETIVO])
        RUTA_MODELO.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"modelo": modelo, "variables": VARIABLES, "metricas": metricas}, RUTA_MODELO)
        RUTA_METRICAS.write_text(json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info("Modelo guardado en %s", RUTA_MODELO)
    return metricas


def predecir(artefacto: dict, fila: dict) -> tuple[int, dict[int, float]]:
    """Predice la categoría (1-3) para un diccionario de variables y devuelve sus probabilidades."""
    X = pd.DataFrame([fila])[artefacto["variables"]]
    modelo = artefacto["modelo"]
    clase = int(modelo.predict(X)[0])
    probs = {int(c): float(p) for c, p in zip(modelo.classes_, modelo.predict_proba(X)[0])}
    return clase, probs


def explicar(artefacto: dict, fila: dict, referencia: pd.DataFrame, clase: int = 3, top: int = 8) -> pd.DataFrame:
    """IA explicable por sustitución: ¿cuánto cambia la probabilidad de la clase si cada variable tomara un valor típico?

    Para cada variable se reemplaza su valor por la mediana (o la moda, si es categórica) de `referencia` y se mide
    la caída o subida de P(clase). Contribución > 0 = esa variable empuja el riesgo hacia arriba respecto de lo normal.
    No necesita librerías extra y se interpreta directamente en puntos porcentuales.
    """
    variables = artefacto["variables"]
    modelo = artefacto["modelo"]
    ref = {v: (referencia[v].mode().iloc[0] if v == COL_CATEGORICA else float(referencia[v].median())) for v in variables}
    base = pd.DataFrame([{v: fila[v] for v in variables}])
    variantes = pd.concat([base] * len(variables), ignore_index=True)
    for i, v in enumerate(variables):
        variantes.loc[i, v] = ref[v]
    col = list(modelo.classes_).index(clase)
    p0 = float(modelo.predict_proba(base)[0, col])
    p = modelo.predict_proba(variantes)[:, col]
    out = pd.DataFrame({"variable": variables, "valor": [fila[v] for v in variables],
                        "valor_tipico": [ref[v] for v in variables], "contribucion": p0 - p})
    out = out.reindex(out["contribucion"].abs().sort_values(ascending=False).index).head(top)
    out.attrs["probabilidad"] = p0
    return out.reset_index(drop=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import pandas as _pd
    from .config import RUTA_PANEL
    m = entrenar(_pd.read_csv(RUTA_PANEL))
    print({k: v for k, v in m.items() if k not in ("importancias", "matriz_confusion")})
