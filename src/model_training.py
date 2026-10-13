"""Misión 2 (Nivel Medio): feature engineering y entrenamiento del RandomForest."""
import json
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import (
    COL_CATEGORICA, COLS_NUMERICAS, ETIQUETAS_RIESGO, OBJETIVO, RUTA_METRICAS,
    RUTA_MODELO, SEMILLA, TIPOS_SUELO, VARIABLES,
)

log = logging.getLogger(__name__)


def construir_pipeline() -> Pipeline:
    """One-Hot de tipo_suelo + RandomForest con pesos balanceados.

    Meter el encoder dentro del Pipeline evita fuga de datos en la validación cruzada y
    permite inferir en la app con un único objeto serializado.
    """
    # Se fijan las categorías para que el simulador funcione aunque el entrenamiento
    # no haya visto alguna (handle_unknown evita errores con valores nuevos).
    codificador = OneHotEncoder(categories=[TIPOS_SUELO], handle_unknown="ignore", sparse_output=False)
    pre = ColumnTransformer(
        [("suelo", codificador, [COL_CATEGORICA])],
        remainder="passthrough",
    )
    rf = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2, class_weight="balanced",
        n_jobs=-1, random_state=SEMILLA,
    )
    return Pipeline([("pre", pre), ("rf", rf)])


def entrenar(df: pd.DataFrame, guardar: bool = True) -> dict:
    """Entrena, evalúa (hold-out 20 % + CV estratificada 5 pliegues) y guarda el modelo.

    Devuelve el diccionario de métricas.
    """
    X, y = df[VARIABLES], df[OBJETIVO]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEMILLA
    )

    modelo = construir_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    puntajes_cv = cross_val_score(modelo, X_tr, y_tr, cv=cv, scoring="f1_weighted", n_jobs=1)

    modelo.fit(X_tr, y_tr)
    pred = modelo.predict(X_te)
    prec, rec, f1, _ = precision_recall_fscore_support(y_te, pred, average="weighted", zero_division=0)

    # Importancia de variables con los nombres ya expandidos por el One-Hot
    nombres = list(modelo.named_steps["pre"].get_feature_names_out())
    nombres = [n.split("__", 1)[-1] for n in nombres]
    importancias = dict(sorted(
        zip(nombres, modelo.named_steps["rf"].feature_importances_.round(4).tolist()),
        key=lambda kv: kv[1], reverse=True,
    ))

    metricas = {
        "accuracy": round(float(accuracy_score(y_te, pred)), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "cv_f1_media": round(float(np.mean(puntajes_cv)), 4),
        "cv_f1_desv": round(float(np.std(puntajes_cv)), 4),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "importancias": importancias,
        "matriz_confusion": pd.crosstab(
            y_te.map(ETIQUETAS_RIESGO), pd.Series(pred, index=y_te.index).map(ETIQUETAS_RIESGO)
        ).to_dict(),
    }

    if guardar:
        RUTA_MODELO.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"modelo": modelo, "variables": VARIABLES, "metricas": metricas}, RUTA_MODELO)
        RUTA_METRICAS.write_text(json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info("Modelo guardado en %s", RUTA_MODELO)
    return metricas


def predecir(artefacto: dict, fila: dict) -> tuple[int, dict[int, float]]:
    """Predice la categoría (1-4) para un diccionario de variables y devuelve sus probabilidades."""
    X = pd.DataFrame([fila])[artefacto["variables"]]
    modelo = artefacto["modelo"]
    clase = int(modelo.predict(X)[0])
    probs = {int(c): float(p) for c, p in zip(modelo.classes_, modelo.predict_proba(X)[0])}
    return clase, probs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from .data_processing import ejecutar_etl

    datos, _ = ejecutar_etl()
    m = entrenar(datos)
    print({k: v for k, v in m.items() if k not in ("importancias", "matriz_confusion")})
