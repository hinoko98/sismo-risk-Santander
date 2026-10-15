"""Reto supervisado: predice la categoría de riesgo (Bajo/Medio/Alto) del AÑO SIGUIENTE con datos hasta el año t.

v2: se comparan tres familias de modelos con la misma validación cruzada temporal, se afinan con
RandomizedSearchCV (siempre sobre TimeSeriesSplit y sin tocar el test), se calibran las probabilidades y se
ajusta el umbral de decisión de la clase "Alto". El test (2019–2024) solo se usa para informar.
"""
import json
import logging
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import (
    COL_CATEGORICA, ETIQUETAS_RIESGO, OBJETIVO, RUTA_METRICAS, RUTA_METRICAS_V1, RUTA_MODELO, SEMILLA, TIPOS_SUELO,
    UMBRAL_ALTO, UMBRAL_MEDIO, VARIABLES,
)

log = logging.getLogger(__name__)
CLASES = [1, 2, 3]
NOMBRES = [ETIQUETAS_RIESGO[c] for c in CLASES]
IDX_ALTO = 2
PRECISION_MINIMA_ALTO = 0.50   # al subir el recall de "Alto" no se acepta una precisión menor a esta
TOLERANCIA_F1 = 0.01          # se acepta perder hasta 1 punto de F1 macro (en validación) por ganar recall de «Alto»
N_ITER_BUSQUEDA = 15


# ------------------------------------------------------------------ Modelos candidatos
def _preprocesador(variables: list[str]):
    """One-Hot de suelo_predominante (si sigue en el conjunto); el resto pasa sin cambios."""
    if COL_CATEGORICA not in variables:
        return "passthrough"
    return ColumnTransformer(
        [("suelo", OneHotEncoder(categories=[TIPOS_SUELO], handle_unknown="ignore", sparse_output=False), [COL_CATEGORICA])],
        remainder="passthrough")


def _clasificador(familia: str, **params):
    if familia == "RandomForest":
        base = dict(n_estimators=300, min_samples_leaf=3, class_weight="balanced", n_jobs=-1, random_state=SEMILLA)
        return RandomForestClassifier(**{**base, **params})
    if familia == "ExtraTrees":
        base = dict(n_estimators=300, min_samples_leaf=3, class_weight="balanced", n_jobs=-1, random_state=SEMILLA)
        return ExtraTreesClassifier(**{**base, **params})
    if familia == "HistGradientBoosting":
        base = dict(class_weight="balanced", random_state=SEMILLA)
        return HistGradientBoostingClassifier(**{**base, **params})
    raise ValueError(familia)


def construir_pipeline(familia: str = "RandomForest", variables: list[str] | None = None, **params) -> Pipeline:
    """El encoder va dentro del Pipeline para evitar fuga en la validación cruzada y para inferir con un solo objeto."""
    return Pipeline([("pre", _preprocesador(variables or VARIABLES)), ("clf", _clasificador(familia, **params))])


ESPACIO_BUSQUEDA = {  # hiperparámetros explorados por familia (prefijo clf__ = paso del Pipeline)
    "RandomForest": {"clf__n_estimators": randint(150, 500), "clf__max_depth": [None, 6, 10, 16],
                     "clf__min_samples_leaf": randint(1, 12), "clf__max_features": ["sqrt", "log2", 0.5]},
    "ExtraTrees": {"clf__n_estimators": randint(150, 500), "clf__max_depth": [None, 6, 10, 16],
                   "clf__min_samples_leaf": randint(1, 12), "clf__max_features": ["sqrt", "log2", 0.5]},
    "HistGradientBoosting": {"clf__learning_rate": loguniform(0.02, 0.3), "clf__max_iter": randint(80, 400),
                             "clf__max_leaf_nodes": randint(4, 32), "clf__min_samples_leaf": randint(10, 80),
                             "clf__l2_regularization": uniform(0, 5), "clf__max_depth": [None, 3, 5, 8]},
}


# ------------------------------------------------------------------ Utilidades de evaluación
def dividir_temporal(df: pd.DataFrame, frac_test: float = 0.20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train = años más antiguos (≈80 %), test = años más recientes (≈20 %)."""
    anios = np.sort(df["anio"].unique())
    n_test = max(1, round(len(anios) * frac_test))
    corte = anios[-n_test]
    return df[df["anio"] < corte], df[df["anio"] >= corte]


def _clase_por_persistencia(mag_max_actual: pd.Series) -> np.ndarray:
    """Línea base ingenua: asumir que el próximo año repetirá la magnitud máxima de este año."""
    return np.digitize(mag_max_actual.fillna(0), [UMBRAL_MEDIO, UMBRAL_ALTO]) + 1


def aplicar_umbral(probs: np.ndarray, umbral: float, umbral_medio: float | None = None) -> np.ndarray:
    """Regla de decisión: «Alto» si P(Alto) ≥ umbral; si no, «Medio» si P(Medio) ≥ umbral_medio; si no, «Bajo».

    Sin `umbral_medio`, entre Bajo y Medio gana la más probable. Como las probabilidades calibradas de las clases
    raras rara vez superan 0,5, un umbral propio por clase evita que «Medio» y «Alto» casi nunca se declaren.
    """
    if umbral_medio is None:
        resto = np.array(CLASES)[np.argmax(probs[:, :IDX_ALTO], axis=1)]
    else:
        resto = np.where(probs[:, 1] >= umbral_medio, 2, 1)
    return np.where(probs[:, IDX_ALTO] >= umbral, CLASES[IDX_ALTO], resto).astype(int)


def brier_multiclase(y, probs: np.ndarray) -> float:
    """Brier multiclase: media de Σ_k (p_k − 1[y=k])². 0 = perfecto; menor es mejor."""
    onehot = (np.asarray(y)[:, None] == np.array(CLASES)[None, :]).astype(float)
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def _por_clase(y, pred) -> dict:
    p, r, f, n = precision_recall_fscore_support(y, pred, labels=CLASES, zero_division=0)
    return {NOMBRES[i]: {"precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4),
                         "f1": round(float(f[i]), 4), "n": int(n[i])} for i in range(3)}


def _matriz(y, pred) -> dict:
    cm = confusion_matrix(y, pred, labels=CLASES)
    return {NOMBRES[i]: {NOMBRES[j]: int(cm[i, j]) for j in range(3)} for i in range(3)}


def _calibrado(pipe: Pipeline, metodo: str) -> CalibratedClassifierCV:
    # cv temporal: cada calibrador se ajusta con años posteriores a los que entrenaron el modelo base
    return CalibratedClassifierCV(pipe, method=metodo, cv=TimeSeriesSplit(n_splits=3), ensemble=True)


def _probs_fuera_de_muestra(pipe, X, y, metodo, n_splits=4) -> tuple[np.ndarray, np.ndarray]:
    """Probabilidades calibradas de cada pliegue temporal (el modelo nunca vio esos años). Devuelve (índices, probs)."""
    idx, out = [], []
    for tr, va in TimeSeriesSplit(n_splits=n_splits).split(X):
        m = _calibrado(_clonar(pipe), metodo).fit(X.iloc[tr], y.iloc[tr])
        idx.append(va)
        out.append(m.predict_proba(X.iloc[va]))
    return np.concatenate(idx), np.vstack(out)


def _clonar(est):
    from sklearn.base import clone
    return clone(est)


def elegir_umbral(y, probs) -> tuple[float, float, list[dict]]:
    """Umbrales (Alto, Medio): máximo recall de «Alto» entre los que casi igualan el mejor F1 macro temporal, con precisión de «Alto» ≥ 50 %.

    Devuelve (umbral_alto, umbral_medio, curva), donde la curva es el barrido del umbral de «Alto» con el mejor umbral de Medio.
    """
    filas = []
    for t in np.round(np.arange(0.05, 0.71, 0.01), 2):
        for tm in np.round(np.arange(0.10, 0.61, 0.02), 2):
            pred = aplicar_umbral(probs, t, tm)
            p, r, _, _ = precision_recall_fscore_support(y, pred, labels=[3], zero_division=0)
            filas.append({"umbral": float(t), "umbral_medio": float(tm),
                          "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
                          "precision_alto": float(p[0]), "recall_alto": float(r[0])})
    validos = [f for f in filas if f["precision_alto"] >= PRECISION_MINIMA_ALTO] or filas
    f1_top = max(f["f1_macro"] for f in validos)
    cercanos = [f for f in validos if f["f1_macro"] >= f1_top - TOLERANCIA_F1]   # prácticamente el mejor F1 macro
    mejor = max(cercanos, key=lambda f: (round(f["recall_alto"], 3), f["f1_macro"]))   # y, entre ellos, el mayor recall de Alto
    curva = [f for f in filas if f["umbral_medio"] == mejor["umbral_medio"]]
    return mejor["umbral"], mejor["umbral_medio"], curva


# ------------------------------------------------------------------ Entrenamiento
def entrenar(panel: pd.DataFrame, guardar: bool = True, n_iter: int = N_ITER_BUSQUEDA) -> dict:
    """Pipeline completo de la v2. Devuelve el diccionario de métricas (y guarda modelo y metricas.json)."""
    warnings.filterwarnings("ignore", category=UserWarning)
    datos = panel.dropna(subset=[OBJETIVO]).sort_values(["anio", "municipio"]).copy()
    datos[OBJETIVO] = datos[OBJETIVO].astype(int)
    train, test = dividir_temporal(datos)
    X_tr, y_tr, X_te, y_te = train[VARIABLES], train[OBJETIVO], test[VARIABLES], test[OBJETIVO]
    cv = TimeSeriesSplit(n_splits=4)  # cada pliegue valida con años posteriores a los de entrenamiento

    # 1) Comparación de familias con la misma CV temporal: sin afinar y afinadas (métrica: F1 macro)
    comparacion, afinados = {}, {}
    for fam in ESPACIO_BUSQUEDA:
        base = cross_val_score(construir_pipeline(fam), X_tr, y_tr, cv=cv, scoring="f1_macro")
        busq = RandomizedSearchCV(construir_pipeline(fam), ESPACIO_BUSQUEDA[fam], n_iter=n_iter, cv=cv,
                                  scoring="f1_macro", random_state=SEMILLA, n_jobs=1, refit=True)
        busq.fit(X_tr, y_tr)
        params = {k.split("__", 1)[1]: v for k, v in busq.best_params_.items()}
        afinados[fam] = params
        comparacion[fam] = {"cv_f1_macro_defecto": round(float(base.mean()), 4),
                            "cv_f1_macro_afinado": round(float(busq.best_score_), 4),
                            "hiperparametros": {k: (v if isinstance(v, (str, type(None))) else float(v) if isinstance(v, float) else int(v))
                                                for k, v in params.items()}}
    # La familia se elige por lo que realmente se despliega: probabilidades calibradas + umbrales, medido con
    # predicciones fuera de muestra de la CV temporal (el F1 de argmax sin calibrar no es el modelo final).
    oof_fam = {}
    for fam in ESPACIO_BUSQUEDA:
        for met in ("sigmoid", "isotonic"):
            idx, pr = _probs_fuera_de_muestra(construir_pipeline(fam, **afinados[fam]), X_tr, y_tr, met)
            ta, tm, curva = elegir_umbral(y_tr.iloc[idx].to_numpy(), pr)
            f1_desp = float(f1_score(y_tr.iloc[idx], aplicar_umbral(pr, ta, tm), average="macro", zero_division=0))
            oof_fam[(fam, met)] = {"f1": f1_desp, "brier": brier_multiclase(y_tr.iloc[idx], pr)}
        comparacion[fam]["cv_f1_macro_calibrado_umbral"] = round(max(oof_fam[(fam, m)]["f1"] for m in ("sigmoid", "isotonic")), 4)
    ganador, metodo = max(oof_fam, key=lambda k: (round(oof_fam[k]["f1"], 3), -oof_fam[k]["brier"]))
    log.info("Familia ganadora (CV F1 macro con calibración y umbral): %s + %s", ganador, metodo)

    # 2) Selección de variables SIN tocar el test: importancia por permutación en los últimos años del train
    anios_tr = np.sort(train["anio"].unique())
    corte_val = anios_tr[-5]
    sub_tr, sub_va = train[train["anio"] < corte_val], train[train["anio"] >= corte_val]
    modelo_sel = construir_pipeline(ganador, **afinados[ganador]).fit(sub_tr[VARIABLES], sub_tr[OBJETIVO])
    imp_val = permutation_importance(modelo_sel, sub_va[VARIABLES], sub_va[OBJETIVO], scoring="f1_macro",
                                     n_repeats=10, random_state=SEMILLA)
    candidatas = [v for v, m in zip(VARIABLES, imp_val.importances_mean) if m > 0]
    f1_todas = cross_val_score(construir_pipeline(ganador, **afinados[ganador]), X_tr, y_tr, cv=cv, scoring="f1_macro").mean()
    f1_red = cross_val_score(construir_pipeline(ganador, variables=candidatas, **afinados[ganador]),
                             X_tr[candidatas], y_tr, cv=cv, scoring="f1_macro").mean()
    usar_reducido = f1_red >= f1_todas - 0.002   # «si mejora o empata»
    variables = candidatas if usar_reducido else list(VARIABLES)
    descartadas = [v for v in VARIABLES if v not in variables]

    # 3) Calibración (isotónica vs sigmoide, la de menor Brier fuera de muestra) y umbral de «Alto»
    pipe = construir_pipeline(ganador, variables=variables, **afinados[ganador])
    oof = {m: _probs_fuera_de_muestra(pipe, X_tr[variables], y_tr, m) for m in ("sigmoid", "isotonic")}
    brier_cal = {m: brier_multiclase(y_tr.iloc[i], p) for m, (i, p) in oof.items()}
    idx_oof, probs_oof = oof[metodo]   # el método ya se eligió junto con la familia
    umbral, umbral_medio, curva_umbral = elegir_umbral(y_tr.iloc[idx_oof].to_numpy(), probs_oof)

    # 4) Evaluación en el test (una sola vez)
    cal = _calibrado(construir_pipeline(ganador, variables=variables, **afinados[ganador]), metodo).fit(X_tr[variables], y_tr)
    probs = cal.predict_proba(X_te[variables])
    pred = aplicar_umbral(probs, umbral, umbral_medio)
    pred_050 = aplicar_umbral(probs, 0.5)   # sin ajuste de umbrales (Alto con P ≥ 0,5; Bajo/Medio por la más probable)
    pred_argmax = np.array(CLASES)[probs.argmax(axis=1)]

    def puntaje(est, X, y):  # F1 macro con la regla del umbral, para la importancia por permutación
        return f1_score(y, aplicar_umbral(est.predict_proba(X), umbral, umbral_medio), average="macro", zero_division=0)

    imp_te = permutation_importance(cal, X_te[variables], y_te, scoring=puntaje, n_repeats=20, random_state=SEMILLA)
    importancias = dict(sorted(zip(variables, imp_te.importances_mean.round(4).tolist()), key=lambda kv: kv[1], reverse=True))
    importancias_desv = dict(zip(variables, imp_te.importances_std.round(4).tolist()))

    # Comparación en el test de las 3 familias afinadas (sin calibrar), solo informativa
    test_familias = {}
    for fam, p in afinados.items():
        m = construir_pipeline(fam, **p).fit(X_tr, y_tr)
        pr = m.predict(X_te)
        test_familias[fam] = {"f1_macro": round(float(f1_score(y_te, pr, average="macro", zero_division=0)), 4),
                              "recall_alto": _por_clase(y_te, pr)["Alto"]["recall"]}
    # Líneas base
    base_pers = _clase_por_persistencia(test["mag_max"])
    moda = int(y_tr.mode().iloc[0])
    base_may = np.full(len(y_te), moda)
    v1 = json.loads(RUTA_METRICAS_V1.read_text(encoding="utf-8"))["supervisado"] if RUTA_METRICAS_V1.exists() else None

    # Curva de calibración de «Alto» (una contra el resto) en el test
    frac, medio = calibration_curve((y_te == 3).astype(int), probs[:, IDX_ALTO], n_bins=8, strategy="quantile")

    metricas = {
        "modelo": ganador, "calibracion": metodo, "umbral_alto": umbral, "umbral_medio": umbral_medio,
        "umbral_justificacion": (f"Se eligió «Alto» si P ≥ {umbral:.2f} (y «Medio» si P ≥ {umbral_medio:.2f}) porque da el mayor recall de «Alto» entre los cortes "
                                 f"que casi igualan el mejor F1 macro (±{TOLERANCIA_F1:.2f}) en las predicciones temporales de la validación cruzada (años que el modelo no vio), exigiendo una "
                                 f"precisión de «Alto» ≥ {PRECISION_MINIMA_ALTO:.0%}. Con el corte por defecto (0,50) las clases raras "
                                 f"casi nunca se declaran, porque sus probabilidades calibradas rara vez llegan a 0,5."),
        "accuracy": round(float(accuracy_score(y_te, pred)), 4),
        "f1_macro": round(float(f1_score(y_te, pred, average="macro", zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_te, pred, average="weighted", zero_division=0)), 4),
        "por_clase": _por_clase(y_te, pred),
        "recall_alto": _por_clase(y_te, pred)["Alto"]["recall"],
        "precision_alto": _por_clase(y_te, pred)["Alto"]["precision"],
        "brier": round(brier_multiclase(y_te, probs), 4),
        "brier_alto": round(float(np.mean((probs[:, IDX_ALTO] - (y_te == 3)) ** 2)), 4),
        "brier_por_metodo_cv": {k: round(v, 4) for k, v in brier_cal.items()},
        "sin_ajuste_umbral": {"f1_macro": round(float(f1_score(y_te, pred_050, average="macro", zero_division=0)), 4),
                              "recall_alto": _por_clase(y_te, pred_050)["Alto"]["recall"],
                              "precision_alto": _por_clase(y_te, pred_050)["Alto"]["precision"]},
        "argmax_calibrado": {"f1_macro": round(float(f1_score(y_te, pred_argmax, average="macro", zero_division=0)), 4)},
        "cv_f1_macro_media": comparacion[ganador]["cv_f1_macro_calibrado_umbral"],
        "comparacion_modelos_cv": comparacion, "comparacion_modelos_test": test_familias,
        "lineas_base": {
            "mayoritaria": {"accuracy": round(float((y_te == moda).mean()), 4),
                            "f1_macro": round(float(f1_score(y_te, base_may, average="macro", zero_division=0)), 4)},
            "persistencia": {"accuracy": round(float(accuracy_score(y_te, base_pers)), 4),
                             "f1_macro": round(float(f1_score(y_te, base_pers, average="macro", zero_division=0)), 4),
                             "recall_alto": _por_clase(y_te, base_pers)["Alto"]["recall"]}},
        "v1": ({"f1_macro": v1["f1_macro"], "accuracy": v1["accuracy"], "recall_alto": v1["por_clase"]["Alto"]["recall"],
                "precision_alto": v1["por_clase"]["Alto"]["precision"]} if v1 else None),
        "variables_usadas": variables, "variables_descartadas": descartadas,
        "seleccion_variables": {"cv_f1_macro_todas": round(float(f1_todas), 4), "cv_f1_macro_reducido": round(float(f1_red), 4),
                                "se_uso_reducido": bool(usar_reducido)},
        "importancia_permutacion": importancias, "importancia_permutacion_desv": importancias_desv,
        "curva_calibracion_alto": {"prob_predicha": medio.round(4).tolist(), "frecuencia_real": frac.round(4).tolist()},
        "curva_umbral": curva_umbral,
        "anios_train": f"{int(train['anio'].min())}–{int(train['anio'].max())}",
        "anios_test": f"{int(test['anio'].min())}–{int(test['anio'].max())}",
        "n_train": int(len(train)), "n_test": int(len(test)),
        "matriz_confusion": _matriz(y_te, pred),
    }
    if guardar:
        # Para producción se reajusta con todos los años etiquetados (el test ya sirvió para medir)
        final = _calibrado(construir_pipeline(ganador, variables=variables, **afinados[ganador]), metodo)
        final.fit(datos[variables], datos[OBJETIVO])
        RUTA_MODELO.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"modelo": final, "variables": variables, "umbral_alto": umbral, "umbral_medio": umbral_medio, "metricas": metricas}, RUTA_MODELO)
        RUTA_METRICAS.write_text(json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info("Modelo guardado en %s", RUTA_MODELO)
    return metricas


# ------------------------------------------------------------------ Inferencia (firmas sin cambios respecto de v1)
def clases_con_umbral(artefacto: dict, X: pd.DataFrame) -> np.ndarray:
    """Clase (1-3) de cada fila de X aplicando el umbral de «Alto» del artefacto."""
    return aplicar_umbral(artefacto["modelo"].predict_proba(X[artefacto["variables"]]),
                          artefacto.get("umbral_alto", 0.5), artefacto.get("umbral_medio"))


def predecir(artefacto: dict, fila: dict) -> tuple[int, dict[int, float]]:
    """Predice la categoría (1-3) para un diccionario de variables y devuelve sus probabilidades calibradas."""
    X = pd.DataFrame([fila])[artefacto["variables"]]
    modelo = artefacto["modelo"]
    probs = modelo.predict_proba(X)
    clase = int(aplicar_umbral(probs, artefacto.get("umbral_alto", 0.5), artefacto.get("umbral_medio"))[0])
    return clase, {int(c): float(p) for c, p in zip(modelo.classes_, probs[0])}


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
    from .config import RUTA_PANEL
    m = entrenar(pd.read_csv(RUTA_PANEL))
    print({k: v for k, v in m.items() if k in ("modelo", "calibracion", "umbral_alto", "f1_macro", "recall_alto",
                                               "precision_alto", "brier", "variables_descartadas", "v1", "sin_ajuste_umbral")})
    print(json.dumps(m["comparacion_modelos_cv"], indent=1, ensure_ascii=False))
