"""Reto no supervisado: agrupamiento de municipios por perfil sísmico y territorial.

v2: log1p en variables muy sesgadas antes de estandarizar, comparación con/sin coordenadas, tres algoritmos
(K-Means, GaussianMixture y Agglomerative-Ward) evaluados con silueta, Davies-Bouldin y estabilidad (ARI), y un
perfil en lenguaje claro de cada grupo. No usa la etiqueta de riesgo: los grupos se ordenan por un índice de
actividad/amenaza solo para nombrarlos Bajo / Medio / Alto.
"""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from .config import ANIO_ACTUAL, ANIO_MIN, DIR_MODELOS, ETIQUETAS_RIESGO, SEMILLA, VARS_TERRITORIALES

# Variables agregadas por municipio sobre todo el periodo (≥15 exigidas por el reto)
VARS_AGREGADAS = [
    "frecuencia_anual", "frecuencia_anual_m3", "mag_media", "mag_max", "profundidad_media", "frac_someros",
    "rms_medio", "gap_medio", "energia_total_log10",
]
VARS_AGRUPAMIENTO = VARS_AGREGADAS + VARS_TERRITORIALES
VARS_COORDENADAS = ["latitud", "longitud"]
# Muy sesgadas (cola larga a la derecha): frecuencias, población y densidad. La energía ya viene en log10.
VARS_LOG = ["frecuencia_anual", "frecuencia_anual_m3", "poblacion", "densidad_poblacional"]
METODOS = ["K-Means", "Mezcla gaussiana", "Aglomerativo (Ward)"]
METODO_DEFECTO = "Mezcla gaussiana"   # mejor silueta, Davies-Bouldin y estabilidad (ver comparar_metodos)
USAR_COORDENADAS_DEFECTO = False   # ver comparar_configuraciones(): sin ellas los grupos describen comportamiento, no ubicación
RUTA_METRICAS_NS = DIR_MODELOS / "metricas_no_supervisado.json"


def agregar_municipios(eventos: pd.DataFrame, feats: pd.DataFrame, anio_max: int = ANIO_ACTUAL) -> pd.DataFrame:
    """Una fila por municipio: sismicidad 1994-`anio_max` (derivada del catálogo) + variables territoriales."""
    e = eventos[(eventos["anio"] >= ANIO_MIN) & (eventos["anio"] <= anio_max)]
    n_anios = anio_max - ANIO_MIN + 1
    g = e.groupby("municipio").agg(
        n=("magnitud", "size"), n_m3=("magnitud", lambda s: (s >= 3).sum()), mag_media=("magnitud", "mean"),
        mag_max=("magnitud", "max"), profundidad_media=("profundidad_km", "mean"),
        frac_someros=("profundidad_km", lambda s: (s < 30).mean()), rms_medio=("rms", "mean"),
        gap_medio=("gap_azimutal", "mean"), energia=("magnitud", lambda s: (10 ** (1.5 * s + 4.8)).sum()),
    )
    g["frecuencia_anual"] = g["n"] / n_anios
    g["frecuencia_anual_m3"] = g["n_m3"] / n_anios
    g["energia_total_log10"] = np.log10(g["energia"])
    g = g.drop(columns=["n", "n_m3", "energia"])
    out = feats.set_index("municipio").join(g, how="left")
    # Municipios sin ningún evento: actividad cero; promedios con la mediana
    for c in ["frecuencia_anual", "frecuencia_anual_m3", "frac_someros"]:
        out[c] = out[c].fillna(0)
    for c in VARS_AGRUPAMIENTO:
        out[c] = out[c].fillna(out[c].median())
    return out.reset_index()


def variables_usadas(usar_coordenadas: bool) -> list[str]:
    return VARS_AGRUPAMIENTO if usar_coordenadas else [v for v in VARS_AGRUPAMIENTO if v not in VARS_COORDENADAS]


def matriz_estandarizada(agg: pd.DataFrame, usar_coordenadas: bool = USAR_COORDENADAS_DEFECTO) -> tuple[np.ndarray, list[str]]:
    """log1p a las variables sesgadas y luego StandardScaler."""
    cols = variables_usadas(usar_coordenadas)
    d = agg[cols].copy()
    for c in VARS_LOG:
        if c in d:
            d[c] = np.log1p(d[c])
    return StandardScaler().fit_transform(d), cols


def _ajustar(metodo: str, X: np.ndarray, k: int, semilla: int = SEMILLA) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if metodo == "K-Means":
            return KMeans(k, n_init=10, random_state=semilla).fit_predict(X)
        if metodo == "Mezcla gaussiana":
            return GaussianMixture(k, covariance_type="diag", n_init=3, random_state=semilla).fit(X).predict(X)
        return AgglomerativeClustering(k, linkage="ward").fit_predict(X)


def _estabilidad(metodo: str, X: np.ndarray, k: int, ref: np.ndarray, n: int = 20) -> float:
    """ARI medio entre la solución completa y la obtenida con submuestras del 80 % (y otra semilla si el método es aleatorio)."""
    rng = np.random.default_rng(SEMILLA)
    aris = []
    for s in range(n):
        idx = np.sort(rng.choice(len(X), int(0.8 * len(X)), replace=False))
        aris.append(adjusted_rand_score(ref[idx], _ajustar(metodo, X[idx], k, semilla=s)))
    return float(np.mean(aris))


def comparar_metodos(agg: pd.DataFrame, k: int = 3, usar_coordenadas: bool = USAR_COORDENADAS_DEFECTO) -> pd.DataFrame:
    """Silueta (↑), Davies-Bouldin (↓) y estabilidad ARI (↑) de los tres algoritmos con los mismos datos."""
    X, _ = matriz_estandarizada(agg, usar_coordenadas)
    filas = []
    for m in METODOS:
        et = _ajustar(m, X, k)
        filas.append({"metodo": m, "silueta": round(float(silhouette_score(X, et)), 3),
                      "davies_bouldin": round(float(davies_bouldin_score(X, et)), 3),
                      "estabilidad_ari": round(_estabilidad(m, X, k, et), 3),
                      "tamanos": sorted(np.bincount(et).tolist(), reverse=True)})
    return pd.DataFrame(filas)


def comparar_configuraciones(agg: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Con y sin latitud/longitud, y con y sin log1p, para justificar las decisiones de preprocesamiento (K-Means)."""
    filas = []
    for usar_coords in (True, False):
        for usar_log in (False, True):
            cols = variables_usadas(usar_coords)
            d = agg[cols].copy()
            if usar_log:
                for c in VARS_LOG:
                    d[c] = np.log1p(d[c])
            X = StandardScaler().fit_transform(d)
            et = _ajustar("K-Means", X, k)
            filas.append({"coordenadas": usar_coords, "log1p": usar_log, "silueta": round(float(silhouette_score(X, et)), 3),
                          "davies_bouldin": round(float(davies_bouldin_score(X, et)), 3),
                          "estabilidad_ari": round(_estabilidad("K-Means", X, k, et), 3)})
    out = pd.DataFrame(filas)
    # ¿Los grupos se explican por la ubicación? ARI entre la solución con y sin coordenadas (alto = coordenadas irrelevantes)
    Xc, _ = matriz_estandarizada(agg, True)
    Xs, _ = matriz_estandarizada(agg, False)
    out.attrs["ari_con_vs_sin_coordenadas"] = round(float(adjusted_rand_score(_ajustar("K-Means", Xc, k), _ajustar("K-Means", Xs, k))), 3)
    pca = PCA(2, random_state=SEMILLA).fit(Xc)
    cols = variables_usadas(True)
    carga = pd.Series(np.abs(pca.components_).sum(axis=0), index=cols)
    out.attrs["peso_coordenadas_en_pca"] = round(float(carga[VARS_COORDENADAS].sum() / carga.sum()), 3)
    return out


def agrupar(agg: pd.DataFrame, k: int = 3, metodo: str = METODO_DEFECTO,
            usar_coordenadas: bool = USAR_COORDENADAS_DEFECTO) -> tuple[pd.DataFrame, dict]:
    """Agrupa con `metodo` + proyección PCA 2D. Devuelve (tabla con grupo, diagnósticos)."""
    X, cols = matriz_estandarizada(agg, usar_coordenadas)
    silueta = {kk: round(float(silhouette_score(X, _ajustar(metodo, X, kk))), 3) for kk in range(2, 7)}
    etiquetas = _ajustar(metodo, X, k)
    pca = PCA(2, random_state=SEMILLA).fit(X)
    pc = pca.transform(X)

    out = agg.copy()
    out["pc1"], out["pc2"] = pc[:, 0], pc[:, 1]
    # Índice de actividad para ordenar los grupos (mayor = más activo y amenazado)
    z = pd.DataFrame(X, columns=cols)
    indice = z[["frecuencia_anual_m3", "mag_max", "energia_total_log10", "PGA_medio"]].mean(axis=1)
    orden = pd.Series(indice.groupby(etiquetas).mean()).rank().astype(int)  # 1 = menos activo
    nivel = pd.Series(etiquetas).map(orden)
    out["grupo"] = nivel.map(lambda r: ETIQUETAS_RIESGO[min(int(r), 3)] if k == 3 else f"Grupo {int(r)}").to_numpy()
    cargas = pd.DataFrame(pca.components_.T, index=cols, columns=["PC1", "PC2"]).abs().sum(axis=1)
    diag = {
        "metodo": metodo, "usa_coordenadas": usar_coordenadas,
        "silueta_por_k": silueta,
        "silueta_k": silueta.get(k),
        "davies_bouldin": round(float(davies_bouldin_score(X, etiquetas)), 3),
        "estabilidad_ari": round(_estabilidad(metodo, X, k, etiquetas), 3),
        "varianza_explicada": [round(float(v), 3) for v in pca.explained_variance_ratio_],
        "variables_mas_influyentes": cargas.sort_values(ascending=False).head(6).round(3).to_dict(),
        "n_variables": len(cols),
    }
    return out, diag


# ------------------------------------------------------------------ Perfiles en lenguaje claro
def _nivel(valor: float, serie: pd.Series, bajo: str, medio: str, alto: str) -> str:
    q1, q2 = serie.quantile([1 / 3, 2 / 3])
    return alto if valor >= q2 else (bajo if valor <= q1 else medio)


def _num(x: float, dec: int = 1) -> str:
    """Formato colombiano: coma decimal y punto de miles."""
    t = f"{x:,.{dec}f}"
    return t.replace(",", "#").replace(".", ",").replace("#", ".")


def describir_grupos(tabla: pd.DataFrame) -> dict[str, str]:
    """Una frase por grupo a partir de las medianas de sus municipios frente al resto del departamento."""
    textos = {}
    for g, d in tabla.groupby("grupo"):
        med = d.median(numeric_only=True)
        prof = med["profundidad_media"]
        if prof >= 100:
            tipo = f"sismos profundos (~{prof:.0f} km), del tipo del Nido de Bucaramanga"
        elif prof >= 40:
            tipo = f"sismos de profundidad intermedia (~{prof:.0f} km)"
        else:
            tipo = f"sismos someros (~{prof:.0f} km), que sacuden cerca de la superficie"
        frec = _nivel(med["frecuencia_anual_m3"], tabla["frecuencia_anual_m3"], "poco frecuentes", "de frecuencia moderada", "muy frecuentes")
        pga = _nivel(med["PGA_medio"], tabla["PGA_medio"], "amenaza oficial baja", "amenaza oficial media", "amenaza oficial alta")
        pob = _nivel(med["poblacion"], tabla["poblacion"], "poblados pequeños", "población mediana", "los más poblados")
        falla = "cerca de fallas activas" if med["distancia_falla_km"] <= tabla["distancia_falla_km"].median() else "lejos de fallas activas"
        nido = f", a ~{med['distancia_bucaramanga_km']:.0f} km de Bucaramanga" if med["distancia_bucaramanga_km"] < 60 else ""
        textos[g] = (f"{len(d)} municipios con {tipo}, {frec} (mediana {_num(med['frecuencia_anual_m3'])} sismos M≥3 por año), "
                     f"magnitud máxima mediana {_num(med['mag_max'])}, {pga} ({_num(med['PGA_medio'], 2)} g), {pob} "
                     f"(mediana {_num(med['poblacion'], 0)} hab.), {falla}{nido}.")
    return textos


if __name__ == "__main__":
    from .config import RUTA_MUN_FEATURES, RUTA_SISMOS_LIMPIO
    agg = agregar_municipios(pd.read_csv(RUTA_SISMOS_LIMPIO), pd.read_csv(RUTA_MUN_FEATURES))
    conf = comparar_configuraciones(agg)
    metodos = comparar_metodos(agg)
    tabla, diag = agrupar(agg, 3)
    resultado = {"configuraciones": conf.to_dict(orient="records"), "ari_con_vs_sin_coordenadas": conf.attrs["ari_con_vs_sin_coordenadas"],
                 "peso_coordenadas_en_pca": conf.attrs["peso_coordenadas_en_pca"], "metodos": metodos.to_dict(orient="records"),
                 "elegido": {"metodo": METODO_DEFECTO, "coordenadas": USAR_COORDENADAS_DEFECTO, **diag},
                 "perfiles": describir_grupos(tabla)}
    RUTA_METRICAS_NS.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(conf.to_string(), conf.attrs, sep="\n")
    print(metodos.to_string())
    print(json.dumps(diag, ensure_ascii=False))
    for g, t in resultado["perfiles"].items():
        print(g, "->", t)
