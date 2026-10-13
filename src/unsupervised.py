"""Reto no supervisado: agrupamiento de municipios por perfil sísmico y territorial (K-Means + PCA).

No usa la etiqueta de riesgo. Los grupos se ordenan por un índice de actividad/amenaza para poder
nombrarlos Bajo / Medio / Alto, pero la formación de los grupos es puramente por similitud.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .config import ANIO_ACTUAL, ANIO_MIN, ETIQUETAS_RIESGO, SEMILLA, VARS_TERRITORIALES

# Variables agregadas por municipio sobre todo el periodo (≥15 exigidas por el reto)
VARS_AGREGADAS = [
    "frecuencia_anual", "frecuencia_anual_m3", "mag_media", "mag_max", "profundidad_media", "frac_someros",
    "rms_medio", "gap_medio", "energia_total_log10",
]
VARS_AGRUPAMIENTO = VARS_AGREGADAS + VARS_TERRITORIALES


def agregar_municipios(eventos: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Una fila por municipio: sismicidad 1994-2017 (derivada del catálogo) + variables territoriales."""
    e = eventos[(eventos["anio"] >= ANIO_MIN) & (eventos["anio"] <= ANIO_ACTUAL)]
    n_anios = ANIO_ACTUAL - ANIO_MIN + 1
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


def agrupar(agg: pd.DataFrame, k: int = 3) -> tuple[pd.DataFrame, dict]:
    """K-Means con k grupos + proyección PCA 2D. Devuelve (tabla con grupo, diagnósticos)."""
    X = StandardScaler().fit_transform(agg[VARS_AGRUPAMIENTO])
    silueta = {kk: round(float(silhouette_score(X, KMeans(kk, n_init=10, random_state=SEMILLA).fit_predict(X))), 3)
               for kk in range(2, 7)}
    etiquetas = KMeans(k, n_init=10, random_state=SEMILLA).fit_predict(X)
    pca = PCA(2, random_state=SEMILLA).fit(X)
    pc = pca.transform(X)

    out = agg.copy()
    out["pc1"], out["pc2"] = pc[:, 0], pc[:, 1]
    # Índice de actividad para ordenar los grupos (mayor = más activo y amenazado)
    z = pd.DataFrame(X, columns=VARS_AGRUPAMIENTO)
    indice = z[["frecuencia_anual_m3", "mag_max", "energia_total_log10", "PGA_medio"]].mean(axis=1)
    orden = pd.Series(indice.groupby(etiquetas).mean()).rank().astype(int)  # 1 = menos activo
    nivel = pd.Series(etiquetas).map(orden)
    out["grupo"] = nivel.map(lambda r: ETIQUETAS_RIESGO[min(int(r), 3)] if k == 3 else f"Grupo {int(r)}").to_numpy()
    cargas = pd.DataFrame(pca.components_.T, index=VARS_AGRUPAMIENTO, columns=["PC1", "PC2"]).abs().sum(axis=1)
    diag = {
        "silueta_por_k": silueta,
        "silueta_k": silueta.get(k),
        "varianza_explicada": [round(float(v), 3) for v in pca.explained_variance_ratio_],
        "variables_mas_influyentes": cargas.sort_values(ascending=False).head(6).round(3).to_dict(),
        "n_variables": len(VARS_AGRUPAMIENTO),
    }
    return out, diag
