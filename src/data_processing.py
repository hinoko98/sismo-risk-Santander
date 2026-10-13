"""Misión 1 (Nivel Básico): ETL y limpieza.

Carga el catálogo bruto (SGC + capas IGAC/DANE), imputa nulos, filtra espacialmente con
geopandas y calcula la CATEGORIA_RIESGO cuando no existe.
"""
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from .config import (
    COL_CATEGORICA, COLS_NUMERICAS, ETIQUETAS_RIESGO, OBJETIVO, POLIGONO_APROX,
    RUTA_POLIGONO_IGAC, RUTA_SISMOS_BRUTO, RUTA_SISMOS_LIMPIO, SEMILLA, TIPOS_SUELO,
    VARIABLES,
)
from .data_generation import generar_dataset, guardar_dataset_bruto

log = logging.getLogger(__name__)

# Nombres típicos del catálogo del SGC -> nombres de nuestro esquema
EQUIVALENCIAS_SGC = {
    "Latitud(grados)": "latitud", "Longitud(grados)": "longitud",
    "Profundidad(Km)": "profundidad_km", "Magnitud": "magnitud",
    "RMS(Seg)": "rms", "GAP": "gap_azimutal", "FASES": "fases_asociadas",
    "lat": "latitud", "lon": "longitud", "mag": "magnitud", "depth": "profundidad_km",
}


# ------------------------------------------------------------------ Carga
def cargar_bruto(ruta: Path = RUTA_SISMOS_BRUTO) -> pd.DataFrame:
    """Lee el CSV bruto; si no existe genera uno simulado para poder trabajar."""
    if not Path(ruta).exists():
        log.warning("No se encontró %s; se genera un dataset simulado.", ruta)
        return guardar_dataset_bruto(Path(ruta))
    df = pd.read_csv(ruta)
    return df.rename(columns=EQUIVALENCIAS_SGC)


def cargar_poligono_santander() -> gpd.GeoDataFrame:
    """Polígono oficial (IGAC) si está en data/, de lo contrario el contorno aproximado."""
    try:
        if RUTA_POLIGONO_IGAC.exists():
            gdf = gpd.read_file(RUTA_POLIGONO_IGAC).to_crs(4326)
            return gdf[["geometry"]].dissolve().reset_index(drop=True)
    except Exception as exc:  # archivo corrupto, CRS ausente, etc.
        log.warning("No se pudo leer el polígono IGAC (%s); se usa el aproximado.", exc)
    return gpd.GeoDataFrame(geometry=[POLIGONO_APROX], crs="EPSG:4326")


# ------------------------------------------------------------------ Pasos del ETL
def filtrar_santander(df: pd.DataFrame) -> pd.DataFrame:
    """Conserva solo los eventos cuyo punto (lon, lat) cae dentro del polígono de Santander."""
    df = df.dropna(subset=["latitud", "longitud"]).copy()
    puntos = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["longitud"], df["latitud"]), crs="EPSG:4326"
    )
    poligono = cargar_poligono_santander()
    dentro = gpd.sjoin(puntos, poligono, how="inner", predicate="within")
    # sjoin puede duplicar filas si el polígono tuviera varias partes: se deduplica por índice
    dentro = dentro[~dentro.index.duplicated()]
    return pd.DataFrame(dentro.drop(columns=["geometry", "index_right"], errors="ignore"))


def imputar_nulos(df: pd.DataFrame, referencia: pd.DataFrame) -> pd.DataFrame:
    """Mediana para numéricas y moda para categóricas.

    Si una columna llega vacía por completo (p. ej. un CSV del SGC sin datos de suelo),
    se toma la estadística de `referencia` (dataset simulado) en lugar de quedar en NaN.
    """
    df = df.copy()
    for col in COLS_NUMERICAS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        mediana = df[col].median()
        if pd.isna(mediana):
            mediana = pd.to_numeric(referencia[col], errors="coerce").median()
        df[col] = df[col].fillna(mediana)
    moda = df[COL_CATEGORICA].mode()
    moda = moda.iloc[0] if not moda.empty else referencia[COL_CATEGORICA].mode().iloc[0]
    df[COL_CATEGORICA] = df[COL_CATEGORICA].where(df[COL_CATEGORICA].isin(TIPOS_SUELO), moda)
    return df


def calcular_categoria_riesgo(df: pd.DataFrame, seed: int = SEMILLA) -> pd.Series:
    """Índice compuesto amenaza × vulnerabilidad × exposición -> clases 1..4.

    Criterio (pesos justificados por práctica común en microzonificación sísmica):
      amenaza     : magnitud, PGA, cercanía a falla, poca profundidad
      sitio       : suelo blando, pendiente y nivel freático somero
      exposición  : densidad poblacional, antigüedad de construcciones, infraestructura crítica
    Se añade ruido gaussiano pequeño (incertidumbre del dato) y se corta por cuantiles
    (≈40 % Bajo, 35 % Medio, 18 % Alto, 7 % Extremo) para tener clases con sentido.
    """
    rng = np.random.default_rng(seed)
    peso_suelo = df[COL_CATEGORICA].map({
        "Roca": 0.0, "Suelo duro": 0.25, "Suelo intermedio": 0.55,
        "Suelo blando": 0.9, "Aluvial": 1.0,
    }).astype(float)

    amenaza = (
        0.45 * ((df["magnitud"] - 2) / 4.8).clip(0, 1)
        + 0.25 * (df["PGA_historico"] / 0.5).clip(0, 1)
        + 0.20 * np.exp(-df["distancia_falla_km"] / 30)
        + 0.10 * np.exp(-df["profundidad_km"] / 60)
    )
    sitio = (
        0.60 * peso_suelo
        + 0.20 * (df["pendiente_terreno"] / 40).clip(0, 1)
        + 0.20 * np.exp(-df["nivel_freatico"] / 10)
    )
    exposicion = (
        0.50 * (np.log1p(df["densidad_poblacional"]) / np.log1p(5000)).clip(0, 1)
        + 0.30 * (df["antiguedad_construcciones"] / 100).clip(0, 1)
        + 0.20 * (df["infraestructura_critica"] / 6).clip(0, 1)
    )
    puntaje = 0.50 * amenaza + 0.20 * sitio + 0.30 * exposicion
    puntaje = puntaje + rng.normal(0, 0.015, len(df))
    cortes = puntaje.quantile([0.40, 0.75, 0.93]).to_numpy()
    return pd.Series(np.digitize(puntaje, cortes) + 1, index=df.index, name=OBJETIVO)


# ------------------------------------------------------------------ Orquestador
def ejecutar_etl(ruta_bruto: Path = RUTA_SISMOS_BRUTO, ruta_salida: Path = RUTA_SISMOS_LIMPIO,
                 guardar: bool = True) -> tuple[pd.DataFrame, dict]:
    """Pipeline completo. Devuelve (dataframe limpio, informe con conteos de cada paso)."""
    bruto = cargar_bruto(ruta_bruto)
    informe = {"filas_brutas": len(bruto), "nulos_antes": int(bruto.isna().sum().sum())}

    # Columnas ausentes en el CSV (p. ej. catálogo real del SGC sin capas IGAC/DANE) quedan
    # como NaN y se completan con la referencia simulada durante la imputación
    for col in VARIABLES:
        if col not in bruto.columns:
            bruto[col] = np.nan
    informe["columnas_completadas"] = [c for c in VARIABLES if bruto[c].isna().all()]

    filtrado = filtrar_santander(bruto)
    informe["filas_fuera_santander"] = len(bruto) - len(filtrado)

    referencia = generar_dataset(500)
    limpio = imputar_nulos(filtrado, referencia)

    if OBJETIVO not in limpio.columns or limpio[OBJETIVO].isna().any():
        limpio[OBJETIVO] = calcular_categoria_riesgo(limpio)
    limpio[OBJETIVO] = limpio[OBJETIVO].astype(int)

    limpio = limpio[VARIABLES + [OBJETIVO]].reset_index(drop=True)
    limpio["riesgo_etiqueta"] = limpio[OBJETIVO].map(ETIQUETAS_RIESGO)
    informe["filas_limpias"] = len(limpio)
    informe["nulos_despues"] = int(limpio.isna().sum().sum())

    if guardar:
        Path(ruta_salida).parent.mkdir(parents=True, exist_ok=True)
        limpio.to_csv(ruta_salida, index=False)
    return limpio, informe


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df, info = ejecutar_etl()
    print(info)
    print(df[OBJETIVO].value_counts().sort_index())
