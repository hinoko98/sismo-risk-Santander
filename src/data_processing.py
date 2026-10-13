"""Misión 1 (Nivel Básico): ETL, limpieza y construcción del dataset municipio-año.

Flujo:
  1. Carga el catálogo real del SGC y calcula la magnitud (Mw si existe, si no Ml).
  2. Filtra espacialmente con geopandas: solo eventos dentro de los municipios de Santander.
  3. Imputa nulos (mediana) y guarda el catálogo limpio.
  4. Agrega variables territoriales por municipio (DANE, DEM, fallas GEM, PGA SGC, OSM).
  5. Construye el panel municipio-año con variables hasta el año t y la etiqueta del año t+1.
"""
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from .config import (
    ANIO_ACTUAL, ANIO_MAX_ENTRADA, ANIO_MIN, BUCARAMANGA, COLS_NUMERICAS, COL_CATEGORICA,
    ETIQUETAS_RIESGO, OBJETIVO, RUTA_MUN_FEATURES, RUTA_MUNICIPIOS, RUTA_PANEL, RUTA_POLIGONO_IGAC,
    RUTA_SISMOS_BRUTO, RUTA_SISMOS_LIMPIO, RUTA_SITIOS, UMBRAL_ALTO, UMBRAL_MEDIO, VARIABLES,
)

log = logging.getLogger(__name__)
CRS_M = "EPSG:3116"  # metros (MAGNA-SIRGAS Colombia Bogotá)


# ------------------------------------------------------------------ Carga
def cargar_bruto(ruta: Path = RUTA_SISMOS_BRUTO) -> pd.DataFrame:
    """Lee el catálogo SGC; si no existe lo descarga (requiere internet)."""
    if not Path(ruta).exists():
        from .real_data import descargar_catalogo_sgc
        log.warning("No existe %s; se descarga del SGC.", ruta)
        return descargar_catalogo_sgc()
    return pd.read_csv(ruta)


def cargar_municipios() -> gpd.GeoDataFrame:
    """Municipios de Santander (polígonos MGN-DANE, población DANE 2025); se construyen si faltan."""
    if not RUTA_MUNICIPIOS.exists():
        from .real_data import construir_municipios
        construir_municipios()
    return gpd.read_file(RUTA_MUNICIPIOS).to_crs(4326)


def cargar_sitios() -> pd.DataFrame:
    """Variables de terreno por celda de ~2 km; se construyen con las fuentes reales si faltan."""
    if not RUTA_SITIOS.exists():
        from .real_data import construir_sitios
        log.warning("No existe %s; se construye (tarda varios minutos).", RUTA_SITIOS)
        return construir_sitios()
    return pd.read_csv(RUTA_SITIOS)


# ------------------------------------------------------------------ 1-3. Catálogo limpio
def filtrar_santander(df: pd.DataFrame, municipios: gpd.GeoDataFrame) -> pd.DataFrame:
    """Cruce espacial: conserva solo eventos cuyo punto (lon, lat) cae dentro de un municipio
    de Santander y les asigna el municipio (DANE) donde ocurrieron."""
    puntos = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["longitud"], df["latitud"]), crs=4326)
    dentro = gpd.sjoin(puntos, municipios[["municipio", "geometry"]].rename(columns={"municipio": "municipio_dane"}),
                       how="inner", predicate="within")
    dentro = dentro[~dentro.index.duplicated()]
    return pd.DataFrame(dentro.drop(columns=["geometry", "index_right"]))


def limpiar_catalogo() -> tuple[pd.DataFrame, dict]:
    bruto = cargar_bruto()
    informe = {"filas_brutas": len(bruto)}
    bruto = bruto.dropna(subset=["magnitud", "latitud", "longitud", "fecha"])
    bruto = bruto.drop_duplicates(subset=["fecha", "hora_utc", "latitud", "longitud"])
    informe["descartados_sin_magnitud_o_duplicados"] = informe["filas_brutas"] - len(bruto)

    dentro = filtrar_santander(bruto, cargar_municipios())
    informe["descartados_fuera_santander"] = len(bruto) - len(dentro)

    cols_imputar = ["profundidad_km", "rms", "gap_azimutal", "fases_asociadas", "error_lat_km", "error_lon_km", "error_prof_km"]
    informe["nulos_imputados"] = int(dentro[cols_imputar].isna().sum().sum())
    for c in cols_imputar:
        dentro[c] = pd.to_numeric(dentro[c], errors="coerce")
        dentro[c] = dentro[c].fillna(dentro[c].median())

    dentro["anio"] = pd.to_datetime(dentro["fecha"], errors="coerce").dt.year
    dentro = dentro.drop(columns=["municipio"], errors="ignore")  # el nombre del SGC se reemplaza por el del DANE
    limpio = dentro.rename(columns={"municipio_dane": "municipio"})[
        ["fecha", "hora_utc", "anio", "municipio", "latitud", "longitud", "profundidad_km", "magnitud",
         "rms", "gap_azimutal", "fases_asociadas", "error_lat_km", "error_lon_km", "error_prof_km", "fuente"]
    ].dropna(subset=["anio"]).astype({"anio": int}).reset_index(drop=True)
    informe["filas_limpias"] = len(limpio)
    return limpio, informe


# ------------------------------------------------------------------ 4. Variables territoriales por municipio
def construir_features_municipio(municipios: gpd.GeoDataFrame, sitios: pd.DataFrame) -> pd.DataFrame:
    """Resume por municipio las celdas de terreno y suma las variables DANE (población, área)."""
    blando = sitios["tipo_suelo"].str.startswith(("D", "E"))
    agg = sitios.assign(suelo_blando=blando).groupby("municipio").agg(
        PGA_medio=("PGA_historico", "mean"),
        elevacion_media=("elevacion", "mean"),
        pendiente_media=("pendiente_terreno", "mean"),
        distancia_falla_km=("distancia_falla_km", "mean"),
        distancia_cuerpos_agua=("distancia_cuerpos_agua", "mean"),
        pct_suelo_blando=("suelo_blando", "mean"),
        suelo_predominante=("tipo_suelo", lambda s: s.mode().iloc[0]),
    )
    m = municipios.copy()
    cen = m.geometry.representative_point()
    m["latitud"], m["longitud"] = cen.y, cen.x
    bu = gpd.GeoSeries(gpd.points_from_xy([BUCARAMANGA[1]], [BUCARAMANGA[0]]), crs=4326).to_crs(CRS_M).iloc[0]
    m["distancia_bucaramanga_km"] = cen.to_crs(CRS_M).distance(bu) / 1000
    base = pd.DataFrame(m.drop(columns="geometry")).set_index("municipio")
    out = base.join(agg, how="left")
    # Un municipio sin celda propia (p. ej. muy pequeño) hereda la mediana del departamento
    for c in out.columns:
        if out[c].isna().any():
            out[c] = out[c].fillna(out[c].median() if out[c].dtype != object else out[c].mode().iloc[0])
    return out.reset_index()


# ------------------------------------------------------------------ 5. Panel municipio-año
def _historial_anual(eventos: pd.DataFrame, municipios: list[str], anios: range) -> pd.DataFrame:
    """Variables sísmicas de cada municipio y año (todas calculadas con eventos de ese año)."""
    e = eventos.copy()
    e["m3"] = e["magnitud"] >= 3
    e["someros"] = e["profundidad_km"] < 30
    e["energia"] = 10 ** (1.5 * e["magnitud"] + 4.8)  # energía sísmica (J), Gutenberg-Richter
    g = e.groupby(["municipio", "anio"]).agg(
        n_sismos=("magnitud", "size"), n_sismos_m3=("m3", "sum"), mag_media=("magnitud", "mean"),
        mag_max=("magnitud", "max"), profundidad_media=("profundidad_km", "mean"), frac_someros=("someros", "mean"),
        rms_medio=("rms", "mean"), gap_medio=("gap_azimutal", "mean"), fases_medias=("fases_asociadas", "mean"),
        error_lat_medio=("error_lat_km", "mean"), error_prof_medio=("error_prof_km", "mean"),
        energia=("energia", "sum"),
    )
    malla = pd.MultiIndex.from_product([municipios, anios], names=["municipio", "anio"])
    g = g.reindex(malla)
    g["n_sismos"] = g["n_sismos"].fillna(0)
    g["n_sismos_m3"] = g["n_sismos_m3"].fillna(0)
    g["energia_log10"] = np.log10(g["energia"].fillna(0) + 1)
    return g.drop(columns="energia").reset_index()


def construir_panel(eventos: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Panel municipio-año con variables hasta t y etiqueta del año t+1."""
    municipios = sorted(feats["municipio"])
    h = _historial_anual(eventos, municipios, range(ANIO_MIN - 1, ANIO_ACTUAL + 2))
    h = h.sort_values(["municipio", "anio"])
    gr = h.groupby("municipio")

    # Variables de memoria: solo miran años ≤ t
    h["n_m3_prev3"] = gr["n_sismos_m3"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    h["mag_max_prev3"] = gr["mag_max"].transform(lambda s: s.rolling(3, min_periods=1).max())
    h["tendencia_m3"] = gr["n_sismos_m3"].diff()
    h["anios_desde_m4"] = gr["mag_max"].transform(_anios_desde_m4)
    # Etiqueta: magnitud máxima del año SIGUIENTE (no se usa como entrada)
    h["mag_max_siguiente"] = gr["mag_max"].shift(-1)
    h["CATEGORIA_RIESGO_NUM"] = np.digitize(h["mag_max_siguiente"].fillna(0), [UMBRAL_MEDIO, UMBRAL_ALTO]) + 1

    panel = h[(h["anio"] >= ANIO_MIN) & (h["anio"] <= ANIO_ACTUAL)].merge(feats, on="municipio", how="left")
    # Huecos: años sin sismos -> no hubo magnitud/profundidad que promediar. Se imputa con la mediana de lo
    # observado HASTA ese año (nunca con años posteriores, para no filtrar información del futuro).
    for c in ["mag_media", "mag_max", "profundidad_media", "frac_someros", "rms_medio", "gap_medio", "fases_medias",
              "error_lat_medio", "error_prof_medio", "mag_max_prev3", "tendencia_m3", "n_m3_prev3"]:
        panel[c] = _imputar_solo_pasado(panel, c)
    panel[OBJETIVO] = panel["CATEGORIA_RIESGO_NUM"].astype(int)
    panel["riesgo_etiqueta"] = panel[OBJETIVO].map(ETIQUETAS_RIESGO)
    # El último año completo no tiene etiqueta (el siguiente está incompleto): se conserva como estado actual
    panel.loc[panel["anio"] > ANIO_MAX_ENTRADA, [OBJETIVO, "riesgo_etiqueta", "mag_max_siguiente"]] = np.nan
    return panel.drop(columns="CATEGORIA_RIESGO_NUM")


def _imputar_solo_pasado(panel: pd.DataFrame, col: str) -> pd.Series:
    """Rellena los nulos de `col` con la mediana de los valores observados en años ≤ al de cada fila."""
    mediana_hasta = {a: panel.loc[panel["anio"] <= a, col].median() for a in sorted(panel["anio"].unique())}
    llenado = panel["anio"].map(mediana_hasta).fillna(0.0)  # sin observaciones previas: 0, no estadística futura
    return panel[col].fillna(llenado)


def _anios_desde_m4(mag_max: pd.Series) -> pd.Series:
    """Años transcurridos desde el último año con M≥4 (tope 30) usando solo información ≤ t."""
    out, ultimo = [], None
    for k, m in enumerate(mag_max.to_numpy()):
        if not np.isnan(m) and m >= 4:
            ultimo = k
        out.append(30 if ultimo is None else min(k - ultimo, 30))
    return pd.Series(out, index=mag_max.index)


# ------------------------------------------------------------------ Orquestador
def ejecutar_etl(guardar: bool = True) -> tuple[pd.DataFrame, dict]:
    """Pipeline completo. Devuelve (panel municipio-año, informe con conteos de cada paso)."""
    limpio, informe = limpiar_catalogo()
    municipios = cargar_municipios()
    feats = construir_features_municipio(municipios, cargar_sitios())
    panel = construir_panel(limpio, feats)
    informe["filas_panel"] = len(panel)
    informe["municipios"] = int(feats["municipio"].nunique())
    informe["nulos_panel"] = int(panel[VARIABLES].isna().sum().sum())
    if guardar:
        RUTA_SISMOS_LIMPIO.parent.mkdir(parents=True, exist_ok=True)
        limpio.to_csv(RUTA_SISMOS_LIMPIO, index=False)
        feats.to_csv(RUTA_MUN_FEATURES, index=False)
        panel.to_csv(RUTA_PANEL, index=False)
    return panel, informe


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    p, info = ejecutar_etl()
    print(info)
    print(p[OBJETIVO].value_counts(dropna=False).sort_index())
