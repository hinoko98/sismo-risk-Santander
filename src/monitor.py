"""Monitor de actividad reciente: ¿qué municipios están más activos de lo normal?

El catálogo del SGC (Consulta Experta) se detiene en marzo de 2018. Este módulo lo complementa con sismos
reales del USGS hasta la fecha (descarga en vivo si hay internet; si no, el archivo local) y compara, por municipio,
los eventos M≥4 de los últimos 12 meses con lo esperable según su propia historia (2008–2017, misma agencia, por lo
que las magnitudes son comparables). La comparación usa la cola de Poisson: P(X ≥ observados | esperados).

Importante: una actividad inusual es una señal ESTADÍSTICA para vigilar, no un pronóstico de un sismo mayor.
"""
import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from scipy.stats import poisson

from .config import BBOX_SANTANDER, RUTA_USGS, UMBRAL_ALTO

log = logging.getLogger(__name__)
VENTANA_BASE = (2008, 2017)
PESO_PRIOR = 5.0


def cargar_usgs(en_vivo: bool = True) -> tuple[pd.DataFrame, str]:
    """USGS local + (si hay internet) los eventos nuevos desde la última fecha guardada. Devuelve (df, fuente)."""
    local = pd.read_csv(RUTA_USGS)
    local["time"] = pd.to_datetime(local["time"], utc=True)
    fuente = f"archivo local (hasta {local['time'].max():%d-%m-%Y})"
    if not en_vivo:
        return local, fuente
    try:
        lat0, lat1, lon0, lon1 = BBOX_SANTANDER
        desde = (local["time"].max() - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
        r = requests.get("https://earthquake.usgs.gov/fdsnws/event/1/query", timeout=8, params={
            "format": "csv", "minlatitude": lat0, "maxlatitude": lat1, "minlongitude": lon0, "maxlongitude": lon1,
            "starttime": desde, "minmagnitude": 3.5, "orderby": "time-asc"})
        r.raise_for_status()
        from io import StringIO
        nuevo = pd.read_csv(StringIO(r.text))
        nuevo["time"] = pd.to_datetime(nuevo["time"], utc=True)
        df = pd.concat([local, nuevo]).drop_duplicates("id").sort_values("time").reset_index(drop=True)
        return df, f"USGS en vivo (hasta {df['time'].max():%d-%m-%Y})"
    except Exception as exc:  # sin internet o servicio caído: se sigue con el archivo local
        log.warning("USGS en vivo no disponible: %s", exc)
        return local, fuente + " · sin conexión al USGS"


def asignar_municipio(usgs: pd.DataFrame, municipios: gpd.GeoDataFrame) -> pd.DataFrame:
    pts = gpd.GeoDataFrame(usgs, geometry=gpd.points_from_xy(usgs["longitude"], usgs["latitude"]), crs=4326)
    j = gpd.sjoin(pts, municipios[["municipio", "geometry"]], how="inner", predicate="within")
    return pd.DataFrame(j.drop(columns=["geometry", "index_right"]))


def tasas_base(usgs_mun: pd.DataFrame, municipios: list[str]) -> pd.Series:
    """Eventos M≥4 por año y municipio en 2008–2017, suavizados hacia el promedio departamental."""
    ini, fin = VENTANA_BASE
    e = usgs_mun[(usgs_mun["mag"] >= UMBRAL_ALTO) & (usgs_mun["time"].dt.year.between(ini, fin))]
    n = e.groupby("municipio").size().reindex(municipios, fill_value=0)
    T = fin - ini + 1
    return (n + PESO_PRIOR * n.mean() / T) / (T + PESO_PRIOR)


def evaluar_actividad(usgs_mun: pd.DataFrame, municipios: list[str], dias: int = 365) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Compara los últimos `dias` días con lo esperado. Semáforo: Rojo p<0,01 · Amarillo p<0,05 · Verde."""
    fin = usgs_mun["time"].max()
    ini = fin - pd.Timedelta(days=dias)
    lam = tasas_base(usgs_mun, municipios)
    obs = usgs_mun[(usgs_mun["mag"] >= UMBRAL_ALTO) & (usgs_mun["time"] > ini)].groupby("municipio").size() \
        .reindex(municipios, fill_value=0)
    esperado = lam * dias / 365.0
    p = pd.Series(poisson.sf(obs.to_numpy() - 1, esperado.to_numpy()), index=municipios)
    mayor = usgs_mun[usgs_mun["time"] > ini].groupby("municipio")["mag"].max().reindex(municipios)
    out = pd.DataFrame({"municipio": municipios, "observados_12m": obs.to_numpy(), "esperados_12m": esperado.round(2).to_numpy(),
                        "razon": (obs / esperado.replace(0, np.nan)).round(2).to_numpy(), "p_valor": p.round(4).to_numpy(),
                        "mag_max_12m": mayor.to_numpy()})
    out["semaforo"] = np.where((out["p_valor"] < 0.01) & (out["observados_12m"] >= 2), "Rojo",
                               np.where((out["p_valor"] < 0.05) & (out["observados_12m"] >= 2), "Amarillo", "Verde"))
    return out.sort_values(["p_valor", "observados_12m"], ascending=[True, False]).reset_index(drop=True), fin


def serie_mensual(usgs_mun: pd.DataFrame, lam_total_anual: float) -> pd.DataFrame:
    """Sismos M≥4 por mes en todo Santander desde 2018 frente al promedio esperado."""
    e = usgs_mun[(usgs_mun["mag"] >= UMBRAL_ALTO) & (usgs_mun["time"] >= "2018-01-01")]
    m = e.set_index("time").resample("MS").size().rename("observados").reset_index()
    m["esperados"] = lam_total_anual / 12
    return m
