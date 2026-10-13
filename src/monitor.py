"""Monitor de actividad reciente: ¿qué municipios están más activos de lo normal?

Usa el catálogo del SGC ya limpio (hasta la fecha de la última descarga) y, si hay internet, lo completa con los
eventos más nuevos que publique el SGC (consulta SeisComP). Compara, por municipio, los eventos M≥4 de los últimos
12 meses con lo esperable según su propia historia (2014–2023, misma agencia y escala). La comparación usa la cola
de Poisson: P(X ≥ observados | esperados).

Importante: una actividad inusual es una señal ESTADÍSTICA para vigilar, no un pronóstico de un sismo mayor.
"""
import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import poisson

from .config import RUTA_SISMOS_LIMPIO, UMBRAL_ALTO

log = logging.getLogger(__name__)
VENTANA_BASE = (2014, 2023)
PESO_PRIOR = 5.0


def cargar_catalogo(municipios: gpd.GeoDataFrame, en_vivo: bool = True) -> tuple[pd.DataFrame, str]:
    """Catálogo SGC limpio (+ eventos nuevos si el SGC responde). Columnas: time, mag, profundidad_km, municipio."""
    base = pd.read_csv(RUTA_SISMOS_LIMPIO, usecols=["fecha", "hora_utc", "municipio", "magnitud", "profundidad_km", "latitud", "longitud"])
    base["time"] = pd.to_datetime(base["fecha"] + " " + base["hora_utc"].astype(str), errors="coerce")
    base = base.dropna(subset=["time"]).rename(columns={"magnitud": "mag"})
    fuente = f"catálogo SGC local (hasta {base['time'].max():%d-%m-%Y})"
    if en_vivo:
        try:
            from .real_data import actualizar_reciente
            nuevo = actualizar_reciente(base["time"].max())
            if len(nuevo):
                pts = gpd.GeoDataFrame(nuevo, geometry=gpd.points_from_xy(nuevo["longitud"], nuevo["latitud"]), crs=4326)
                j = gpd.sjoin(pts, municipios[["municipio", "geometry"]], how="inner", predicate="within")
                add = pd.DataFrame({"time": j["fecha_hora"], "mag": j["magnitud"], "profundidad_km": j["profundidad_km"],
                                    "municipio": j["municipio"], "latitud": j["latitud"], "longitud": j["longitud"]})
                base = pd.concat([base, add], ignore_index=True).drop_duplicates(["time", "latitud", "longitud"])
                fuente = f"SGC en vivo (hasta {base['time'].max():%d-%m-%Y})"
        except Exception as exc:  # sin internet o servicio caído: se sigue con lo local
            log.warning("SGC en vivo no disponible: %s", exc)
            fuente += " · sin conexión al SGC"
    return base.sort_values("time").reset_index(drop=True), fuente


def tasas_base(cat: pd.DataFrame, municipios: list[str]) -> pd.Series:
    """Eventos M≥4 por año y municipio en la ventana base, suavizados hacia el promedio departamental."""
    ini, fin = VENTANA_BASE
    e = cat[(cat["mag"] >= UMBRAL_ALTO) & (cat["time"].dt.year.between(ini, fin))]
    n = e.groupby("municipio").size().reindex(municipios, fill_value=0)
    T = fin - ini + 1
    return (n + PESO_PRIOR * n.mean() / T) / (T + PESO_PRIOR)


def evaluar_actividad(cat: pd.DataFrame, municipios: list[str], dias: int = 365) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Compara los últimos `dias` días con lo esperado. Semáforo: Rojo p<0,01 · Amarillo p<0,05 · Verde."""
    fin = cat["time"].max()
    ini = fin - pd.Timedelta(days=dias)
    lam = tasas_base(cat, municipios)
    obs = cat[(cat["mag"] >= UMBRAL_ALTO) & (cat["time"] > ini)].groupby("municipio").size().reindex(municipios, fill_value=0)
    esperado = lam * dias / 365.0
    p = pd.Series(poisson.sf(obs.to_numpy() - 1, esperado.to_numpy()), index=municipios)
    mayor = cat[cat["time"] > ini].groupby("municipio")["mag"].max().reindex(municipios)
    out = pd.DataFrame({"municipio": municipios, "observados_12m": obs.to_numpy(), "esperados_12m": esperado.round(2).to_numpy(),
                        "razon": (obs / esperado.replace(0, np.nan)).round(2).to_numpy(), "p_valor": p.round(4).to_numpy(),
                        "mag_max_12m": mayor.to_numpy()})
    out["semaforo"] = np.where((out["p_valor"] < 0.01) & (out["observados_12m"] >= 2), "Rojo",
                               np.where((out["p_valor"] < 0.05) & (out["observados_12m"] >= 2), "Amarillo", "Verde"))
    return out.sort_values(["p_valor", "observados_12m"], ascending=[True, False]).reset_index(drop=True), fin


def serie_mensual(cat: pd.DataFrame, lam_total_anual: float, desde: str = "2018-01-01") -> pd.DataFrame:
    """Sismos M≥4 por mes en todo Santander frente al promedio esperado."""
    e = cat[(cat["mag"] >= UMBRAL_ALTO) & (cat["time"] >= desde)]
    m = e.set_index("time").resample("MS").size().rename("observados").reset_index()
    m["esperados"] = lam_total_anual / 12
    return m
