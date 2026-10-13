"""Proyección probabilística hasta 2027 con datos reales del SGC (modelo de Poisson por municipio).

Idea: un sismo concreto no se puede predecir, pero sí la probabilidad de que ocurra al menos uno en un
periodo, a partir de su tasa histórica (enfoque estándar en amenaza sísmica):  P = 1 - exp(-λ·T).

  * λ se estima por municipio con la última década completa del catálogo (2016–2025). Antes de 2008 el SGC
    reportaba magnitudes sistemáticamente más altas (cambio de método), así que usar todo el historial inflaría λ.
  * Como muchos municipios tienen pocos eventos, la tasa se "encoge" hacia el promedio departamental
    (estimador bayesiano empírico gamma-Poisson) para no declarar probabilidad cero por falta de datos.
  * Las clases son las mismas del modelo supervisado, definidas por la magnitud máxima del año:
    Bajo < 3 · Medio 3–<4 · Alto ≥ 4.
  * Validación retrospectiva (backtest): se estima λ con 2008–2017 y se compara con lo que realmente registró
    el SGC entre 2018 y 2025, años que no se usaron para estimarlo.

Supuestos: los eventos llegan de forma independiente y estacionaria (no modela réplicas ni cambios del
régimen sísmico). Es una extrapolación de tasas, no una predicción de eventos.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import ANIO_ACTUAL, ANIO_PROYECCION, ETIQUETAS_RIESGO, UMBRAL_ALTO, UMBRAL_MEDIO

VENTANA_ACTUAL = (2016, ANIO_ACTUAL)   # años para la proyección real
VENTANA_BACKTEST = (2008, 2017)        # años para el backtest
ANIOS_BACKTEST = list(range(2018, ANIO_ACTUAL + 1))
PESO_PRIOR = 5.0                       # años "virtuales" de información del promedio departamental


def estimar_tasas(eventos: pd.DataFrame, municipios: list[str], ventana: tuple[int, int] = VENTANA_ACTUAL) -> pd.DataFrame:
    """Tasa anual λ de sismos M≥3 y M≥4 por municipio (con encogimiento hacia el promedio)."""
    ini, fin = ventana
    T = fin - ini + 1
    e = eventos[(eventos["anio"] >= ini) & (eventos["anio"] <= fin)]
    out = pd.DataFrame(index=pd.Index(municipios, name="municipio"))
    for nombre, umbral in [("lambda_m3", UMBRAL_MEDIO), ("lambda_m4", UMBRAL_ALTO)]:
        n = e[e["magnitud"] >= umbral].groupby("municipio").size().reindex(municipios, fill_value=0)
        media = n.mean() / T
        out["n_" + nombre[-2:]] = n
        out[nombre] = (n + PESO_PRIOR * media) / (T + PESO_PRIOR)
    out["lambda_m3"] = np.maximum(out["lambda_m3"], out["lambda_m4"])  # M≥3 incluye a M≥4
    return out.reset_index()


def clases_anuales(tasas: pd.DataFrame) -> pd.DataFrame:
    """Probabilidad de cada clase de riesgo en un año cualquiera (modelo estacionario)."""
    t = tasas.copy()
    t["p_alto"] = 1 - np.exp(-t["lambda_m4"])
    t["p_bajo"] = np.exp(-t["lambda_m3"])
    t["p_medio"] = 1 - t["p_alto"] - t["p_bajo"]
    nombres = [ETIQUETAS_RIESGO[1], ETIQUETAS_RIESGO[2], ETIQUETAS_RIESGO[3]]
    t["clase_probable"] = t[["p_bajo", "p_medio", "p_alto"]].to_numpy().argmax(axis=1)
    t["clase_probable"] = t["clase_probable"].map(dict(enumerate(nombres)))
    return t


def proyectar(tasas: pd.DataFrame, hasta: int = ANIO_PROYECCION) -> pd.DataFrame:
    """Probabilidad acumulada (desde enero del año siguiente al último completo) de ≥1 sismo M≥4 y M≥3, por año."""
    filas = []
    for anio in range(ANIO_ACTUAL + 1, hasta + 1):
        T = anio - ANIO_ACTUAL
        d = tasas[["municipio", "lambda_m3", "lambda_m4"]].copy()
        d["anio"] = anio
        d["p_acum_m4"] = 1 - np.exp(-d["lambda_m4"] * T)
        d["p_acum_m3"] = 1 - np.exp(-d["lambda_m3"] * T)
        d["esperados_m4"] = d["lambda_m4"] * T
        d["p_anual_m4"] = 1 - np.exp(-d["lambda_m4"])
        filas.append(d)
    return pd.concat(filas, ignore_index=True)


def validar(tasas_backtest: pd.DataFrame, eventos: pd.DataFrame) -> dict:
    """Backtest: compara las tasas estimadas con 2008–2017 contra los sismos M≥4 REALES del SGC de 2018–2025."""
    e = eventos[(eventos["magnitud"] >= UMBRAL_ALTO) & (eventos["anio"].isin(ANIOS_BACKTEST))]
    obs = e.groupby(["municipio", "anio"]).size()
    mun = tasas_backtest["municipio"].tolist()
    idx = pd.MultiIndex.from_product([mun, ANIOS_BACKTEST], names=["municipio", "anio"])
    obs = obs.reindex(idx, fill_value=0)
    lam = tasas_backtest.set_index("municipio")["lambda_m4"].reindex(mun)
    p_mun_anio = np.repeat((1 - np.exp(-lam)).to_numpy(), len(ANIOS_BACKTEST))
    hubo = (obs.to_numpy() > 0).astype(float)

    brier = float(np.mean((p_mun_anio - hubo) ** 2))
    brier_ref = float(np.mean((hubo.mean() - hubo) ** 2))  # referencia: misma probabilidad para todos
    por_mun = obs.groupby("municipio").sum().reindex(mun)
    esperado = lam * len(ANIOS_BACKTEST)
    por_anio = obs.groupby("anio").sum().rename("observados").reset_index().assign(esperados=float(lam.sum()))
    return {
        "anios": f"{ANIOS_BACKTEST[0]}–{ANIOS_BACKTEST[-1]}",
        "ventana": f"{VENTANA_BACKTEST[0]}–{VENTANA_BACKTEST[1]}",
        "observados_total": int(por_mun.sum()), "esperados_total": float(esperado.sum()),
        "spearman_municipios": float(spearmanr(esperado, por_mun).statistic),
        "brier": brier, "brier_referencia": brier_ref, "habilidad_brier": 1 - brier / brier_ref,
        "tabla_municipios": pd.DataFrame({"municipio": mun, "esperados": esperado.to_numpy(), "observados": por_mun.to_numpy()}),
        "por_anio": por_anio,
    }
