"""Índice de Riesgo Integral (IRI): de la amenaza a la decisión.

El riesgo sísmico no depende solo de qué tan fuerte puede sacudir el suelo, sino de cuánta gente y qué terreno lo reciben:

    IRI ∝  Amenaza  ×  Exposición  ×  (0,5 + 0,5 · Vulnerabilidad del sitio)

  * Amenaza        : PGA esperado a 475 años (modelo oficial de amenaza sísmica del SGC, 2020), relativo al máximo
                     del departamento. Se usa el PGA y no el conteo de epicentros porque el modelo del SGC ya incorpora
                     profundidad y atenuación: casi todos los M≥4 de Santander son del Nido de Bucaramanga (~150 km de
                     profundidad) y sacuden también a los municipios vecinos, no solo a donde está el epicentro.
  * Exposición     : población DANE 2025 en escala logarítmica (0–1), para que las grandes ciudades no eclipsen todo.
  * Vulnerabilidad : fracción de suelo blando/rígido (amplifica la sacudida) y pendiente media (deslizamientos), 0–1.

Es un índice RELATIVO de priorización con pesos propuestos por el equipo; no estima daños ni víctimas.
"""
import numpy as np
import pandas as pd


def calcular_iri(feats: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una fila por municipio con componentes, IRI (0–100), nivel relativo y ranking."""
    d = feats[["municipio", "poblacion", "area_km2", "PGA_medio", "pct_suelo_blando", "pendiente_media",
               "latitud", "longitud", "distancia_bucaramanga_km"]].copy()
    d["poblacion"] = d["poblacion"].round().astype("int64")  # habitantes enteros (sin decimales)
    d["amenaza"] = d["PGA_medio"] / d["PGA_medio"].max()
    d["exposicion"] = np.log10(d["poblacion"]) / np.log10(d["poblacion"].max())
    pend = (d["pendiente_media"] / d["pendiente_media"].max()).clip(0, 1)
    d["vulnerabilidad"] = (0.6 * d["pct_suelo_blando"].clip(0, 1) + 0.4 * pend).clip(0, 1)
    crudo = d["amenaza"] * d["exposicion"] * (0.5 + 0.5 * d["vulnerabilidad"])
    d["iri"] = (100 * crudo / crudo.max()).round(1)
    # Niveles relativos: Alto = 10 % superior · Medio = siguiente 25 % · Bajo = resto
    c_alto, c_medio = d["iri"].quantile([0.90, 0.65])
    d["nivel"] = np.where(d["iri"] >= c_alto, "Alto", np.where(d["iri"] >= c_medio, "Medio", "Bajo"))
    d["ranking"] = d["iri"].rank(ascending=False, method="min").astype(int)
    return d.sort_values("iri", ascending=False).reset_index(drop=True)


def resumen_poblacion(iri: pd.DataFrame) -> dict:
    """Cifras de titular: cuánta gente vive en los municipios de nivel Alto y en el tercio con mayor PGA."""
    alto = iri[iri["nivel"] == "Alto"]
    corte_pga = iri["PGA_medio"].quantile(2 / 3)
    pga_alto = iri[iri["PGA_medio"] >= corte_pga]
    total = iri["poblacion"].sum()
    return {
        "n_alto": int(len(alto)), "pob_alto": int(alto["poblacion"].sum()), "frac_alto": float(alto["poblacion"].sum() / total),
        "n_pga": int(len(pga_alto)), "pob_pga": int(pga_alto["poblacion"].sum()), "frac_pga": float(pga_alto["poblacion"].sum() / total),
        "corte_pga": float(corte_pga), "poblacion_total": int(total),
    }
