"""Generador de un dataset sintético realista (simula SGC + IGAC + DANE).

Se usa cuando no hay CSV real en data/. Reproduce rasgos conocidos de la sismicidad
de Santander: el "Nido de Bucaramanga" (sismos intermedios, 120-180 km de profundidad)
y la sismicidad superficial asociada al sistema de fallas Bucaramanga–Santa Marta.
"""
import numpy as np
import pandas as pd
from shapely.geometry import Point

from .config import (
    BBOX_SANTANDER, MUNICIPIOS, POLIGONO_APROX, SEMILLA, TIPOS_SUELO, TRAZO_FALLA,
)

KM_POR_GRADO = 111.0
NIDO_BUCARAMANGA = (6.82, -73.12)  # lat, lon aproximados del nido sísmico


def _punto_en_poligono(rng: np.random.Generator) -> tuple[float, float]:
    """Muestreo por rechazo dentro del polígono aproximado del departamento."""
    lat_min, lat_max, lon_min, lon_max = BBOX_SANTANDER
    while True:
        lat, lon = rng.uniform(lat_min, lat_max), rng.uniform(lon_min, lon_max)
        if POLIGONO_APROX.contains(Point(lon, lat)):
            return lat, lon


def distancia_falla_km(lat: float, lon: float) -> float:
    """Distancia aproximada (km) al trazo de falla de referencia."""
    return TRAZO_FALLA.distance(Point(lon, lat)) * KM_POR_GRADO


def _datos_municipales(lat: float, lon: float) -> tuple[float, float]:
    """Devuelve (distancia_km al casco urbano más cercano, densidad hab/km² del municipio)."""
    mejor = min(
        MUNICIPIOS.items(),
        key=lambda kv: (kv[1][0] - lat) ** 2 + (kv[1][1] - lon) ** 2,
    )
    _, (mlat, mlon, poblacion, area) = mejor
    dist = float(np.hypot(mlat - lat, mlon - lon) * KM_POR_GRADO)
    return dist, poblacion / area


def generar_dataset(n: int = 3000, seed: int = SEMILLA, frac_fuera: float = 0.04,
                    frac_nulos: float = 0.03) -> pd.DataFrame:
    """Crea `n` eventos con las 19 variables predictoras (sin la categoría de riesgo).

    frac_fuera: proporción de eventos con coordenadas fuera de Santander (para que el
        filtro espacial de la Misión 1 tenga trabajo real).
    frac_nulos: proporción de celdas nulas inyectadas (para ejercitar la imputación).
    """
    rng = np.random.default_rng(seed)
    filas = []
    for _ in range(n):
        en_nido = rng.random() < 0.40
        if en_nido:  # sismos intermedios concentrados en el nido
            lat = rng.normal(NIDO_BUCARAMANGA[0], 0.08)
            lon = rng.normal(NIDO_BUCARAMANGA[1], 0.08)
            prof = rng.normal(150, 15)
        else:  # sismicidad superficial repartida, más densa cerca de la falla
            lat, lon = _punto_en_poligono(rng)
            prof = rng.gamma(2.2, 8.0) + 2
        # Eventos fuera del departamento (se descartarán en la limpieza)
        if rng.random() < frac_fuera:
            lat = rng.uniform(BBOX_SANTANDER[0] - 1.5, BBOX_SANTANDER[1] + 1.5)
            lon = rng.uniform(BBOX_SANTANDER[2] - 1.5, BBOX_SANTANDER[3] + 1.5)
            if POLIGONO_APROX.contains(Point(lon, lat)):
                lon -= 3.0

        dist_falla = distancia_falla_km(lat, lon)
        dist_urbana, dens_pob = _datos_municipales(lat, lon)
        dist_urbana = max(dist_urbana, 0.2)
        # La densidad cae con la distancia al casco urbano (modelo exponencial)
        densidad = float(dens_pob * np.exp(-dist_urbana / 12) * rng.lognormal(0, 0.25))

        # Ley de Gutenberg–Richter: pocas magnitudes grandes
        mag = float(np.clip(2.0 + rng.exponential(0.7), 2.0, 6.8))
        fases = int(np.clip(rng.normal(12 + 4 * mag, 5), 6, 80))
        gap = float(np.clip(rng.normal(150 - 8 * mag, 50), 30, 330))
        rms = float(np.clip(rng.gamma(2.0, 0.18), 0.03, 1.5))

        elevacion = float(np.clip(rng.normal(1100, 700) + (lon + 73.5) * 400, 60, 3800))
        pendiente = float(np.clip(rng.gamma(2.0, 7.0) + elevacion / 400, 0.5, 60))

        # Suelo: más blando en zonas bajas / cerca de agua
        dist_agua = float(rng.gamma(1.8, 4.0))
        p_blando = np.clip(0.55 - elevacion / 4000 - dist_agua / 60, 0.05, 0.6)
        probs = np.array([0.22, 0.28, 0.25, p_blando * 0.6, p_blando * 0.4])
        suelo = rng.choice(TIPOS_SUELO, p=probs / probs.sum())
        nivel_fre = float(np.clip(rng.gamma(2.0, 5.0) - (elevacion / 600), 0.3, 60))

        # PGA histórico (g): atenuación con distancia a falla + amplificación por suelo
        amp = {"Roca": 0.8, "Suelo duro": 1.0, "Suelo intermedio": 1.25,
               "Suelo blando": 1.6, "Aluvial": 1.5}[suelo]
        pga = float(np.clip(0.35 * np.exp(-dist_falla / 40) * amp * rng.lognormal(0, 0.2), 0.01, 0.9))

        antig = float(np.clip(rng.normal(32, 15) + (8 if dens_pob > 1000 else 0), 1, 100))
        infra = int(rng.poisson(0.6 + densidad / 1500))

        filas.append({
            "magnitud": round(mag, 2), "profundidad_km": round(float(np.clip(prof, 1, 200)), 1),
            "rms": round(rms, 3), "gap_azimutal": round(gap, 1), "fases_asociadas": fases,
            "latitud": round(lat, 4), "longitud": round(lon, 4), "elevacion": round(elevacion, 1),
            "distancia_falla_km": round(dist_falla, 2),
            "densidad_historica_5km": 0,  # se calcula luego con los vecinos reales
            "tipo_suelo": suelo, "pendiente_terreno": round(pendiente, 1),
            "PGA_historico": round(pga, 3), "nivel_freatico": round(nivel_fre, 1),
            "distancia_cuerpos_agua": round(dist_agua, 2),
            "densidad_poblacional": round(densidad, 1), "distancia_zona_urbana": round(dist_urbana, 2),
            "antiguedad_construcciones": round(antig, 1), "infraestructura_critica": infra,
        })

    df = pd.DataFrame(filas)
    df["densidad_historica_5km"] = _contar_vecinos_5km(df)

    # Inyección de nulos (solo en columnas numéricas no espaciales)
    candidatas = ["rms", "gap_azimutal", "elevacion", "pendiente_terreno", "nivel_freatico",
                  "distancia_cuerpos_agua", "antiguedad_construcciones", "PGA_historico"]
    mascara = rng.random((len(df), len(candidatas))) < frac_nulos
    for j, col in enumerate(candidatas):
        df.loc[mascara[:, j], col] = np.nan
    return df


def _contar_vecinos_5km(df: pd.DataFrame) -> np.ndarray:
    """Número de eventos históricos en un radio de ~5 km (aprox. planar, vectorizado)."""
    xy = np.column_stack([df["latitud"], df["longitud"]]) * KM_POR_GRADO
    # Se procesa por bloques para no construir una matriz n×n gigante
    conteos = np.zeros(len(df), dtype=int)
    for i in range(0, len(df), 500):
        d = np.linalg.norm(xy[i:i + 500, None, :] - xy[None, :, :], axis=2)
        conteos[i:i + 500] = (d <= 5).sum(axis=1) - 1
    return conteos


def guardar_dataset_bruto(ruta, n: int = 3000) -> pd.DataFrame:
    """Genera y guarda el CSV bruto simulado."""
    df = generar_dataset(n)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta, index=False)
    return df
