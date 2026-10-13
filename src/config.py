"""Configuración central de Sismo-Risk Santander.

Concentra rutas, listas de variables y la geometría de referencia del departamento
para que todos los módulos compartan una única fuente de verdad.
"""
from pathlib import Path

from shapely.geometry import LineString, Polygon

# ---------------------------------------------------------------- Rutas
RAIZ = Path(__file__).resolve().parent.parent
DIR_DATOS = RAIZ / "data"
DIR_MODELOS = RAIZ / "models"

RUTA_SISMOS_BRUTO = DIR_DATOS / "sismos_SGC.csv"
RUTA_SISMOS_LIMPIO = DIR_DATOS / "santander_limpio.csv"
# Si el usuario descarga el polígono real del IGAC/DANE lo coloca aquí y se usa en lugar del aproximado
RUTA_POLIGONO_IGAC = DIR_DATOS / "santander_poligono.geojson"
RUTA_MODELO = DIR_MODELOS / "sismo_rf_model.pkl"
RUTA_METRICAS = DIR_MODELOS / "metricas.json"

# ---------------------------------------------------------------- Variables (20 features)
VARS_SISMICAS = ["magnitud", "profundidad_km", "rms", "gap_azimutal", "fases_asociadas"]
VARS_ESPACIALES = ["latitud", "longitud", "elevacion", "distancia_falla_km", "densidad_historica_5km"]
VARS_GEOLOGICAS = ["tipo_suelo", "pendiente_terreno", "PGA_historico", "nivel_freatico", "distancia_cuerpos_agua"]
VARS_EXPOSICION = ["densidad_poblacional", "distancia_zona_urbana", "antiguedad_construcciones", "infraestructura_critica"]

VARIABLES = VARS_SISMICAS + VARS_ESPACIALES + VARS_GEOLOGICAS + VARS_EXPOSICION  # 19 predictoras; con el objetivo suman las 20 variables
COL_CATEGORICA = "tipo_suelo"
COLS_NUMERICAS = [c for c in VARIABLES if c != COL_CATEGORICA]
OBJETIVO = "CATEGORIA_RIESGO"  # 20.ª columna: variable objetivo

TIPOS_SUELO = ["Roca", "Suelo duro", "Suelo intermedio", "Suelo blando", "Aluvial"]
ETIQUETAS_RIESGO = {1: "Bajo", 2: "Medio", 3: "Alto", 4: "Extremo"}
COLORES_RIESGO = {1: "#2ecc71", 2: "#f1c40f", 3: "#e67e22", 4: "#e74c3c"}

# ---------------------------------------------------------------- Geografía
# Bounding box del departamento (lat_min, lat_max, lon_min, lon_max)
BBOX_SANTANDER = (5.70, 8.14, -74.55, -72.35)
CENTRO_SANTANDER = (6.85, -73.45)

# Contorno simplificado (lon, lat) usado cuando no hay shapefile oficial del IGAC.
# Es una aproximación para el prototipo: sustituir por el polígono oficial en data/.
POLIGONO_APROX = Polygon([
    (-74.10, 6.40), (-74.25, 6.90), (-73.95, 7.35), (-73.80, 7.75), (-73.35, 7.85),
    (-73.10, 7.55), (-72.95, 7.30), (-72.70, 7.10), (-72.55, 6.75), (-72.60, 6.35),
    (-72.90, 6.10), (-73.30, 5.90), (-73.65, 5.70), (-73.95, 5.85), (-74.05, 6.10),
])

# Trazo esquemático del sistema de fallas Bucaramanga–Santa Marta / Suárez (lon, lat)
TRAZO_FALLA = LineString([(-72.95, 6.40), (-73.05, 6.80), (-73.12, 7.12), (-73.20, 7.50), (-73.30, 7.85)])

# Principales municipios: nombre -> (lat, lon, población aprox. hab., área km² aprox.)
MUNICIPIOS = {
    "Bucaramanga": (7.119, -73.122, 580_000, 165),
    "Floridablanca": (7.062, -73.087, 270_000, 97),
    "Girón": (7.068, -73.169, 200_000, 475),
    "Piedecuesta": (6.987, -73.049, 165_000, 344),
    "Barrancabermeja": (7.065, -73.854, 205_000, 1_347),
    "San Gil": (6.556, -73.134, 55_000, 150),
    "Socorro": (6.467, -73.260, 32_000, 124),
    "Málaga": (6.703, -72.733, 19_000, 561),
    "Vélez": (6.010, -73.675, 20_000, 270),
    "Barbosa": (5.932, -73.617, 30_000, 183),
    "Cimitarra": (6.315, -73.950, 35_000, 2_600),
    "Sabana de Torres": (7.393, -73.498, 26_000, 1_000),
}

SEMILLA = 42
