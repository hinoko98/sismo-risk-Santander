"""Configuración central de Sismo-Risk Santander.

Concentra rutas, listas de variables y constantes para que todos los módulos
compartan una única fuente de verdad.
"""
from pathlib import Path

# ---------------------------------------------------------------- Rutas
RAIZ = Path(__file__).resolve().parent.parent
DIR_DATOS = RAIZ / "data"
DIR_MODELOS = RAIZ / "models"
DIR_ASSETS = RAIZ / "assets"

RUTA_SISMOS_BRUTO = DIR_DATOS / "sismos_SGC.csv"                # catálogo SGC (Consulta Experta)
RUTA_USGS = DIR_DATOS / "usgs_santander.csv"                       # sismos reales M≥3,5 del USGS (validación posterior a 2018)
RUTA_SITIOS = DIR_DATOS / "sitios_santander.csv"                # variables de terreno por celda de ~2 km
RUTA_MUNICIPIOS = DIR_DATOS / "municipios_santander.geojson"    # polígonos MGN-DANE + población DANE
RUTA_POLIGONO_IGAC = DIR_DATOS / "santander_poligono.geojson"   # unión de municipios
RUTA_SISMOS_LIMPIO = DIR_DATOS / "santander_limpio.csv"         # eventos limpios con municipio (ETL)
RUTA_MUN_FEATURES = DIR_DATOS / "municipios_features.csv"       # variables estáticas por municipio
RUTA_PANEL = DIR_DATOS / "panel_municipio_anio.csv"             # dataset supervisado (municipio-año)
RUTA_MODELO = DIR_MODELOS / "sismo_rf_model.pkl"
RUTA_METRICAS = DIR_MODELOS / "metricas.json"
RUTA_AGENTE_RL = DIR_MODELOS / "agente_rl.pkl"
RUTA_LOGO = DIR_ASSETS / "logo.png"
RUTA_ICONO = DIR_ASSETS / "icono.png"

# ---------------------------------------------------------------- Definición del problema
# Unidad de análisis: municipio-año. Las variables describen lo ocurrido hasta el año t y la
# etiqueta se calcula SOLO con el catálogo del año t+1, de modo que ninguna entrada revela la clase.
ANIO_MIN, ANIO_MAX_ENTRADA = 1994, 2016   # años t con etiqueta (1993 y 2018 son parciales en el catálogo)
ANIO_ACTUAL = 2017                        # último año completo: estado "actual" para predecir 2018
UMBRAL_MEDIO, UMBRAL_ALTO = 3.0, 4.0      # magnitud máxima del año t+1: <3 Bajo · 3–<4 Medio · ≥4 Alto
ETIQUETAS_RIESGO = {1: "Bajo", 2: "Medio", 3: "Alto"}
COLORES_RIESGO = {1: "#2ecc71", 2: "#f1c40f", 3: "#e74c3c"}
OBJETIVO = "CATEGORIA_RIESGO"

# ---------------------------------------------------------------- Variables (panel municipio-año)
VARS_SISMICAS = [  # historial del catálogo SGC hasta el año t
    "n_sismos", "n_sismos_m3", "mag_media", "mag_max", "profundidad_media", "frac_someros",
    "rms_medio", "gap_medio", "fases_medias", "energia_log10",
    "n_m3_prev3", "mag_max_prev3", "tendencia_m3", "anios_desde_m4",
]
VARS_TERRITORIALES = [  # estáticas por municipio
    "latitud", "longitud", "area_km2", "poblacion", "densidad_poblacional",
    "PGA_medio", "elevacion_media", "pendiente_media", "distancia_falla_km", "distancia_cuerpos_agua",
    "pct_suelo_blando", "distancia_bucaramanga_km",
]
COL_CATEGORICA = "suelo_predominante"
VARIABLES = VARS_SISMICAS + VARS_TERRITORIALES + [COL_CATEGORICA]
COLS_NUMERICAS = VARS_SISMICAS + VARS_TERRITORIALES

TIPOS_SUELO = ["B - Roca", "C - Suelo denso / roca blanda", "D - Suelo rígido", "E - Suelo blando"]

# Colores tomados del logo
AZUL_MARCA = "#1B3A5C"
NARANJA_MARCA = "#E8601C"

# ---------------------------------------------------------------- Geografía
BBOX_SANTANDER = (5.70, 8.14, -74.55, -72.35)   # (lat_min, lat_max, lon_min, lon_max)
CENTRO_SANTANDER = (6.85, -73.45)
BUCARAMANGA = (7.119, -73.122)                  # "zona de interés" para distancias

SEMILLA = 42
