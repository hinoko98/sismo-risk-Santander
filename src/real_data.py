"""Descarga de datos REALES (sin simulación) desde fuentes oficiales y abiertas.

Fuentes:
  * SGC  – Catálogo sísmico, Consulta Experta (magnitud, profundidad, RMS, GAP, fases).
           https://bdrsnc.sgc.gov.co/paginas1/catalogo/Consulta_Experta/consultaexperta_2.php
  * SGC  – Amenaza sísmica de Colombia 2020 (PGA, periodo de retorno 475 años, en roca).
           https://amenazasismica.sgc.gov.co/
  * DANE – Proyecciones de población municipal 2018-2042 y Marco Geoestadístico Nacional
           (límites municipales, distribuidos por geoBoundaries, CC BY 4.0).
  * GEM  – Global Active Faults Database (trazas de fallas activas).
  * OSM  – Cuerpos de agua, hospitales/bomberos/policía y cascos urbanos (Overpass API).
  * OpenTopoData – Elevación SRTM 90 m (NASA).

Todo se cachea en data/raw/ para no repetir descargas. Ejecutar:  python -m src.real_data
"""
import io
import json
import logging
import re
import time
import unicodedata
import warnings
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
from shapely.geometry import Point

from .config import (
    BBOX_SANTANDER, DIR_DATOS, RUTA_MUNICIPIOS, RUTA_USGS, RUTA_POLIGONO_IGAC, RUTA_SISMOS_BRUTO, RUTA_SITIOS,
)

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message="Unverified HTTPS request")

DIR_RAW = DIR_DATOS / "raw"
CRS_M = "EPSG:3116"  # MAGNA-SIRGAS / Colombia Bogotá (metros) para distancias y áreas
HEADERS = {"User-Agent": "sismo-risk-santander/1.0 (hackathon CIAI UTSmart 2026)"}
PASO_GRILLA = 0.02  # grados (~2,2 km): resolución de la tabla de sitios

URL_SGC = "https://bdrsnc.sgc.gov.co/paginas1/catalogo/Consulta_Experta/"
URL_PGA = "https://amenazasismica.sgc.gov.co/mapproxy/co18_results_rock_grid/ows"
URL_DANE = "https://www.dane.gov.co/files/censo2018/proyecciones-de-poblacion/Municipal/PPED-AreaMun-2018-2042_VP.xlsx"
URL_ADM2 = "https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/COL/ADM2/geoBoundaries-COL-ADM2_simplified.geojson"
URL_GEM = "https://raw.githubusercontent.com/GEMScienceTools/gem-global-active-faults/master/geojson/gem_active_faults_harmonized.geojson"
URL_OVERPASS = [  # servidores espejo: el público principal se satura con frecuencia
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
URL_ELEVACION = "https://api.opentopodata.org/v1/srtm90m"

ANIO_INI, ANIO_FIN = 1993, 2018  # rango con datos en la Consulta Experta (2019+ devuelve 0 registros)
ANIO_POBLACION = 2025


# ------------------------------------------------------------------ utilidades
def _get(url, reintentos=4, **kw):
    """GET con reintentos y espera creciente; el SGC y Overpass fallan de forma intermitente."""
    kw.setdefault("timeout", 180)
    kw.setdefault("headers", HEADERS)
    kw.setdefault("verify", False)  # el certificado de algunos servidores del SGC es incompleto
    for i in range(reintentos):
        try:
            r = requests.get(url, **kw)
            r.raise_for_status()
            return r
        except requests.RequestException as exc:
            if i == reintentos - 1:
                raise
            log.warning("Reintento %d de %s (%s)", i + 1, url[:70], exc)
            time.sleep(3 * (i + 1))


def _normalizar(txt: str) -> str:
    """Minúsculas sin tildes ni signos, para cruzar nombres de municipios entre fuentes."""
    txt = unicodedata.normalize("NFKD", str(txt)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", txt.lower())


def _overpass(consulta: str, nombre: str) -> dict:
    ruta = DIR_RAW / f"osm_{nombre}.json"
    if ruta.exists():
        return json.loads(ruta.read_text(encoding="utf-8"))
    for i in range(8):
        url = URL_OVERPASS[i % len(URL_OVERPASS)]
        try:
            r = requests.post(url, data={"data": consulta}, headers=HEADERS, timeout=240)
        except requests.RequestException as exc:
            log.warning("Overpass %s (%s) falló: %s", nombre, url, exc)
            continue
        if r.status_code == 200 and r.text.lstrip().startswith("{"):
            ruta.write_text(r.text, encoding="utf-8")
            return r.json()
        log.warning("Overpass %s (%s) -> %s; se prueba otro servidor", nombre, url, r.status_code)
        time.sleep(5 * (i + 1))
    raise RuntimeError(f"Overpass no respondió para {nombre}")


# ------------------------------------------------------------------ 1. Catálogo SGC
def _leer_xlsx_sin_estilos(contenido: bytes) -> pd.DataFrame:
    """Lee la primera hoja de un .xlsx ignorando styles.xml.

    El Excel que genera el SGC trae una hoja de estilos inválida y openpyxl/pandas fallan,
    pero los datos están bien: se leen directamente del XML.
    """
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    z = zipfile.ZipFile(io.BytesIO(contenido))
    textos = [
        "".join(t.text or "" for t in si.iter("{%s}t" % ns["m"]))
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", ns)
    ]
    filas = []
    for fila in ET.fromstring(z.read("xl/worksheets/sheet1.xml")).iter("{%s}row" % ns["m"]):
        valores = {}
        for c in fila.findall("m:c", ns):
            v = c.find("m:v", ns)
            if v is None:
                continue
            col = re.match(r"[A-Z]+", c.get("r")).group()
            valores[col] = textos[int(v.text)] if c.get("t") == "s" else v.text
        filas.append(valores)
    cols = list(filas[0].keys())
    return pd.DataFrame([[f.get(c) for c in cols] for f in filas[1:]], columns=[filas[0][c] for c in cols])


def _descargar_anio_sgc(anio: int) -> pd.DataFrame:
    ruta = DIR_RAW / f"sgc_{anio}.csv"
    if ruta.exists():
        return pd.read_csv(ruta)
    lat0, lat1, lon0, lon1 = BBOX_SANTANDER
    datos = {
        "inicial": f"01/01/{anio}", "final": f"31/12/{anio}", "radiobutton": "cuadrante",
        "longitudStart": lon0, "longitudEnd": lon1, "latitudStart": lat0, "latitudEnd": lat1,
        "latitudCentral": 5, "longitudCentral": -76, "radio": 100,
        "magnitudStart": 0, "magnitudEnd": 9, "magnitudmwStart": -9, "magnitudmwEnd": 9,
        "depthStart": 0, "depthEnd": 700, "rmsStart": 0, "rmsEnd": 10, "gapStart": 0, "gapEnd": 360,
        "eprofmin": 0, "eprofmax": 999, "elongmin": 0, "elongmax": 999, "elatmin": 0, "elatmax": 999,
        "Submit": "Consultar",
    }
    html = requests.post(URL_SGC + "consulsismo3.php", data=datos, headers=HEADERS, timeout=240, verify=False).text
    if " 0 registros" in html or "descargar_exel_experta" not in html:
        df = pd.DataFrame()
    else:
        enlace = re.search(r"descargar_exel_experta\.php\?[^\"' >]+", html).group().replace("&amp;", "&")
        df = _leer_xlsx_sin_estilos(_get(URL_SGC + enlace, timeout=300).content)
    df.to_csv(ruta, index=False)
    return df


def descargar_catalogo_sgc() -> pd.DataFrame:
    """Catálogo SGC 1993-2018 para el bounding box de Santander -> data/sismos_SGC.csv."""
    DIR_RAW.mkdir(parents=True, exist_ok=True)
    partes = []
    for anio in range(ANIO_INI, ANIO_FIN + 1):
        d = _descargar_anio_sgc(anio)
        log.info("SGC %d: %d eventos", anio, len(d))
        if len(d):
            partes.append(d)
    df = pd.concat(partes, ignore_index=True)
    df = df.rename(columns={
        "FECHA": "fecha", "HORA_UTC": "hora_utc", "LATITUD (grados)": "latitud",
        "LONGITUD (grados)": "longitud", "PROFUNDIDAD (Km)": "profundidad_km",
        "MAGNITUD Ml": "magnitud_ml", "MAGNITUD Mw": "magnitud_mw", "DEPARTAMENTO": "departamento",
        "MUNICIPIO": "municipio", "# FASES": "fases_asociadas", "RMS (Seg)": "rms",
        "GAP (grados)": "gap_azimutal", "ESTADO": "estado",
    }).drop(columns=["ERROR LATITUD (Km)", "ERROR LONGITUD (Km)", "ERROR PROFUNDIDAD (Km)"], errors="ignore")
    for c in ["latitud", "longitud", "profundidad_km", "magnitud_ml", "magnitud_mw", "fases_asociadas", "rms", "gap_azimutal"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.to_csv(RUTA_SISMOS_BRUTO, index=False)
    return df


def descargar_usgs() -> pd.DataFrame:
    """Sismos reales del USGS (M≥3,5, bounding box de Santander) hasta hoy.

    Sirve para validar la proyección con eventos posteriores a marzo de 2018, que es donde termina la
    Consulta Experta del SGC. Las magnitudes del USGS (mayormente mb) no son idénticas a las del SGC (Ml).
    """
    lat0, lat1, lon0, lon1 = BBOX_SANTANDER
    r = _get("https://earthquake.usgs.gov/fdsnws/event/1/query", params={
        "format": "csv", "minlatitude": lat0, "maxlatitude": lat1, "minlongitude": lon0, "maxlongitude": lon1,
        "starttime": "1993-01-01", "endtime": pd.Timestamp.today().strftime("%Y-%m-%d"),
        "minmagnitude": 3.5, "orderby": "time-asc"})
    RUTA_USGS.write_bytes(r.content)
    return pd.read_csv(RUTA_USGS)


# ------------------------------------------------------------------ 2. Límites (DANE MGN) y población (DANE)
def construir_municipios() -> gpd.GeoDataFrame:
    """Municipios de Santander con polígono (MGN-DANE), área km² y población proyectada (DANE)."""
    ruta = RUTA_MUNICIPIOS
    if ruta.exists():
        return gpd.read_file(ruta)

    # Población: hoja PobMunicipalx área, departamento 68 (Santander), total del año elegido
    xlsx = _get(URL_DANE).content
    libro = pd.ExcelFile(io.BytesIO(xlsx))
    hoja = next(h for h in libro.sheet_names if h.startswith("PobMunicipal"))  # el nombre lleva tilde
    pob = libro.parse(hoja, header=7)
    pob.columns = [str(c).strip() for c in pob.columns]
    col_anio = next(c for c in pob.columns if c.upper().startswith("A") and c.upper().endswith("O"))
    col_area = next(c for c in pob.columns if "REA" in c.upper())
    pob = pob[(pob["DP"].astype(str).str.strip().isin(["68", "68.0"])) & (pob[col_anio] == ANIO_POBLACION)
              & (pob[col_area].astype(str).str.lower() == "total")].copy()
    pob["clave"] = pob["DPMP"].map(_normalizar)  # DPMP = nombre del municipio (MPIO es el código)

    # Polígonos: se escoge, por nombre, el municipio cuyo centroide cae en el bbox de Santander
    adm2 = gpd.read_file(_descargar(URL_ADM2, "adm2_col.geojson")).to_crs(4326)
    adm2["clave"] = adm2["shapeName"].map(_normalizar)
    lat0, lat1, lon0, lon1 = BBOX_SANTANDER
    cen = adm2.geometry.representative_point()
    adm2 = adm2[(cen.y.between(lat0, lat1)) & (cen.x.between(lon0, lon1))]
    adm2 = adm2[adm2["clave"].isin(pob["clave"])].drop_duplicates("clave")

    mun = adm2.merge(pob[["clave", "DPMP", "TOTAL"]], on="clave")
    faltan = set(pob["clave"]) - set(mun["clave"])
    if faltan:
        log.warning("Municipios DANE sin polígono: %s", sorted(pob.loc[pob["clave"].isin(faltan), "DPMP"]))
    mun["area_km2"] = mun.to_crs(CRS_M).area / 1e6
    mun["poblacion"] = pd.to_numeric(mun["TOTAL"])
    mun["densidad_poblacional"] = mun["poblacion"] / mun["area_km2"]
    mun = mun.rename(columns={"DPMP": "municipio"})[["municipio", "poblacion", "area_km2", "densidad_poblacional", "geometry"]]
    mun.to_file(ruta, driver="GeoJSON")
    return mun


def _descargar(url: str, nombre: str):
    ruta = DIR_RAW / nombre
    if not ruta.exists():
        ruta.write_bytes(_get(url).content)
    return ruta


def guardar_poligono_santander(mun: gpd.GeoDataFrame):
    """Contorno del departamento = unión de sus municipios (Marco Geoestadístico DANE)."""
    gpd.GeoDataFrame(geometry=[mun.geometry.union_all()], crs=4326).to_file(RUTA_POLIGONO_IGAC, driver="GeoJSON")


# ------------------------------------------------------------------ 3. Variables de sitio (grilla de ~2 km)
def _crear_grilla(poligono) -> pd.DataFrame:
    """Centros de celda de PASO_GRILLA grados que caen dentro de Santander."""
    import shapely
    lat0, lat1, lon0, lon1 = BBOX_SANTANDER
    lats = np.arange(lat0 + PASO_GRILLA / 2, lat1, PASO_GRILLA)
    lons = np.arange(lon0 + PASO_GRILLA / 2, lon1, PASO_GRILLA)
    la, lo = np.meshgrid(lats, lons, indexing="ij")
    df = pd.DataFrame({"lat_c": la.ravel().round(4), "lon_c": lo.ravel().round(4)})
    return df[shapely.contains_xy(poligono, df["lon_c"], df["lat_c"])].reset_index(drop=True)


def _guardar_cache_elevacion(ruta, cache):
    pd.DataFrame([(a, b, z) for (a, b), z in cache.items()], columns=["lat", "lon", "z"]).to_csv(ruta, index=False)


def _elevaciones(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Elevación (m) SRTM 90 m vía OpenTopoData, en bloques de 100 puntos (máx. 1 consulta/s) con caché."""
    ruta = DIR_RAW / "elevacion.csv"
    cache = pd.read_csv(ruta).set_index(["lat", "lon"])["z"].to_dict() if ruta.exists() else {}
    clave = list(zip(np.round(lats, 4), np.round(lons, 4)))
    pendientes = [k for k in dict.fromkeys(clave) if k not in cache]
    for i in range(0, len(pendientes), 100):
        bloque = pendientes[i:i + 100]
        params = {"locations": "|".join(f"{a},{b}" for a, b in bloque)}
        for intento in range(8):
            r = requests.get(URL_ELEVACION, params=params, headers=HEADERS, timeout=60)
            if r.status_code == 200:
                cache.update({k: res["elevation"] for k, res in zip(bloque, r.json()["results"])})
                break
            time.sleep(10 * (intento + 1))  # límite de uso de la API pública: esperar y reintentar
        else:
            _guardar_cache_elevacion(ruta, cache)
            raise RuntimeError(f"OpenTopoData no respondió: {r.status_code} {r.text[:120]}")
        time.sleep(1.1)
        if (i // 100) % 20 == 0:
            _guardar_cache_elevacion(ruta, cache)
            log.info("Elevación: %d/%d puntos", min(i + 100, len(pendientes)), len(pendientes))
    _guardar_cache_elevacion(ruta, cache)
    z = np.array([cache[k] if cache[k] is not None else np.nan for k in clave], dtype=float)
    return np.where(np.isnan(z), np.nanmedian(z), z)  # huecos del DEM -> mediana regional


def _pendiente_y_suelo(df: pd.DataFrame) -> pd.DataFrame:
    """Elevación, pendiente (°) y clase de sitio NEHRP estimada.

    La pendiente se calcula por diferencias centrales con vecinos a ±0,01° (~1,1 km). La clase de
    sitio usa el proxy topográfico de Wald & Allen (2007) para regiones tectónicamente activas:
    pendiente (m/m) -> Vs30 -> clase NEHRP. Es una ESTIMACIÓN regional, no una microzonificación.
    """
    d = 0.01
    la, lo = df["lat_c"].to_numpy(), df["lon_c"].to_numpy()
    z = _elevaciones(np.concatenate([la, la + d, la - d, la, la]), np.concatenate([lo, lo, lo, lo + d, lo - d]))
    n = len(df)
    z0, zn, zs, ze, zo = (z[i * n:(i + 1) * n] for i in range(5))
    dy = 2 * d * 111_320
    dx = 2 * d * 111_320 * np.cos(np.radians(la))
    grad = np.hypot((zn - zs) / dy, (ze - zo) / dx)  # m/m
    df["elevacion"] = z0
    df["pendiente_terreno"] = np.degrees(np.arctan(grad)).round(2)
    # Umbrales (m/m) del proxy de Wald & Allen 2007 (tabla "active tectonic")
    df["tipo_suelo"] = np.select(
        [grad >= 0.138, grad >= 0.0022, grad >= 1e-4],
        ["B - Roca", "C - Suelo denso / roca blanda", "D - Suelo rígido"],
        default="E - Suelo blando",
    )
    return df


def _distancia_a_geometrias(puntos_m: gpd.GeoSeries, geoms_m) -> np.ndarray:
    """Distancia (m) de cada punto a la geometría más cercana (índice espacial STRtree)."""
    from shapely import STRtree
    arbol = STRtree(list(geoms_m))
    _, dist = arbol.query_nearest(list(puntos_m), return_distance=True, all_matches=False)
    return dist


def _distancia_fallas(puntos_m: gpd.GeoSeries) -> np.ndarray:
    ruta = DIR_RAW / "fallas_gem.geojson"
    if not ruta.exists():
        f = gpd.read_file(_descargar(URL_GEM, "gem_global.geojson"))
        lat0, lat1, lon0, lon1 = BBOX_SANTANDER
        f = f.cx[lon0 - 1.5:lon1 + 1.5, lat0 - 1.5:lat1 + 1.5]  # margen: una falla fuera del bbox también influye
        f.to_file(ruta, driver="GeoJSON")
    fallas = gpd.read_file(ruta).to_crs(CRS_M)
    return _distancia_a_geometrias(puntos_m, fallas.geometry) / 1000


def _distancia_agua(puntos_m: gpd.GeoSeries) -> np.ndarray:
    from shapely.geometry import LineString
    lat0, lat1, lon0, lon1 = BBOX_SANTANDER
    caja = f"({lat0},{lon0},{lat1},{lon1})"
    datos = _overpass(
        f'[out:json][timeout:200];(way["waterway"~"river|canal"]{caja};way["natural"="water"]{caja};);out geom;', "agua"
    )
    lineas = [LineString([(p["lon"], p["lat"]) for p in e["geometry"]])
              for e in datos["elements"] if e.get("geometry") and len(e["geometry"]) > 1]
    agua = gpd.GeoSeries(lineas, crs=4326).to_crs(CRS_M)
    return _distancia_a_geometrias(puntos_m, agua) / 1000


def _osm_puntos(filtro: str, nombre: str, con_ways: bool = True) -> np.ndarray:
    """Coordenadas (lon, lat) de nodos/ways OSM que cumplen el filtro (ways -> centro)."""
    lat0, lat1, lon0, lon1 = BBOX_SANTANDER
    caja = f"({lat0},{lon0},{lat1},{lon1})"
    q = (f"[out:json][timeout:120];(node{filtro}{caja};way{filtro}{caja};);out center;" if con_ways
         else f"[out:json][timeout:120];node{filtro}{caja};out;")
    pts = []
    for e in _overpass(q, nombre)["elements"]:
        c = e if "lat" in e else e.get("center")
        if c:
            pts.append((c["lon"], c["lat"]))
    return np.array(pts)


def _a_metros(lon_lat: np.ndarray) -> np.ndarray:
    g = gpd.GeoSeries(gpd.points_from_xy(lon_lat[:, 0], lon_lat[:, 1]), crs=4326).to_crs(CRS_M)
    return np.column_stack([g.x, g.y])


def _pga_sgc(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """PGA (g) en roca, T=475 años, modelo SGC 2020, interpolado bilinealmente.

    El servicio publica una malla de 0,1° (nodos en coordenadas redondas); se consulta cada nodo
    con GetFeatureInfo y se interpola entre ellos.
    """
    ruta = DIR_RAW / "pga_sgc_475.csv"
    if ruta.exists():
        malla = pd.read_csv(ruta)
    else:
        lats = np.round(np.arange(5.5, 8.4, 0.1), 1)
        lons = np.round(np.arange(-74.7, -72.2, 0.1), 1)

        def consulta(par):
            la, lo = par
            h = 0.005
            p = dict(service="WMS", version="1.1.1", request="GetFeatureInfo", layers="hmap-mean-pga-475yr",
                     query_layers="hmap-mean-pga-475yr", styles="", srs="EPSG:4326",
                     bbox=f"{lo - h},{la - h},{lo + h},{la + h}", width=101, height=101, x=50, y=50,
                     feature_count=1, info_format="application/json")
            for _ in range(3):
                try:
                    f = requests.get(URL_PGA, params=p, headers=HEADERS, verify=False, timeout=60).json()["features"]
                    return (la, lo, f[0]["properties"]["PGA-0.002107"] if f else np.nan)
                except (requests.RequestException, ValueError):
                    time.sleep(2)
            return (la, lo, np.nan)

        with ThreadPoolExecutor(6) as ex:
            malla = pd.DataFrame(list(ex.map(consulta, [(a, b) for a in lats for b in lons])),
                                 columns=["lat", "lon", "pga"])
        malla.to_csv(ruta, index=False)
    valido = malla.dropna()
    # Los nodos sin dato (fuera de la cobertura del modelo) se rellenan con el vecino válido más cercano
    piv = malla.pivot(index="lat", columns="lon", values="pga").sort_index().sort_index(axis=1)
    arbol = cKDTree(valido[["lat", "lon"]].to_numpy())
    for la in piv.index:
        for lo in piv.columns:
            if np.isnan(piv.loc[la, lo]):
                piv.loc[la, lo] = valido.iloc[arbol.query([la, lo])[1]]["pga"]
    f = RegularGridInterpolator((piv.index.values, piv.columns.values), piv.values, bounds_error=False, fill_value=None)
    return f(np.column_stack([lat, lon]))


def construir_sitios() -> pd.DataFrame:
    """Tabla de variables de sitio por celda de ~2 km (todas derivadas de fuentes reales)."""
    municipios = construir_municipios()
    sitios = _crear_grilla(municipios.geometry.union_all())
    log.info("Celdas de sitio dentro de Santander: %d", len(sitios))

    sitios = _pendiente_y_suelo(sitios)
    pts = gpd.GeoSeries(gpd.points_from_xy(sitios["lon_c"], sitios["lat_c"]), crs=4326).to_crs(CRS_M)
    sitios["distancia_falla_km"] = _distancia_fallas(pts).round(2)
    sitios["distancia_cuerpos_agua"] = _distancia_agua(pts).round(2)
    sitios["PGA_historico"] = _pga_sgc(sitios["lat_c"].to_numpy(), sitios["lon_c"].to_numpy()).round(4)

    xy = np.column_stack([pts.x, pts.y])
    poblados = _a_metros(_osm_puntos('["place"~"^(city|town|village)$"]', "poblados", con_ways=False))
    sitios["distancia_zona_urbana"] = (cKDTree(poblados).query(xy)[0] / 1000).round(2)
    # PENDIENTE: infraestructura crítica (hospitales, bomberos, policía) desde OSM; Overpass estaba saturado.
    # Se puede reactivar con _osm_puntos('["amenity"~"^(hospital|clinic|fire_station|police)$"]', "infraestructura").

    # Municipio y densidad poblacional (DANE) por cruce espacial; si cae fuera de polígonos, el más cercano
    pg = gpd.GeoDataFrame(sitios[["lat_c"]], geometry=gpd.points_from_xy(sitios["lon_c"], sitios["lat_c"]), crs=4326)
    union = gpd.sjoin_nearest(pg.to_crs(CRS_M), municipios.to_crs(CRS_M), how="left")
    union = union[~union.index.duplicated()]
    sitios["municipio"] = union["municipio"].to_numpy()
    sitios["densidad_poblacional"] = union["densidad_poblacional"].round(2).to_numpy()

    sitios.insert(0, "celda_id", np.arange(len(sitios)))
    sitios.to_csv(RUTA_SITIOS, index=False)
    return sitios


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    DIR_RAW.mkdir(parents=True, exist_ok=True)
    cat = descargar_catalogo_sgc()
    print("Catálogo SGC:", cat.shape, cat["fecha"].min(), "->", cat["fecha"].max())
    mun = construir_municipios()
    print("Municipios:", len(mun), "población:", int(mun["poblacion"].sum()))
    guardar_poligono_santander(mun)
    print("USGS:", len(descargar_usgs()), "eventos")
    sit = construir_sitios()
    print("Sitios:", sit.shape)
    print(sit.describe().T.to_string())
