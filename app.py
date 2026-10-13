"""Sismo-Risk Santander · dashboard Streamlit con datos reales (SGC, DANE, IGAC/MGN, GEM, OSM).

Cubre los tres retos: supervisado (Random Forest), no supervisado (K-Means) y refuerzo (Q-learning).
Ejecutar con:  streamlit run app.py
"""
import json
import logging

import folium
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from folium.plugins import HeatMap, MarkerCluster
from shapely.geometry import Point, shape
from streamlit_folium import st_folium

from src.config import (
    ANIO_ACTUAL, AZUL_MARCA, CENTRO_SANTANDER, COLORES_RIESGO, ETIQUETAS_RIESGO, NARANJA_MARCA, OBJETIVO,
    RUTA_AGENTE_RL, RUTA_ICONO, RUTA_USGS, RUTA_LOGO, RUTA_MODELO, RUTA_MUNICIPIOS, RUTA_PANEL, RUTA_SISMOS_LIMPIO,
    UMBRAL_ALTO, UMBRAL_MEDIO, VARIABLES, VARS_SISMICAS, VARS_TERRITORIALES,
)
from src.model_training import entrenar, predecir

logging.basicConfig(level=logging.INFO)
st.set_page_config(page_title="Sismo-Risk Santander", page_icon=str(RUTA_ICONO) if RUTA_ICONO.exists() else "🌋", layout="wide")
COLOR_ETQ = {ETIQUETAS_RIESGO[k]: v for k, v in COLORES_RIESGO.items()}
ORDEN_ETQ = list(ETIQUETAS_RIESGO.values())

st.markdown(
    f"""<style>
    h1, h2, h3 {{ color: {AZUL_MARCA}; }}
    div[data-testid="stMetricValue"] {{ color: {AZUL_MARCA}; }}
    .stTabs [aria-selected="true"] {{ color: {NARANJA_MARCA}; }}
    </style>""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ Carga con caché y tolerancia a fallos
@st.cache_data(show_spinner="Cargando y limpiando datos reales…")
def cargar_datos() -> tuple[pd.DataFrame | None, pd.DataFrame | None, dict, str | None]:
    """Devuelve (panel, eventos, informe_etl, error). Nunca lanza: ante un fallo retorna el mensaje."""
    try:
        if RUTA_PANEL.exists() and RUTA_SISMOS_LIMPIO.exists():
            panel = pd.read_csv(RUTA_PANEL)
            if set(VARIABLES + [OBJETIVO, "anio", "municipio"]).issubset(panel.columns):
                return panel, pd.read_csv(RUTA_SISMOS_LIMPIO), {}, None
        from src.data_processing import ejecutar_etl  # import diferido: geopandas solo si hay que regenerar
        panel, informe = ejecutar_etl()
        return panel, pd.read_csv(RUTA_SISMOS_LIMPIO), informe, None
    except Exception as exc:
        logging.exception("Fallo al cargar datos")
        return None, None, {}, f"{type(exc).__name__}: {exc}"


@st.cache_data(show_spinner=False)
def cargar_municipios_geo() -> dict | None:
    try:
        return json.loads(RUTA_MUNICIPIOS.read_text(encoding="utf-8"))
    except Exception:
        logging.exception("No se pudo leer los municipios")
        return None


@st.cache_resource(show_spinner="Cargando modelo…")
def cargar_modelo(_panel: pd.DataFrame | None) -> tuple[dict | None, str | None]:
    """Carga el .pkl; si no existe (o es incompatible con esta versión de scikit-learn) reentrena."""
    try:
        if RUTA_MODELO.exists():
            try:
                return joblib.load(RUTA_MODELO), None
            except Exception:
                logging.warning("Modelo ilegible; se reentrena.")
        if _panel is None:
            return None, "No hay datos para entrenar el modelo."
        entrenar(_panel)
        return joblib.load(RUTA_MODELO), None
    except Exception as exc:
        logging.exception("Fallo al cargar el modelo")
        return None, f"{type(exc).__name__}: {exc}"


@st.cache_data(show_spinner="Agrupando municipios…")
def cargar_agrupamiento(k: int) -> tuple[pd.DataFrame | None, dict, str | None]:
    try:
        from src.config import RUTA_MUN_FEATURES
        from src.unsupervised import agregar_municipios, agrupar
        agg = agregar_municipios(pd.read_csv(RUTA_SISMOS_LIMPIO), pd.read_csv(RUTA_MUN_FEATURES))
        tabla, diag = agrupar(agg, k)
        return tabla, diag, None
    except Exception as exc:
        logging.exception("Fallo en el agrupamiento")
        return None, {}, f"{type(exc).__name__}: {exc}"


@st.cache_resource(show_spinner="Cargando agente de refuerzo…")
def cargar_agente(_panel: pd.DataFrame | None) -> tuple[dict | None, str | None]:
    try:
        if RUTA_AGENTE_RL.exists():
            return joblib.load(RUTA_AGENTE_RL), None
        from src.reinforcement import entrenar_y_guardar
        return entrenar_y_guardar(_panel), None
    except Exception as exc:
        logging.exception("Fallo con el agente RL")
        return None, f"{type(exc).__name__}: {exc}"


@st.cache_data(show_spinner="Calculando proyección…")
def cargar_proyeccion() -> tuple[dict | None, str | None]:
    """Tasas de Poisson por municipio, proyección a 2027 y validación con sismos reales del USGS."""
    try:
        import geopandas as gpd
        from src.proyeccion import clases_anuales, estimar_tasas, proyectar, validar
        mun = gpd.read_file(RUTA_MUNICIPIOS)
        tasas = clases_anuales(estimar_tasas(pd.read_csv(RUTA_SISMOS_LIMPIO), sorted(mun["municipio"])))
        val = validar(tasas, pd.read_csv(RUTA_USGS), mun) if RUTA_USGS.exists() else None
        return {"tasas": tasas, "proy": proyectar(tasas, 2027), "val": val}, None
    except Exception as exc:
        logging.exception("Fallo en la proyección")
        return None, f"{type(exc).__name__}: {exc}"


def badge(clase: int) -> str:
    return (f"<span style='background:{COLORES_RIESGO[clase]};color:#fff;padding:4px 12px;"
            f"border-radius:12px;font-weight:600'>{ETIQUETAS_RIESGO[clase]}</span>")


def predecir_todos(artefacto: dict, actual: pd.DataFrame) -> pd.DataFrame:
    """Clase esperada para el año siguiente en todos los municipios (estado del último año completo)."""
    X = actual[artefacto["variables"]]
    clases = artefacto["modelo"].predict(X)
    return actual[["municipio"]].assign(clase=clases.astype(int), riesgo=[ETIQUETAS_RIESGO[int(c)] for c in clases])


# ------------------------------------------------------------------ Arranque
panel, eventos, informe_etl, err_datos = cargar_datos()
geo_mun = cargar_municipios_geo()
artefacto, err_modelo = cargar_modelo(panel)

if RUTA_LOGO.exists():
    st.logo(str(RUTA_LOGO), size="large")
    st.image(str(RUTA_LOGO), width=360)
else:
    st.title("🌋 Sismo-Risk Santander")
st.caption("Gestión del riesgo sísmico · Hackathon CIAI UTSmart 2026 · datos reales: SGC · DANE · IGAC/MGN · GEM · OpenStreetMap")

if err_datos or panel is None:
    st.error("No fue posible cargar los datos reales. Ejecuta `python -m src.real_data` y luego "
             f"`python -m src.data_processing` (requieren internet) y recarga la página. Detalle: {err_datos}")
    st.stop()
if err_modelo:
    st.warning(f"El simulador no está disponible: {err_modelo}")

actual = panel[panel["anio"] == ANIO_ACTUAL].reset_index(drop=True)
etiquetado = panel.dropna(subset=[OBJETIVO]).copy()
etiquetado["riesgo_etiqueta"] = etiquetado[OBJETIVO].astype(int).map(ETIQUETAS_RIESGO)
polys = {f["properties"]["municipio"]: shape(f["geometry"]) for f in geo_mun["features"]} if geo_mun else {}
if "predicciones" not in st.session_state:
    st.session_state.predicciones = []


def municipio_en(lat: float, lon: float) -> str | None:
    p = Point(lon, lat)
    return next((m for m, g in polys.items() if g.contains(p)), None)


# ------------------------------------------------------------------ Sidebar: simulador
with st.sidebar:
    st.header("🎛️ Simulador de riesgo")
    st.caption(f"Elige un punto de Santander y un escenario para el último año. El modelo estima el nivel de riesgo "
               f"sísmico del municipio para el año siguiente. Las variables del terreno y la población se toman de las fuentes oficiales.")
    with st.form("form_simulador"):
        lat = st.number_input("Latitud", 5.70, 8.14, 7.119, 0.001, format="%.4f")
        lon = st.number_input("Longitud", -74.55, -72.35, -73.122, 0.001, format="%.4f")
        mag = st.slider("Magnitud máxima del año en curso", 1.0, 7.5, 4.0, 0.1)
        n3 = st.slider("Sismos M≥3 en el año en curso", 0, 400, 10)
        prof = st.slider("Profundidad media (km)", 1.0, 200.0, 30.0, 1.0,
                         help="El Nido de Bucaramanga produce sismos a ~150 km de profundidad.")
        enviar = st.form_submit_button("Predecir riesgo", width="stretch")

    if enviar:
        try:
            mun = municipio_en(lat, lon)
            if mun is None:
                st.error("Las coordenadas quedan fuera del departamento de Santander.")
            elif artefacto is None:
                st.error("Modelo no disponible.")
            else:
                fila = actual[actual["municipio"] == mun].iloc[0].to_dict()
                fila.update({"mag_max": mag, "n_sismos_m3": n3, "profundidad_media": prof,
                             "n_sismos": max(fila["n_sismos"], n3), "frac_someros": float(prof < 30),
                             "anios_desde_m4": 0 if mag >= 4 else fila["anios_desde_m4"]})
                clase, probs = predecir(artefacto, fila)
                st.session_state.predicciones.append({"lat": lat, "lon": lon, "mag": mag, "clase": clase, "municipio": mun})
                st.session_state.ultima = (clase, probs, mun, fila)
        except Exception as exc:
            st.error(f"No se pudo calcular la predicción: {exc}")

    if "ultima" in st.session_state:
        clase, probs, mun, _ = st.session_state.ultima
        st.markdown(f"**{mun}:** riesgo esperado {badge(clase)}", unsafe_allow_html=True)
        for c, p in probs.items():
            st.progress(p, text=f"{ETIQUETAS_RIESGO[c]}: {p:.0%}")
    if st.session_state.predicciones and st.button("Borrar predicciones del mapa"):
        st.session_state.predicciones = []
        st.session_state.pop("ultima", None)
        st.rerun()

# ------------------------------------------------------------------ Tabs
tab_eda, tab_mapa, tab_sup, tab_uns, tab_rl, tab_proy = st.tabs(
    ["📊 Exploración (EDA)", "🗺️ Mapa de riesgo", "🎯 Supervisado", "🧩 No supervisado", "🤖 Refuerzo", "🔭 Proyección 2027"])

with tab_eda:
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Sismos SGC en Santander", f"{len(eventos):,}")
    k2.metric("Periodo", f"{int(eventos['anio'].min())}–{int(eventos['anio'].max())}")
    k3.metric("Magnitud máxima", f"{eventos['magnitud'].max():.1f}")
    k4.metric("Municipios", f"{panel['municipio'].nunique()}")
    k5.metric("Variables por municipio-año", f"{len(VARIABLES)}")
    if informe_etl:
        st.info(f"ETL: {informe_etl['filas_brutas']:,} eventos del SGC → {informe_etl['filas_limpias']:,} dentro de Santander "
                f"({informe_etl['descartados_fuera_santander']:,} fuera del departamento y "
                f"{informe_etl['descartados_sin_magnitud_o_duplicados']:,} sin magnitud/duplicados descartados; "
                f"{informe_etl['nulos_imputados']:,} nulos imputados con la mediana).")
    st.markdown(f"""**Definición de la etiqueta (reproducible).** Para cada municipio y año *t*, la categoría de riesgo es la **magnitud máxima
registrada en el año *t+1***: **Bajo** si < {UMBRAL_MEDIO}, **Medio** si {UMBRAL_MEDIO}–<{UMBRAL_ALTO}, **Alto** si ≥ {UMBRAL_ALTO}. Todas las variables de
entrada se calculan con datos hasta el año *t*, por lo que ninguna revela la clase.""")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(eventos, x="magnitud", nbins=40, log_y=True, title="Distribución de sismos por magnitud",
                           color_discrete_sequence=[NARANJA_MARCA])
        fig.update_layout(yaxis_title="Eventos (escala log)")
        st.plotly_chart(fig, width="stretch")
    with c2:
        por_anio = eventos.groupby("anio").agg(total=("magnitud", "size"), m3=("magnitud", lambda s: (s >= 3).sum())).reset_index()
        fig = px.line(por_anio.melt("anio", var_name="Serie", value_name="Eventos").replace({"total": "Todos", "m3": "M≥3"}),
                      x="anio", y="Eventos", color="Serie", title="Sismos por año (el total sube por mejor cobertura de la red)",
                      color_discrete_sequence=[AZUL_MARCA, NARANJA_MARCA])
        st.plotly_chart(fig, width="stretch")
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(px.histogram(eventos, x="profundidad_km", nbins=60, title="Profundidad (km): sismos someros y Nido de Bucaramanga",
                                     color_discrete_sequence=[AZUL_MARCA]), width="stretch")
    with c4:
        cl = etiquetado["riesgo_etiqueta"].value_counts().reindex(ORDEN_ETQ).reset_index()
        cl.columns = ["Riesgo", "Municipio-años"]
        st.plotly_chart(px.bar(cl, x="Riesgo", y="Municipio-años", color="Riesgo", title="Clases de riesgo en el dataset (desbalanceadas)",
                               color_discrete_map=COLOR_ETQ), width="stretch")
    c5, c6 = st.columns(2)
    with c5:
        top = eventos["municipio"].value_counts().head(12).sort_values().reset_index()
        top.columns = ["Municipio", "Eventos"]
        st.plotly_chart(px.bar(top, x="Eventos", y="Municipio", orientation="h", title="Municipios con más sismos",
                               color_discrete_sequence=[NARANJA_MARCA]), width="stretch")
    with c6:
        st.plotly_chart(px.scatter(actual, x="poblacion", y="PGA_medio", size="area_km2", hover_name="municipio", log_x=True,
                                   title="Población vs. PGA (amenaza SGC 475 años)", color_discrete_sequence=[AZUL_MARCA]),
                        width="stretch")

    st.subheader("Dataset limpio (municipio-año)")
    st.dataframe(panel.drop(columns=["riesgo_etiqueta"], errors="ignore").head(3000), width="stretch", height=320)

    with st.expander("📚 Fuentes de datos (todas reales)"):
        st.markdown("""
| Variable(s) | Fuente |
|---|---|
| Actividad sísmica: frecuencia, magnitud, profundidad, RMS, GAP, fases, energía | [SGC · Catálogo sísmico, Consulta Experta](https://bdrsnc.sgc.gov.co/paginas1/catalogo/Consulta_Experta/consultaexperta_2.php) (jun-1993 a mar-2018; el servicio no entrega fechas posteriores) |
| PGA_medio | [SGC · Amenaza sísmica de Colombia 2020](https://amenazasismica.sgc.gov.co/) — PGA en roca, retorno 475 años (malla 0,1°) |
| poblacion, densidad_poblacional | [DANE · Proyecciones de población 2025](https://www.dane.gov.co/index.php/estadisticas-por-tema/demografia-y-poblacion/proyecciones-de-poblacion) |
| area_km2, límites, municipio de cada evento | Marco Geoestadístico Nacional (DANE) vía geoBoundaries; referencia oficial: [IGAC datos abiertos](https://www.igac.gov.co/datos-abiertos/datos-abiertos-geoespaciales) |
| elevacion_media, pendiente_media | SRTM 90 m (NASA) vía OpenTopoData |
| suelo_predominante, pct_suelo_blando | Clase NEHRP **estimada** con el proxy topográfico de Wald & Allen (2007) a partir de la pendiente |
| distancia_falla_km | GEM Global Active Faults Database |
| distancia_cuerpos_agua | OpenStreetMap (ríos y cuerpos de agua) |
| distancia_bucaramanga_km | Calculada desde el centroide municipal a Bucaramanga (zona de interés) |
""")

with tab_mapa:
    try:
        capa = st.radio("Colorear municipios por", ["Riesgo esperado para el año siguiente (Random Forest)", "Grupo (K-Means)"],
                        horizontal=True)
        mapa = folium.Map(location=CENTRO_SANTANDER, zoom_start=8, tiles="OpenStreetMap")
        color_mun = {}
        if capa.startswith("Riesgo") and artefacto is not None:
            pr = predecir_todos(artefacto, actual).set_index("municipio")
            color_mun = {m: (COLORES_RIESGO[int(r["clase"])], r["riesgo"]) for m, r in pr.iterrows()}
        else:
            tabla, _, _ = cargar_agrupamiento(3)
            if tabla is not None:
                color_mun = {r["municipio"]: (COLOR_ETQ[r["grupo"]], r["grupo"]) for _, r in tabla.iterrows()}
        if geo_mun and color_mun:
            folium.GeoJson(
                geo_mun, name="Municipios",
                style_function=lambda f: {"color": "#555", "weight": 0.8, "fillOpacity": 0.45,
                                          "fillColor": color_mun.get(f["properties"]["municipio"], ("#bbb", ""))[0]},
                tooltip=folium.GeoJsonTooltip(fields=["municipio", "poblacion"], aliases=["Municipio", "Población (DANE)"]),
            ).add_to(mapa)
        # Mapa de calor: se agrega el catálogo por celdas de ~2 km para no saturar el navegador
        celdas = (eventos.assign(la=(eventos["latitud"] / 0.02).round() * 0.02, lo=(eventos["longitud"] / 0.02).round() * 0.02)
                  .groupby(["la", "lo"]).size().reset_index(name="n"))
        celdas["w"] = np.log1p(celdas["n"]) / np.log1p(celdas["n"]).max()
        HeatMap(celdas[["la", "lo", "w"]].values.tolist(), name="Calor histórico (densidad de sismos)",
                radius=14, blur=18, min_opacity=0.3).add_to(mapa)
        grupo = MarkerCluster(name="Mayores sismos (M≥5)").add_to(mapa)
        for r in eventos.nlargest(400, "magnitud").query("magnitud >= 5").itertuples():
            folium.CircleMarker([r.latitud, r.longitud], radius=3 + r.magnitud, color="#c0392b", fill=True, fill_opacity=0.7,
                                tooltip=f"{r.fecha} · M{r.magnitud:.1f} · {r.profundidad_km:.0f} km · {r.municipio}").add_to(grupo)
        capa_pred = folium.FeatureGroup(name="Predicciones simuladas").add_to(mapa)
        for p in st.session_state.predicciones:
            folium.Marker([p["lat"], p["lon"]],
                          icon=folium.Icon(color={1: "green", 2: "beige", 3: "red"}[p["clase"]], icon="bolt", prefix="fa"),
                          tooltip=f"{p['municipio']}: {ETIQUETAS_RIESGO[p['clase']]} (escenario M{p['mag']})").add_to(capa_pred)
        folium.LayerControl().add_to(mapa)
        st_folium(mapa, height=620, width=1400, returned_objects=[])
        st.caption(f"{len(st.session_state.predicciones)} predicción(es) en el mapa. Usa el panel lateral para añadir más.")
    except Exception as exc:
        st.error(f"No se pudo dibujar el mapa: {exc}")

with tab_sup:
    st.subheader("Aprendizaje supervisado · Random Forest")
    if artefacto is None:
        st.warning("El modelo no está disponible.")
    else:
        m = artefacto["metricas"]
        st.markdown(f"Entrenado con **{m['n_train']:,}** municipio-años ({m['anios_train']}) y evaluado con **{m['n_test']:,}** "
                    f"posteriores ({m['anios_test']}). `class_weight='balanced'` compensa que \"Alto\" es poco frecuente; "
                    f"`suelo_predominante` se codifica con One-Hot.")
        a, b, c, d, e = st.columns(5)
        a.metric("Accuracy", f"{m['accuracy']:.1%}")
        b.metric("Precision", f"{m['precision']:.1%}")
        c.metric("Recall", f"{m['recall']:.1%}")
        d.metric("F1 ponderado", f"{m['f1_score']:.1%}")
        e.metric("F1 macro", f"{m['f1_macro']:.1%}")
        st.caption(f"Validación cruzada temporal (4 pliegues): F1 = {m['cv_f1_media']:.1%} ± {m['cv_f1_desv']:.1%}")
        st.markdown(f"**Líneas base para comparar:** acertar siempre la clase más frecuente = {m['baseline_mayoritaria_accuracy']:.1%} de accuracy · "
                    f"repetir la magnitud máxima de este año (persistencia) = {m['baseline_persistencia_accuracy']:.1%} de accuracy, "
                    f"F1 {m['baseline_persistencia_f1']:.1%}.")
        c1, c2 = st.columns(2)
        with c1:
            imp = pd.Series(m["importancias"]).head(12).sort_values().reset_index()
            imp.columns = ["Variable", "Importancia"]
            st.plotly_chart(px.bar(imp, x="Importancia", y="Variable", orientation="h", title="Variables más influyentes",
                                   color_discrete_sequence=[AZUL_MARCA]), width="stretch")
        with c2:
            cm = pd.DataFrame(m["matriz_confusion"]).reindex(index=ORDEN_ETQ, columns=ORDEN_ETQ).fillna(0)
            st.plotly_chart(px.imshow(cm, text_auto=True, color_continuous_scale="Blues", title="Matriz de confusión (test)",
                                      labels=dict(x="Predicho", y="Real")), width="stretch")
        st.divider()
        if "ultima" not in st.session_state:
            st.info("Usa el panel lateral y pulsa **Predecir riesgo** para simular un escenario.")
        else:
            clase, probs, mun, fila = st.session_state.ultima
            st.markdown(f"### {mun}: {badge(clase)} el próximo año", unsafe_allow_html=True)
            dfp = pd.DataFrame({"Categoría": [ETIQUETAS_RIESGO[c] for c in probs], "Probabilidad": list(probs.values())})
            st.plotly_chart(px.bar(dfp, x="Categoría", y="Probabilidad", color="Categoría", color_discrete_map=COLOR_ETQ,
                                   range_y=[0, 1]), width="stretch")
            st.markdown("**Variables del municipio consultadas automáticamente**")
            mostrar = {"poblacion": "Población (DANE 2025)", "area_km2": "Área (km²)", "PGA_medio": "PGA 475 años en roca (g)",
                       "distancia_falla_km": "Distancia media a falla activa (km)", "elevacion_media": "Elevación media (m)",
                       "pendiente_media": "Pendiente media (°)", "pct_suelo_blando": "Fracción de suelo blando/rígido",
                       "suelo_predominante": "Suelo predominante (NEHRP, estimado)"}
            st.dataframe(pd.DataFrame({"Variable": list(mostrar.values()),
                                       "Valor": [fila[k] if isinstance(fila[k], str) else f"{float(fila[k]):,.3f}" for k in mostrar]}),
                         hide_index=True, width="stretch")
        st.subheader(f"Riesgo esperado para {ANIO_ACTUAL + 1} (con datos hasta {ANIO_ACTUAL})")
        pr = predecir_todos(artefacto, actual)
        st.dataframe(pr.merge(actual[["municipio", "poblacion", "mag_max", "n_sismos_m3"]], on="municipio")
                     .sort_values(["clase", "poblacion"], ascending=False).drop(columns="clase")
                     .rename(columns={"riesgo": "Riesgo esperado", "mag_max": f"Mag. máx. {ANIO_ACTUAL}",
                                      "n_sismos_m3": f"Sismos M≥3 en {ANIO_ACTUAL}", "poblacion": "Población"}),
                     hide_index=True, width="stretch", height=300)
        st.caption("La categoría mide la magnitud máxima esperada en el municipio, es decir, amenaza sísmica próxima; "
                   "la exposición de la población se usa en la pestaña de refuerzo para priorizar.")

with tab_uns:
    st.subheader("Aprendizaje no supervisado · K-Means + PCA")
    k = st.slider("Número de grupos", 2, 6, 3)
    tabla, diag, err = cargar_agrupamiento(k)
    if err or tabla is None:
        st.error(f"No fue posible agrupar: {err}")
    else:
        st.markdown(f"Se agrupan **{len(tabla)} municipios** con **{diag['n_variables']} variables** (sismicidad "
                    f"agregada 1994–{ANIO_ACTUAL} + territorio), estandarizadas. Coeficiente de silueta con k={k}: **{diag['silueta_k']}**. "
                    f"Con k=3 los grupos se nombran Bajo/Medio/Alto según su índice de actividad (frecuencia M≥3, magnitud máx., energía y PGA).")
        c1, c2 = st.columns(2)
        with c1:
            fig = px.scatter(tabla, x="pc1", y="pc2", color="grupo", hover_name="municipio", size="poblacion", size_max=28,
                             color_discrete_map=COLOR_ETQ, title=f"Proyección PCA (varianza explicada {sum(diag['varianza_explicada']):.0%})",
                             category_orders={"grupo": ORDEN_ETQ})
            st.plotly_chart(fig, width="stretch")
        with c2:
            sil = pd.DataFrame({"k": list(diag["silueta_por_k"]), "Silueta": list(diag["silueta_por_k"].values())})
            st.plotly_chart(px.line(sil, x="k", y="Silueta", markers=True, title="Silueta según k",
                                    color_discrete_sequence=[NARANJA_MARCA]), width="stretch")
        perfil = tabla.groupby("grupo")[["frecuencia_anual_m3", "mag_max", "profundidad_media", "poblacion", "PGA_medio", "distancia_falla_km"]].mean().round(2)
        st.markdown("**Perfil medio de cada grupo**")
        st.dataframe(perfil, width="stretch")
        st.markdown("**Municipios por grupo**")
        for g in [x for x in ORDEN_ETQ if x in set(tabla["grupo"])] or sorted(set(tabla["grupo"])):
            st.markdown(f"- **{g}** ({(tabla['grupo'] == g).sum()}): " + ", ".join(sorted(tabla.loc[tabla["grupo"] == g, "municipio"])))

with tab_rl:
    st.subheader("Aprendizaje por refuerzo · priorización de evaluaciones con presupuesto")
    agente, err_rl = cargar_agente(panel)
    if err_rl or agente is None:
        st.error(f"No fue posible cargar el agente: {err_rl}")
    else:
        st.markdown(
            "Cada año hay un **presupuesto limitado** para enviar equipos de evaluación. El agente (Q-learning) ve los municipios uno a uno y decide "
            "evaluar o no. **Recompensa = beneficio − costo.** El beneficio usa resultados reales (la categoría que tuvo el municipio al año siguiente × "
            "su población DANE). ⚠️ **El costo es simulado** (crece con el área y la distancia a Bucaramanga) y se expresa en unidades relativas, no en pesos.")
        st.caption(f"Entrenado con {agente['anios_train'][0]}–{agente['anios_train'][-1]} y evaluado en {agente['anios_test'][0]}–{agente['anios_test'][-1]}.")
        c1, c2 = st.columns(2)
        with c1:
            curva = pd.DataFrame({"Época": range(1, len(agente["curva"]) + 1), "Recompensa media": agente["curva"]})
            curva["Suavizada"] = curva["Recompensa media"].rolling(15, min_periods=1).mean()
            st.plotly_chart(px.line(curva, x="Época", y="Suavizada", title="Curva de aprendizaje", color_discrete_sequence=[AZUL_MARCA]),
                            width="stretch")
        with c2:
            comp = agente["comparacion"]
            st.plotly_chart(px.bar(comp, x="Política", y="Recompensa neta media/año", title="Recompensa en años de prueba",
                                   color_discrete_sequence=[NARANJA_MARCA]), width="stretch")
        st.dataframe(agente["comparacion"], hide_index=True, width="stretch")
        st.markdown(f"**Prioridad de evaluación para {ANIO_ACTUAL + 1}** (ventaja aprendida de evaluar, con presupuesto completo)")
        pri = agente["prioridades"].rename(columns={"ventaja_evaluar": "Ventaja de evaluar", "costo_simulado": "Costo (simulado)",
                                                    "poblacion": "Población", "sismos_M3_prom_3a": "Sismos M≥3 prom. 3 años"}).round(2)
        st.dataframe(pri.head(15), hide_index=True, width="stretch")

with tab_proy:
    st.subheader("Proyección probabilística hasta 2027")
    res, err_p = cargar_proyeccion()
    if err_p or res is None:
        st.error(f"No fue posible calcular la proyección: {err_p}")
    else:
        st.markdown(
            "El catálogo del SGC termina en marzo de 2018, y **un sismo concreto no se puede predecir**. Lo que sí se puede es proyectar la "
            "**probabilidad** de que ocurra al menos uno en cada municipio, a partir de su tasa histórica (modelo de Poisson: P = 1 − e^(−λ·T)). "
            "λ se estima con la última década completa (2008–2017), porque antes de 2008 el SGC reportaba magnitudes más altas, y se suaviza hacia el "
            "promedio departamental en municipios con pocos eventos. Supone una actividad estacionaria: no modela réplicas ni cambios del régimen sísmico.")
        proy, tasas = res["proy"], res["tasas"]
        c1, c2 = st.columns([2, 1])
        with c2:
            anio_p = st.slider("Año objetivo", ANIO_ACTUAL + 1, 2027, 2027)
            medida = st.radio("Medida", ["Probabilidad acumulada desde 2018 (≥1 sismo M≥4)", "Probabilidad solo en ese año (≥1 sismo M≥4)"])
        col = "p_acum_m4" if medida.startswith("Probabilidad acum") else "p_anual_m4"
        d = proy[proy["anio"] == anio_p].set_index("municipio")
        with c1:
            try:
                from branca.colormap import linear
                cmap = linear.YlOrRd_09.scale(0, 1)
                cmap.caption = "Probabilidad"
                mp = folium.Map(location=CENTRO_SANTANDER, zoom_start=8, tiles="OpenStreetMap")
                if geo_mun:
                    folium.GeoJson(
                        geo_mun,
                        style_function=lambda f: {"color": "#555", "weight": 0.7, "fillOpacity": 0.7,
                                                  "fillColor": cmap(float(d[col].get(f["properties"]["municipio"], 0)))},
                        tooltip=folium.GeoJsonTooltip(fields=["municipio"], aliases=["Municipio"]),
                    ).add_to(mp)
                cmap.add_to(mp)
                st_folium(mp, height=480, width=900, returned_objects=[], key=f"mapa_proy_{anio_p}_{col}")
            except Exception as exc:
                st.error(f"No se pudo dibujar el mapa de proyección: {exc}")
        m_sel = st.selectbox("Municipio", sorted(tasas["municipio"]), index=sorted(tasas["municipio"]).index("Bucaramanga"))
        serie = proy[proy["municipio"] == m_sel]
        t_m = tasas[tasas["municipio"] == m_sel].iloc[0]
        a, b, c = st.columns(3)
        a.metric("Sismos M≥4 esperados por año", f"{t_m['lambda_m4']:.2f}")
        b.metric(f"Prob. acumulada de ≥1 M≥4 a {anio_p}", f"{float(serie[serie['anio'] == anio_p]['p_acum_m4'].iloc[0]):.0%}")
        c.metric("Clase más probable en un año", t_m["clase_probable"])
        fig = px.line(serie.melt("anio", ["p_acum_m4", "p_acum_m3"], var_name="Evento", value_name="Probabilidad")
                      .replace({"p_acum_m4": "≥1 sismo M≥4", "p_acum_m3": "≥1 sismo M≥3"}),
                      x="anio", y="Probabilidad", color="Evento", markers=True, range_y=[0, 1.02],
                      title=f"{m_sel}: probabilidad acumulada desde 2018", color_discrete_sequence=[NARANJA_MARCA, AZUL_MARCA])
        st.plotly_chart(fig, width="stretch")
        top = d.reset_index().sort_values(col, ascending=False).head(15)[["municipio", col, "esperados_m4"]]
        st.markdown(f"**Municipios con mayor probabilidad a {anio_p}**")
        st.dataframe(top.rename(columns={col: "Probabilidad", "esperados_m4": "Sismos M≥4 esperados (acum.)"}).round(3),
                     hide_index=True, width="stretch")

        v = res["val"]
        st.divider()
        st.subheader("¿Qué tan bien habría funcionado? Validación con sismos reales")
        if v is None:
            st.info("Falta data/usgs_santander.csv (ejecuta `python -m src.real_data`).")
        else:
            st.markdown(f"Se compara la proyección con los sismos M≥4 **realmente registrados por el USGS en {v['anios']}**, "
                        "años que no se usaron para estimar λ.")
            a, b, c = st.columns(3)
            a.metric("Sismos M≥4 observados / esperados", f"{v['observados_total']} / {v['esperados_total']:.0f}")
            b.metric("Correlación entre municipios (Spearman)", f"{v['spearman_municipios']:.2f}")
            c.metric("Mejora sobre una probabilidad uniforme (Brier)", f"{v['habilidad_brier']:.0%}")
            pa = v["por_anio"].melt("anio", ["observados", "esperados"], var_name="Serie", value_name="Sismos M≥4")
            st.plotly_chart(px.bar(pa[pa["Serie"] == "observados"], x="anio", y="Sismos M≥4", title="Sismos M≥4 por año en Santander: observados (USGS) vs. esperados",
                                   color_discrete_sequence=[AZUL_MARCA]).add_hline(y=float(v["por_anio"]["esperados"].iloc[0]),
                                   line_color=NARANJA_MARCA, annotation_text="esperados (proyección)"), width="stretch")
            tv = v["tabla_municipios"].sort_values("observados", ascending=False).head(8).round(1)
            st.dataframe(tv.rename(columns={"esperados": "Esperados", "observados": "Observados (USGS)"}), hide_index=True, width="stretch")
            st.warning("El **total** departamental coincide razonablemente, pero la **distribución por municipio** es sensible a la agencia: "
                       "el USGS ubica muchos eventos del Nido de Bucaramanga en Aratoca, Piedecuesta o Cepitá, mientras el SGC los asigna a Los Santos. "
                       "Además, el SGC usa magnitud local (Ml) y el USGS mayormente mb. Por eso conviene leer la proyección municipal como orden de "
                       "magnitud y no como valor exacto.")
