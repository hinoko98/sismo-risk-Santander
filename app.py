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
from shapely.geometry import mapping, shape
from streamlit_folium import st_folium

from src.config import (
    ANIO_ACTUAL, ANIO_PROYECCION, AZUL_MARCA, CENTRO_SANTANDER, COLORES_RIESGO, ETIQUETAS_RIESGO, NARANJA_MARCA, OBJETIVO,
    RUTA_AGENTE_RL, RUTA_MUN_FEATURES, RUTA_ICONO, RUTA_LOGO, RUTA_MODELO, RUTA_MUNICIPIOS, RUTA_PANEL, RUTA_SISMOS_LIMPIO,
    UMBRAL_ALTO, UMBRAL_MEDIO, VARIABLES, VARS_SISMICAS, VARS_TERRITORIALES,
)
from src.decision import ACCIONES, recomendar, resumen as resumen_decision
from src.ficha import generar_ficha
from src.model_training import entrenar, explicar, predecir
from src.riesgo_integral import calcular_iri, resumen_poblacion

logging.basicConfig(level=logging.INFO)
st.set_page_config(page_title="Sismo-Risk Santander", page_icon=str(RUTA_ICONO) if RUTA_ICONO.exists() else "🌋", layout="wide")
COLOR_ETQ = {ETIQUETAS_RIESGO[k]: v for k, v in COLORES_RIESGO.items()}
ORDEN_ETQ = list(ETIQUETAS_RIESGO.values())


def plot(fig, **kw):
    """Muestra una figura Plotly con separadores colombianos (1.234,5) en lugar de abreviaturas como «1M» o «10k»."""
    fig.update_layout(separators=",.")
    st.plotly_chart(fig, **kw)


def miles(n) -> str:
    """Habitantes con separador de miles colombiano y sin decimales: 617659.0 -> '617.659'."""
    return f"{int(round(float(n))):,}".replace(",", ".")


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
        geo = json.loads(RUTA_MUNICIPIOS.read_text(encoding="utf-8"))
        for f in geo["features"]:  # se simplifica la geometría (~200 m) y se envían solo los campos necesarios
            f["geometry"] = mapping(shape(f["geometry"]).simplify(0.002, preserve_topology=True))
            pob = int(round(float(f["properties"]["poblacion"])))
            f["properties"] = {"municipio": f["properties"]["municipio"], "poblacion": pob, "habitantes": miles(pob)}
        return geo
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
    """Tasas de Poisson por municipio, proyección y prueba retrospectiva, todo con el catálogo del SGC."""
    try:
        from src.proyeccion import VENTANA_ACTUAL, VENTANA_BACKTEST, clases_anuales, estimar_tasas, proyectar, validar
        lista = sorted(pd.read_csv(RUTA_MUN_FEATURES)["municipio"])
        ev = pd.read_csv(RUTA_SISMOS_LIMPIO, usecols=["municipio", "anio", "magnitud"])
        tasas = clases_anuales(estimar_tasas(ev, lista))
        return {"tasas": tasas, "proy": proyectar(tasas), "ventana": f"{VENTANA_ACTUAL[0]}–{VENTANA_ACTUAL[1]}",
                "val": validar(estimar_tasas(ev, lista, VENTANA_BACKTEST), ev)}, None
    except Exception as exc:
        logging.exception("Fallo en la proyección")
        return None, f"{type(exc).__name__}: {exc}"


@st.cache_data(ttl=3600, show_spinner="Consultando actividad reciente (SGC)…")
def cargar_decision() -> tuple[dict | None, str | None]:
    """Índice de Riesgo Integral + monitor de actividad reciente (catálogo SGC, con actualización en vivo si hay internet)."""
    try:
        import geopandas as gpd
        from src import monitor
        mun = gpd.read_file(RUTA_MUNICIPIOS)
        lista = sorted(mun["municipio"])
        iri = calcular_iri(pd.read_csv(RUTA_MUN_FEATURES))
        cat, fuente = monitor.cargar_catalogo(mun, en_vivo=True)
        act, fin = monitor.evaluar_actividad(cat, lista)
        base = monitor.tasas_base(cat, lista)
        return {"iri": iri, "act": act, "fin": fin, "fuente": fuente,
                "serie": monitor.serie_mensual(cat, float(base.sum()))}, None
    except Exception as exc:
        logging.exception("Fallo en el centro de decisión")
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

panel["poblacion"] = panel["poblacion"].round().astype("int64")  # habitantes enteros
actual = panel[panel["anio"] == ANIO_ACTUAL].reset_index(drop=True)
feats_mun = pd.read_csv(RUTA_MUN_FEATURES)
feats_mun["poblacion"] = feats_mun["poblacion"].round().astype("int64")
etiquetado = panel.dropna(subset=[OBJETIVO]).copy()
etiquetado["riesgo_etiqueta"] = etiquetado[OBJETIVO].astype(int).map(ETIQUETAS_RIESGO)
if "predicciones" not in st.session_state:
    st.session_state.predicciones = []


# ------------------------------------------------------------------ Sidebar: simulador
with st.sidebar:
    st.header("🎛️ Simulador de riesgo")
    st.caption("Elige un municipio y un escenario para el último año. El modelo estima el nivel de riesgo sísmico del "
               "municipio para el año siguiente. Las variables del terreno y la población se toman de las fuentes oficiales.")
    with st.form("form_simulador"):
        municipios_lista = sorted(actual["municipio"])
        mun = st.selectbox("Municipio", municipios_lista,
                           index=municipios_lista.index("Bucaramanga") if "Bucaramanga" in municipios_lista else 0)
        mag = st.slider("Magnitud máxima del año en curso", 1.0, 7.5, 4.0, 0.1)
        n3 = st.slider("Sismos M≥3 en el año en curso", 0, 400, 10)
        prof = st.slider("Profundidad media (km)", 1.0, 200.0, 30.0, 1.0,
                         help="El Nido de Bucaramanga produce sismos a ~150 km de profundidad.")
        enviar = st.form_submit_button("Predecir riesgo", width="stretch")

    if enviar:
        try:
            if artefacto is None:
                st.error("Modelo no disponible.")
            else:
                fila = actual[actual["municipio"] == mun].iloc[0].to_dict()
                fila.update({"mag_max": mag, "n_sismos_m3": n3, "profundidad_media": prof,
                             "n_sismos": max(fila["n_sismos"], n3), "frac_someros": float(prof < 30),
                             "anios_desde_m4": 0 if mag >= 4 else fila["anios_desde_m4"]})
                clase, probs = predecir(artefacto, fila)
                st.session_state.predicciones.append({"lat": fila["latitud"], "lon": fila["longitud"], "mag": mag,
                                                      "clase": clase, "municipio": mun})
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
V_DEC = "🧭 Centro de decisión"
VISTAS = ["📊 Exploración (EDA)", "🗺️ Mapa de riesgo", "🎯 Supervisado", "🧩 No supervisado", "🤖 Refuerzo", "🔭 Proyección 2027", V_DEC]
ORDEN = [V_DEC] + VISTAS[:-1]  # el centro de decisión va primero
# En vez de st.tabs: Streamlit ejecuta todas las pestañas a la vez y un mapa en una pestaña oculta se queda en blanco.
vista = st.segmented_control("Sección", ORDEN, default=ORDEN[0], key="vista", label_visibility="collapsed") or ORDEN[0]

if vista == VISTAS[0]:
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
        plot(fig, width="stretch")
    with c2:
        por_anio = eventos.groupby("anio").agg(total=("magnitud", "size"), m3=("magnitud", lambda s: (s >= 3).sum())).reset_index()
        fig = px.line(por_anio.melt("anio", var_name="Serie", value_name="Eventos").replace({"total": "Todos", "m3": "M≥3"}),
                      x="anio", y="Eventos", color="Serie", title="Sismos por año (el total sube por mejor cobertura de la red)",
                      color_discrete_sequence=[AZUL_MARCA, NARANJA_MARCA])
        plot(fig, width="stretch")
    c3, c4 = st.columns(2)
    with c3:
        plot(px.histogram(eventos, x="profundidad_km", nbins=60, title="Profundidad (km): sismos someros y Nido de Bucaramanga",
                                     color_discrete_sequence=[AZUL_MARCA]), width="stretch")
    with c4:
        cl = etiquetado["riesgo_etiqueta"].value_counts().reindex(ORDEN_ETQ).reset_index()
        cl.columns = ["Riesgo", "Municipio-años"]
        plot(px.bar(cl, x="Riesgo", y="Municipio-años", color="Riesgo", title="Clases de riesgo en el dataset (desbalanceadas)",
                               color_discrete_map=COLOR_ETQ), width="stretch")
    c5, c6 = st.columns(2)
    with c5:
        top = eventos["municipio"].value_counts().head(12).sort_values().reset_index()
        top.columns = ["Municipio", "Eventos"]
        plot(px.bar(top, x="Eventos", y="Municipio", orientation="h", title="Municipios con más sismos",
                               color_discrete_sequence=[NARANJA_MARCA]), width="stretch")
    with c6:
        plot(px.scatter(actual, x="poblacion", y="PGA_medio", size="area_km2", hover_name="municipio", log_x=True,
                                   title="Habitantes vs. PGA (amenaza SGC 475 años)", color_discrete_sequence=[AZUL_MARCA],
                                   labels={"poblacion": "Habitantes (escala log)", "PGA_medio": "PGA (g)"}).update_xaxes(tickformat=",d"),
                        width="stretch")

    st.subheader("Dataset limpio (municipio-año)")
    st.dataframe(panel.drop(columns=["riesgo_etiqueta"], errors="ignore").head(3000), width="stretch", height=320)

    with st.expander("📚 Fuentes de datos (todas reales)"):
        st.markdown("""
| Variable(s) | Fuente |
|---|---|
| Actividad sísmica: frecuencia, magnitud, profundidad, RMS, GAP, fases, errores de localización, energía | [SGC · Catálogo sísmico, Consulta Experta](https://bdrsnc.sgc.gov.co/paginas1/catalogo/Consulta_Experta/consultaexperta_2.php) (1993-2017) y su sucesor [SGC · SeisComP](https://bdrsnc.sgc.gov.co/paginas1/catalogo/Consulta_Experta_Seiscomp/consultaexperta.php) (2018 hasta hoy) |
| PGA_medio | [SGC · Amenaza sísmica de Colombia 2020](https://amenazasismica.sgc.gov.co/) — PGA en roca, retorno 475 años (malla 0,1°) |
| poblacion, densidad_poblacional | [DANE · Proyecciones de población 2025](https://www.dane.gov.co/index.php/estadisticas-por-tema/demografia-y-poblacion/proyecciones-de-poblacion) |
| area_km2, límites, municipio de cada evento | Marco Geoestadístico Nacional (DANE) vía geoBoundaries; referencia oficial: [IGAC datos abiertos](https://www.igac.gov.co/datos-abiertos/datos-abiertos-geoespaciales) |
| elevacion_media, pendiente_media | SRTM 90 m (NASA) vía OpenTopoData |
| suelo_predominante, pct_suelo_blando | Clase NEHRP **estimada** con el proxy topográfico de Wald & Allen (2007) a partir de la pendiente |
| distancia_falla_km | GEM Global Active Faults Database |
| distancia_cuerpos_agua | OpenStreetMap (ríos y cuerpos de agua) |
| distancia_bucaramanga_km | Calculada desde el centroide municipal a Bucaramanga (zona de interés) |
""")

if vista == VISTAS[1]:
    try:
        capa = st.radio("Colorear municipios por", ["Riesgo esperado para el año siguiente (Random Forest)", "Grupo (K-Means)"],
                        horizontal=True)
        mapa = folium.Map(location=CENTRO_SANTANDER, zoom_start=8, tiles="OpenStreetMap", prefer_canvas=True)
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
                radius=14, blur=18, min_opacity=0.3, show=False).add_to(mapa)  # apagado: se activa en el control de capas
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
        leyenda = "".join(f"<div><span style='background:{c};width:12px;height:12px;display:inline-block;margin-right:6px;"
                          f"border-radius:2px'></span>{n}</div>" for n, c in COLOR_ETQ.items())
        mapa.get_root().html.add_child(folium.Element(
            "<div style='position:fixed;bottom:24px;left:24px;z-index:9999;background:white;padding:8px 12px;"
            f"border-radius:6px;box-shadow:0 0 6px rgba(0,0,0,.3);font-size:13px'><b>Riesgo</b>{leyenda}</div>"))
        st_folium(mapa, height=620, width=1400, returned_objects=[], key=f"mapa_riesgo_{capa[:6]}_{len(st.session_state.predicciones)}")
        st.caption(f"{len(st.session_state.predicciones)} predicción(es) en el mapa. Usa el panel lateral para añadir más.")
    except Exception as exc:
        st.error(f"No se pudo dibujar el mapa: {exc}")

if vista == VISTAS[2]:
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
            plot(px.bar(imp, x="Importancia", y="Variable", orientation="h", title="Variables más influyentes",
                                   color_discrete_sequence=[AZUL_MARCA]), width="stretch")
        with c2:
            cm = pd.DataFrame(m["matriz_confusion"]).reindex(index=ORDEN_ETQ, columns=ORDEN_ETQ).fillna(0)
            plot(px.imshow(cm, text_auto=True, color_continuous_scale="Blues", title="Matriz de confusión (test)",
                                      labels=dict(x="Predicho", y="Real")), width="stretch")
        st.divider()
        if "ultima" not in st.session_state:
            st.info("Usa el panel lateral y pulsa **Predecir riesgo** para simular un escenario.")
        else:
            clase, probs, mun, fila = st.session_state.ultima
            st.markdown(f"### {mun}: {badge(clase)} el próximo año", unsafe_allow_html=True)
            dfp = pd.DataFrame({"Categoría": [ETIQUETAS_RIESGO[c] for c in probs], "Probabilidad": list(probs.values())})
            plot(px.bar(dfp, x="Categoría", y="Probabilidad", color="Categoría", color_discrete_map=COLOR_ETQ,
                                   range_y=[0, 1]), width="stretch")
            st.markdown("**¿Por qué este resultado?** (IA explicable)")
            try:
                ex = explicar(artefacto, fila, etiquetado, clase=3)
                ex["pp"] = (ex["contribucion"] * 100).round(1)
                ex["etiqueta"] = ex["variable"] + " = " + ex["valor"].map(lambda v: v if isinstance(v, str) else f"{float(v):,.2f}")
                fig = px.bar(ex.sort_values("pp"), x="pp", y="etiqueta", orientation="h", color="pp",
                             color_continuous_scale=["#2ecc71", "#f5f5f5", "#e74c3c"], color_continuous_midpoint=0,
                             title=f"Probabilidad de riesgo Alto: {ex.attrs['probabilidad']:.0%}. Efecto de cada variable frente a un valor típico (puntos porcentuales)",
                             labels={"pp": "Puntos porcentuales", "etiqueta": ""})
                fig.update_layout(coloraxis_showscale=False)
                plot(fig, width="stretch")
                st.caption("Rojo: la variable aumenta la probabilidad de riesgo Alto respecto de un municipio típico; verde: la reduce. "
                           "Se calcula reemplazando cada variable por su mediana y midiendo cuánto cambia la predicción.")
            except Exception as exc:
                st.caption(f"No se pudo calcular la explicación: {exc}")
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

if vista == VISTAS[3]:
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
            plot(fig, width="stretch")
        with c2:
            sil = pd.DataFrame({"k": list(diag["silueta_por_k"]), "Silueta": list(diag["silueta_por_k"].values())})
            plot(px.line(sil, x="k", y="Silueta", markers=True, title="Silueta según k",
                                    color_discrete_sequence=[NARANJA_MARCA]), width="stretch")
        perfil = tabla.groupby("grupo")[["frecuencia_anual_m3", "mag_max", "profundidad_media", "poblacion", "PGA_medio", "distancia_falla_km"]].mean().round(2)
        st.markdown("**Perfil medio de cada grupo**")
        st.dataframe(perfil, width="stretch")
        st.markdown("**Municipios por grupo**")
        for g in [x for x in ORDEN_ETQ if x in set(tabla["grupo"])] or sorted(set(tabla["grupo"])):
            st.markdown(f"- **{g}** ({(tabla['grupo'] == g).sum()}): " + ", ".join(sorted(tabla.loc[tabla["grupo"] == g, "municipio"])))

if vista == VISTAS[4]:
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
            plot(px.line(curva, x="Época", y="Suavizada", title="Curva de aprendizaje", color_discrete_sequence=[AZUL_MARCA]),
                            width="stretch")
        with c2:
            comp = agente["comparacion"]
            plot(px.bar(comp, x="Política", y="Recompensa neta media/año", title="Recompensa en años de prueba",
                                   color_discrete_sequence=[NARANJA_MARCA]), width="stretch")
        st.dataframe(agente["comparacion"], hide_index=True, width="stretch")
        st.markdown(f"**Prioridad de evaluación para {ANIO_ACTUAL + 1}** (ventaja aprendida de evaluar, con presupuesto completo)")
        pri = agente["prioridades"].rename(columns={"ventaja_evaluar": "Ventaja de evaluar", "costo_simulado": "Costo (simulado)",
                                                    "poblacion": "Población", "sismos_M3_prom_3a": "Sismos M≥3 prom. 3 años"}).round(2)
        st.dataframe(pri.head(15), hide_index=True, width="stretch")

if vista == VISTAS[5]:
    st.subheader(f"Proyección probabilística hasta {ANIO_PROYECCION}")
    res, err_p = cargar_proyeccion()
    if err_p or res is None:
        st.error(f"No fue posible calcular la proyección: {err_p}")
    else:
        v = res["val"]
        st.markdown(
            "**Un sismo concreto no se puede predecir.** Lo que sí se puede es proyectar la **probabilidad** de que ocurra al menos uno en cada "
            "municipio, a partir de su tasa histórica (modelo de Poisson: P = 1 − e^(−λ·T)). "
            f"λ se estima con la última década completa ({res['ventana']}), porque antes de 2008 el SGC reportaba magnitudes más altas, y se suaviza hacia el "
            "promedio departamental en municipios con pocos eventos. Supone una actividad estacionaria: no modela réplicas ni cambios del régimen sísmico.")
        proy, tasas = res["proy"], res["tasas"]
        c1, c2 = st.columns([2, 1])
        with c2:
            anio_p = st.slider("Año objetivo", ANIO_ACTUAL + 1, ANIO_PROYECCION, ANIO_PROYECCION)
            medida = st.radio("Medida", [f"Probabilidad acumulada desde {ANIO_ACTUAL + 1} (≥1 sismo M≥4)", "Probabilidad solo en ese año (≥1 sismo M≥4)"])
        col = "p_acum_m4" if medida.startswith("Probabilidad acum") else "p_anual_m4"
        d = proy[proy["anio"] == anio_p].set_index("municipio")
        with c1:
            try:
                from branca.colormap import linear
                cmap = linear.YlOrRd_09.scale(0, 1)
                cmap.caption = "Probabilidad"
                mp = folium.Map(location=CENTRO_SANTANDER, zoom_start=8, tiles="OpenStreetMap", prefer_canvas=True)
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
        lista = sorted(tasas["municipio"])
        m_sel = st.selectbox("Municipio", lista, index=lista.index("Bucaramanga") if "Bucaramanga" in lista else 0)
        serie = proy[proy["municipio"] == m_sel]
        t_m = tasas[tasas["municipio"] == m_sel].iloc[0]
        a, b, c = st.columns(3)
        a.metric("Sismos M≥4 esperados por año", f"{t_m['lambda_m4']:.2f}")
        b.metric(f"Prob. acumulada de ≥1 M≥4 a {anio_p}", f"{float(serie[serie['anio'] == anio_p]['p_acum_m4'].iloc[0]):.0%}")
        c.metric("Clase más probable en un año", t_m["clase_probable"])
        fig = px.line(serie.melt("anio", ["p_acum_m4", "p_acum_m3"], var_name="Evento", value_name="Probabilidad")
                      .replace({"p_acum_m4": "≥1 sismo M≥4", "p_acum_m3": "≥1 sismo M≥3"}),
                      x="anio", y="Probabilidad", color="Evento", markers=True, range_y=[0, 1.02],
                      title=f"{m_sel}: probabilidad acumulada desde {ANIO_ACTUAL + 1}", color_discrete_sequence=[NARANJA_MARCA, AZUL_MARCA])
        fig.update_xaxes(dtick=1)
        plot(fig, width="stretch")
        top = d.reset_index().sort_values(col, ascending=False).head(15)[["municipio", col, "esperados_m4"]]
        st.markdown(f"**Municipios con mayor probabilidad a {anio_p}**")
        st.dataframe(top.rename(columns={col: "Probabilidad", "esperados_m4": "Sismos M≥4 esperados (acum.)"}).round(3),
                     hide_index=True, width="stretch")

        st.divider()
        st.subheader("¿Qué tan bien habría funcionado? Prueba retrospectiva")
        st.markdown(f"Se estimaron las tasas **solo con {v['ventana']}** y se compararon con los sismos M≥4 que el SGC **realmente registró en {v['anios']}**, "
                    "años que el modelo no vio.")
        a, b, c = st.columns(3)
        a.metric("Sismos M≥4 observados / esperados", f"{v['observados_total']} / {v['esperados_total']:.0f}")
        b.metric("Correlación entre municipios (Spearman)", f"{v['spearman_municipios']:.2f}")
        c.metric("Mejora sobre una probabilidad uniforme (Brier)", f"{v['habilidad_brier']:.0%}")
        fig = px.bar(v["por_anio"], x="anio", y="observados", title="Sismos M≥4 por año en Santander: observados vs. esperados",
                     color_discrete_sequence=[AZUL_MARCA], labels={"observados": "Sismos M≥4", "anio": "Año"})
        fig.add_hline(y=float(v["por_anio"]["esperados"].iloc[0]), line_color=NARANJA_MARCA, annotation_text="esperados (proyección)")
        plot(fig, width="stretch")
        tv = v["tabla_municipios"].sort_values("observados", ascending=False).head(8).round(1)
        st.dataframe(tv.rename(columns={"esperados": "Esperados", "observados": "Observados (SGC)"}), hide_index=True, width="stretch")
        st.caption("Si el total observado se aleja del esperado, la actividad cambió respecto de la década usada para estimar; "
                   "a nivel de municipio hay más ruido porque la mayoría registra pocos eventos M≥4 por año.")


if vista == V_DEC:
    st.subheader("Centro de decisión · del dato a la acción")
    st.markdown(
        "Una predicción sola no ayuda a un gestor del riesgo. Aquí se combinan **cuatro capas de datos reales** para responder "
        "*¿dónde actuar primero y qué hacer?*: la **amenaza** oficial del SGC, la **población expuesta** del DANE, la **vulnerabilidad del terreno** "
        "y la **actividad sísmica de los últimos 12 meses** frente a lo normal en cada municipio.")
    dec, err_dec = cargar_decision()
    if err_dec or dec is None:
        st.error(f"No fue posible armar el centro de decisión: {err_dec}")
    else:
        iri, act = dec["iri"], dec["act"]
        tabla = iri.merge(act[["municipio", "observados_12m", "esperados_12m", "razon", "p_valor", "semaforo", "mag_max_12m"]], on="municipio")
        agente_d, _ = cargar_agente(panel)
        if agente_d is not None:
            pr = agente_d["prioridades"].copy()
            pr["prioridad_rl"] = pr["ventaja_evaluar"].rank(ascending=False, method="min").astype(int)
            tabla = tabla.merge(pr[["municipio", "prioridad_rl"]], on="municipio", how="left")
        res = resumen_poblacion(iri)
        reco = recomendar(tabla)
        en_alerta = tabla[tabla["semaforo"] != "Verde"]

        st.success("📌 **Decisión recomendada.** " + resumen_decision(reco))
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Habitantes en municipios de nivel Alto", miles(res["pob_alto"]), f"{res['n_alto']} municipios · {res['frac_alto']:.0%} de Santander", delta_color="off")
        k2.metric("Municipios con actividad inusual (12 m)", f"{len(en_alerta)}", ", ".join(en_alerta["municipio"].head(2)) or "ninguno", delta_color="off")
        k3.metric("Mayor Índice de Riesgo Integral", iri.iloc[0]["municipio"], f"{iri.iloc[0]['iri']:.0f}/100", delta_color="off")
        k4.metric("Último sismo del catálogo", f"{dec['fin']:%d-%m-%Y}")
        st.caption(f"Actividad reciente: {dec['fuente']}. Se actualiza sola cada hora si hay internet.")

        st.markdown("**Plan de acción y por qué se recomienda**")
        iconos = {1: "🔴", 2: "🟠", 3: "🟡"}
        for prio, grupo in reco.groupby("prioridad"):
            n_g = len(grupo)
            st.markdown(f"##### {iconos[prio]} {grupo.iloc[0]['decision']} · {n_g} municipio{'s' if n_g > 1 else ''}")
            st.markdown(f"**Qué hacer:** {ACCIONES[prio][1]}")
            for f in grupo.itertuples():
                marca = "✅ dentro del presupuesto" if f.en_plan else "⏳ fuera del presupuesto"
                extra = " *Además, estudiar la microzonificación de sus suelos.*" if "microzonificación" in f.accion else ""
                st.markdown(f"- **{f.municipio}** · {miles(f.poblacion)} habitantes · {marca}  \n  {f.porque}{extra}")
        with st.expander("Ver el plan como tabla y descargarlo"):
            plan = reco.rename(columns={"prioridad": "Prioridad", "decision": "Decisión", "municipio": "Municipio", "accion": "Acción recomendada",
                                        "porque": "¿Por qué? (evidencia)", "costo_simulado": "Costo (simulado)", "en_plan": "Dentro del presupuesto"})
            plan["Habitantes"] = plan["poblacion"].map(miles)
            st.dataframe(plan[["Prioridad", "Decisión", "Municipio", "Habitantes", "Costo (simulado)", "Dentro del presupuesto", "Acción recomendada",
                               "¿Por qué? (evidencia)"]], hide_index=True, width="stretch")
            st.download_button("⬇️ Descargar plan (CSV)", plan.drop(columns=["poblacion"]).to_csv(index=False).encode("utf-8-sig"),
                               "plan_de_accion_santander.csv", "text/csv")
        with st.expander("¿Cómo se toma la decisión? (reglas)"):
            st.markdown(
                "- **Prioridad 1 · Actuar ya:** nivel de riesgo **Alto** *y* actividad sísmica inusual en los últimos 12 meses.\n"
                "- **Prioridad 2 · Preparar:** nivel de riesgo **Alto** con actividad normal; el riesgo viene de la amenaza, la población y el terreno, no de un repunte.\n"
                "- **Prioridad 3 · Vigilar:** actividad inusual en un municipio de nivel Medio o Bajo (señal para observar, no para alarmar).\n"
                "- Si la vulnerabilidad del terreno está en el 10 % más alto, se añade *estudiar la microzonificación de sus suelos*.\n"
                "- El plan cubre los municipios por orden de prioridad e IRI hasta agotar un presupuesto de 30 unidades (costo **simulado**, las mismas del agente de refuerzo).\n\n"
                "Son recomendaciones de apoyo a la decisión: los umbrales los propone el equipo y deben validarlos especialistas en gestión del riesgo.")

        capa_d = st.radio("Colorear el mapa por", ["Índice de Riesgo Integral", "Semáforo de actividad (últimos 12 meses)"], horizontal=True)
        try:
            t_idx = tabla.set_index("municipio")
            sem_col = {"Verde": "#2ecc71", "Amarillo": "#f1c40f", "Rojo": "#e74c3c"}

            def color_d(m):
                if capa_d.startswith("Índice"):
                    return COLOR_ETQ[t_idx["nivel"].get(m, "Bajo")]
                return sem_col[t_idx["semaforo"].get(m, "Verde")]

            md = folium.Map(location=CENTRO_SANTANDER, zoom_start=8, tiles="OpenStreetMap", prefer_canvas=True)
            if geo_mun:
                folium.GeoJson(
                    geo_mun, style_function=lambda f: {"color": "#555", "weight": 0.7, "fillOpacity": 0.75,
                                                       "fillColor": color_d(f["properties"]["municipio"])},
                    tooltip=folium.GeoJsonTooltip(fields=["municipio", "habitantes"], aliases=["Municipio", "Habitantes"]),
                ).add_to(md)
            niveles = COLOR_ETQ if capa_d.startswith("Índice") else sem_col
            titulo = "Nivel de riesgo integral (relativo)" if capa_d.startswith("Índice") else "Actividad vs. lo normal"
            items = "".join(f"<div><span style='background:{c};width:12px;height:12px;display:inline-block;margin-right:6px;"
                            f"border-radius:2px'></span>{n}</div>" for n, c in niveles.items())
            md.get_root().html.add_child(folium.Element(
                "<div style='position:fixed;bottom:24px;left:24px;z-index:9999;background:white;padding:8px 12px;border-radius:6px;"
                f"box-shadow:0 0 6px rgba(0,0,0,.3);font-size:13px'><b>{titulo}</b>{items}</div>"))
            st_folium(md, height=520, width=1400, returned_objects=[], key=f"mapa_dec_{capa_d[:5]}")
        except Exception as exc:
            st.error(f"No se pudo dibujar el mapa: {exc}")

        c1, c2 = st.columns(2)
        with c1:
            fig = px.scatter(tabla, x="PGA_medio", y="poblacion", log_y=True, color="nivel", size="iri", size_max=26,
                             hover_name="municipio", color_discrete_map=COLOR_ETQ, category_orders={"nivel": ORDEN_ETQ},
                             title="Amenaza (PGA) vs. población: arriba a la derecha = crítico",
                             labels={"PGA_medio": "PGA esperado a 475 años (g)", "poblacion": "Habitantes (escala log)", "nivel": "Nivel"})
            fig.update_yaxes(tickformat=",d")
            plot(fig, width="stretch")
        with c2:
            m = dec["serie"]
            fig = px.bar(m, x="time", y="observados", title="Sismos M≥4 por mes en Santander (SGC) vs. promedio esperado",
                         color_discrete_sequence=[AZUL_MARCA], labels={"time": "Mes", "observados": "Sismos M≥4"})
            fig.add_hline(y=float(m["esperados"].iloc[0]), line_color=NARANJA_MARCA, annotation_text="esperado")
            plot(fig, width="stretch")

        st.markdown("**Ranking de prioridad**")
        cols = ["ranking", "municipio", "iri", "nivel", "poblacion", "PGA_medio", "vulnerabilidad", "semaforo", "observados_12m", "esperados_12m"]
        nombres = {"ranking": "Puesto", "municipio": "Municipio", "iri": "IRI", "nivel": "Nivel", "poblacion": "Habitantes", "PGA_medio": "PGA (g)",
                   "vulnerabilidad": "Vulnerabilidad", "semaforo": "Semáforo 12 m", "observados_12m": "M≥4 observados", "esperados_12m": "M≥4 esperados"}
        if "prioridad_rl" in tabla:
            cols.append("prioridad_rl")
            nombres["prioridad_rl"] = "Prioridad agente RL"
        vista_t = tabla[cols].rename(columns=nombres)
        mostrar_t = vista_t.round(3).assign(Habitantes=vista_t["Habitantes"].map(miles))
        st.dataframe(mostrar_t, hide_index=True, width="stretch", height=330)
        st.download_button("⬇️ Descargar ranking (CSV)", vista_t.to_csv(index=False).encode("utf-8-sig"), "ranking_riesgo_santander.csv", "text/csv")
        st.caption("IRI = amenaza × exposición × (0,5 + 0,5·vulnerabilidad). Es un índice relativo para priorizar, con pesos propuestos por el equipo; "
                   "no estima daños. Una actividad inusual es una señal estadística para vigilar, no un pronóstico.")

        st.markdown("**Ficha municipal para gestores**")
        sel_f = st.selectbox("Municipio de la ficha", sorted(tabla["municipio"]),
                             index=sorted(tabla["municipio"]).index("Bucaramanga") if "Bucaramanga" in set(tabla["municipio"]) else 0, key="sel_ficha")
        try:
            fila_f = tabla[tabla["municipio"] == sel_f].iloc[0]
            res_p, _ = cargar_proyeccion()
            p_fin = float(res_p["proy"].query("municipio == @sel_f and anio == @ANIO_PROYECCION")["p_acum_m4"].iloc[0]) if res_p else float("nan")
            clase_rf = predecir_todos(artefacto, actual).set_index("municipio")["riesgo"].get(sel_f, "—") if artefacto else "—"
            base = feats_mun.set_index("municipio").loc[sel_f]
            decision_f = reco[reco["municipio"] == sel_f]
            html = generar_ficha({
                "municipio": sel_f, "iri": float(fila_f["iri"]), "nivel": fila_f["nivel"], "ranking": int(fila_f["ranking"]), "total": len(tabla),
                "indicadores": {
                    "Población (DANE 2025)": miles(base["poblacion"]), "Área (km²)": f"{base['area_km2']:.0f}",
                    "Densidad (hab/km²)": f"{base['densidad_poblacional']:.0f}", "PGA esperado a 475 años (g, SGC)": f"{base['PGA_medio']:.3f}",
                    "Distancia media a falla activa (km)": f"{base['distancia_falla_km']:.1f}",
                    "Suelo predominante (estimado)": base["suelo_predominante"], "Pendiente media (°)": f"{base['pendiente_media']:.1f}",
                    "Vulnerabilidad del sitio (0–1)": f"{fila_f['vulnerabilidad']:.2f}"},
                "obs12": int(fila_f["observados_12m"]), "esp12": f"{fila_f['esperados_12m']:.1f}", "semaforo": fila_f["semaforo"],
                "p2027": p_fin, "anio_fin": ANIO_PROYECCION, "anio_ini": ANIO_ACTUAL + 1, "clase_rf": clase_rf,
                "decision": (decision_f.iloc[0]["decision"] + ": " + decision_f.iloc[0]["accion"]) if len(decision_f) else "Sin acción extraordinaria; seguimiento de rutina.",
                "porque": decision_f.iloc[0]["porque"] if len(decision_f) else ""})
            st.download_button(f"⬇️ Descargar ficha de {sel_f} (HTML, imprimible a PDF)", html.encode("utf-8"), f"ficha_{sel_f.replace(' ', '_')}.html", "text/html")
        except Exception as exc:
            st.warning(f"No se pudo generar la ficha: {exc}")
