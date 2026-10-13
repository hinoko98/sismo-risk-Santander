"""Sismo-Risk Santander · Misión 3 (Nivel Avanzado): dashboard Streamlit.

Ejecutar con:  streamlit run app.py
"""
import logging

import folium
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium

from src.config import (
    CENTRO_SANTANDER, COLORES_RIESGO, ETIQUETAS_RIESGO, OBJETIVO, POLIGONO_APROX,
    RUTA_MODELO, RUTA_SISMOS_LIMPIO, TIPOS_SUELO, VARIABLES,
)
from src.data_generation import distancia_falla_km
from src.model_training import entrenar, predecir

logging.basicConfig(level=logging.INFO)
st.set_page_config(page_title="Sismo-Risk Santander", page_icon="🌋", layout="wide")


# ------------------------------------------------------------------ Carga con caché y tolerancia a fallos
@st.cache_data(show_spinner="Cargando y limpiando datos…")
def cargar_datos() -> tuple[pd.DataFrame | None, dict, str | None]:
    """Devuelve (df, informe_etl, error). Nunca lanza: ante un fallo retorna el mensaje."""
    try:
        if RUTA_SISMOS_LIMPIO.exists():
            df = pd.read_csv(RUTA_SISMOS_LIMPIO)
            if set(VARIABLES + [OBJETIVO]).issubset(df.columns):
                return df, {}, None
        # Import diferido: geopandas solo se necesita si hay que regenerar el CSV limpio
        from src.data_processing import ejecutar_etl
        df, informe = ejecutar_etl()
        return df, informe, None
    except Exception as exc:
        logging.exception("Fallo al cargar datos")
        return None, {}, f"{type(exc).__name__}: {exc}"


@st.cache_resource(show_spinner="Cargando modelo…")
def cargar_modelo(_df: pd.DataFrame | None) -> tuple[dict | None, str | None]:
    """Carga el .pkl; si no existe (o está dañado) reentrena automáticamente con los datos."""
    try:
        if RUTA_MODELO.exists():
            try:
                return joblib.load(RUTA_MODELO), None
            except Exception:
                logging.warning("Modelo ilegible; se reentrena.")
        if _df is None:
            return None, "No hay datos para entrenar el modelo."
        entrenar(_df)
        return joblib.load(RUTA_MODELO), None
    except Exception as exc:
        logging.exception("Fallo al cargar el modelo")
        return None, f"{type(exc).__name__}: {exc}"


def en_santander(lat: float, lon: float) -> bool:
    from shapely.geometry import Point
    return POLIGONO_APROX.contains(Point(lon, lat))


def badge(clase: int) -> str:
    return f"<span style='background:{COLORES_RIESGO[clase]};color:#fff;padding:4px 12px;" \
           f"border-radius:12px;font-weight:600'>{ETIQUETAS_RIESGO[clase]}</span>"


# ------------------------------------------------------------------ Arranque
df, informe_etl, err_datos = cargar_datos()
artefacto, err_modelo = cargar_modelo(df)

st.title("🌋 Sismo-Risk Santander")
st.caption("Gestión del riesgo sísmico · Hackathon CIAI UTSmart 2026 · datos simulados con base en SGC, IGAC y DANE")

if err_datos or df is None:
    st.error(f"No fue posible cargar los datos. Detalle: {err_datos}")
    st.stop()
if err_modelo:
    st.warning(f"El simulador no está disponible: {err_modelo}")

if "predicciones" not in st.session_state:
    st.session_state.predicciones = []

# ------------------------------------------------------------------ Sidebar: simulador (Tab 3 lo muestra)
medianas = df[[c for c in VARIABLES if c != "tipo_suelo"]].median()
with st.sidebar:
    st.header("🎛️ Simulador de riesgo")
    with st.form("form_simulador"):
        lat = st.number_input("Latitud", 5.70, 8.14, 7.119, 0.001, format="%.4f")
        lon = st.number_input("Longitud", -74.55, -72.35, -73.122, 0.001, format="%.4f")
        mag = st.slider("Magnitud (Mw)", 2.0, 7.5, 4.5, 0.1)
        prof = st.slider("Profundidad (km)", 1.0, 200.0, float(medianas["profundidad_km"]), 1.0)
        suelo = st.selectbox("Tipo de suelo", TIPOS_SUELO, index=2)
        pga = st.slider("PGA histórico (g)", 0.01, 0.90, float(medianas["PGA_historico"]), 0.01)
        dens = st.number_input("Densidad poblacional (hab/km²)", 0.0, 20000.0,
                               float(medianas["densidad_poblacional"]), 10.0)
        antig = st.slider("Antigüedad de construcciones (años)", 1, 100, int(medianas["antiguedad_construcciones"]))
        pend = st.slider("Pendiente del terreno (°)", 0.5, 60.0, float(medianas["pendiente_terreno"]), 0.5)
        infra = st.number_input("Infraestructura crítica (unidades)", 0, 30, int(medianas["infraestructura_critica"]))
        enviar = st.form_submit_button("Predecir riesgo", width="stretch")

    if enviar:
        try:
            if not en_santander(lat, lon):
                st.error("Las coordenadas quedan fuera del departamento de Santander.")
            elif artefacto is None:
                st.error("Modelo no disponible.")
            else:
                # Variables no ingresadas se completan con la mediana; la distancia a falla se calcula
                fila = medianas.to_dict()
                fila.update({
                    "latitud": lat, "longitud": lon, "magnitud": mag, "profundidad_km": prof,
                    "tipo_suelo": suelo, "PGA_historico": pga, "densidad_poblacional": dens,
                    "antiguedad_construcciones": antig, "pendiente_terreno": pend,
                    "infraestructura_critica": infra,
                    "distancia_falla_km": distancia_falla_km(lat, lon),
                })
                clase, probs = predecir(artefacto, fila)
                st.session_state.predicciones.append(
                    {"lat": lat, "lon": lon, "mag": mag, "clase": clase, "probs": probs}
                )
                st.session_state.ultima = (clase, probs)
        except Exception as exc:
            st.error(f"No se pudo calcular la predicción: {exc}")

    if "ultima" in st.session_state:
        clase, probs = st.session_state.ultima
        st.markdown(f"**Riesgo estimado:** {badge(clase)}", unsafe_allow_html=True)
        for c, p in probs.items():
            st.progress(p, text=f"{ETIQUETAS_RIESGO[c]}: {p:.0%}")
    if st.session_state.predicciones and st.button("Borrar predicciones del mapa"):
        st.session_state.predicciones = []
        st.session_state.pop("ultima", None)
        st.rerun()

# ------------------------------------------------------------------ Tabs
tab_eda, tab_mapa, tab_sim = st.tabs(["📊 Exploración de datos", "🗺️ Mapa de riesgo", "🎯 Simulador"])

with tab_eda:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Eventos analizados", f"{len(df):,}")
    k2.metric("Magnitud máxima", f"{df['magnitud'].max():.1f}")
    k3.metric("Profundidad media", f"{df['profundidad_km'].mean():.0f} km")
    k4.metric("Riesgo Alto/Extremo", f"{(df[OBJETIVO] >= 3).mean():.0%}")
    if informe_etl:
        st.info(f"ETL: {informe_etl['filas_brutas']:,} filas brutas → {informe_etl['filas_limpias']:,} limpias "
                f"({informe_etl['filas_fuera_santander']} fuera de Santander descartadas, "
                f"{informe_etl['nulos_antes']} nulos imputados).")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="magnitud", nbins=30, color="riesgo_etiqueta",
                           title="Distribución de sismos por magnitud",
                           category_orders={"riesgo_etiqueta": list(ETIQUETAS_RIESGO.values())},
                           color_discrete_map={ETIQUETAS_RIESGO[k]: v for k, v in COLORES_RIESGO.items()})
        st.plotly_chart(fig, width="stretch")
    with c2:
        conteo = df["riesgo_etiqueta"].value_counts().reindex(ETIQUETAS_RIESGO.values()).reset_index()
        conteo.columns = ["Riesgo", "Eventos"]
        fig = px.bar(conteo, x="Riesgo", y="Eventos", color="Riesgo", title="Eventos por categoría de riesgo",
                     color_discrete_map={ETIQUETAS_RIESGO[k]: v for k, v in COLORES_RIESGO.items()})
        st.plotly_chart(fig, width="stretch")
    c3, c4 = st.columns(2)
    with c3:
        fig = px.scatter(df.sample(min(len(df), 1500), random_state=1), x="profundidad_km", y="magnitud",
                         color="riesgo_etiqueta", opacity=0.6, title="Profundidad vs magnitud (Nido de Bucaramanga ≈150 km)",
                         color_discrete_map={ETIQUETAS_RIESGO[k]: v for k, v in COLORES_RIESGO.items()})
        st.plotly_chart(fig, width="stretch")
    with c4:
        if artefacto:
            imp = pd.Series(artefacto["metricas"]["importancias"]).head(10).sort_values().reset_index()
            imp.columns = ["Variable", "Importancia"]
            st.plotly_chart(px.bar(imp, x="Importancia", y="Variable", orientation="h",
                                   title="Variables más influyentes (Random Forest)"), width="stretch")

    if artefacto:
        m = artefacto["metricas"]
        st.subheader("Desempeño del modelo (hold-out 20 %)")
        a, b, c, d, e = st.columns(5)
        a.metric("Accuracy", f"{m['accuracy']:.2%}")
        b.metric("Precision", f"{m['precision']:.2%}")
        c.metric("Recall", f"{m['recall']:.2%}")
        d.metric("F1-Score", f"{m['f1_score']:.2%}")
        e.metric("F1 CV (5 pliegues)", f"{m['cv_f1_media']:.2%} ± {m['cv_f1_desv']:.2%}")

    st.subheader("Dataset limpio")
    filtro = st.multiselect("Filtrar por riesgo", list(ETIQUETAS_RIESGO.values()), default=list(ETIQUETAS_RIESGO.values()))
    st.dataframe(df[df["riesgo_etiqueta"].isin(filtro)], width="stretch", height=320)

with tab_mapa:
    try:
        opciones = st.multiselect("Capas de riesgo a mostrar", list(ETIQUETAS_RIESGO.values()),
                                  default=["Alto", "Extremo"])
        mapa = folium.Map(location=CENTRO_SANTANDER, zoom_start=8, tiles="OpenStreetMap")
        folium.GeoJson(POLIGONO_APROX.__geo_interface__, name="Santander",
                       style_function=lambda _: {"color": "#34495e", "weight": 2, "fillOpacity": 0.03}).add_to(mapa)
        # Mapa de calor ponderado por magnitud
        HeatMap(df[["latitud", "longitud", "magnitud"]].values.tolist(), name="Calor histórico",
                radius=14, blur=18, min_opacity=0.3).add_to(mapa)
        # Clústeres de eventos según las categorías elegidas (tope para mantener fluidez)
        grupo = MarkerCluster(name="Eventos por riesgo").add_to(mapa)
        sel = df[df["riesgo_etiqueta"].isin(opciones)].head(800)
        for r in sel.itertuples():
            folium.CircleMarker(
                [r.latitud, r.longitud], radius=4 + r.magnitud / 1.5, color=COLORES_RIESGO[getattr(r, OBJETIVO)],
                fill=True, fill_opacity=0.7,
                tooltip=f"M{r.magnitud} · {r.profundidad_km} km · Riesgo {r.riesgo_etiqueta}",
            ).add_to(grupo)
        # Predicciones hechas desde el simulador
        capa_pred = folium.FeatureGroup(name="Predicciones simuladas").add_to(mapa)
        for p in st.session_state.predicciones:
            folium.Marker(
                [p["lat"], p["lon"]],
                icon=folium.Icon(color={1: "green", 2: "beige", 3: "orange", 4: "red"}[p["clase"]], icon="bolt", prefix="fa"),
                tooltip=f"Predicción: {ETIQUETAS_RIESGO[p['clase']]} (M{p['mag']})",
            ).add_to(capa_pred)
        folium.LayerControl().add_to(mapa)
        st_folium(mapa, height=600, width=1400, returned_objects=[])
        st.caption(f"{len(st.session_state.predicciones)} predicción(es) en el mapa. Usa el panel lateral para añadir más.")
    except Exception as exc:
        st.error(f"No se pudo dibujar el mapa: {exc}")

with tab_sim:
    st.subheader("Simulador de riesgo en tiempo real")
    if artefacto is None:
        st.warning("El modelo no está disponible.")
    elif "ultima" not in st.session_state:
        st.info("Completa el formulario del panel lateral y pulsa **Predecir riesgo**.")
    else:
        clase, probs = st.session_state.ultima
        st.markdown(f"### Resultado: {badge(clase)}", unsafe_allow_html=True)
        dfp = pd.DataFrame({"Categoría": [ETIQUETAS_RIESGO[c] for c in probs], "Probabilidad": list(probs.values())})
        st.plotly_chart(px.bar(dfp, x="Categoría", y="Probabilidad", color="Categoría",
                               color_discrete_map={ETIQUETAS_RIESGO[k]: v for k, v in COLORES_RIESGO.items()},
                               range_y=[0, 1]), width="stretch")
        st.caption("El punto aparece como marcador en la pestaña «Mapa de riesgo».")
    if st.session_state.predicciones:
        hist = pd.DataFrame(st.session_state.predicciones)
        hist["riesgo"] = hist["clase"].map(ETIQUETAS_RIESGO)
        st.dataframe(hist[["lat", "lon", "mag", "riesgo"]], width="stretch")
