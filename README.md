# 🌋 Sismo-Risk Santander

Plataforma web para la gestión del riesgo sísmico en Santander — Hackathon CIAI UTSmart 2026.
Todos los datos son **reales** (SGC, DANE, IGAC/MGN, GEM, OpenStreetMap, SRTM); solo el costo de evaluación del reto de refuerzo es simulado, como permite el reto.

## Ejecución rápida
```bash
pip install -r requirements.txt
streamlit run app.py
```
La carpeta `data/` incluye los datos ya descargados. Si falta algo, la app lo regenera (requiere internet):
`python -m src.real_data` (descarga) → `python -m src.data_processing` (ETL) → `python -m src.model_training` → `python -m src.reinforcement`.

## Los tres retos
| Reto | Módulo | Qué hace |
|---|---|---|
| Básico · supervisado | `src/data_processing.py`, `src/model_training.py` | ETL con geopandas, imputación, panel municipio-año, Random Forest (`class_weight='balanced'`, One-Hot, split temporal 80/20, CV de series de tiempo) |
| Medio · no supervisado | `src/unsupervised.py` | K-Means (k configurable) + PCA sobre 22 variables agregadas por municipio |
| Avanzado · refuerzo | `src/reinforcement.py` | Q-learning: priorizar evaluaciones municipales con presupuesto limitado |
| Centro de decisión (diferenciador) | `src/riesgo_integral.py`, `src/monitor.py`, `src/ficha.py` | Índice de Riesgo Integral, monitor de actividad reciente con USGS en vivo, IA explicable y ficha municipal descargable |
| Proyección a 2027 | `src/proyeccion.py` | Probabilidad por municipio con modelo de Poisson (tasas 2008–2017) y validación con sismos reales del USGS 2019–2025 |

## Etiqueta (reproducible y sin fuga de datos)
Unidad: **municipio-año**. Las variables de entrada usan información hasta el año *t*; la categoría es la **magnitud máxima registrada en el año *t+1***:
**Bajo** < 3,0 · **Medio** 3,0–<4,0 · **Alto** ≥ 4,0. Ninguna entrada contiene datos del año siguiente. La evaluación es temporal (entrena con 1994–2011, prueba con 2012–2016) y se compara con líneas base (clase mayoritaria y persistencia).

## Fuentes
| Variable(s) | Fuente |
|---|---|
| Actividad sísmica (frecuencia, magnitud, profundidad, RMS, GAP, fases) | [SGC Consulta Experta](https://bdrsnc.sgc.gov.co/paginas1/catalogo/Consulta_Experta/consultaexperta_2.php) — jun-1993 a mar-2018 (el servicio no devuelve fechas posteriores) |
| PGA (475 años, roca) | [SGC Amenaza sísmica 2020](https://amenazasismica.sgc.gov.co/) |
| Población, área, límites | [DANE](https://www.dane.gov.co/index.php/estadisticas-por-tema/demografia-y-poblacion/proyecciones-de-poblacion) (proyecciones 2025 y Marco Geoestadístico vía geoBoundaries); [IGAC](https://www.igac.gov.co/datos-abiertos/datos-abiertos-geoespaciales) como referencia |
| Elevación y pendiente | SRTM 90 m (OpenTopoData) |
| Distancia a fallas | GEM Global Active Faults |
| Distancia a cuerpos de agua | OpenStreetMap |
| Sismos posteriores a 2018 (solo para validar) | [USGS ComCat](https://earthquake.usgs.gov/fdsnws/event/1/) M≥3,5 |

## Limitaciones conocidas
- `suelo_predominante` es una **estimación** (clase NEHRP con el proxy topográfico de Wald & Allen 2007), no un mapa de suelos; conviene verificar los umbrales.
- Quedaron fuera por no tener fuente abierta descargable: nivel freático y antigüedad de construcciones. **Pendiente:** infraestructura crítica (OSM/Overpass estaba saturado).
- Falta el municipio de Jordán (no se halló su polígono en la capa de límites).
- La categoría mide amenaza sísmica próxima, no daños; la exposición (población) entra en el reto de refuerzo.
- El costo y la escala de beneficio del agente de refuerzo son **simulados** (unidades relativas).
- **Navegación:** la app muestra una sección a la vez (no `st.tabs`) para que los mapas se dibujen bien.
- **Proyección a 2027:** es una extrapolación de tasas (Poisson estacionario), no una predicción de eventos. El SGC reportaba magnitudes más altas antes de 2008, por eso λ usa solo 2008–2017. En la validación (USGS 2019–2025) el total departamental coincide (337 observados vs 302 esperados), pero la distribución por municipio es sensible a la agencia: el USGS ubica parte del Nido de Bucaramanga en Aratoca/Piedecuesta y el SGC en Los Santos.

## Lo diferenciador: Centro de decisión
- **Índice de Riesgo Integral (IRI):** amenaza (PGA oficial del SGC) × población expuesta (DANE) × vulnerabilidad del terreno. Convierte la predicción en una lista de dónde actuar primero. Es relativo, con pesos propuestos por el equipo.
- **Monitor de actividad reciente:** el catálogo del SGC se detiene en 2018; se complementa con el USGS en vivo y se compara, municipio por municipio, lo ocurrido en 12 meses con lo normal (cola de Poisson). Es una señal estadística, no un pronóstico.
- **IA explicable:** cada predicción muestra cuánto empuja cada variable la probabilidad de riesgo Alto (en puntos porcentuales).
- **Ficha municipal** en HTML (imprimible a PDF) y **ranking en CSV** para gestores.
- El IRI usa el PGA y no el conteo de epicentros porque casi todos los M≥4 de Santander pertenecen al Nido de Bucaramanga (~150 km de profundidad) y sacuden también a los municipios vecinos.
