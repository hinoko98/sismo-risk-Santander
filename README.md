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

## Limitaciones conocidas
- `suelo_predominante` es una **estimación** (clase NEHRP con el proxy topográfico de Wald & Allen 2007), no un mapa de suelos; conviene verificar los umbrales.
- Quedaron fuera por no tener fuente abierta descargable: nivel freático y antigüedad de construcciones. **Pendiente:** infraestructura crítica (OSM/Overpass estaba saturado).
- Falta el municipio de Jordán (no se halló su polígono en la capa de límites).
- La categoría mide amenaza sísmica próxima, no daños; la exposición (población) entra en el reto de refuerzo.
- El costo y la escala de beneficio del agente de refuerzo son **simulados** (unidades relativas).
