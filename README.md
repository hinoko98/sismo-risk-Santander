# 🌋 Sismo-Risk Santander

Plataforma web para la gestión del riesgo sísmico en Santander — Hackathon CIAI UTSmart 2026.
Todos los datos son **reales** (SGC, DANE, IGAC/MGN, GEM, OpenStreetMap, SRTM). Solo el costo de evaluación del reto de refuerzo es simulado, como permite el reto.

## Ejecución rápida
```bash
pip install -r requirements.txt
streamlit run app.py
```
La carpeta `data/` incluye los datos ya descargados (catálogo sísmico hasta octubre de 2026). Si falta algo, la app lo regenera (requiere internet):
`python -m src.real_data` (descarga, tarda varios minutos) → `python -m src.data_processing` (ETL) → `python -m src.model_training` → `python -m src.reinforcement`.

## Secciones de la app
| Sección | Qué muestra |
|---|---|
| 🧭 Centro de decisión | Decisión recomendada por municipio **con su evidencia**, Índice de Riesgo Integral, semáforo de actividad de los últimos 12 meses, ranking, ficha municipal descargable |
| 📊 Exploración (EDA) | KPIs, gráficos, dataset limpio y fuentes |
| 🗺️ Mapa de riesgo | Riesgo esperado por municipio (Random Forest) o grupo (K-Means), mapa de calor y mayores sismos |
| 🎯 Supervisado | Random Forest, métricas contra líneas base, explicación de cada predicción |
| 🧩 No supervisado | K-Means + PCA sobre 21 variables |
| 🤖 Refuerzo | Q-learning para priorizar evaluaciones con presupuesto limitado |
| 🔭 Proyección 2027 | Probabilidad por municipio (Poisson) y prueba retrospectiva |

## Los tres retos
| Reto | Módulo | Qué hace |
|---|---|---|
| Básico · supervisado | `src/data_processing.py`, `src/model_training.py` | ETL con geopandas, imputación, panel municipio-año (29 variables), Random Forest (`class_weight='balanced'`, One-Hot, split temporal, CV de series de tiempo) |
| Medio · no supervisado | `src/unsupervised.py` | K-Means (k configurable) + PCA sobre 21 variables agregadas por municipio |
| Avanzado · refuerzo | `src/reinforcement.py` | Q-learning; el estado usa actividad, población, costo, presupuesto, amenaza (PGA) y el grupo del K-Means |
| Diferenciador | `src/riesgo_integral.py`, `src/decision.py`, `src/monitor.py`, `src/ficha.py` | Índice de Riesgo Integral, reglas de decisión auditables, monitor de actividad y ficha municipal |
| Proyección | `src/proyeccion.py` | Poisson por municipio (tasas 2016–2025) y prueba retrospectiva (tasas 2008–2017 contra lo ocurrido en 2018–2025) |

## Etiqueta (reproducible y sin fuga de datos)
Unidad: **municipio-año**. Las variables de entrada usan información hasta el año *t*; la categoría es la **magnitud máxima registrada en el año *t+1***:
**Bajo** < 3,0 · **Medio** 3,0–<4,0 · **Alto** ≥ 4,0. Ninguna entrada contiene datos del año siguiente (incluida la imputación, que solo usa años previos). La evaluación es temporal (entrena con 1994–2018, prueba con 2019–2024) y se compara con líneas base (clase mayoritaria y persistencia).

## Fuentes
| Variable(s) | Fuente |
|---|---|
| Actividad sísmica (frecuencia, magnitud, profundidad, RMS, GAP, fases, errores de localización) | **SGC**, dos catálogos oficiales: [Consulta Experta](https://bdrsnc.sgc.gov.co/paginas1/catalogo/Consulta_Experta/consultaexperta_2.php) (jun-1993 a 2017) y su sucesor [SeisComP](https://bdrsnc.sgc.gov.co/paginas1/catalogo/Consulta_Experta_Seiscomp/consultaexperta.php) (2018 a hoy). Se verificó que empalman: el volumen de sismos M≥3 y M≥4 por año es el mismo antes y después del cambio |
| PGA (475 años, roca) | [SGC Amenaza sísmica 2020](https://amenazasismica.sgc.gov.co/) |
| Población, área, límites | [DANE](https://www.dane.gov.co/index.php/estadisticas-por-tema/demografia-y-poblacion/proyecciones-de-poblacion) (proyecciones 2025 y Marco Geoestadístico vía geoBoundaries); [IGAC](https://www.igac.gov.co/datos-abiertos/datos-abiertos-geoespaciales) como referencia |
| Elevación y pendiente | SRTM 90 m (OpenTopoData) |
| Distancia a fallas | GEM Global Active Faults |
| Distancia a cuerpos de agua | OpenStreetMap |

No se usa ninguna otra base sísmica: el USGS se retiró del proyecto porque el SGC ya cubre hasta hoy con la misma agencia y escala. Con internet, el Centro de decisión pide al SGC los eventos más nuevos cada hora.

## Limitaciones conocidas
- `suelo_predominante` es una **estimación** (clase NEHRP con el proxy topográfico de Wald & Allen 2007), no un mapa de suelos; conviene verificar los umbrales.
- Quedaron fuera por no tener fuente abierta descargable: nivel freático y antigüedad de construcciones. **Pendiente:** infraestructura crítica (OSM/Overpass estaba saturado).
- Falta el municipio de Jordán (no se halló su polígono en la capa de límites).
- La categoría mide amenaza sísmica próxima, no daños; la exposición (población) entra en el Índice de Riesgo Integral y en el reto de refuerzo.
- El Índice de Riesgo Integral y las reglas de decisión usan pesos y umbrales propuestos por el equipo; son un apoyo a la decisión y deben validarlos especialistas.
- El costo y la escala de beneficio del agente de refuerzo son **simulados** (unidades relativas).
- **Proyección:** extrapolación de tasas (Poisson estacionario), no una predicción de eventos. El SGC reportaba magnitudes más altas antes de 2008, por eso λ usa solo años recientes.
- **Navegación:** la app muestra una sección a la vez (no `st.tabs`) para que los mapas se dibujen bien.
