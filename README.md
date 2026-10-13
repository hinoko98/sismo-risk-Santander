# 🌋 Sismo-Risk Santander

Plataforma web para la gestión del riesgo sísmico en Santander — Hackathon CIAI UTSmart 2026.

## Ejecución rápida
```bash
pip install -r requirements.txt
streamlit run app.py
```
Si `data/` o `models/` están vacíos, la app genera los datos, ejecuta el ETL y entrena el modelo sola.

## Estructura
| Ruta | Misión | Contenido |
|---|---|---|
| `src/data_processing.py` | 1 · Básico | Carga, imputación (mediana/moda), filtro espacial con geopandas, cálculo de `CATEGORIA_RIESGO` |
| `src/model_training.py` | 2 · Medio | One-Hot de `tipo_suelo`, split 80/20, `RandomForestClassifier(class_weight='balanced')`, CV estratificada, métricas, `models/sismo_rf_model.pkl` |
| `app.py` | 3 · Avanzado | Dashboard con 3 pestañas (EDA, mapa folium, simulador en sidebar), caché y manejo de errores |
| `src/data_generation.py` | — | Dataset sintético (SGC + IGAC + DANE) con Nido de Bucaramanga |
| `src/config.py` | — | Rutas, variables, polígono y falla de referencia |

Pasos por consola: `python -m src.data_processing` y `python -m src.model_training`.

## ⚠️ Datos
Los datos incluidos son **simulados**, y la categoría de riesgo es un índice calculado, por lo que las métricas del modelo miden
cuánto reproduce ese índice, no el riesgo real. Para usar datos reales: pon el catálogo en `data/sismos_SGC.csv`
(columnas del SGC se renombran automáticamente; las variables faltantes se imputan) y, opcionalmente, el polígono oficial
del IGAC en `data/santander_poligono.geojson`. Borra `data/santander_limpio.csv` y `models/*.pkl` para regenerarlos.
El contorno de Santander por defecto es una **aproximación**.

Fuentes: [SGC](https://www.sgc.gov.co/catalogo) · [Catálogo sísmico](https://catalogosismico.sgc.gov.co/visor/) · [DANE](https://www.dane.gov.co/index.php/estadisticas-por-tema/demografia-y-poblacion/proyecciones-de-poblacion) · [IGAC](https://www.igac.gov.co/datos-abiertos/datos-abiertos-geoespaciales) · [Amenaza sísmica](https://amenazasismica.sgc.gov.co/)
