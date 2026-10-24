# Airbnb en Ciudad de México: ETL, modelos predictivos y dashboard

Análisis de ~27,700 alojamientos de Airbnb en la CDMX (datos de
[Inside Airbnb](https://insideairbnb.com/get-the-data/), snapshot 2026-09-23).

**Pregunta central:** ¿qué determina el precio por noche de un Airbnb en la CDMX?

## Estructura

```
ProyectoJulia/
├── run_pipeline.py        # Ejecuta todo: ETL -> regresión -> árbol
├── app.py                 # Dashboard (Streamlit)
├── src/
│   ├── config.py          # Rutas y lista de variables
│   ├── etl.py             # 1. Proceso ETL
│   ├── datos.py           # Lectura de SQLite + preprocesamiento común
│   ├── regresion.py       # 2. Regresión lineal
│   └── arbol.py           # 3. Árbol de decisión
├── data/
│   ├── raw/               # Datos originales descargados
│   └── processed/         # airbnb_cdmx.db (SQLite) y CSV limpio
├── models/                # Modelos entrenados (.joblib)
└── reports/               # Métricas, coeficientes, imagen del árbol
```

## Cómo ejecutarlo

```bash
pip install -r requirements.txt
python run_pipeline.py      # descarga, limpia y entrena (~1 min)
streamlit run app.py        # abre el dashboard en el navegador
```

## 1. Proceso ETL (`src/etl.py`)

| Etapa | Qué hace |
|---|---|
| **Extract** | Descarga `listings.csv.gz` (30,468 anuncios, 90 columnas) y el mapa de alcaldías (GeoJSON). |
| **Transform** | Convierte el precio de texto (`"$1,250.00"`) a número · elimina anuncios sin precio (2,108) · recorta el 1 % de precios extremos · quita estancias de más de 30 noches · extrae los baños del texto (`"1.5 shared baths"`) · cuenta las amenidades de una lista JSON · convierte booleanos `t/f` · imputa faltantes con la mediana · elimina valores imposibles (p. ej. 50 recámaras) · traduce y renombra columnas · crea la variable objetivo `es_caro`. |
| **Load** | Guarda la tabla `alojamientos` en SQLite (`data/processed/airbnb_cdmx.db`) y una copia en CSV. |

Resultado: **27,672 anuncios limpios (90.8 % del total)**.

## 2. Regresión lineal (`src/regresion.py`)

- **Objetivo:** predecir el precio por noche (MXN).
- Se modela `log(precio)` porque el precio tiene una cola larga a la derecha.
- Variables: alcaldía, tipo de alojamiento, huéspedes, recámaras, camas, baños,
  amenidades, reseñas, calificación, superhost, etc. (16 en total).
- Partición 80 % entrenamiento / 20 % prueba.

| Métrica | Valor |
|---|---|
| R² (escala log) | 0.53 |
| Error absoluto medio | $690 MXN |
| RMSE | $1,266 MXN |

**Hallazgos:** con todo lo demás igual, un alojamiento en Tláhuac o Iztapalapa
cuesta entre 40 % y 50 % menos que uno en Cuauhtémoc. Una habitación privada
cuesta ~25 % menos que un depto completo.

## 3. Árbol de decisión (`src/arbol.py`)

- **Objetivo:** clasificar un alojamiento como **Caro** (más de $1,501 MXN,
  la mediana) o **Económico**.
- Profundidad y tamaño mínimo de hoja elegidos con `GridSearchCV` (validación
  cruzada de 5 folds).

| Métrica | Valor |
|---|---|
| Accuracy | 78.6 % |
| Precisión (Caro) | 79.1 % |
| Recall (Caro) | 77.6 % |

**Hallazgos:** la primera división es si el alojamiento es un depto completo
(41 % de la importancia), seguida del número de baños (20 %) y de amenidades (9 %).

## 4. Dashboard (`app.py`)

- **Exploración:** KPIs, mapa coroplético del precio mediano por alcaldía, mapa
  de anuncios, ranking de alcaldías, distribución de precios. Con filtros.
- **Regresión lineal:** métricas, gráfico real vs. predicho, efecto de cada variable.
- **Árbol de decisión:** métricas, matriz de confusión, importancia de variables,
  diagrama del árbol y sus reglas.
- **Predictor:** describes un alojamiento y ambos modelos estiman su precio.
