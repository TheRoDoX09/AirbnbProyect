"""Rutas y parámetros compartidos por todo el proyecto."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

DATA_RAW = RAIZ / "data" / "raw"
DATA_PROCESSED = RAIZ / "data" / "processed"
MODELOS = RAIZ / "models"
REPORTES = RAIZ / "reports"

# Snapshot de Inside Airbnb para Ciudad de México
FECHA_SNAPSHOT = "2026-09-23"
URL_BASE = f"https://data.insideairbnb.com/mexico/df/mexico-city/{FECHA_SNAPSHOT}"
ARCHIVOS_FUENTE = {
    "listings.csv.gz": f"{URL_BASE}/data/listings.csv.gz",
    "neighbourhoods.geojson": f"{URL_BASE}/visualisations/neighbourhoods.geojson",
}

DB_PATH = DATA_PROCESSED / "airbnb_cdmx.db"
CSV_PATH = DATA_PROCESSED / "airbnb_cdmx_limpio.csv"
GEOJSON_PATH = DATA_RAW / "neighbourhoods.geojson"
TABLA = "alojamientos"

SEMILLA = 42

# Variables que usan los modelos
VARS_CATEGORICAS = ["alcaldia", "tipo_habitacion"]
VARS_NUMERICAS = [
    "huespedes",
    "recamaras",
    "camas",
    "banos",
    "bano_compartido",
    "num_amenidades",
    "anios_anfitrion",
    "superhost",
    "noches_minimas",
    "num_resenas",
    "calificacion",
    "sin_resenas",
    "anuncios_del_anfitrion",
    "disponibilidad_365",
]
