"""
Proceso ETL: Airbnb Ciudad de México.

  Extract   -> descarga los datos crudos de Inside Airbnb (si no existen)
  Transform -> limpia tipos, valores faltantes y atípicos; crea variables nuevas
  Load      -> guarda el resultado en SQLite y en CSV

Uso:  python src/etl.py
"""
import json
import re
import sqlite3

import numpy as np
import pandas as pd
import requests

from config import (ARCHIVOS_FUENTE, CSV_PATH, DATA_PROCESSED, DATA_RAW,
                    DB_PATH, TABLA)

COLUMNAS_FUENTE = [
    "id", "name", "host_id", "neighbourhood_cleansed", "latitude", "longitude",
    "property_type", "room_type", "accommodates", "bathrooms_text", "bedrooms",
    "beds", "amenities", "price", "minimum_nights", "availability_365",
    "number_of_reviews", "review_scores_rating", "host_is_superhost",
    "hosts_time_as_host_years",
    "calculated_host_listings_count",
]

TIPOS_HABITACION = {
    "Entire home/apt": "Casa/depto completo",
    "Private room": "Habitación privada",
    "Shared room": "Habitación compartida",
    "Hotel room": "Habitación de hotel",
}


# --------------------------------------------------------------------------- #
# EXTRACT
# --------------------------------------------------------------------------- #
def extraer() -> pd.DataFrame:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    for nombre, url in ARCHIVOS_FUENTE.items():
        destino = DATA_RAW / nombre
        if destino.exists():
            print(f"[extract] {nombre} ya existe, se omite la descarga")
            continue
        print(f"[extract] Descargando {url}")
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        destino.write_bytes(resp.content)

    df = pd.read_csv(DATA_RAW / "listings.csv.gz", usecols=COLUMNAS_FUENTE)
    print(f"[extract] {len(df):,} anuncios leídos, {df.shape[1]} columnas")
    return df


# --------------------------------------------------------------------------- #
# TRANSFORM
# --------------------------------------------------------------------------- #
def _banos_desde_texto(texto) -> float:
    """'1.5 baths' -> 1.5, 'Half-bath' -> 0.5, 'Shared half-bath' -> 0.5."""
    if pd.isna(texto):
        return np.nan
    numero = re.search(r"\d+(\.\d+)?", texto)
    if numero:
        return float(numero.group())
    if "half" in texto.lower():
        return 0.5
    return np.nan


def transformar(df: pd.DataFrame) -> pd.DataFrame:
    n_inicial = len(df)
    df = df.copy()

    # Precio: "$1,250.00" -> 1250.0
    df["precio"] = (df["price"].str.replace(r"[\$,]", "", regex=True)
                               .astype(float))
    df = df.dropna(subset=["precio"])
    print(f"[transform] Sin precio: se eliminan {n_inicial - len(df):,} filas")

    # Atípicos: se recortan el 1 % más barato y el 1 % más caro
    p01, p99 = df["precio"].quantile([0.01, 0.99])
    antes = len(df)
    df = df[df["precio"].between(p01, p99)]
    print(f"[transform] Precios fuera de [{p01:,.0f}, {p99:,.0f}] MXN: "
          f"se eliminan {antes - len(df):,} filas")

    # Solo estancias cortas (Airbnb "turístico")
    antes = len(df)
    df = df[df["minimum_nights"] <= 30]
    print(f"[transform] Noches mínimas > 30: se eliminan {antes - len(df):,} filas")

    # Baños a partir del texto libre
    df["banos"] = df["bathrooms_text"].apply(_banos_desde_texto)
    df["bano_compartido"] = (df["bathrooms_text"].str.contains("shared", case=False)
                                                 .fillna(False).astype(int))

    # Amenidades: lista JSON -> número de amenidades
    df["num_amenidades"] = df["amenities"].apply(
        lambda s: len(json.loads(s)) if isinstance(s, str) else 0)

    # Booleano "t"/"f" -> 1/0
    df["host_is_superhost"] = (df["host_is_superhost"] == "t").astype(int)

    # Reseñas: si no hay calificación, se marca y se imputa la mediana
    df["sin_resenas"] = df["review_scores_rating"].isna().astype(int)
    df["review_scores_rating"] = df["review_scores_rating"].fillna(
        df["review_scores_rating"].median())

    # Faltantes numéricos: mediana según la capacidad de huéspedes
    for col in ["bedrooms", "beds", "banos"]:
        df[col] = df[col].fillna(
            df.groupby("accommodates")[col].transform("median"))
        df[col] = df[col].fillna(df[col].median())

    # Valores imposibles (errores de captura): más de 10 recámaras/baños o 16 camas
    antes = len(df)
    df = df[(df["bedrooms"] <= 10) & (df["banos"] <= 10) & (df["beds"] <= 16)]
    print(f"[transform] Recámaras/baños/camas imposibles: se eliminan "
          f"{antes - len(df):,} filas")

    df["room_type"] = df["room_type"].map(TIPOS_HABITACION)

    df = df.rename(columns={
        "neighbourhood_cleansed": "alcaldia",
        "name": "nombre",
        "latitude": "latitud",
        "longitude": "longitud",
        "property_type": "tipo_propiedad",
        "room_type": "tipo_habitacion",
        "accommodates": "huespedes",
        "bedrooms": "recamaras",
        "beds": "camas",
        "minimum_nights": "noches_minimas",
        "availability_365": "disponibilidad_365",
        "number_of_reviews": "num_resenas",
        "review_scores_rating": "calificacion",
        "host_is_superhost": "superhost",
        "hosts_time_as_host_years": "anios_anfitrion",
        "calculated_host_listings_count": "anuncios_del_anfitrion",
    })

    # Variable objetivo para el árbol de decisión: ¿es caro?
    mediana = df["precio"].median()
    df["es_caro"] = (df["precio"] > mediana).astype(int)
    print(f"[transform] Precio mediano: {mediana:,.0f} MXN "
          f"(umbral para 'es_caro')")

    df = df.drop(columns=["price", "bathrooms_text", "amenities"])
    print(f"[transform] Resultado: {len(df):,} filas "
          f"({len(df) / n_inicial:.1%} del total)")
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# LOAD
# --------------------------------------------------------------------------- #
def cargar(df: pd.DataFrame) -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as con:
        df.to_sql(TABLA, con, if_exists="replace", index=False)
    df.to_csv(CSV_PATH, index=False, encoding="utf-8")
    print(f"[load] Tabla '{TABLA}' guardada en {DB_PATH.name}")
    print(f"[load] Copia en CSV: {CSV_PATH.name}")


def main():
    crudo = extraer()
    limpio = transformar(crudo)
    cargar(limpio)


if __name__ == "__main__":
    main()
