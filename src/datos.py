"""Lectura de los datos limpios y preprocesamiento común a ambos modelos."""
import sqlite3

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import DB_PATH, TABLA, VARS_CATEGORICAS, VARS_NUMERICAS

# Categorías de referencia para la regresión: cada coeficiente se lee como
# "cuánto cambia el precio frente a un depto completo en Cuauhtémoc"
CATEGORIAS_BASE = ["Cuauhtémoc", "Casa/depto completo"]


def cargar_datos() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as con:
        return pd.read_sql(f"SELECT * FROM {TABLA}", con)


def preprocesador(para_regresion: bool) -> ColumnTransformer:
    """One-hot para las categóricas; en la regresión además estandariza.

    La regresión lineal necesita una categoría base (si no, las dummies son
    colineales con el intercepto) y escalar para comparar coeficientes.
    Al árbol de decisión no le afecta nada de eso, así que ahí las numéricas
    quedan en su escala original y las reglas se leen en unidades reales.
    """
    if para_regresion:
        onehot = OneHotEncoder(drop=CATEGORIAS_BASE, sparse_output=False,
                               handle_unknown="ignore")
        numericas = StandardScaler()
    else:
        onehot = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        numericas = "passthrough"
    return ColumnTransformer([
        ("cat", onehot, VARS_CATEGORICAS),
        ("num", numericas, VARS_NUMERICAS),
    ], verbose_feature_names_out=False)
