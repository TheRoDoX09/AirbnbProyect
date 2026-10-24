"""
Árbol de decisión: clasificar si un alojamiento es "caro"
(precio por encima de la mediana de la ciudad).

La profundidad del árbol se elige con validación cruzada (5 folds)
para equilibrar precisión e interpretabilidad.

Uso:  python src/arbol.py
"""
import json

import joblib
import matplotlib
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

from config import MODELOS, REPORTES, SEMILLA, VARS_CATEGORICAS, VARS_NUMERICAS
from datos import cargar_datos, preprocesador

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CLASES = ["Económico", "Caro"]


def main():
    df = cargar_datos()
    X = df[VARS_CATEGORICAS + VARS_NUMERICAS]
    y = df["es_caro"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEMILLA, stratify=y)

    pipeline = Pipeline([
        ("prep", preprocesador(para_regresion=False)),
        ("arbol", DecisionTreeClassifier(random_state=SEMILLA)),
    ])
    busqueda = GridSearchCV(
        pipeline,
        {"arbol__max_depth": [3, 4, 5, 6, 8, 10],
         "arbol__min_samples_leaf": [20, 50, 100]},
        cv=5, scoring="accuracy", n_jobs=-1)
    busqueda.fit(X_train, y_train)
    modelo = busqueda.best_estimator_
    arbol = modelo.named_steps["arbol"]
    print(f"Mejores hiperparámetros: {busqueda.best_params_}")

    pred = modelo.predict(X_test)
    metricas = {
        "accuracy": accuracy_score(y_test, pred),
        "mejores_parametros": busqueda.best_params_,
        "umbral_caro_mxn": float(df["precio"].median()),
        "matriz_confusion": confusion_matrix(y_test, pred).tolist(),
        "reporte": classification_report(y_test, pred, target_names=CLASES,
                                         output_dict=True),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    print(f"Accuracy en prueba: {metricas['accuracy']:.3f}")
    print(classification_report(y_test, pred, target_names=CLASES))

    nombres = modelo.named_steps["prep"].get_feature_names_out()
    importancias = (pd.DataFrame({"variable": nombres,
                                  "importancia": arbol.feature_importances_})
                      .query("importancia > 0")
                      .sort_values("importancia", ascending=False))

    MODELOS.mkdir(exist_ok=True)
    REPORTES.mkdir(exist_ok=True)
    joblib.dump(modelo, MODELOS / "arbol_decision.joblib")
    (REPORTES / "arbol_metricas.json").write_text(
        json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8")
    importancias.to_csv(REPORTES / "arbol_importancias.csv", index=False)
    (REPORTES / "arbol_reglas.txt").write_text(
        export_text(arbol, feature_names=list(nombres), max_depth=4),
        encoding="utf-8")

    # Solo los primeros 2 niveles, para que la imagen sea legible
    fig, ax = plt.subplots(figsize=(16, 7))
    plot_tree(arbol, feature_names=nombres, class_names=CLASES, max_depth=2,
              filled=True, rounded=True, impurity=False, proportion=True,
              precision=2, fontsize=12, ax=ax)
    fig.tight_layout()
    fig.savefig(REPORTES / "arbol_decision.png", dpi=130)
    print(f"Modelo y reportes guardados en {MODELOS.name}/ y {REPORTES.name}/")


if __name__ == "__main__":
    main()
