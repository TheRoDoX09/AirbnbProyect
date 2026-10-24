"""
Regresión lineal: predecir el precio por noche (MXN).

El precio tiene una cola larga a la derecha, así que el modelo se entrena
sobre log(precio) y las predicciones se regresan a pesos con exp().

Uso:  python src/regresion.py
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from config import MODELOS, REPORTES, SEMILLA, VARS_CATEGORICAS, VARS_NUMERICAS
from datos import cargar_datos, preprocesador


def main():
    df = cargar_datos()
    X = df[VARS_CATEGORICAS + VARS_NUMERICAS]
    y = np.log(df["precio"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEMILLA)

    modelo = Pipeline([
        ("prep", preprocesador(para_regresion=True)),
        ("reg", LinearRegression()),
    ])
    modelo.fit(X_train, y_train)

    pred_log = modelo.predict(X_test)
    real = np.exp(y_test)
    pred = np.exp(pred_log)

    metricas = {
        "r2_log": r2_score(y_test, pred_log),
        "r2_pesos": r2_score(real, pred),
        "rmse_pesos": float(np.sqrt(mean_squared_error(real, pred))),
        "mae_pesos": mean_absolute_error(real, pred),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    print("Regresión lineal sobre log(precio)")
    print(f"  R² (escala log):  {metricas['r2_log']:.3f}")
    print(f"  R² (pesos):       {metricas['r2_pesos']:.3f}")
    print(f"  RMSE:             {metricas['rmse_pesos']:,.0f} MXN")
    print(f"  MAE:              {metricas['mae_pesos']:,.0f} MXN")

    # Coeficientes: con log(precio), exp(coef) - 1 = cambio % en el precio
    # por cada desviación estándar (numéricas) o frente a la categoría base
    nombres = modelo.named_steps["prep"].get_feature_names_out()
    coefs = pd.DataFrame({
        "variable": nombres,
        "coeficiente": modelo.named_steps["reg"].coef_,
    })
    coefs["efecto_pct"] = (np.exp(coefs["coeficiente"]) - 1) * 100
    coefs = coefs.sort_values("coeficiente", key=abs, ascending=False)

    predicciones = X_test.assign(precio_real=real.values, precio_predicho=pred)

    MODELOS.mkdir(exist_ok=True)
    REPORTES.mkdir(exist_ok=True)
    joblib.dump(modelo, MODELOS / "regresion_lineal.joblib")
    (REPORTES / "regresion_metricas.json").write_text(
        json.dumps(metricas, indent=2), encoding="utf-8")
    coefs.to_csv(REPORTES / "regresion_coeficientes.csv", index=False)
    predicciones.to_csv(REPORTES / "regresion_predicciones.csv", index=False)
    print(f"Modelo y reportes guardados en {MODELOS.name}/ y {REPORTES.name}/")


if __name__ == "__main__":
    main()
