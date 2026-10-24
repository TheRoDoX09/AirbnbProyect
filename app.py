"""
Dashboard: Airbnb en Ciudad de México.

Uso:  streamlit run app.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from config import (FECHA_SNAPSHOT, GEOJSON_PATH, MODELOS,  # noqa: E402
                    REPORTES, VARS_CATEGORICAS, VARS_NUMERICAS)
from datos import cargar_datos  # noqa: E402

# Paleta: azul secuencial para magnitudes; 3 colores categóricos para el tipo
# de alojamiento (en el mapa los puntos se mezclan, más de 3 no se distinguen)
AZULES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95",
          "#0d366b"]
AZUL = "#2a78d6"
NARANJA = "#eb6834"
COLORES_TIPO = {
    "Casa/depto completo": "#2a78d6",
    "Habitación privada": "#eb6834",
    "Otro (compartida/hotel)": "#1baf7a",
}

st.set_page_config(page_title="Airbnb CDMX", page_icon=":material/apartment:",
                   layout="wide")

st.markdown("""
<style>
.block-container { padding-top: 2rem; }
.encabezado {
    border-left: 4px solid #1f3a5f; padding: 0.2rem 0 0.2rem 1rem;
    margin-bottom: 1.2rem;
}
.encabezado h1 {
    font-size: 1.9rem; font-weight: 700; color: #1f3a5f;
    margin: 0; padding: 0; letter-spacing: -0.01em;
}
.encabezado p { color: #5b6573; font-size: 0.95rem; margin: 0.3rem 0 0; }
div[data-testid="stMetric"] { background: #f9fafc; }
div[data-testid="stMetricLabel"] p {
    text-transform: uppercase; letter-spacing: 0.04em; font-size: 0.75rem;
    color: #5b6573; font-weight: 600;
}
button[data-baseweb="tab"] p { font-weight: 500; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def datos():
    df = cargar_datos()
    df["tipo_grupo"] = df["tipo_habitacion"].where(
        df["tipo_habitacion"].isin(["Casa/depto completo", "Habitación privada"]),
        "Otro (compartida/hotel)")
    return df


@st.cache_data
def geojson():
    return json.loads(GEOJSON_PATH.read_text(encoding="utf-8"))


@st.cache_resource
def modelos():
    return (joblib.load(MODELOS / "regresion_lineal.joblib"),
            joblib.load(MODELOS / "arbol_decision.joblib"))


@st.cache_data
def reportes():
    leer_json = lambda f: json.loads((REPORTES / f).read_text(encoding="utf-8"))
    return {
        "reg_met": leer_json("regresion_metricas.json"),
        "reg_coef": pd.read_csv(REPORTES / "regresion_coeficientes.csv"),
        "reg_pred": pd.read_csv(REPORTES / "regresion_predicciones.csv"),
        "arb_met": leer_json("arbol_metricas.json"),
        "arb_imp": pd.read_csv(REPORTES / "arbol_importancias.csv"),
    }


ETIQUETAS = {
    "huespedes": "Huéspedes", "recamaras": "Recámaras", "camas": "Camas",
    "banos": "Baños", "bano_compartido": "Baño compartido",
    "num_amenidades": "Número de amenidades",
    "anios_anfitrion": "Años como anfitrión", "superhost": "Superhost",
    "noches_minimas": "Noches mínimas", "num_resenas": "Número de reseñas",
    "calificacion": "Calificación", "sin_resenas": "Sin reseñas",
    "anuncios_del_anfitrion": "Anuncios del anfitrión",
    "disponibilidad_365": "Días disponibles al año",
}


def nombre_legible(variable: str) -> str:
    for prefijo in ("alcaldia_", "tipo_habitacion_"):
        if variable.startswith(prefijo):
            return variable.removeprefix(prefijo)
    return ETIQUETAS.get(variable, variable)


def ajustar(fig, alto=380):
    fig.update_layout(height=alto, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=-0.15),
                      font=dict(family="Inter, sans-serif", color="#1c2430"),
                      title_font=dict(size=15, color="#1f3a5f"))
    return fig


df = datos()
rep = reportes()

# --------------------------------------------------------------------------- #
# Filtros
# --------------------------------------------------------------------------- #
st.markdown(
    f'<div class="encabezado"><h1>Airbnb en Ciudad de México</h1>'
    f'<p>Análisis de precios y modelos predictivos · Datos de Inside Airbnb, '
    f'snapshot del {FECHA_SNAPSHOT} · {len(df):,} anuncios después del '
    f'proceso ETL</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header(":material/tune: Filtros")
    alcaldias = sorted(df["alcaldia"].unique())
    sel_alc = st.multiselect("Alcaldía", alcaldias, placeholder="Todas")
    sel_tipo = st.multiselect("Tipo de alojamiento",
                              list(COLORES_TIPO), placeholder="Todos")
    # Redondeados a centenas para que el slider cubra todo el rango
    pmin = int(df["precio"].min() // 100 * 100)
    pmax = int(-(-df["precio"].max() // 100) * 100)
    rango = st.slider("Precio por noche (MXN)", pmin, pmax, (pmin, pmax),
                      step=100)
    st.caption("Los filtros aplican a la pestaña **Exploración**.")

filtrado = df[df["precio"].between(*rango)]
if sel_alc:
    filtrado = filtrado[filtrado["alcaldia"].isin(sel_alc)]
if sel_tipo:
    filtrado = filtrado[filtrado["tipo_grupo"].isin(sel_tipo)]

tab_exp, tab_reg, tab_arb, tab_pred = st.tabs(
    [":material/bar_chart: Exploración",
     ":material/trending_up: Regresión lineal",
     ":material/account_tree: Árbol de decisión",
     ":material/calculate: Predictor"])

# --------------------------------------------------------------------------- #
# Exploración
# --------------------------------------------------------------------------- #
with tab_exp:
    if filtrado.empty:
        st.warning("No hay anuncios con esos filtros.",
                   icon=":material/filter_alt_off:")
        st.stop()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Anuncios", f"{len(filtrado):,}", border=True)
    k2.metric("Precio mediano", f"${filtrado['precio'].median():,.0f}",
              border=True)
    k3.metric("Calificación promedio",
              f"{filtrado.loc[filtrado['sin_resenas'] == 0, 'calificacion'].mean():.2f} / 5",
              border=True)
    k4.metric("Superhosts", f"{filtrado['superhost'].mean():.0%}", border=True)

    c1, c2 = st.columns(2)
    por_alc = (filtrado.groupby("alcaldia")
                       .agg(precio_mediano=("precio", "median"),
                            anuncios=("precio", "size"))
                       .reset_index())
    with c1:
        fig = px.choropleth_map(
            por_alc, geojson=geojson(), locations="alcaldia",
            featureidkey="properties.neighbourhood", color="precio_mediano",
            color_continuous_scale=AZULES, map_style="carto-positron",
            center={"lat": 19.36, "lon": -99.13}, zoom=8.6, opacity=0.8,
            hover_data={"anuncios": ":,"},
            labels={"precio_mediano": "Precio mediano (MXN)",
                    "alcaldia": "Alcaldía", "anuncios": "Anuncios"},
            title="Precio mediano por alcaldía")
        fig.update_coloraxes(colorbar=dict(orientation="h", y=-0.12,
                                           thickness=12, title_side="top"))
        st.plotly_chart(ajustar(fig, 450), width="stretch")
    with c2:
        muestra = filtrado.sample(min(len(filtrado), 6000), random_state=0)
        fig = px.scatter_map(
            muestra, lat="latitud", lon="longitud", color="tipo_grupo",
            color_discrete_map=COLORES_TIPO, map_style="carto-positron",
            category_orders={"tipo_grupo": list(COLORES_TIPO)},
            center={"lat": 19.40, "lon": -99.16}, zoom=10.3, opacity=0.6,
            hover_name="nombre",
            hover_data={"precio": ":$,.0f", "alcaldia": True,
                        "latitud": False, "longitud": False, "tipo_grupo": False},
            labels={"tipo_grupo": "Tipo", "precio": "Precio",
                    "alcaldia": "Alcaldía"},
            title=f"Ubicación de los anuncios (muestra de {len(muestra):,})")
        fig.update_traces(marker=dict(size=6))
        st.plotly_chart(ajustar(fig, 450), width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        orden = por_alc.sort_values("precio_mediano")
        fig = px.bar(orden, x="precio_mediano", y="alcaldia", orientation="h",
                     text_auto="$,.0f", color_discrete_sequence=[AZUL],
                     hover_data={"anuncios": ":,"},
                     labels={"precio_mediano": "Precio mediano (MXN)",
                             "alcaldia": "", "anuncios": "Anuncios"},
                     title="Ranking de alcaldías por precio mediano")
        st.plotly_chart(ajustar(fig, 480), width="stretch")
    with c4:
        fig = px.histogram(filtrado, x="precio", color="tipo_grupo",
                           color_discrete_map=COLORES_TIPO, nbins=60,
                           category_orders={"tipo_grupo": list(COLORES_TIPO)},
                           barmode="stack",
                           labels={"precio": "Precio por noche (MXN)",
                                   "tipo_grupo": "Tipo", "count": "Anuncios"},
                           title="Distribución de precios")
        fig.update_layout(bargap=0.05, yaxis_title="Anuncios")
        st.plotly_chart(ajustar(fig, 480), width="stretch")

    with st.expander("Ver tabla de datos filtrados"):
        st.dataframe(filtrado.drop(columns=["tipo_grupo"]),
                     width="stretch", hide_index=True)

# --------------------------------------------------------------------------- #
# Regresión lineal
# --------------------------------------------------------------------------- #
with tab_reg:
    m = rep["reg_met"]
    st.markdown("**Objetivo:** predecir el precio por noche. El modelo se "
                "entrena sobre *log(precio)* porque los precios tienen una "
                "cola larga de alojamientos muy caros.")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("R² (escala log)", f"{m['r2_log']:.3f}", border=True)
    k2.metric("R² (pesos)", f"{m['r2_pesos']:.3f}", border=True)
    k3.metric("Error absoluto medio", f"${m['mae_pesos']:,.0f}", border=True)
    k4.metric("RMSE", f"${m['rmse_pesos']:,.0f}", border=True)

    c1, c2 = st.columns(2)
    with c1:
        pred = rep["reg_pred"]
        tope = float(pred[["precio_real", "precio_predicho"]].max().max())
        fig = px.scatter(pred, x="precio_real", y="precio_predicho",
                         opacity=0.35, color_discrete_sequence=[AZUL],
                         labels={"precio_real": "Precio real (MXN)",
                                 "precio_predicho": "Precio predicho (MXN)"},
                         title="Real vs. predicho (conjunto de prueba)")
        fig.add_trace(go.Scatter(x=[0, tope], y=[0, tope], mode="lines",
                                 line=dict(color="#898781", dash="dash", width=2),
                                 name="Predicción perfecta"))
        st.plotly_chart(ajustar(fig, 480), width="stretch")
    with c2:
        coef = rep["reg_coef"].head(15).copy()
        coef["variable"] = coef["variable"].map(nombre_legible)
        coef["direccion"] = np.where(coef["efecto_pct"] > 0, "Sube el precio",
                                     "Baja el precio")
        coef = coef.iloc[::-1]
        fig = px.bar(coef, x="efecto_pct", y="variable", orientation="h",
                     color="direccion",
                     color_discrete_map={"Sube el precio": AZUL,
                                         "Baja el precio": NARANJA},
                     text_auto="+.0f",
                     labels={"efecto_pct": "Efecto en el precio (%)",
                             "variable": "", "direccion": ""},
                     title="Las 15 variables con más efecto")
        fig.update_traces(textposition="outside", textangle=0, cliponaxis=False)
        st.plotly_chart(ajustar(fig, 480), width="stretch")

    st.info("**Cómo leer los efectos:** las alcaldías y los tipos de "
            "alojamiento se comparan contra *un depto completo en Cuauhtémoc*. "
            "Las variables numéricas indican el cambio en el precio cuando la "
            "variable aumenta una desviación estándar, con todo lo demás igual.",
            icon=":material/info:")

# --------------------------------------------------------------------------- #
# Árbol de decisión
# --------------------------------------------------------------------------- #
with tab_arb:
    m = rep["arb_met"]
    umbral = m["umbral_caro_mxn"]
    st.markdown(f"**Objetivo:** clasificar un alojamiento como **Caro** "
                f"(más de ${umbral:,.0f} MXN por noche, la mediana de la "
                f"ciudad) o **Económico**.")
    p = m["mejores_parametros"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Accuracy", f"{m['accuracy']:.1%}", border=True)
    k2.metric("Precisión (Caro)", f"{m['reporte']['Caro']['precision']:.1%}",
              border=True)
    k3.metric("Recall (Caro)", f"{m['reporte']['Caro']['recall']:.1%}",
              border=True)
    k4.metric("Profundidad elegida", p["arbol__max_depth"], border=True,
              help="Elegida por validación cruzada de 5 folds")

    c1, c2 = st.columns(2)
    with c1:
        cm = np.array(m["matriz_confusion"])
        etiquetas = ["Económico", "Caro"]
        fig = px.imshow(cm, x=etiquetas, y=etiquetas, text_auto=",",
                        color_continuous_scale=AZULES,
                        labels=dict(x="Predicción", y="Real", color="Anuncios"),
                        title="Matriz de confusión (conjunto de prueba)")
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(ajustar(fig, 420), width="stretch")
    with c2:
        imp = rep["arb_imp"].head(12).copy()
        imp["variable"] = imp["variable"].map(nombre_legible)
        fig = px.bar(imp.iloc[::-1], x="importancia", y="variable",
                     orientation="h", text_auto=".0%",
                     color_discrete_sequence=[AZUL],
                     labels={"importancia": "Importancia", "variable": ""},
                     title="Variables más importantes para el árbol")
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(ajustar(fig, 420), width="stretch")

    st.subheader("Primeros niveles del árbol")
    st.image(str(REPORTES / "arbol_decision.png"), width="stretch")
    with st.expander("Ver las reglas en texto"):
        st.code((REPORTES / "arbol_reglas.txt").read_text(encoding="utf-8"),
                language=None)

# --------------------------------------------------------------------------- #
# Predictor
# --------------------------------------------------------------------------- #
with tab_pred:
    st.markdown("Describe un alojamiento y los dos modelos estiman su precio.")
    reg, arbol = modelos()
    tipos = sorted(df["tipo_habitacion"].unique())

    c1, c2, c3 = st.columns(3)
    entrada = {
        "alcaldia": c1.selectbox("Alcaldía", alcaldias,
                                 index=alcaldias.index("Cuauhtémoc")),
        "tipo_habitacion": c1.selectbox(
            "Tipo de alojamiento", tipos,
            index=tipos.index("Casa/depto completo")),
        "huespedes": c1.number_input("Huéspedes", 1, 16, 2),
        "recamaras": c2.number_input("Recámaras", 0, 10, 1),
        "camas": c2.number_input("Camas", 1, 16, 1),
        "banos": c2.number_input("Baños", 0.0, 10.0, 1.0, step=0.5),
        "num_amenidades": c3.slider("Número de amenidades", 0, 100, 30),
        "superhost": int(c3.toggle("Es Superhost")),
        "calificacion": c3.slider("Calificación", 1.0, 5.0, 4.8, step=0.1),
    }
    entrada["bano_compartido"] = int(entrada["tipo_habitacion"]
                                     == "Habitación compartida")
    # El resto de las variables toma el valor típico (mediana)
    for var in VARS_NUMERICAS:
        entrada.setdefault(var, float(df[var].median()))
    X = pd.DataFrame([entrada])[VARS_CATEGORICAS + VARS_NUMERICAS]

    precio = float(np.exp(reg.predict(X)[0]))
    prob_caro = float(arbol.predict_proba(X)[0][1])

    st.divider()
    r1, r2 = st.columns(2)
    r1.metric("Precio estimado (regresión lineal)", f"${precio:,.0f} MXN",
              border=True)
    r2.metric("Clasificación (árbol de decisión)",
              "Caro" if prob_caro >= 0.5 else "Económico", border=True)
    r2.caption(f"Probabilidad de ser caro: **{prob_caro:.0%}**")
    st.caption("Las demás características (reseñas, disponibilidad, años "
               "como anfitrión, etc.) se fijan en el valor típico de la ciudad.")
