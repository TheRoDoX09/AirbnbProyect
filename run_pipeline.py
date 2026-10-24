"""Ejecuta todo el proyecto en orden: ETL -> regresión -> árbol.

Uso:  python run_pipeline.py
Después:  streamlit run app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import arbol  # noqa: E402
import etl  # noqa: E402
import regresion  # noqa: E402

for titulo, paso in [("1. ETL", etl.main),
                     ("2. Regresión lineal", regresion.main),
                     ("3. Árbol de decisión", arbol.main)]:
    print(f"\n{'=' * 60}\n{titulo}\n{'=' * 60}")
    paso()

print("\nListo. Ahora ejecuta:  streamlit run app.py")
