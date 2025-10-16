import pandas as pd
import os

# === VARIABLES DE ENTRADA/SALIDA ===
DICCIONARIO_PATH = "data/Diccionario_CUP.xlsx"
PRED_FILE = "output/predictions_chilecompra_2024.xlsx"

# === LECTURA DE DICCIONARIO ===
dicc = pd.read_excel(DICCIONARIO_PATH)
assert "CUP_2013" in dicc.columns, f"Falta columna CUP_2013 en {DICCIONARIO_PATH}"
assert "GLOSA_CUP_2013" in dicc.columns, f"Falta columna GLOSA_CUP_2013 en {DICCIONARIO_PATH}"

# === LECTURA DE PREDICCIONES ===
if PRED_FILE.lower().endswith('.xlsx'):
    preds = pd.read_excel(PRED_FILE)
else:
    preds = pd.read_csv(PRED_FILE, encoding="utf-8")
assert "prediction" in preds.columns, f"Falta columna prediction en archivo de predicciones: {PRED_FILE}"

# === UNIÓN Y GUARDADO ===
preds["prediction"] = preds["prediction"].astype(str)
dicc["CUP_2013"] = dicc["CUP_2013"].astype(str)
merged = preds.merge(dicc[["CUP_2013", "GLOSA_CUP_2013"]], left_on="prediction", right_on="CUP_2013", how="left")
merged = merged.drop(columns=["CUP_2013"])

if PRED_FILE.lower().endswith('.xlsx'):
    merged.to_excel(PRED_FILE, index=False)
else:
    merged.to_csv(PRED_FILE, index=False, encoding="utf-8")
print(f"Columna GLOSA_CUP_2013 agregada a {PRED_FILE}")