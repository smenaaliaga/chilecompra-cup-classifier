import pandas as pd
from setfit import SetFitModel
import os
import numpy as np

# ==== PARÁMETROS ====
# Ruta al modelo entrenado 
MODEL_PATH = "models/setfit_model_mpnet"
# Archivo CSV de entrada con columna 'text'
INPUT_CSV = "data/chilecompra_2024_processed.csv"
# Archivo CSV de salida con predicciones
OUTPUT_CSV = "output/predictions_chilecompra_2024_mpnet.csv"

# ==== CARGA DE MODELO ====
print(f"Cargando modelo desde: {MODEL_PATH}")
model = SetFitModel.from_pretrained(MODEL_PATH)

# ==== LECTURA DE TEXTOS ====
if not os.path.exists(INPUT_CSV):
    print(f"ERROR: No se encuentra el archivo {INPUT_CSV}")
    exit(1)
df = pd.read_csv(INPUT_CSV)
if "glosa" not in df.columns:
    print("ERROR: El archivo debe tener una columna llamada 'text'.")
    exit(1)

# ==== CLASIFICACIÓN ====
print(f"Clasificando {len(df)} textos...")
preds = model.predict(df["glosa"].tolist())
probas = model.predict_proba(df["glosa"].tolist())
confidences = np.round(probas.max(dim=1).values.numpy(), 3)
df["prediction"] = preds
df["confidence"] = confidences

# ==== GUARDADO DE RESULTADOS ====
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df.to_csv(OUTPUT_CSV, encoding="latin1", index=False)
print(f"Predicciones guardadas en {OUTPUT_CSV}")

preds = pd.read_csv("output/predictions_chilecompra_2024_mpnet.csv", encoding="utf-8")
merged = df.merge(preds, on="id", suffixes=("", "_pred"))
merged.to_csv("output/predictions_chilecompra_2024_mpnet.csv", index=False, encoding="utf-8")

# ==== ANÁLISIS ESTADÍSTICO (si existe columna CUP) ====
if "cup" in df.columns:
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
    y_true = df["cup"].astype(str).tolist()
    y_pred = df["prediction"].astype(str).tolist()
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    f1w = f1_score(y_true, y_pred, average="weighted")
    print(f"\nEstadísticos de predicción (cup vs prediction):")
    print(f"Accuracy: {acc:.4f} | F1 Macro: {f1m:.4f} | F1 Weighted: {f1w:.4f}")
    print("\nReporte detallado:\n", classification_report(y_true, y_pred))
    print("\nMatriz de confusión:\n", confusion_matrix(y_true, y_pred))
