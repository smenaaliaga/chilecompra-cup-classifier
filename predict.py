import pandas as pd
from setfit import SetFitModel
import os
import numpy as np
from utils.config_loader import get_predict_config

# ==== CARGA DE CONFIGURACIÓN ====
predict_config = get_predict_config()

# ==== PARÁMETROS DESDE CONFIGURACIÓN ====
# Ruta al modelo entrenado 
MODEL_PATH = predict_config.get('model_dir', 'models/setfit_model_mpnet')
# Archivo CSV de entrada con columna 'text'
INPUT_CSV = predict_config.get('input_file', 'data/chilecompra_2024_processed.csv')
# Archivo CSV de salida con predicciones
OUTPUT_CSV = predict_config.get('output_file', 'output/predictions_chilecompra_2024_mpnet.csv')

# ==== CARGA DE MODELO ====
print(f"Cargando modelo desde: {MODEL_PATH}")
model = SetFitModel.from_pretrained(MODEL_PATH)

# ==== LECTURA DE TEXTOS ====
if not os.path.exists(INPUT_CSV):
    print(f"ERROR: No se encuentra el archivo {INPUT_CSV}")
    exit(1)
df = pd.read_csv(INPUT_CSV)

# Obtener nombres de columnas desde configuración
text_column = predict_config.get('text_column', 'glosa')
prediction_column = predict_config.get('prediction_column', 'prediction')
confidence_column = predict_config.get('confidence_column', 'confidence')

if text_column not in df.columns:
    print(f"ERROR: El archivo debe tener una columna llamada '{text_column}'.")
    exit(1)

# ==== CLASIFICACIÓN ====
print(f"Clasificando {len(df)} textos...")
preds = model.predict(df[text_column].tolist())
probas = model.predict_proba(df[text_column].tolist())
confidences = np.round(probas.max(dim=1).values.numpy(), 3)
df[prediction_column] = preds
df[confidence_column] = confidences

# ==== GUARDADO DE RESULTADOS ====
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df.to_csv(OUTPUT_CSV, encoding="latin1", index=False)
print(f"Predicciones guardadas en {OUTPUT_CSV}")

preds = pd.read_csv("output/predictions_chilecompra_2024_mpnet.csv", encoding="utf-8")
merged = df.merge(preds, on="id", suffixes=("", "_pred"))
merged.to_csv("output/predictions_chilecompra_2024_mpnet.csv", index=False, encoding="utf-8")

# ==== ANÁLISIS ESTADÍSTICO (si existe columna de etiquetas) ====
label_column = predict_config.get('label_column', 'cup')
if label_column in df.columns:
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
    y_true = df[label_column].astype(str).tolist()
    y_pred = df[prediction_column].astype(str).tolist()
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    f1w = f1_score(y_true, y_pred, average="weighted")
    print(f"\nEstadísticos de predicción ({label_column} vs {prediction_column}):")
    print(f"Accuracy: {acc:.4f} | F1 Macro: {f1m:.4f} | F1 Weighted: {f1w:.4f}")
    print("\nReporte detallado:\n", classification_report(y_true, y_pred))
    print("\nMatriz de confusión:\n", confusion_matrix(y_true, y_pred))
