from setfit import SetFitModel
import pandas as pd
from utils.config_loader import get_predict_config
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# === CONFIGURACIÓN DESDE config.yaml ===
predict_config = get_predict_config()
MODEL_PATH = predict_config.get('model_dir', 'models/setfit_model_mpnet')

# === CARGA DEL MODELO ===
print(f"Cargando modelo desde: {MODEL_PATH}")
model = SetFitModel.from_pretrained(MODEL_PATH)

# === TEXTOS DE PRUEBA ===
textos = [
    "ACCESORIO DE IMPRESIÓN DENTAL 3M KIT MASILLAS C/CUCHARA DISPENSADORA 400 GR KIT 2 UNIDADES",
    "COMPRA DE INSUMOS MÉDICOS",
    "SERVICIO DE MANTENIMIENTO DE COMPUTADORES"
]

# === PREDICCIÓN ===
predicciones = model.predict(textos)
print("Predicciones:", predicciones)

# === PROBABILIDADES DE CADA CLASE ===
probas = model.predict_proba(textos)
import numpy as np
confianzas = np.round(probas.max(dim=1).values.numpy(), 3)
print("\nResultados detallados:")
for texto, clase, confianza in zip(textos, predicciones, confianzas):
    print(f"Texto: {texto}\nClase: {clase}\nConfianza: {confianza}\n---")
