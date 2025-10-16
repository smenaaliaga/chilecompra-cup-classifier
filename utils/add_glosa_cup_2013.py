import pandas as pd

# Leer diccionario CUP
dicc = pd.read_excel("data/Diccionario_CUP.xlsx")
# Asegura que las columnas relevantes existen
assert "CUP_2013" in dicc.columns, "Falta columna CUP_2013 en Diccionario_CUP.xlsx"
assert "GLOSA_CUP_2013" in dicc.columns, "Falta columna GLOSA_CUP_2013 en Diccionario_CUP.xlsx"

# Leer predicciones
preds = pd.read_csv("output/predictions_chilecompra_2024.csv", encoding="latin")
assert "prediction" in preds.columns, "Falta columna prediction en archivo de predicciones"

# Unir por CUP_2013 == prediction
preds["prediction"] = preds["prediction"].astype(str)
dicc["CUP_2013"] = dicc["CUP_2013"].astype(str)
merged = preds.merge(dicc[["CUP_2013", "GLOSA_CUP_2013"]], left_on="prediction", right_on="CUP_2013", how="left")
merged = merged.drop(columns=["CUP_2013"])

# Agrega la columna GLOSA_CUP_2013 al archivo de predicciones
merged.to_csv("output/predictions_chilecompra_2024.csv", index=False, encoding="latin")
print("Columna GLOSA_CUP_2013 agregada a output/predictions_chilecompra_2024.csv")
