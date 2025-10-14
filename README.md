# ChileCompra CUP Classifier

Implementación de un modelo de clasificación semántica para glosas de Ordenes de Compra de Chilecompra, utilizando de base el framework SetFit + Sentence Transformers.

## Estructura principal
- **data/**: Datos de entrada (CSV, Excel) para entrenamiento y predicción.
- **models/**: Modelos entrenados.
- **output/**: Resultados y predicciones generadas por los modelos.
- **report/**: Reportes de métricas y análisis de clasificación.
- **train_mpnet.py | train_e5.py**: Scripts de entrenamiento de modelos.
- **predict.py | predict_with_setfit.py**: Scripts para clasificar nuevas glosas usando modelos entrenados.
- **utils/process_text_batch.py**: Limpieza y normalización masiva de glosas.

## Uso básico
1. Preprocesa las glosas con `utils/process_text_batch.py`.
2. Entrena el modelo con `train_mpnet.py` o variantes.
3. Clasifica nuevas glosas con `predict.py`.
4. Revisa los reportes en la carpeta `report/` y las predicciones en `output/`.

## Requisitos
- Python 3.10+ (recomendado)
- Instalar dependencias con `pip install -r requirements.txt` 

## Notas
- Los scripts están preparados para trabajar con GPU si está disponible.
