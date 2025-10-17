# ChileCompra CUP Classifier

Implementación de un modelo de clasificación semántica de las glosas de órdenes de compra de ChileCompra, con el objetivo de asignar automáticamente el Código Único de Producto (CUP 2013).

La implementación se basa en el framework de _few-shot learning_ [SetFit](https://github.com/huggingface/setfit)
, que combina _Sentence Transformers_, _Contrastive Learning_ y modelos de _Machine Learning_ clásico para aprender de manera eficiente incluso con pocas muestras por clase.

## Estructura del proyecto

```
bc_chilecompra_setfit/
├── data/                    # Archivos de datos 
├── models/                  # Modelos entrenados
├── output/                  # Predicciones generadas
├── report/                  # Reportes de métricas
├── config/                  # Configuración centralizada
│   └── config.yaml         # Rutas de archivos
├── utils/                  # Scripts auxiliares
├── train_mpnet.py          # Entrenar modelo
├── predict.py              # Clasificar textos
└── requirements.txt        # Dependencias
```

## Configuración

### Archivo `config/config.yaml`
Este archivo controla todo el proyecto. Tiene dos secciones principales:

**`train:` - Para entrenar el modelo**
```yaml
train:
  # Dataset a entrada
  input_file: "data/chilecompra.csv"
  # Donde guardar modelo
  output_model_dir: "models/setfit_model_mpnet" 
  # Reporte de métricas
  output_report_file: "report/classification_report_mpnet.txt"
  # Modelo base
  base_model: "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
  ...
```

**`predict:` - Para hacer predicciones**
```yaml
predict:
  # Dataset a predecir
  input_file: "data/chilecompra_2024.csv" 
  # Donde guardar resultados
  output_file: "output/predictions.csv"
  # Modelo a utilizar
  model_dir: "models/setfit_model_mpnet"
  ...

```

## Uso

### Entrenar modelo
```bash
python train_mpnet.py
```
**Proceso completo:**
- Lee datos desde `train.input_file`
- Usa columnas `train.text_column` y `train.label_column`
- Filtra clases con menos de `train.min_class_size` muestras
- Reduce clases con más de `train.max_samples` muestras (undersampling)
- Entrena modelo base `train.base_model` con parámetros `train.batch_size`, `train.num_epochs`
- Guarda modelo entrenado en `train.output_model_dir`
- Genera reporte de métricas en `train.output_report_file`

### Hacer predicciones
```bash
python predict.py
```
- Lee datos desde `predict.input_file`
- Usa modelo desde `predict.model_dir`
- Guarda predicciones en `predict.output_file`

### Ver resultados
- **Modelo entrenado:** `models/setfit_model_mpnet/`
- **Predicciones:** `output/predictions.csv`
- **Reporte:** `report/classification_report_mpnet.txt`

## Instalación

```bash
pip install -r requirements.txt
```
