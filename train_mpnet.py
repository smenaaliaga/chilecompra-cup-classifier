import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from datasets import Dataset, DatasetDict
from setfit import SetFitModel, SetFitTrainer
from collections import Counter
import random
import time
from utils.config_loader import get_train_config

def balance_classes(df, max_samples=50):
    """
    Balance classes usando solo undersampling para clases mayoritarias
    """
    balanced_dfs = []
    
    for class_label in df['label'].unique():
        class_df = df[df['label'] == class_label].copy()
        n_samples = len(class_df)
        
        if n_samples > max_samples:
            # Undersample: tomar muestra aleatoria
            balanced_dfs.append(class_df.sample(n=max_samples, random_state=42))
        else:
            balanced_dfs.append(class_df)
    
    return pd.concat(balanced_dfs, ignore_index=True)


def main():
    # ---- INICIO DEL CRONOMETRO ----
    start_time = time.time()
    
    # ---- CARGA DE CONFIGURACIÓN ----
    train_config = get_train_config()
    
    # ---- GPU INFO ----
    print(f"CUDA available: {torch.cuda.is_available()}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---- DATA ----
    train_input_file = train_config.get('input_file', 'data/chilecompra.csv')
    text_column = train_config.get('text_column', 'glosa')
    label_column = train_config.get('label_column', 'cup')
    
    df = pd.read_csv(train_input_file)[[text_column, label_column]].dropna()
    df = df.rename(columns={text_column: "text", label_column: "label"})
    df["label"] = df["label"].astype(str)
    
    print(f"Dataset original: {len(df)} muestras, {df['label'].nunique()} clases")
    
    # Análisis de distribución
    class_counts = df['label'].value_counts()
    print(f"Clases con <= 7 muestras: {sum(class_counts <= 7)}")
    
    # ---- FILTRAR CLASES MUY PEQUEÑAS ----
    min_class_size = train_config.get('min_class_size', 8)
    valid_classes = class_counts[class_counts >= min_class_size].index
    df_filtered = df[df['label'].isin(valid_classes)].copy()
    
    print(f"Después del filtrado (>={min_class_size} muestras): {len(df_filtered)} muestras, {df_filtered['label'].nunique()} clases")
    
    # ---- BALANCING ----
    max_samples = train_config.get('max_samples', 30)
    df_balanced = balance_classes(df_filtered, max_samples=max_samples)
    print(f"Después del balancing: {len(df_balanced)} muestras")
    
    # ---- SPLIT ----
    test_size = train_config.get('test_size', 0.2)
    random_state = train_config.get('random_state', 42)
    train_df, test_df = train_test_split(
        df_balanced, 
        test_size=test_size, 
        stratify=df_balanced["label"], 
        random_state=random_state
    )

    def to_ds(dframe):
        return Dataset.from_pandas(dframe[["text","label"]].reset_index(drop=True))

    dset = DatasetDict(train=to_ds(train_df), validation=to_ds(test_df))

    # ---- MODEL ----
    base_model = train_config.get('base_model', 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    labels = sorted(df_balanced["label"].unique())
    model = SetFitModel.from_pretrained(base_model, labels=labels)
    # Ajustes del modelo
    max_seq_length = train_config.get('max_seq_length', 256)
    model.model_body.max_seq_length = max_seq_length 
    model = model.to(device)

    # ---- TRAINER ----
    batch_size = train_config.get('batch_size', 64)
    num_epochs = train_config.get('num_epochs', 5)
    num_iterations = train_config.get('num_iterations', 400)
    learning_rate = float(train_config.get('learning_rate', 2e-5))
    warmup_proportion = float(train_config.get('warmup_proportion', 0.1))
    seed = train_config.get('random_state', 42)

    trainer = SetFitTrainer(
        model=model,
        train_dataset=dset["train"],
        eval_dataset=dset["validation"],
        batch_size=batch_size,
        num_epochs=num_epochs,
        num_iterations=num_iterations,
        learning_rate=learning_rate,
        warmup_proportion=warmup_proportion,
        seed=seed,
    )

    print("Training (mpnet-improved...")
    training_start = time.time()
    trainer.train()
    training_end = time.time()
    training_time = training_end - training_start
    print("Training completed!")

    # ---- EVAL ----
    y_true = test_df["label"].tolist()
    predictions = trainer.model.predict(test_df["text"].tolist())

    acc = accuracy_score(y_true, predictions)
    f1m = f1_score(y_true, predictions, average="macro")
    f1w = f1_score(y_true, predictions, average="weighted")
    
    print(f"Accuracy: {acc:.4f} | F1 Macro: {f1m:.4f} | F1 Weighted: {f1w:.4f}")
    
    # Análisis por clase
    report = classification_report(y_true, predictions, output_dict=True)
    
    # Clases problemáticas
    problematic_classes = []
    for label, metrics in report.items():
        if isinstance(metrics, dict) and 'f1-score' in metrics:
            if metrics['f1-score'] == 0.0:
                problematic_classes.append(label)
    
    print(f"\nClases con F1=0: {len(problematic_classes)}")
    if problematic_classes:
        print("Clases problemáticas:", problematic_classes[:10])

    # ---- SAVE RESULTS ----
    report_file = train_config.get('output_report_file', 'report/classification_report_mpnet.txt')
    
    # Calcular tiempo total
    total_time = time.time() - start_time
    
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("REPORTE DE CLASIFICACIÓN SETFIT (mpnet-improved)\n")
        f.write("="*60 + "\n\n")
        f.write(f"Dataset original: {len(df)} muestras, {df['label'].nunique()} clases\n")
        f.write(f"Dataset filtrado: {len(df_filtered)} muestras, {df_filtered['label'].nunique()} clases\n")
        f.write(f"Dataset final: {len(df_balanced)} muestras\n\n")
        f.write(f"Accuracy: {acc:.4f}\nF1 Macro: {f1m:.4f}\nF1 Weighted: {f1w:.4f}\n\n")
        f.write("REPORTE DETALLADO:\n")
        f.write(classification_report(y_true, predictions))
        f.write(f"\n\nClases problemáticas (F1=0): {len(problematic_classes)}\n")
        if problematic_classes:
            f.write(f"Clases: {problematic_classes}\n")
        
        # Información del modelo y parámetros
        f.write("\n" + "="*60 + "\n")
        f.write("INFORMACIÓN DEL MODELO Y ENTRENAMIENTO\n")
        f.write("="*60 + "\n\n")
        f.write(f"Modelo base: {base_model}\n")
        f.write(f"Dispositivo: {device}\n")
        f.write(f"Archivo de entrada: {train_input_file}\n")
        f.write(f"Columnas: texto='{text_column}', etiqueta='{label_column}'\n")
        f.write(f"- Tamaño de train: {len(train_df)}\n")
        f.write(f"- Tamaño de test: {len(test_df)}\n\n")
        
        f.write("Parámetros de entrenamiento:\n")
        f.write(f"- batch_size: {batch_size}\n")
        f.write(f"- num_epochs: {num_epochs}\n")
        f.write(f"- num_iterations: {num_iterations}\n\n")
        
        f.write("TIEMPOS DE EJECUCIÓN:\n")
        f.write(f"- Tiempo de entrenamiento: {training_time:.2f} segundos ({training_time/60:.2f} minutos)\n")
        f.write(f"- Tiempo total: {total_time:.2f} segundos ({total_time/60:.2f} minutos)\n")

    # Guardar modelo
    out_dir = train_config.get('output_model_dir', './models/setfit_model_mpnet')
    trainer.model.save_pretrained(out_dir)
    
    # Guardar predicciones
    probas = trainer.model.predict_proba(test_df["text"].tolist())
    top1_idx = np.argmax(probas, axis=1)

    top1_conf = probas[np.arange(len(probas)), top1_idx]

    results_df = pd.DataFrame({
        "text": test_df["text"],
        "label": y_true,
        "pred": predictions,
        "confidence": top1_conf,
        "correct": [yt == yp for yt, yp in zip(y_true, predictions)]
    })
    predictions_file = train_config.get('output_predictions_test', 'output/predictions_test.csv')
    results_df.to_csv(predictions_file, index=False)
    print("Modelo y reportes guardados ✓ (mpnet)")


if __name__ == "__main__":
    main()