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
    max_seq_length = train_config.get('max_seq_length', 128)
    model.model_body.max_seq_length = max_seq_length 
    model = model.to(device)

    # ---- TRAINER ----
    batch_size = train_config.get('batch_size', 96)
    num_epochs = train_config.get('num_epochs', 3)
    num_iterations = train_config.get('num_iterations', 200)
    seed = train_config.get('random_state', 42)
    
    trainer = SetFitTrainer(
        model=model,
        train_dataset=dset["train"],
        eval_dataset=dset["validation"],
        batch_size=batch_size,
        num_epochs=num_epochs, 
        num_iterations=num_iterations,   
        # learning_rate=2e-5,   
        # warmup_proportion=0.1,  
        seed=seed,
    )

    print("Training (mpnet-improved)...")
    trainer.train()
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

    # Guardar modelo
    out_dir = train_config.get('output_model_dir', './models/setfit_model_mpnet')
    trainer.model.save_pretrained(out_dir)
    
    # Guardar predicciones
    probas = trainer.model.predict_proba(test_df["text"].tolist())
    top1_idx = np.argmax(probas, axis=1)
    top2_idx = np.argsort(probas, axis=1)[:, -2]

    # Obtener lista de labels
    labels_list = trainer.model.config["labels"] if hasattr(trainer.model, "config") else labels

    top1_conf = probas[np.arange(len(probas)), top1_idx]
    top2_conf = probas[np.arange(len(probas)), top2_idx]
    top2_class = [labels_list[i] for i in top2_idx]

    results_df = pd.DataFrame({
        "text": test_df["text"],
        "label": y_true,
        "pred": predictions,
        "confidence": top1_conf,
        "correct": [yt == yp for yt, yp in zip(y_true, predictions)],
        # "second_pred": top2_class,
        # "second_confidence": top2_conf
    })
    predictions_file = train_config.get('output_predictions_test', 'output/predictions_test.csv')
    results_df.to_csv(predictions_file, index=False)
    print("Modelo y reportes guardados ✓ (mpnet)")

if __name__ == "__main__":
    main()