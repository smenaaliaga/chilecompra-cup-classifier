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

def balance_classes(df, min_samples=5, max_samples=50):
    balanced_dfs = []
    for class_label in df['label'].unique():
        class_df = df[df['label'] == class_label].copy()
        n_samples = len(class_df)
        if n_samples < min_samples:
            multiplier = min_samples // n_samples + 1
            oversampled = pd.concat([class_df] * multiplier, ignore_index=True)
            balanced_dfs.append(oversampled.head(min_samples))
        elif n_samples > max_samples:
            balanced_dfs.append(class_df.sample(n=max_samples, random_state=42))
        else:
            balanced_dfs.append(class_df)
    return pd.concat(balanced_dfs, ignore_index=True)

def add_e5_prefix(df: pd.DataFrame, col="text", mode="passage") -> pd.DataFrame:
    df = df.copy()
    df[col] = df[col].astype(str).map(lambda s: f"{mode}: {s}" if not s.startswith(f"{mode}: ") else s)
    return df

def main():
    print(f"CUDA available: {torch.cuda.is_available()}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---- DATA LOADING ----
    df = pd.read_csv("data/chilecompra_final_rnd.csv")[["glosa", "cup"]].dropna()
    df = df.rename(columns={"glosa": "text", "cup": "label"})
    print(f"Dataset original: {len(df)} muestras, {df['label'].nunique()} clases")
    class_counts = df['label'].value_counts()
    print(f"Clases con <= 7 muestras: {sum(class_counts <= 7)}")

    # ---- FILTRAR CLASES MUY PEQUEÑAS ----
    min_class_size = 8
    valid_classes = class_counts[class_counts >= min_class_size].index
    df_filtered = df[df['label'].isin(valid_classes)].copy()
    print(f"Después del filtrado (>={min_class_size} muestras): {len(df_filtered)} muestras, {df_filtered['label'].nunique()} clases")

    # ---- BALANCING ----
    df_balanced = balance_classes(df_filtered, min_samples=5, max_samples=30)
    print(f"Después del balancing: {len(df_balanced)} muestras")

    # ---- Añadir prefijos e5 ----
    df_augmented = add_e5_prefix(df_augmented, "text", mode="passage")

    # ---- SPLIT ----
    train_df, test_df = train_test_split(
        df_augmented, 
        test_size=0.2, 
        stratify=df_augmented["label"], 
        random_state=42
    )
    raw_test_texts = test_df["text"].tolist()

    def to_ds(dframe):
        return Dataset.from_pandas(dframe[["text","label"]].reset_index(drop=True))

    dset = DatasetDict(train=to_ds(train_df), validation=to_ds(test_df))

    # ---- MODEL CON E5 ----
    BASE = "intfloat/multilingual-e5-base"
    labels = sorted(df_balanced["label"].unique())
    model = SetFitModel.from_pretrained(BASE, labels=labels)
    model.model_body.max_seq_length = 128
    model = model.to(device)

    # ---- TRAINER ----
    trainer = SetFitTrainer(
        model=model,
        train_dataset=dset["train"],
        eval_dataset=dset["validation"],
        batch_size=96,
        num_epochs=5,
        num_iterations=500,
        learning_rate=2e-5,
        warmup_proportion=0.1,
        seed=42,
    )

    print("Training (e5-improved)...")
    trainer.train()
    print("Training completed!")

    # ---- EVALUACIÓN ----
    # Usar 'query:' en inferencia
    query_texts = [f"query: {t[8:]}" if t.startswith("passage: ") else f"query: {t}" for t in raw_test_texts]
    y_true = test_df["label"].tolist()
    predictions = trainer.model.predict(query_texts)

    acc = accuracy_score(y_true, predictions)
    f1m = f1_score(y_true, predictions, average="macro")
    f1w = f1_score(y_true, predictions, average="weighted")
    print(f"Accuracy: {acc:.4f} | F1 Macro: {f1m:.4f} | F1 Weighted: {f1w:.4f}")
    report = classification_report(y_true, predictions, output_dict=True)
    problematic_classes = []
    for label, metrics in report.items():
        if isinstance(metrics, dict) and 'f1-score' in metrics:
            if metrics['f1-score'] == 0.0:
                problematic_classes.append(label)
    print(f"\nClases con F1=0: {len(problematic_classes)}")
    if problematic_classes:
        print("Clases problemáticas:", problematic_classes[:10])

    # ---- SAVE RESULTS ----
    os.makedirs("report", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    with open("report/classification_report_e5.txt", "w", encoding="utf-8") as f:
        f.write("REPORTE DE CLASIFICACIÓN SETFIT (e5)\n")
        f.write("="*60 + "\n\n")
        f.write(f"Dataset original: {len(df)} muestras, {df['label'].nunique()} clases\n")
        f.write(f"Dataset filtrado: {len(df_filtered)} muestras, {df_filtered['label'].nunique()} clases\n")
        f.write(f"Dataset final: {len(df_augmented)} muestras\n\n")
        f.write(f"Accuracy: {acc:.4f}\nF1 Macro: {f1m:.4f}\nF1 Weighted: {f1w:.4f}\n\n")
        f.write("REPORTE DETALLADO:\n")
        f.write(classification_report(y_true, predictions))
        f.write(f"\n\nClases problemáticas (F1=0): {len(problematic_classes)}\n")
        if problematic_classes:
            f.write(f"Clases: {problematic_classes}\n")
    out_dir = "./models/setfit_model_e5"
    trainer.model.save_pretrained(out_dir)
    results_df = pd.DataFrame({
        "text": test_df["text"], 
        "label": y_true, 
        "pred": predictions,
        "correct": [yt == yp for yt, yp in zip(y_true, predictions)]
    })
    results_df.to_csv("output/predictions_e5.csv", index=False)
    print("Modelo y reportes guardados ✓ (e5)")

if __name__ == "__main__":
    main()
