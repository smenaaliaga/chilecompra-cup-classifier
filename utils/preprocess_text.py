import pandas as pd
import re
import unicodedata

# Configuración de normalización de texto
preprocessing_config = {
    'normalize_text': {
        'to_uppercase': True,
        'remove_accents': True,
        'remove_special_chars': True,
        'normalize_spaces': True
    }
}

def normalize_text(text: str) -> str:
    if pd.isna(text) or text == ".":
        return ""
    text = str(text).strip()
    cfg = preprocessing_config['normalize_text']
    if cfg.get('to_uppercase', True):
        text = text.upper()
    if cfg.get('remove_accents', True):
        text = unicodedata.normalize('NFD', text)
        text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
    if cfg.get('remove_special_chars', True):
        text = re.sub(r'[^A-Z0-9\s]', ' ', text)
    if cfg.get('normalize_spaces', True):
        text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def remove_region_and_roman_numbers(text: str) -> str:
    if pd.isna(text) or text == ".":
        return ""
    text = str(text)
    region_words = ['REGION', 'NACIONAL', 'RM', 'MACROZONA', 'CENTRO', 'NORTE', 'SUR', 'AUSTRAL']
    for word in region_words:
        text = re.sub(rf'\b{word}\b', '', text)
    roman_pattern = r'\b(M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3}))\b'
    text = re.sub(roman_pattern, '', text)
    simple_roman = r'\b(I{1,4}|V|X{1,4}|L|C{1,4}|D|M{1,4})\b'
    text = re.sub(simple_roman, '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_quantity_references(text: str) -> str:
    if pd.isna(text) or text == ".":
        return ""
    text = str(text)
    keywords = ['UNIDAD', 'UNIDADES', 'PAQUETE', 'PAQUETES', 'PQTES']
    words = text.split()
    filtered_words = []
    skip_until_idx = -1
    i = 0
    while i < len(words):
        if i <= skip_until_idx:
            i += 1
            continue
        current_word = words[i]
        if current_word in keywords:
            found_number_before = False
            start_idx = max(0, i - 2)
            for j in range(start_idx, i):
                if re.match(r'^\d+$', words[j]) or re.match(r'^\d+[.,]\d+$', words[j]):
                    found_number_before = True
                    skip_until_idx = i
                    break
            if not found_number_before:
                if i + 1 < len(words) and (re.match(r'^\d+$', words[i+1]) or re.match(r'^\d+[.,]\d+$', words[i+1])):
                    skip_until_idx = i + 1
                else:
                    filtered_words.append(current_word)
        else:
            filtered_words.append(current_word)
        i += 1
    result = ' '.join(filtered_words)
    result = re.sub(r'\s+', ' ', result).strip()
    return result

def process_text(text: str) -> str:
    t = normalize_text(text)
    t = remove_region_and_roman_numbers(t)
    t = clean_quantity_references(t)
    return t

def process_csv(input_csv, output_csv, text_column='text'):
    df = pd.read_csv(input_csv)
    if text_column not in df.columns:
        print(f"ERROR: El archivo debe tener una columna llamada '{text_column}'.")
        return
    # Guarda el texto original en una nueva columna
    df[f"{text_column}_original"] = df[text_column]
    df[text_column] = df[text_column].map(process_text)
    # Remueve duplicados basados en la columna procesada
    df = df.drop_duplicates(subset=[text_column])
    df.to_csv(output_path, index=False)
    print(f"Archivo procesado guardado en {output_path}")

if __name__ == "__main__":
    # Configura los nombres de archivo aquí
    input_csv = "data/chilecompra_2024.csv"
    output_path = "data/chilecompra_2024_processed.csv"
    text_column = "glosa"  # Cambia si tu columna se llama distinto
    process_csv(input_csv, output_path, text_column)
