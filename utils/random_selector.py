import pandas as pd

def select_random_samples(input_file="data/chilecompra_final.csv", 
                         output_file="data/chilecompra_final_rnd.csv", 
                         max_samples_per_group=50, 
                         random_state=42):
    """Selecciona hasta max_samples_per_group ejemplos aleatorios por cada cup_final"""
    
    # Leer archivo y seleccionar solo las columnas necesarias
    df = pd.read_csv(input_file, delimiter=",")[['NombreroductoGenerico', 'cup_final']]
    
    # Renombrar columnas
    df = df.rename(columns={'NombreroductoGenerico': 'glosa', 'cup_final': 'cup'})
    
    # Seleccionar muestras aleatorias por grupo
    selected_samples = []
    for cup, group in df.groupby('cup'):
        n_samples = min(len(group), max_samples_per_group)
        sampled_group = group.sample(n=n_samples, random_state=random_state)
        selected_samples.append(sampled_group)
    
    # Combinar y guardar
    result_df = pd.concat(selected_samples, ignore_index=True)
    result_df = result_df.sort_values(['cup', 'glosa']).reset_index(drop=True)
    result_df.to_csv(output_file, index=False)
    
    print(f"Procesado: {len(df)} -> {len(result_df)} registros")
    return result_df

if __name__ == "__main__":
    select_random_samples()