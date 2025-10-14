import yaml
from pathlib import Path

def load_paths():
    """
    Carga las rutas desde el archivo de configuración simple
    """
    config_file = "config/config.yaml"
    config_path = Path(config_file)
    
    if not config_path.exists():
        print(f"Archivo de configuración no encontrado: {config_file}")
        return {}
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# Funciones eliminadas - ya no necesarias

def get_train_config():
    """Obtiene la configuración de entrenamiento"""
    config = load_paths()
    return config.get('train', {})

def get_predict_config():
    """Obtiene la configuración de predicción"""
    config = load_paths()
    return config.get('predict', {})
