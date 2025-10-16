import os
from dotenv import load_dotenv
from huggingface_hub import HfApi, login, create_repo, upload_file

# Cargar variables del archivo .env
load_dotenv()

# ==== CARGA DE RUTAS ====
# Configura el token de acceso personal de Hugging Face
HF_TOKEN = os.getenv("HF_TOKEN")
REPO_ID = os.getenv("DATASET_REPO_ID")
DATASET_FILE = 'data/chilecompra.csv'  # Ruta directa al dataset

# Inicia sesión
login(token=HF_TOKEN)

# Crea el repositorio de dataset si no existe
def ensure_dataset_repo(repo_id):
    api = HfApi()
    try:
        api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
        print(f"Repositorio de dataset creado: {repo_id}")
    except Exception as e:
        print(f"Repo ya existe o error: {e}")

# Subir el dataset
def upload_dataset(dataset_file, repo_id):
    try:
        # Obtener el nombre del archivo para el repositorio
        filename = os.path.basename(dataset_file)
        upload_file(
            path_or_fileobj=dataset_file,
            path_in_repo=filename,  # Nombre en el repo
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Subida inicial del dataset ChileCompra 2024"
        )
        print(f"Dataset subido exitosamente a https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"Error al subir el dataset: {e}")

if __name__ == "__main__":
    # Verificar que el archivo existe
    if not os.path.exists(DATASET_FILE):
        print(f"Error: El archivo {DATASET_FILE} no existe")
        exit(1)
    
    # Verificar variables de entorno
    if not HF_TOKEN:
        print("Error: HF_TOKEN no está definido en el archivo .env")
        exit(1)
    
    if not REPO_ID:
        print("Error: DATASET_REPO_ID no está definido en el archivo .env")
        exit(1)
    
    print(f"Subiendo dataset desde: {DATASET_FILE}")
    print(f"Al repositorio: {REPO_ID}")
    
    ensure_dataset_repo(REPO_ID)
    upload_dataset(DATASET_FILE, REPO_ID)
