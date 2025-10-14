import os
from dotenv import load_dotenv
from huggingface_hub import HfApi, login, create_repo, upload_folder
from utils.config_loader import get_model_paths

# Cargar variables del archivo .env
load_dotenv()

# ==== CARGA DE RUTAS ====
model_config = get_model_paths()

# Configura el token de acceso personal de Hugging Face
HF_TOKEN = os.getenv("HF_TOKEN")
REPO_ID = os.getenv("MODEL_REPO_ID")
MODEL_DIR = model_config.get('model_dir', 'models/setfit_model_mpnet')

# Inicia sesión
login(token=HF_TOKEN)

# Crea el repositorio si no existe
def ensure_repo(repo_id):
    api = HfApi()
    try:
        api.create_repo(repo_id, exist_ok=True)
    except Exception as e:
        print(f"Repo ya existe o error: {e}")

# Subir el modelo
def upload_model(model_dir, repo_id):
    upload_folder(
        repo_id=repo_id,
        folder_path=model_dir,
        commit_message="Subida inicial del modelo",
        ignore_patterns=["*.pt", "*.bin", "*.pkl", "*.md"]
    )
    print(f"Modelo subido a https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    ensure_repo(REPO_ID)
    upload_model(MODEL_DIR, REPO_ID)
