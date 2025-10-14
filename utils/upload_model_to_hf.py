import os
from dotenv import load_dotenv
from huggingface_hub import HfApi, login, create_repo, upload_folder

# Cargar variables del archivo .env
load_dotenv()

# Configura el token de acceso personal de Hugging Face
HF_TOKEN = os.getenv("HF_TOKEN") #or "<TU_TOKEN_AQUI>" 
REPO_ID =  os.getenv("MODEL_REPO_ID") #or "<usuario>/<nombre-repo>"
MODEL_DIR = "models/setfit_model_mpnet"

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
        ignore_patterns=["*.pt", "*.bin", "*.pkl", "*.md"]  # Ajusta según lo que quieras subir
    )
    print(f"Modelo subido a https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    ensure_repo(REPO_ID)
    upload_model(MODEL_DIR, REPO_ID)
