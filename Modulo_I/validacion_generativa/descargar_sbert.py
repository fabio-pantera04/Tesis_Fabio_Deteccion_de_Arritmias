"""
Descarga el modelo SBERT a un cache en E: (no C:) para evitar el 
problema de espacio en el disco del sistema.
"""
import os

# CRITICO: configurar ANTES de importar sentence_transformers
os.environ["HF_HOME"] = r"E:\huggingface_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from sentence_transformers import SentenceTransformer

print(f"Descargando modelo a: {os.environ['HF_HOME']}")
modelo = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
print("Descarga OK. Probando embedding...")
emb = modelo.encode(["sinus rhythm, normal ecg"])
print(f"Embedding shape: {emb.shape}")
print("Modelo listo para usar.")