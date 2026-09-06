"""
Configuracion central del sistema. Editar aqui para cambiar
el backend LLM activo, las senales disponibles, o el tema.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DB_DIR = ROOT / "database"

# -----------------------------------------------------------
# Backend LLM activo. Cambiar a 'claude', 'gemini', 'groq'
# cuando se haya elegido y configurado la API key.
# -----------------------------------------------------------
LLM_BACKEND = "mock"

# Lista de backends disponibles para experimentos comparativos
# (analytics permitira comparar entre ellos).
LLM_BACKENDS_DISPONIBLES = ["mock", "claude", "gemini", "groq"]

# -----------------------------------------------------------
# Senales precargadas (cada una es un JSON producido por
# scripts/bootstrap.py o classifier/prepare_signals.py).
# -----------------------------------------------------------
SIGNALS_DIR = DATA_DIR / "signals_processed"


# -----------------------------------------------------------
# Flask
# -----------------------------------------------------------
SECRET_KEY = "cambiar-en-produccion-cimat-tesis"
DEBUG = True
HOST = "127.0.0.1"
PORT = 5000

# -----------------------------------------------------------
# Tema visual: paleta crema/rojo clinica
# -----------------------------------------------------------
THEME = {
    "name": "ecg-paper",
    "cream_bg": "#F4ECD8",
    "cream_paper": "#FBF5E5",
    "ecg_red": "#B22222",
    "ecg_red_dark": "#7A1818",
    "grid_minor": "rgba(178, 34, 34, 0.10)",
    "grid_major": "rgba(178, 34, 34, 0.22)",
    "ink": "#2A1810",
    "ink_soft": "#5C453A",
    "verde": "#3A8F3A",
    "amarillo": "#D6A317",
    "rojo_alerta": "#B22222",
}
