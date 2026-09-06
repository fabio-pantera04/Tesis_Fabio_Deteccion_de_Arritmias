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

LLM_BACKENDS_DISPONIBLES = ["mock", "claude", "gemini", "groq"]

# -----------------------------------------------------------
# Senales precargadas
# -----------------------------------------------------------
SIGNALS_DIR = DATA_DIR / "signals_processed"

# -----------------------------------------------------------
# Notas clinicas LLM-A pre-generadas y persistidas.
# -----------------------------------------------------------
NOTES_DIR = DATA_DIR / "notas_llm_a"
NOTES_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------
# Flask
# -----------------------------------------------------------
SECRET_KEY = "cambiar-en-produccion-cimat-tesis"
DEBUG = True
HOST = "127.0.0.1"
PORT = 5000

# -----------------------------------------------------------
# Panel de administracion (acceso protegido por contrasena
# unica configurable aqui). No linkeado desde la vista medica;
# se accede unicamente por URL directa a /admin/login.
# -----------------------------------------------------------
ADMIN_PASSWORD = "Admin_Fabio_Cimat"

# -----------------------------------------------------------
# Tema visual
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
