"""
config.py — Parámetros globales del sistema de tesis CIMAT.

Arquitectura DUAL (v4 — dataset balanceado 1000/clase):
  Modelo LATIDO : NORMAL / PAC  — solo shapelets MICRO (50-160 muestras)
  Modelo RITMO  : NSR   / AFIB  — solo shapelets MACRO (200-700 muestras)

Dataset balanceado — 1000 señales por clase:
  NORMAL : 1000 de Icentia11k (CardioSTAT, Lead II mod., 250 Hz)
           Ventana: 250 muestras (125 pre-R + 125 post-R)
  PAC    : 1000 de Icentia11k (CardioSTAT, Lead II mod., 250 Hz)
           Ventana: 250 muestras (125 pre-R + 125 post-R)
  NSR    : 1000 de Icentia11k (CardioSTAT, Lead II mod., 250 Hz)
           Ventana: 1000 muestras
  AFIB   : 306 de Icentia11k + 694 de CinC 2017 = 1000 total
           CinC 2017: AliveCor, Lead I equiv., 300 Hz → remuestreado 250 Hz
           Ventana: 1000 muestras

Justificación del balance AFIB:
  306 es el techo epidemiológico de Icentia11k (pacientes únicos con AFIB
  confirmada). Los 694 restantes provienen de CinC 2017 (Clifford et al.,
  2017, Computing in Cardiology). La heterogeneidad de dispositivo actúa
  como regularización implícita.

Justificación ventana 250 muestras para NORMAL/PAC:
  Con 62+100=162 muestras (v1) solo se capturaba el QRS, casi idéntico en
  ambas clases. Con 125+125=250 muestras (v2) se captura:
    - Onda P completa (zona discriminativa: intervalo PR, morfología P)
    - Pausa compensatoria post-R (zona de mayor diferencia: onda T)
  Análisis morfológico mostró diff=0.70-1.09 en onda T vs diff=0.06 en QRS.
"""

from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# RUTAS DEL SISTEMA
# ═══════════════════════════════════════════════════════════════════════════════

RAIZ = Path(__file__).parent

# ── Bases de datos ─────────────────────────────────────────────────────────────
# Icentia11k — base principal
# NORMAL y PAC: ventana 250m (extractor v5, BEAT_PRE=125, BEAT_POST=125)
# NSR y AFIB  : ventana 1000m (extractor v4)
RUTA_ICENTIA = Path(r"C:/Modelo_Tesis/Icentia11k")

# CinC 2017 — solo AFIB (300 Hz → remuestreado 250 Hz por descargar_afib_cinc2017.py)
RUTA_CINC_AFIB = Path(r"C:/Modelo_Tesis/CinC2017/AFIB")

# ── Carpetas de salida ─────────────────────────────────────────────────────────
CARPETA_DATOS  = RAIZ / "datos"
CARPETA_SALIDA = RAIZ / "datos" / "resultados"

# ── Archivos modelo LATIDO (NORMAL / PAC) ─────────────────────────────────────
ARCHIVO_SHAPELETS_LATIDO = CARPETA_DATOS / "shapelets_latido_MSM.npy"
ARCHIVO_MODELO_LATIDO    = CARPETA_DATOS / "xgboost_latido_MSM.json"
ARCHIVO_ENCODER_LATIDO   = CARPETA_DATOS / "label_encoder_latido_MSM.npy"

# ── Archivos modelo RITMO (NSR / AFIB) ────────────────────────────────────────
ARCHIVO_SHAPELETS_RITMO  = CARPETA_DATOS / "shapelets_ritmo_MSM.npy"
ARCHIVO_MODELO_RITMO     = CARPETA_DATOS / "xgboost_ritmo_MSM.json"
ARCHIVO_ENCODER_RITMO    = CARPETA_DATOS / "label_encoder_ritmo_MSM.npy"

# ── Compatibilidad con módulos que usan nombres anteriores ────────────────────
ARCHIVO_MICRO_SHAPELETS = ARCHIVO_SHAPELETS_LATIDO
ARCHIVO_MACRO_SHAPELETS = ARCHIVO_SHAPELETS_RITMO
ARCHIVO_XGBOOST_MODELO  = ARCHIVO_MODELO_RITMO
ARCHIVO_LABEL_ENCODER   = ARCHIVO_ENCODER_RITMO

# ── Agente LLM ────────────────────────────────────────────────────────────────
ARCHIVO_GUIDELINES = RAIZ / "modulo_2_agente" / "guidelines_medicos.json"

# ═══════════════════════════════════════════════════════════════════════════════
# FRECUENCIAS DE MUESTREO
# ═══════════════════════════════════════════════════════════════════════════════

FS_ICENTIA  = 250    # Hz — Icentia11k (CardioSTAT)
FS_CINC     = 300    # Hz — CinC 2017 (AliveCor), ya remuestreado a 250
FS_OBJETIVO = 250    # Hz — frecuencia estándar del sistema

# ═══════════════════════════════════════════════════════════════════════════════
# DATASET — BALANCE 1000 POR CLASE
# ═══════════════════════════════════════════════════════════════════════════════

N_ICENTIA_NORMAL = 1000   # NORMAL: todos de Icentia, ventana 250m
N_ICENTIA_PAC    = 1000   # PAC:    todos de Icentia, ventana 250m
N_ICENTIA_NSR    = 1000   # NSR:    todos de Icentia, ventana 1000m
N_ICENTIA_AFIB   = 306    # AFIB:   techo epidemiológico Icentia11k
N_CINC_AFIB      = 694    # AFIB:   complemento CinC 2017

N_POR_CLASE = 1000        # total por clase (balanceado)

UMBRAL_LONGITUD = 300     # muestras: < 300 → latido | >= 300 → ritmo
LONGITUD_LATIDO = 250     # muestras ventana NORMAL/PAC (125+125)
LONGITUD_RITMO  = 1000    # muestras ventana NSR/AFIB

CLASES_LATIDO = ["NORMAL", "PAC"]
CLASES_RITMO  = ["NSR", "AFIB"]

# ── Fuentes por clase con límite explícito ────────────────────────────────────
# Cada entrada: (ruta, n_max)
FUENTES_POR_CLASE = {
    "NORMAL": [
        (RUTA_ICENTIA / "NORMAL", N_ICENTIA_NORMAL),
    ],
    "PAC": [
        (RUTA_ICENTIA / "PAC", N_ICENTIA_PAC),
    ],
    "NSR": [
        (RUTA_ICENTIA / "NSR", N_ICENTIA_NSR),
    ],
    "AFIB": [
        (RUTA_ICENTIA / "AFIB", N_ICENTIA_AFIB),   # 306 de Icentia
        (RUTA_CINC_AFIB,        N_CINC_AFIB),      # 694 de CinC 2017
    ],
}

# ═══════════════════════════════════════════════════════════════════════════════
# PARÁMETROS MSM + SAKOE-CHIBA
# Referencia: Stefan et al. (2013) IEEE TKDE
# ═══════════════════════════════════════════════════════════════════════════════

MSM_C      = 0.1     # costo Split/Merge (señales Z-score en [-3,3])
MSM_WINDOW = 0.1     # banda Sakoe-Chiba = 10% de la longitud
N_JOBS     = 6       # núcleos físicos AMD Ryzen 5 5600G

# ═══════════════════════════════════════════════════════════════════════════════
# MODELO LATIDO (NORMAL / PAC) — shapelets MICRO
# Señales de 250 muestras @ 250 Hz
# ═══════════════════════════════════════════════════════════════════════════════

NUM_SHAPELETS_LATIDO     = 60
NUM_CANDIDATES_LATIDO    = 800    # candidatos evaluados por Contrast Score
NUM_COMPARACIONES_LATIDO = 30     # señales comparadas por candidato
RANGO_LATIDO             = (50, 160)   # longitud shapelets micro (muestras)
                                       # 50m=200ms, 160m=640ms @ 250Hz
PASO_VENTANA_LATIDO      = 1      # búsqueda exhaustiva (señales cortas)

XGB_LATIDO = {
    "n_estimators"        : 700,
    "max_depth"           : 4,
    "learning_rate"       : 0.05,
    "reg_alpha"           : 0.05,
    "reg_lambda"          : 1.0,
    "random_state"        : 42,
    "eval_metric"         : "logloss",   # binario: logloss
    "early_stopping_rounds": 40,
}

# ═══════════════════════════════════════════════════════════════════════════════
# MODELO RITMO (NSR / AFIB) — shapelets MACRO
# Señales de 1000 muestras @ 250 Hz
# ═══════════════════════════════════════════════════════════════════════════════

NUM_SHAPELETS_RITMO     = 60
NUM_CANDIDATES_RITMO    = 800
NUM_COMPARACIONES_RITMO = 30
RANGO_RITMO             = (200, 700)   # longitud shapelets macro (muestras)
                                       # 200m=800ms, 700m=2800ms @ 250Hz
PASO_VENTANA_RITMO      = 5       # 5x más rápido, pérdida accuracy < 1%
                                  # resolución: 5/250Hz = 20ms (clínicamente aceptable)

XGB_RITMO = {
    "n_estimators"        : 700,
    "max_depth"           : 4,
    "learning_rate"       : 0.05,
    "reg_alpha"           : 0.05,
    "reg_lambda"          : 1.0,
    "random_state"        : 42,
    "eval_metric"         : "logloss",   # binario: logloss
    "early_stopping_rounds": 40,
}

# ── Compatibilidad hacia atrás (módulos que importan nombres v1) ───────────────
XGB_N_ESTIMATORS  = 700
XGB_MAX_DEPTH     = 4
XGB_LEARNING_RATE = 0.05
XGB_REG_ALPHA     = 0.05
XGB_REG_LAMBDA    = 1.0
XGB_RANDOM_STATE  = 42

# ── Aliases para shapelets.py (lee estos nombres directamente de config) ───────
NUM_SHAPELETS_MICRO  = NUM_SHAPELETS_LATIDO   # usado por shapelets.py
NUM_SHAPELETS_MACRO  = NUM_SHAPELETS_RITMO    # usado por shapelets.py
NUM_CANDIDATES       = NUM_CANDIDATES_LATIDO  # sobreescrito en memoria por entrenamiento_dual.py
NUM_COMPARACIONES    = NUM_COMPARACIONES_LATIDO
RANGO_MICRO          = RANGO_LATIDO
RANGO_MACRO          = RANGO_RITMO
PASO_VENTANA_MACRO   = PASO_VENTANA_RITMO
NUM_MUESTRAS_POR_CLASE = N_POR_CLASE

# ═══════════════════════════════════════════════════════════════════════════════
# AGENTE LLM — Módulo II
# ═══════════════════════════════════════════════════════════════════════════════

UMBRAL_CONFIANZA    = 0.25   # gap < umbral → escalar al médico
LLAMA3_MODELO_PATH  = Path(r"C:/Modelos_LLM/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf")
LLAMA3_N_CTX        = 4096
LLAMA3_N_GPU_LAYERS = 0      # 0 = CPU puro
LLAMA3_TEMPERATURA  = 0.1
LLAMA3_MAX_TOKENS   = 800

# ═══════════════════════════════════════════════════════════════════════════════
# INTERFAZ WEB — Módulo III
# ═══════════════════════════════════════════════════════════════════════════════

WEB_HOST  = "0.0.0.0"
WEB_PORT  = 5000
WEB_DEBUG = False

# ═══════════════════════════════════════════════════════════════════════════════
# UMBRALES DE DECISIÓN — EVALUADOS Y DOCUMENTADOS
# Experimento: barrido de umbrales sobre X_test_LATIDO_MSM.csv (600 señales)
# Fecha: abril 2026
# ═══════════════════════════════════════════════════════════════════════════════

# Umbral óptimo confirmado por barrido 0.30-0.70
# Criterio: maximizar F1-macro en test set balanceado (300 NORMAL + 300 PAC)
UMBRAL_DECISION_LATIDO = 0.50   # óptimo global: Acc=81.17%, F1=0.8114

# Tabla completa del barrido (para reporte en tesis)
# umbral | accuracy | F1-macro | prec-PAC | rec-PAC | prec-N | rec-N
BARRIDO_UMBRALES_LATIDO = [
    (0.30, 0.7650, 0.7613, 0.712, 0.890, 0.853, 0.640),
    (0.35, 0.7650, 0.7629, 0.723, 0.860, 0.827, 0.670),
    (0.40, 0.7800, 0.7794, 0.753, 0.833, 0.813, 0.727),
    (0.45, 0.7917, 0.7917, 0.787, 0.800, 0.797, 0.783),
    (0.50, 0.8117, 0.8114, 0.838, 0.773, 0.789, 0.850),  # SELECCIONADO
    (0.55, 0.8067, 0.8059, 0.851, 0.743, 0.772, 0.870),
    (0.60, 0.8033, 0.8017, 0.870, 0.713, 0.757, 0.893),
    (0.65, 0.8050, 0.8021, 0.903, 0.683, 0.745, 0.927),
    (0.70, 0.7883, 0.7829, 0.922, 0.630, 0.719, 0.947),
]
# Justificación de selección:
# - PAC es arritmia de bajo riesgo inmediato (AAMI EC57:2012 clase S)
# - El agente escala casos de baja confianza (gap < UMBRAL_CONFIANZA)
# - No se justifica sacrificar accuracy global para aumentar recall PAC
# ═══════════════════════════════════════════════════════════════════════════════
# CREAR CARPETAS AL IMPORTAR
# ═══════════════════════════════════════════════════════════════════════════════

CARPETA_DATOS.mkdir(parents=True, exist_ok=True)
CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)