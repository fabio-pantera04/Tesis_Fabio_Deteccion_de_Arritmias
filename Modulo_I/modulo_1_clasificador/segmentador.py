"""
modulo_1_clasificador/segmentador.py

Segmentación automática de señales ECG largas en ventanas clasificables.

Convierte una señal ECG completa (30s, 10min, etc.) en:
  - Ventanas de LATIDO  (~162 muestras): para clasificar NORMAL/PAC
  - Ventanas de RITMO   (~1000 muestras): para clasificar NSR/AFIB

El segmentador es el componente que permite al médico subir una señal
ECG completa y obtener etiquetas por cada latido y cada segmento de ritmo,
en lugar de tener que pre-segmentar manualmente.

Flujo:
  señal_larga → remuestreo 250 Hz → detector_qrs → ventanas_latido
                                  → ventaneo_ritmo → ventanas_ritmo

Algoritmo de detección QRS:
  Basado en Pan-Tompkins simplificado usando scipy.signal.find_peaks
  con preprocesamiento por filtro derivativo y suavizado cuadrático.
  Referencia: Pan & Tompkins (1985), IEEE Trans. Biomed. Eng.

Ventaneo de ritmo:
  Ventana deslizante de WINDOW_RITMO muestras con paso STEP_RITMO.
  Ventanas con overlap del 50% para capturar transiciones NSR↔AFIB.
"""

import numpy as np
from scipy import signal as sp_signal
from typing import List, Tuple, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FS_OBJETIVO

# ── Parámetros de segmentación ─────────────────────────────────────────────────
# Latidos
WINDOW_LATIDO   = 162    # muestras totales por ventana de latido (a 250 Hz)
MARGEN_PRE_R    = 65     # muestras antes del pico R
MARGEN_POST_R   = 97     # muestras después del pico R (total = 162)
MIN_RR_MS       = 300    # distancia mínima entre latidos (ms) → 75 muestras a 250 Hz
MAX_RR_MS       = 1500   # distancia máxima entre latidos (ms)

# Ritmos
WINDOW_RITMO    = 1000   # muestras por ventana de ritmo (~4s a 250 Hz)
STEP_RITMO      = 500    # paso del ventaneo (50% overlap)
MIN_VENTANAS    = 2      # mínimo de ventanas de ritmo para clasificar

# QRS detector
FREQ_BAJA       = 5.0    # Hz — filtro pasa-banda inferior
FREQ_ALTA       = 15.0   # Hz — filtro pasa-banda superior
SMOOTH_WINDOW   = 15     # muestras para suavizado cuadrático


# ─────────────────────────────────────────────────────────────────────────────
# DETECTOR QRS
# ─────────────────────────────────────────────────────────────────────────────

def detectar_picos_r(senal: np.ndarray,
                     fs: int = FS_OBJETIVO) -> np.ndarray:
    """
    Detecta los picos R del complejo QRS usando un detector simplificado
    basado en Pan-Tompkins.

    Pasos:
    1. Filtro pasa-banda (5-15 Hz) — elimina ruido de línea base y muscular
    2. Filtro derivativo — resalta las pendientes del QRS
    3. Elevación al cuadrado — amplifica las características del QRS
    4. Integración por ventana móvil — suaviza y crea envolvente
    5. find_peaks con distancia mínima inter-R

    Parámetros
    ----------
    senal : señal ECG normalizada (Z-score)
    fs    : frecuencia de muestreo (default 250 Hz)

    Retorna
    -------
    np.ndarray con índices de los picos R detectados
    """
    # 1. Filtro pasa-banda
    nyq  = fs / 2.0
    low  = FREQ_BAJA / nyq
    high = min(FREQ_ALTA / nyq, 0.99)
    b, a = sp_signal.butter(2, [low, high], btype='band')
    try:
        filtrada = sp_signal.filtfilt(b, a, senal)
    except Exception:
        filtrada = senal.copy()

    # 2. Filtro derivativo (diferencia de primer orden)
    derivada = np.diff(filtrada, prepend=filtrada[0])

    # 3. Elevación al cuadrado
    cuadrada = derivada ** 2

    # 4. Integración por ventana móvil
    ventana   = int(0.15 * fs)  # 150 ms
    kernel    = np.ones(ventana) / ventana
    integrada = np.convolve(cuadrada, kernel, mode='same')

    # 5. Detección de picos con umbral adaptativo
    min_dist  = int(MIN_RR_MS / 1000 * fs)
    max_dist  = int(MAX_RR_MS / 1000 * fs)
    umbral    = np.mean(integrada) * 0.5

    picos, _ = sp_signal.find_peaks(
        integrada,
        distance  = min_dist,
        height    = umbral,
        prominence = umbral * 0.3,
    )

    # Filtrar picos demasiado separados (posibles artefactos)
    if len(picos) > 1:
        rr        = np.diff(picos)
        validos   = np.concatenate([[True], rr < max_dist])
        picos     = picos[validos]

    return picos


def _refinar_pico_r(senal: np.ndarray, idx: int,
                    radio: int = 10) -> int:
    """
    Refina la posición del pico R buscando el máximo absoluto en
    una ventana de ±radio muestras alrededor del índice estimado.
    """
    inicio = max(0, idx - radio)
    fin    = min(len(senal), idx + radio + 1)
    ventana = np.abs(senal[inicio:fin])
    return inicio + np.argmax(ventana)


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACCIÓN DE VENTANAS DE LATIDO
# ─────────────────────────────────────────────────────────────────────────────

def extraer_ventanas_latido(
        senal: np.ndarray,
        fs: int = FS_OBJETIVO
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Extrae ventanas de latido individuales centradas en cada pico R.

    Parámetros
    ----------
    senal : señal ECG normalizada (Z-score), 1D
    fs    : frecuencia de muestreo

    Retorna
    -------
    ventanas  : lista de arrays de WINDOW_LATIDO muestras cada uno
    picos_r   : índices de los picos R usados
    tiempos_s : tiempo en segundos del inicio de cada ventana
    """
    picos_r = detectar_picos_r(senal, fs)
    if len(picos_r) == 0:
        return [], np.array([]), np.array([])

    # Refinar posiciones de picos R
    picos_r = np.array([_refinar_pico_r(senal, p) for p in picos_r])

    ventanas   = []
    picos_usados = []
    tiempos_s  = []

    for p in picos_r:
        inicio = p - MARGEN_PRE_R
        fin    = p + MARGEN_POST_R

        # Verificar que la ventana cabe en la señal
        if inicio < 0 or fin > len(senal):
            continue

        ventana = senal[inicio:fin].copy()
        if len(ventana) == WINDOW_LATIDO:
            ventanas.append(ventana)
            picos_usados.append(p)
            tiempos_s.append(inicio / fs)

    return ventanas, np.array(picos_usados), np.array(tiempos_s)


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACCIÓN DE VENTANAS DE RITMO
# ─────────────────────────────────────────────────────────────────────────────

def extraer_ventanas_ritmo(
        senal: np.ndarray,
        fs: int = FS_OBJETIVO
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Extrae ventanas de ritmo por ventaneo deslizante con overlap 50%.

    Parámetros
    ----------
    senal : señal ECG normalizada (Z-score)
    fs    : frecuencia de muestreo

    Retorna
    -------
    ventanas  : lista de arrays de WINDOW_RITMO muestras
    tiempos_s : tiempo en segundos del inicio de cada ventana
    """
    ventanas  = []
    tiempos_s = []

    n = len(senal)
    if n < WINDOW_RITMO:
        # Señal demasiado corta: rellenar con ceros al final
        pad    = np.zeros(WINDOW_RITMO - n)
        senal  = np.concatenate([senal, pad])
        ventanas.append(senal.copy())
        tiempos_s.append(0.0)
        return ventanas, np.array(tiempos_s)

    for inicio in range(0, n - WINDOW_RITMO + 1, STEP_RITMO):
        fin     = inicio + WINDOW_RITMO
        ventana = senal[inicio:fin].copy()
        ventanas.append(ventana)
        tiempos_s.append(inicio / fs)

    return ventanas, np.array(tiempos_s)


# ─────────────────────────────────────────────────────────────────────────────
# SEGMENTADOR PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def segmentar_senal(
        senal_raw: np.ndarray,
        fs_origen: int,
        modo: str = 'auto'
) -> dict:
    """
    Segmenta una señal ECG completa en ventanas clasificables.

    Parámetros
    ----------
    senal_raw : señal ECG cruda (cualquier escala)
    fs_origen : frecuencia de muestreo original
    modo      : 'latido', 'ritmo', o 'auto' (detecta por longitud)

    Retorna
    -------
    dict con:
        ventanas_latido  : lista de arrays ~162 muestras (NORMAL/PAC)
        tiempos_latido   : tiempos en segundos de cada latido
        picos_r          : índices de los picos R detectados
        ventanas_ritmo   : lista de arrays ~1000 muestras (NSR/AFIB)
        tiempos_ritmo    : tiempos en segundos de cada ventana de ritmo
        fs               : 250 Hz (después del remuestreo)
        n_muestras_orig  : muestras de la señal original
        n_muestras_250   : muestras después del remuestreo
        duracion_s       : duración total en segundos
        modo_usado       : 'latido', 'ritmo', o 'ambos'
    """
    from modulo_1_clasificador.lectura_senal import (
        remuestrear_senal, normalizar_zscore)

    # 1. Remuestreo a 250 Hz
    if fs_origen != FS_OBJETIVO:
        senal_250 = remuestrear_senal(senal_raw,
                                      fs_origen  = fs_origen,
                                      fs_destino = FS_OBJETIVO)
    else:
        senal_250 = senal_raw.copy()

    # 2. Normalización Z-score
    senal_n = normalizar_zscore(senal_250)
    n       = len(senal_n)

    # 3. Determinar modo automáticamente
    if modo == 'auto':
        if n < 300:
            modo_usado = 'latido'
        elif n < 1500:
            modo_usado = 'ritmo'
        else:
            modo_usado = 'ambos'
    else:
        modo_usado = modo

    resultado = {
        'fs'              : FS_OBJETIVO,
        'n_muestras_orig' : len(senal_raw),
        'n_muestras_250'  : n,
        'duracion_s'      : round(n / FS_OBJETIVO, 2),
        'modo_usado'      : modo_usado,
        'ventanas_latido' : [],
        'tiempos_latido'  : np.array([]),
        'picos_r'         : np.array([]),
        'ventanas_ritmo'  : [],
        'tiempos_ritmo'   : np.array([]),
        'senal_normalizada': senal_n,
    }

    # 4. Extraer ventanas según modo
    if modo_usado in ('latido', 'ambos'):
        v_lat, picos_r, t_lat = extraer_ventanas_latido(senal_n)
        resultado['ventanas_latido'] = v_lat
        resultado['tiempos_latido']  = t_lat
        resultado['picos_r']         = picos_r
        print(f"[SEG] Latidos detectados: {len(v_lat)}"
              f" (de {len(picos_r)} picos R)")

    if modo_usado in ('ritmo', 'ambos'):
        v_rit, t_rit = extraer_ventanas_ritmo(senal_n)
        resultado['ventanas_ritmo'] = v_rit
        resultado['tiempos_ritmo']  = t_rit
        print(f"[SEG] Ventanas de ritmo : {len(v_rit)}"
              f" (c/u {WINDOW_RITMO} muestras, paso {STEP_RITMO})")

    return resultado


# ─────────────────────────────────────────────────────────────────────────────
# CLASIFICAR TODAS LAS VENTANAS
# ─────────────────────────────────────────────────────────────────────────────

def clasificar_ventanas(segmentacion: dict) -> dict:
    """
    Clasifica todas las ventanas extraídas por el segmentador.

    Para cada ventana de latido: clasifica como NORMAL o PAC
    Para cada ventana de ritmo:  clasifica como NSR o AFIB

    Retorna
    -------
    dict con listas de resultados por ventana, incluyendo:
        etiquetas, probabilidades, confidence_gaps, tiempos
    """
    from modulo_1_clasificador.embedding import embedding_una_senal
    from modulo_1_clasificador.clasificador import cargar_modelo
    import numpy as np

    modelo, le, clases = cargar_modelo()
    micro = list(np.load(str(__import__('config').ARCHIVO_MICRO_SHAPELETS),
                          allow_pickle=True))
    macro = list(np.load(str(__import__('config').ARCHIVO_MACRO_SHAPELETS),
                          allow_pickle=True))

    def _clasificar_una(ventana: np.ndarray) -> dict:
        """Clasifica una ventana y retorna el resultado estructurado."""
        df_emb  = embedding_una_senal(ventana, micro, macro)
        y_pred  = modelo.predict(df_emb)[0]
        y_proba = modelo.predict_proba(df_emb)[0]
        clase   = str(le.inverse_transform([y_pred])[0])
        probs   = {clases[j]: round(float(y_proba[j]), 4)
                   for j in range(len(clases))}
        ps      = sorted(probs.values(), reverse=True)
        gap     = round(float(ps[0] - ps[1]), 4)
        return {
            'clase'           : clase,
            'probabilidades'  : probs,
            'confidence_gap'  : gap,
            'escalar'         : bool(gap < __import__('config').UMBRAL_CONFIANZA),
        }

    resultado = {
        'latidos'  : [],
        'ritmos'   : [],
        'resumen'  : {},
    }

    # Clasificar latidos
    v_lat = segmentacion.get('ventanas_latido', [])
    t_lat = segmentacion.get('tiempos_latido', [])
    if v_lat:
        print(f"\n[SEG] Clasificando {len(v_lat)} ventanas de latido...")
        for i, (v, t) in enumerate(zip(v_lat, t_lat)):
            r = _clasificar_una(v)
            r['tiempo_s'] = round(float(t), 3)
            r['ventana_id'] = i
            resultado['latidos'].append(r)
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(v_lat)} latidos clasificados")

    # Clasificar ritmos
    v_rit = segmentacion.get('ventanas_ritmo', [])
    t_rit = segmentacion.get('tiempos_ritmo', [])
    if v_rit:
        print(f"\n[SEG] Clasificando {len(v_rit)} ventanas de ritmo...")
        for i, (v, t) in enumerate(zip(v_rit, t_rit)):
            r = _clasificar_una(v)
            r['tiempo_s'] = round(float(t), 3)
            r['ventana_id'] = i
            resultado['ritmos'].append(r)
            if (i + 1) % 5 == 0:
                print(f"  {i+1}/{len(v_rit)} ritmos clasificados")

    # Resumen estadístico
    if resultado['latidos']:
        clases_lat = [r['clase'] for r in resultado['latidos']]
        from collections import Counter
        conteo_lat = Counter(clases_lat)
        escalados_lat = sum(1 for r in resultado['latidos'] if r['escalar'])
        resultado['resumen']['latidos'] = {
            'total'    : len(resultado['latidos']),
            'conteo'   : dict(conteo_lat),
            'escalados': escalados_lat,
            'clase_dominante': conteo_lat.most_common(1)[0][0],
        }

    if resultado['ritmos']:
        clases_rit = [r['clase'] for r in resultado['ritmos']]
        from collections import Counter
        conteo_rit = Counter(clases_rit)
        escalados_rit = sum(1 for r in resultado['ritmos'] if r['escalar'])
        resultado['resumen']['ritmos'] = {
            'total'    : len(resultado['ritmos']),
            'conteo'   : dict(conteo_rit),
            'escalados': escalados_rit,
            'clase_dominante': conteo_rit.most_common(1)[0][0],
        }

    return resultado


# ─────────────────────────────────────────────────────────────────────────────
# DEMO / TEST DE LÍNEA DE COMANDOS
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import pandas as pd

    ruta   = sys.argv[1] if len(sys.argv) > 1 else None
    fs_ori = int(sys.argv[2]) if len(sys.argv) > 2 else 360

    if ruta:
        try:
            df  = pd.read_csv(ruta, header=None)
            col = 0
            for c in range(df.shape[1]):
                vals = pd.to_numeric(df.iloc[:, c], errors='coerce')
                if vals.notna().sum() > 50:
                    col = c
                    break
            senal = pd.to_numeric(df.iloc[:, col],
                                   errors='coerce').dropna().values
            print(f"[SEG] Señal cargada: {len(senal)} muestras @ {fs_ori} Hz")
        except Exception as e:
            print(f"Error leyendo CSV: {e}")
            sys.exit(1)
    else:
        # Señal sintética: 10 segundos de ECG simulado a 360 Hz
        print("[SEG] Usando señal sintética de 10s @ 360 Hz")
        fs_ori = 360
        t      = np.linspace(0, 10, 3600)
        fc     = 1.2   # frecuencia cardíaca ~72 bpm
        senal  = (np.sin(2 * np.pi * fc * t) * 0.8 +
                  np.random.normal(0, 0.05, len(t)))
        # Agregar "picos R" artificiales
        for pico in range(int(fc * 10)):
            idx = int(pico / fc * 360)
            if idx < len(senal):
                senal[idx] += 3.0

    # Segmentar
    seg = segmentar_senal(senal, fs_origen=fs_ori, modo='ambos')

    print(f"\n{'='*55}")
    print(f"  RESULTADO DE SEGMENTACIÓN")
    print(f"{'='*55}")
    print(f"  Duración señal   : {seg['duracion_s']} s")
    print(f"  Muestras @250 Hz : {seg['n_muestras_250']}")
    print(f"  Modo             : {seg['modo_usado']}")
    print(f"  Latidos extraídos: {len(seg['ventanas_latido'])}")
    print(f"  Picos R          : {len(seg['picos_r'])}")
    print(f"  Ventanas ritmo   : {len(seg['ventanas_ritmo'])}")
    print(f"{'='*55}")
    print("\n  Para clasificar las ventanas llama a:")
    print("  from modulo_1_clasificador.segmentador import clasificar_ventanas")
    print("  resultados = clasificar_ventanas(seg)")
