"""
preprocessing.py
================
Carga señales ECG desde CSVs y detecta los puntos fiduciales P, Q, R, S, T
necesarios para extraer las 17 features de Andotra & Sunkaria (2024).

El paper NO especifica algoritmo de detección. Usamos NeuroKit2 que es el
estándar de facto en investigación ECG, con un fallback custom basado en
scipy.signal.find_peaks y ventanas heurísticas cuando NeuroKit2 falla
(común en señales irregulares como AFIB).
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# NeuroKit2 es el estándar para detección de fiducials ECG
try:
    import neurokit2 as nk
    _HAS_NK = True
except ImportError:
    _HAS_NK = False
    warnings.warn("neurokit2 no instalado; usando solo fallback custom")


# =====================================================================
# Carga de CSVs
# =====================================================================

def load_signal_from_csv(csv_path: Path) -> np.ndarray:
    """Carga señal ECG desde CSV de una sola columna, normalizada."""
    df = pd.read_csv(csv_path, header=None)
    if df.shape[1] != 1:
        df = pd.read_csv(csv_path)
        if df.shape[1] != 1:
            raise ValueError(f"{csv_path} tiene {df.shape[1]} columnas; se esperaba 1.")
    return np.ascontiguousarray(df.iloc[:, 0].to_numpy(dtype=np.float64)).copy()


# =====================================================================
# Detección de R-peaks (siempre robusta)
# =====================================================================

def detect_r_peaks_robust(signal: np.ndarray, fs: int = 250) -> np.ndarray:
    """
    Detecta R-peaks usando NeuroKit2 si está disponible, fallback custom si no.
    
    Estrategia:
    1. Intentar nk.ecg_peaks (algoritmo Pan-Tompkins por default)
    2. Si falla o detecta < 1 peak, usar scipy.signal.find_peaks
    """
    if _HAS_NK:
        try:
            # NeuroKit2: detección robusta basada en Pan-Tompkins
            _, info = nk.ecg_peaks(signal, sampling_rate=fs, correct_artifacts=True)
            r_peaks = info["ECG_R_Peaks"]
            r_peaks = r_peaks[~np.isnan(r_peaks)].astype(int)
            if len(r_peaks) > 0:
                return r_peaks
        except Exception:
            pass  # Fallback
    
    # Fallback: scipy con heurísticas conservadoras
    min_distance = int(0.25 * fs)  # 250 ms refractory
    height_threshold = signal.mean() + 0.5 * signal.std()
    peaks, _ = find_peaks(signal, distance=min_distance, height=height_threshold)
    return peaks


def select_main_r_peak(signal: np.ndarray, r_peaks: np.ndarray) -> int:
    """
    Selecciona el R-peak principal (mayor amplitud).
    Para latidos con múltiples R detectados (caso PAC), tomar el más prominente.
    """
    if len(r_peaks) == 0:
        return int(np.argmax(np.abs(signal)))
    if len(r_peaks) == 1:
        return int(r_peaks[0])
    amplitudes = np.abs(signal[r_peaks])
    return int(r_peaks[np.argmax(amplitudes)])


# =====================================================================
# Detección de P, Q, S, T usando NeuroKit2 + fallback
# =====================================================================

def detect_pqst_neurokit(signal: np.ndarray, r_peak: int, fs: int = 250) -> dict:
    """
    Detecta P, Q, S, T peaks usando NeuroKit2.
    Devuelve dict con índices y amplitudes, o None si falla.
    """
    if not _HAS_NK:
        return None
    
    try:
        # nk.ecg_delineate requiere R-peaks ya detectados
        _, waves = nk.ecg_delineate(
            signal,
            rpeaks=np.array([r_peak]),
            sampling_rate=fs,
            method="dwt",  # Discrete Wavelet Transform (robusto)
        )
        
        def safe_extract(key):
            """Extrae el primer valor no-NaN de una clave del dict de NeuroKit2."""
            if key not in waves:
                return None
            vals = waves[key]
            vals = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
            return int(vals[0]) if vals else None
        
        result = {
            'r_idx': r_peak,
            'r_amp': float(signal[r_peak]),
            'p_idx': safe_extract('ECG_P_Peaks'),
            'q_idx': safe_extract('ECG_Q_Peaks'),
            's_idx': safe_extract('ECG_S_Peaks'),
            't_idx': safe_extract('ECG_T_Peaks'),
            'p_onset': safe_extract('ECG_P_Onsets'),
            'p_offset': safe_extract('ECG_P_Offsets'),
            't_onset': safe_extract('ECG_T_Onsets'),
            't_offset': safe_extract('ECG_T_Offsets'),
        }
        
        # Amplitudes
        for key in ['p', 'q', 's', 't']:
            idx = result[f'{key}_idx']
            result[f'{key}_amp'] = float(signal[idx]) if idx is not None else None
        
        return result
    
    except Exception:
        return None


def detect_pqst_fallback(signal: np.ndarray, r_peak: int, fs: int = 250) -> dict:
    """
    Fallback custom basado en ventanas heurísticas y mínimos/máximos locales.
    Usado cuando NeuroKit2 falla (común en AFIB).
    
    Ventanas heurísticas (relativas a R):
    - Q: [-50ms, 0]    → mínimo local
    - S: [0, +50ms]    → mínimo local
    - P: [-200ms, -80ms] → máximo local
    - T: [+150ms, +400ms] → máximo local
    """
    n = len(signal)
    result = {
        'r_idx': r_peak, 'r_amp': float(signal[r_peak]),
        'p_idx': None, 'p_amp': None,
        'q_idx': None, 'q_amp': None,
        's_idx': None, 's_amp': None,
        't_idx': None, 't_amp': None,
        'p_onset': None, 'p_offset': None,
        't_onset': None, 't_offset': None,
    }
    
    # Q: mínimo local en [-50ms, 0]
    q_start = max(0, r_peak - int(0.05 * fs))
    if q_start < r_peak:
        q_local = np.argmin(signal[q_start:r_peak])
        result['q_idx'] = q_start + q_local
        result['q_amp'] = float(signal[result['q_idx']])
    
    # S: mínimo local en [0, +50ms]
    s_end = min(n, r_peak + int(0.05 * fs))
    if r_peak < s_end:
        s_local = np.argmin(signal[r_peak:s_end])
        result['s_idx'] = r_peak + s_local
        result['s_amp'] = float(signal[result['s_idx']])
    
    # P: máximo local en [-200ms, -80ms]
    p_start = max(0, r_peak - int(0.20 * fs))
    p_end = max(0, r_peak - int(0.08 * fs))
    if p_start < p_end:
        p_local = np.argmax(signal[p_start:p_end])
        result['p_idx'] = p_start + p_local
        result['p_amp'] = float(signal[result['p_idx']])
        # Onset y offset aproximados (±20ms del pico)
        result['p_onset'] = max(0, result['p_idx'] - int(0.02 * fs))
        result['p_offset'] = min(n - 1, result['p_idx'] + int(0.02 * fs))
    
    # T: máximo local en [+150ms, +400ms]
    t_start = min(n, r_peak + int(0.15 * fs))
    t_end = min(n, r_peak + int(0.40 * fs))
    if t_start < t_end:
        t_local = np.argmax(signal[t_start:t_end])
        result['t_idx'] = t_start + t_local
        result['t_amp'] = float(signal[result['t_idx']])
        result['t_onset'] = max(0, result['t_idx'] - int(0.04 * fs))
        result['t_offset'] = min(n - 1, result['t_idx'] + int(0.04 * fs))
    
    return result


def detect_all_fiducials(signal: np.ndarray, r_peak: int, fs: int = 250) -> dict:
    """
    Detecta P, Q, R, S, T con NeuroKit2; si falla, usa fallback.
    """
    result = detect_pqst_neurokit(signal, r_peak, fs)
    if result is None or all(result[f'{k}_idx'] is None for k in ['p', 'q', 's', 't']):
        # NeuroKit2 falló completamente; usar fallback
        result = detect_pqst_fallback(signal, r_peak, fs)
    else:
        # NeuroKit2 funcionó parcialmente; rellenar lo que falta con fallback
        fb = detect_pqst_fallback(signal, r_peak, fs)
        for key in ['p_idx', 'q_idx', 's_idx', 't_idx',
                    'p_amp', 'q_amp', 's_amp', 't_amp',
                    'p_onset', 'p_offset', 't_onset', 't_offset']:
            if result.get(key) is None:
                result[key] = fb.get(key)
    return result


# =====================================================================
# Pipeline completo por señal
# =====================================================================

def preprocess_signal(signal: np.ndarray, fs: int = 250) -> dict:
    """
    Detecta R-peaks y los puntos fiduciales para cada R detectado.
    
    Devuelve:
    - signal: señal original (sin denoising; paper no lo aplica)
    - r_peaks: array de índices de R-peaks
    - annotations: lista de dicts con fiducials por cada R
    """
    r_peaks = detect_r_peaks_robust(signal, fs=fs)
    annotations = [detect_all_fiducials(signal, r, fs=fs) for r in r_peaks]
    return {
        'signal': signal,
        'r_peaks': r_peaks,
        'annotations': annotations,
    }


if __name__ == "__main__":
    # Smoke test
    np.random.seed(0)
    t = np.linspace(0, 1, 250)
    fake_ecg = np.sin(2 * np.pi * 5 * t) + 0.1 * np.random.randn(250)
    fake_ecg[120:130] += 2.0  # Pico R sintético
    
    result = preprocess_signal(fake_ecg, fs=250)
    print(f"R-peaks: {result['r_peaks']}")
    if result['annotations']:
        ann = result['annotations'][0]
        print(f"P_idx={ann['p_idx']}, Q_idx={ann['q_idx']}, "
              f"R_idx={ann['r_idx']}, S_idx={ann['s_idx']}, T_idx={ann['t_idx']}")
