"""
preprocessing.py
================
Réplica fiel de la Sección 2.2 del paper Qin et al. (2024).

Carga señales ECG desde CSVs (una columna, ya normalizada) y aplica:
1. Denoising con wavelet de 6 niveles (Eq. 1-3 del paper)
2. Detección de picos R (para validación; las señales ya vienen segmentadas)
3. Detección de QRS, P y T (para features médicas posteriores)

El paper no especifica wavelet madre ni umbral; usamos db6 con soft thresholding
universal por ser el estándar más común en literatura ECG.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pywt
from scipy.signal import find_peaks


# =====================================================================
# Carga de datos
# =====================================================================

def load_signal_from_csv(csv_path: Path) -> np.ndarray:
    """Carga una señal ECG desde un CSV de una sola columna ya normalizada."""
    df = pd.read_csv(csv_path, header=None)
    if df.shape[1] != 1:
        # Asume que la primera fila puede ser un header; reintenta con header=0
        df = pd.read_csv(csv_path)
        if df.shape[1] != 1:
            raise ValueError(f"{csv_path} tiene {df.shape[1]} columnas; se esperaba 1.")
    # np.ascontiguousarray + .copy() asegura array escribible (pywt requiere writeable buffer)
    return np.ascontiguousarray(df.iloc[:, 0].to_numpy(dtype=np.float64)).copy()


def load_class_directory(class_dir: Path, expected_length: int = None) -> tuple:
    """
    Carga todas las señales de un directorio de clase.
    
    Devuelve:
    - signals: array (n_signals, signal_length)
    - filenames: lista de nombres de archivo (para trazabilidad)
    """
    csv_files = sorted(class_dir.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No se encontraron CSVs en {class_dir}")
    
    signals = []
    filenames = []
    for csv_file in csv_files:
        sig = load_signal_from_csv(csv_file)
        if expected_length is not None and len(sig) != expected_length:
            # Pad o truncate; el paper no especifica pero la consistencia es necesaria
            if len(sig) > expected_length:
                sig = sig[:expected_length]
            else:
                sig = np.pad(sig, (0, expected_length - len(sig)), mode='constant')
        signals.append(sig)
        filenames.append(csv_file.name)
    
    return np.array(signals), filenames


# =====================================================================
# Wavelet denoising (Sec. 2.2, Eqs. 1-3)
# =====================================================================

def wavelet_denoise(signal: np.ndarray,
                    wavelet: str = 'db6',
                    level: int = 6,
                    threshold_mode: str = 'soft') -> np.ndarray:
    """
    Aplica denoising por wavelet de 6 niveles.
    
    Parámetros:
    -----------
    signal : array 1D - señal ECG
    wavelet : nombre de la wavelet madre (default 'db6'; el paper no especifica)
    level : niveles de descomposición (paper: 6)
    threshold_mode : 'soft' o 'hard'
    
    Devuelve:
    ---------
    Señal denoised del mismo tamaño que la entrada.
    
    Implementación del paper:
    Eq. 1: cA_{j+1}(n) = sum_m cA_j(m) h(m-2n)        # low-pass
    Eq. 2: cD_{j+1}(n) = sum_m cA_j(m) g(m-2n)        # high-pass
    Eq. 3: cA_j(n) = reconstruction
    
    Para denoising real se requiere thresholding de los coeficientes de detalle,
    paso que el paper omite mencionar explícitamente pero que es estándar.
    """
    # Descomposición
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    # Estimación robusta de sigma usando MAD del nivel más fino
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    
    # Umbral universal de Donoho
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))
    
    # Aplicar umbral solo a coeficientes de detalle (no al de aproximación)
    coeffs_thresholded = [coeffs[0]] + [
        pywt.threshold(c, threshold, mode=threshold_mode) for c in coeffs[1:]
    ]
    
    # Reconstrucción
    denoised = pywt.waverec(coeffs_thresholded, wavelet)
    
    # Asegurar mismo tamaño (waverec puede devolver +1 o +2 muestras)
    return denoised[:len(signal)]


# =====================================================================
# Detección de R, QRS, P, T (Sec. 2.2, paso 3)
# =====================================================================

def detect_r_peaks(signal: np.ndarray, fs: int = 250) -> np.ndarray:
    """
    Detecta picos R en la señal.
    Paso (1)-(2) del proceso de detección descrito en Sec. 2.2.
    
    Para señales de un solo latido (LATIDO, 250 muestras), debería detectar 1 R.
    Para ritmos largos (RITMO, 1000 muestras a 250 Hz = 4 segundos), debería detectar ~3-5 R.
    """
    # Distancia mínima entre picos: ~250 ms = 0.25 * fs (refractario fisiológico)
    min_distance = int(0.25 * fs)
    # Altura mínima: 0.5 desviaciones estándar sobre la media (señal normalizada)
    height_threshold = signal.mean() + 0.5 * signal.std()
    
    peaks, properties = find_peaks(signal,
                                    distance=min_distance,
                                    height=height_threshold)
    return peaks


def select_main_r_peak(signal: np.ndarray,
                       r_peaks: np.ndarray,
                       strategy: str = 'prominent') -> int:
    """
    Cuando hay múltiples R-peaks en una señal de latido (caso común en PAC),
    selecciona el R principal según estrategia.
    
    Parámetros:
    -----------
    signal : señal completa
    r_peaks : array de índices de picos detectados
    strategy : 'prominent' (mayor amplitud), 'central' (más cercano al centro),
               o 'first' (primero detectado)
    
    Devuelve:
    ---------
    Índice del R-peak seleccionado.
    """
    if len(r_peaks) == 0:
        # Fallback: usar el máximo absoluto de la señal como R sintético
        return int(np.argmax(np.abs(signal)))
    
    if len(r_peaks) == 1:
        return int(r_peaks[0])
    
    if strategy == 'prominent':
        # R con mayor amplitud absoluta (más robusto para PAC)
        amplitudes = np.abs(signal[r_peaks])
        return int(r_peaks[np.argmax(amplitudes)])
    elif strategy == 'central':
        center = len(signal) // 2
        return int(r_peaks[np.argmin(np.abs(r_peaks - center))])
    elif strategy == 'first':
        return int(r_peaks[0])
    else:
        raise ValueError(f"strategy desconocida: {strategy}")


def detect_qrs_p_t(signal: np.ndarray, r_peak_idx: int, fs: int = 250) -> dict:
    """
    Detecta los puntos fiduciales QRS, P y T alrededor de un pico R dado.
    
    Sigue el paso (3) de Sec. 2.2: "Specific threshold values were set to detect 
    the QRS complex, P-wave, and T-wave".
    
    Devuelve dict con:
    - q_idx, s_idx: límites del complejo QRS
    - p_idx, p_amp: pico de la onda P (puede ser None si no se detecta)
    - t_idx, t_amp: pico de la onda T
    - qrs_onset, qrs_offset: inicio y fin del QRS
    """
    n = len(signal)
    result = {
        'r_idx': r_peak_idx, 'r_amp': signal[r_peak_idx],
        'q_idx': None, 's_idx': None,
        'qrs_onset': None, 'qrs_offset': None,
        'p_idx': None, 'p_amp': None,
        't_idx': None, 't_amp': None,
    }
    
    # Q: mínimo local en ventana [-50ms, 0] del R
    q_window_start = max(0, r_peak_idx - int(0.05 * fs))
    if q_window_start < r_peak_idx:
        q_local = np.argmin(signal[q_window_start:r_peak_idx])
        result['q_idx'] = q_window_start + q_local
    
    # S: mínimo local en ventana [0, +50ms] del R
    s_window_end = min(n, r_peak_idx + int(0.05 * fs))
    if r_peak_idx < s_window_end:
        s_local = np.argmin(signal[r_peak_idx:s_window_end])
        result['s_idx'] = r_peak_idx + s_local
    
    # QRS onset/offset por umbral de pendiente (~80ms total)
    qrs_half = int(0.04 * fs)
    result['qrs_onset'] = max(0, r_peak_idx - qrs_half)
    result['qrs_offset'] = min(n - 1, r_peak_idx + qrs_half)
    
    # P: máximo local en ventana [-200ms, -80ms] del R
    p_window_end = max(0, r_peak_idx - int(0.08 * fs))
    p_window_start = max(0, r_peak_idx - int(0.20 * fs))
    if p_window_start < p_window_end:
        p_local = np.argmax(signal[p_window_start:p_window_end])
        result['p_idx'] = p_window_start + p_local
        result['p_amp'] = signal[result['p_idx']]
    
    # T: máximo local en ventana [+150ms, +400ms] del R
    t_window_start = min(n, r_peak_idx + int(0.15 * fs))
    t_window_end = min(n, r_peak_idx + int(0.40 * fs))
    if t_window_start < t_window_end:
        t_local = np.argmax(signal[t_window_start:t_window_end])
        result['t_idx'] = t_window_start + t_local
        result['t_amp'] = signal[result['t_idx']]
    
    return result


# =====================================================================
# Pipeline completo de preprocesamiento por señal
# =====================================================================

def preprocess_signal(signal: np.ndarray, fs: int = 250, denoise: bool = True) -> dict:
    """
    Aplica todo el preprocesamiento a una señal y devuelve las anotaciones fiduciales.
    
    Para señales LATIDO (250 muestras) se asume el R-peak central.
    Para señales RITMO (1000 muestras) se detectan todos los R-peaks.
    """
    if denoise:
        signal = wavelet_denoise(signal)
    
    r_peaks = detect_r_peaks(signal, fs=fs)
    
    # Anotaciones por cada R detectado
    annotations = [detect_qrs_p_t(signal, r, fs=fs) for r in r_peaks]
    
    return {
        'signal': signal,
        'r_peaks': r_peaks,
        'annotations': annotations,
    }


if __name__ == "__main__":
    # Smoke test con señal sintética
    t = np.linspace(0, 1, 250)
    fake_ecg = np.sin(2 * np.pi * 5 * t) + 0.3 * np.random.randn(250)
    fake_ecg[120:130] += 3.0  # Pico R sintético
    
    result = preprocess_signal(fake_ecg, fs=250)
    print(f"R peaks detectados: {result['r_peaks']}")
    print(f"Anotaciones del primer R: {result['annotations'][0] if result['annotations'] else 'ninguna'}")
