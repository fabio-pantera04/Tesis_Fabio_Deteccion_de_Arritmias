"""
features.py
===========
Extrae las 17 features del paper Andotra & Sunkaria (2024) por lead.
Como tus datos son single-lead, extraemos solo 17 features (vs 34 del paper que usa II+V5).

Grupos de features:
1. RR Intervals (3): RR, Average RR, Post RR
2. Heartbeat Interval Features (4): PQ, QRS, QT, ST
3. Heartbeat Amplitude Features (5): P, Q, R, S, T peak amplitudes
4. Morphological Features (5): coeficientes de Hermite (REINTERPRETADOS)

Nota crítica: el paper define "QRS-morph feature [0-4]" como un vector de
atributos morfológicos pero NUNCA especifica qué son. Usamos coeficientes de
Hermite que es el estándar de la literatura (Lagerholm et al. 2000).
"""

import math
import numpy as np
from scipy.special import hermite
from preprocessing import (
    detect_r_peaks_robust,
    select_main_r_peak,
    detect_all_fiducials,
)


# =====================================================================
# Helpers
# =====================================================================

def safe_div(a, b, default=0.0):
    """División segura que evita NaN/Inf."""
    if b is None or abs(b) < 1e-12:
        return default
    return a / b


def to_ms(samples, fs):
    """Convierte muestras a milisegundos."""
    if samples is None:
        return np.nan
    return samples / fs * 1000


# =====================================================================
# Grupo 1: RR Intervals (3 features)
# =====================================================================

def rr_features(r_peaks: np.ndarray, central_r_idx: int, fs: int) -> np.ndarray:
    """
    Calcula RR, Average RR y Post RR.
    
    Si solo hay 1 R-peak (latido único), las features RR no se pueden calcular
    desde la señal misma; devuelven NaN para imputación posterior.
    
    Para señales largas (RITMO) con múltiples R, calculamos los RR completos.
    """
    feats = np.full(3, np.nan)
    
    if len(r_peaks) < 2:
        return feats  # No hay RR calculables
    
    # Todos los intervalos RR consecutivos (en ms)
    rrs = np.diff(r_peaks) / fs * 1000
    
    # 1. RR: intervalo RR donde está el R central, o el promedio si es ambiguo
    # Buscamos el RR que incluye al central_r_idx
    central_pos = np.searchsorted(r_peaks, central_r_idx)
    if 0 < central_pos < len(r_peaks):
        feats[0] = rrs[central_pos - 1]  # RR previo (a-b)
    else:
        feats[0] = np.mean(rrs)  # fallback
    
    # 2. Average RR
    feats[1] = np.mean(rrs)
    
    # 3. Post RR: el RR siguiente al central
    if central_pos < len(rrs):
        feats[2] = rrs[central_pos]
    elif len(rrs) > 0:
        feats[2] = rrs[-1]  # fallback
    
    return feats


# =====================================================================
# Grupo 2: Heartbeat Interval Features (4 features)
# =====================================================================

def interval_features(ann: dict, fs: int) -> np.ndarray:
    """
    PQ, QRS, QT, ST intervals (en ms).
    
    Definiciones del paper:
    - PQ = t(Q) - t(P)
    - QRS = t(S) - t(Q)
    - QT = t(T) - t(Q)
    - ST = t(T) - t(S)
    """
    feats = np.full(4, np.nan)
    
    p, q, s, t = ann.get('p_idx'), ann.get('q_idx'), ann.get('s_idx'), ann.get('t_idx')
    
    # PQ Interval
    if p is not None and q is not None and q > p:
        feats[0] = to_ms(q - p, fs)
    
    # QRS Interval
    if q is not None and s is not None and s > q:
        feats[1] = to_ms(s - q, fs)
    
    # QT Interval
    if q is not None and t is not None and t > q:
        feats[2] = to_ms(t - q, fs)
    
    # ST Interval
    if s is not None and t is not None and t > s:
        feats[3] = to_ms(t - s, fs)
    
    return feats


# =====================================================================
# Grupo 3: Heartbeat Amplitude Features (5 features)
# =====================================================================

def amplitude_features(ann: dict) -> np.ndarray:
    """
    Amplitudes de P, Q, R, S, T peaks.
    """
    feats = np.full(5, np.nan)
    
    for i, key in enumerate(['p_amp', 'q_amp', 'r_amp', 's_amp', 't_amp']):
        val = ann.get(key)
        if val is not None:
            feats[i] = val
    
    return feats


# =====================================================================
# Grupo 4: Morphological Features - HERMITE COEFFICIENTS (5 features)
# =====================================================================

def hermite_coefficients(qrs_segment: np.ndarray, n_coeffs: int = 5,
                          sigma: float = 25.0) -> np.ndarray:
    """
    Calcula los primeros n_coeffs coeficientes de la expansión de Hermite
    sobre un segmento QRS.
    
    Reinterpretación de "QRS-morph features [m_0, m_1, m_2, m_3, m_4]" del paper
    (que NO los define). Hermite es estándar para ECG (Lagerholm 2000).
    
    Modelo:
        x(t) ≈ sum_{n=0}^{N-1} c_n * φ_n(t, σ)
    donde φ_n son funciones de Hermite ortonormales:
        φ_n(t, σ) = (1/sqrt(σ * 2^n * n! * sqrt(π))) * H_n(t/σ) * exp(-t²/(2σ²))
    
    Parámetros:
    -----------
    qrs_segment : segmento centrado en R (~80 muestras a 250 Hz = 320ms)
    n_coeffs : número de coeficientes a devolver
    sigma : parámetro de escala de las funciones Hermite (en muestras)
    """
    if qrs_segment is None or len(qrs_segment) < 5:
        return np.full(n_coeffs, np.nan)
    
    # Centrar el tiempo en 0
    n = len(qrs_segment)
    t = np.arange(n) - n // 2
    t_normalized = t / sigma
    
    coeffs = np.zeros(n_coeffs)
    for k in range(n_coeffs):
        # Polinomio de Hermite (físico, He_n)
        H_k = hermite(k)(t_normalized)
        # Función de Hermite ortonormal
        norm_factor = 1.0 / np.sqrt(sigma * (2 ** k) * math.factorial(k) * np.sqrt(np.pi))
        phi_k = norm_factor * H_k * np.exp(-t_normalized ** 2 / 2)
        # Coeficiente = producto interno
        coeffs[k] = np.dot(qrs_segment, phi_k)
    
    return coeffs


def morphological_features(signal: np.ndarray, ann: dict, fs: int = 250,
                            qrs_window_ms: int = 160) -> np.ndarray:
    """
    Calcula los 5 coeficientes de Hermite sobre el complejo QRS.
    
    El segmento QRS se extrae centrado en R con ventana de qrs_window_ms ms
    (80 ms a cada lado = 160 ms total a 250 Hz = 40 muestras).
    """
    r = ann.get('r_idx')
    if r is None:
        return np.full(5, np.nan)
    
    half_win = int(qrs_window_ms / 2 / 1000 * fs)
    start = max(0, r - half_win)
    end = min(len(signal), r + half_win + 1)
    qrs_segment = signal[start:end]
    
    if len(qrs_segment) < 5:
        return np.full(5, np.nan)
    
    return hermite_coefficients(qrs_segment, n_coeffs=5, sigma=half_win / 2)


# =====================================================================
# Pipeline completo: extracción de las 17 features
# =====================================================================

FEATURE_NAMES = [
    # RR (3)
    'RR', 'avg_RR', 'post_RR',
    # Intervals (4)
    'PQ_interval', 'QRS_interval', 'QT_interval', 'ST_interval',
    # Amplitudes (5)
    'P_peak', 'Q_peak', 'R_peak', 'S_peak', 'T_peak',
    # Morphological (5) - Hermite coefficients
    'morph_0', 'morph_1', 'morph_2', 'morph_3', 'morph_4',
]


def extract_17_features(signal: np.ndarray, fs: int = 250,
                          is_rhythm: bool = False) -> np.ndarray:
    """
    Extrae las 17 features del paper para una señal.
    
    Parámetros:
    -----------
    signal : señal ECG (single-lead)
    fs : sample rate (default 250 Hz)
    is_rhythm : si True, la señal es de ritmo (múltiples latidos);
                aplicamos detección sobre todos y agregamos sobre el latido central.
    
    Devuelve:
    ---------
    array (17,) con las features. NaNs donde la detección falló.
    """
    # 1. Detectar R-peaks
    r_peaks = detect_r_peaks_robust(signal, fs=fs)
    
    if len(r_peaks) == 0:
        return np.full(17, np.nan)
    
    # 2. Seleccionar R central (prominente)
    central_r = select_main_r_peak(signal, r_peaks)
    
    # 3. Detectar fiducials P, Q, S, T alrededor del R central
    ann = detect_all_fiducials(signal, central_r, fs=fs)
    
    # 4. Calcular features por grupo
    rr_feats = rr_features(r_peaks, central_r, fs)
    interval_feats = interval_features(ann, fs)
    amp_feats = amplitude_features(ann)
    morph_feats = morphological_features(signal, ann, fs=fs)
    
    return np.concatenate([rr_feats, interval_feats, amp_feats, morph_feats])


def impute_nans_train_test(train_feats: np.ndarray,
                            test_feats: np.ndarray) -> tuple:
    """
    Imputa NaNs usando la MEDIANA del train set, aplicada tanto a train como test.
    
    Esto evita data leakage: el test NO ve estadísticas del propio test.
    """
    imputed_train = train_feats.copy()
    imputed_test = test_feats.copy()
    
    medians = np.zeros(train_feats.shape[1])
    
    for col in range(train_feats.shape[1]):
        col_train = train_feats[:, col]
        median = np.nanmedian(col_train)
        if np.isnan(median):
            median = 0.0
        medians[col] = median
        
        imputed_train[np.isnan(col_train), col] = median
        imputed_test[np.isnan(imputed_test[:, col]), col] = median
    
    return imputed_train, imputed_test, medians


if __name__ == "__main__":
    # Smoke test
    np.random.seed(0)
    fake_ecg = np.sin(np.linspace(0, 8 * np.pi, 250)) + 0.1 * np.random.randn(250)
    fake_ecg[120:130] += 2.0
    
    feats = extract_17_features(fake_ecg, is_rhythm=False)
    print(f"Shape: {feats.shape}")
    for name, val in zip(FEATURE_NAMES, feats):
        print(f"  {name:>18s}: {val:.4f}")
    print(f"NaNs: {np.sum(np.isnan(feats))}/17")
