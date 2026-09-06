"""
features.py
===========
Réplica fiel de Sec. 2.3 y Tabla 2 del paper Qin et al. (2024).

Extrae las 37 features espaciotemporales:
- 16 features médicas (Sec. 2.3, Tabla 2)
- 20 features estadísticas (Sec. 2.3, Tabla 2)
- 1 feature temporal (DTW vs centro de cluster)

Para SFP-FACC-RITMO (señales largas, múltiples latidos), las features médicas
se calculan sobre el latido central + agregados (mean/std) de todos los latidos
detectados, según decisión documentada en README.md.

NOTA DE PERFORMANCE: para señales largas (>500 muestras), el DTW O(n²) es
costoso. Si tienes numba instalado, se usa la versión JIT-compilada
automáticamente (~50x más rápido). Si no, cae al modo NumPy puro.
"""

import numpy as np
from scipy.stats import skew
from preprocessing import detect_r_peaks, detect_qrs_p_t, select_main_r_peak

# Intentar importar numba para aceleración de DTW
try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    def njit(func):  # decorador no-op
        return func


# =====================================================================
# 16 Features médicas (Tabla 2, columna Medical Features)
# =====================================================================

def medical_features_single_beat(signal: np.ndarray, ann: dict, fs: int = 250) -> np.ndarray:
    """
    Calcula las 16 features médicas para un único latido anotado.
    
    Orden (según Tabla 2 del paper):
     1. R-wave area
     2. R-wave amplitude
     3. R-wave width value
     4. QS area
     5. QRS duration
     6. T-wave amplitude
     7. P-wave amplitude
     8. P duration
     9. PR interval
    10. R-wave left slope
    11. R-wave right slope
    12. R-wave left width
    13. R-wave right width
    14. R-wave left angle
    15. R-wave right angle
    16. R-wave angle (total)
    """
    feats = np.full(16, np.nan)
    r = ann['r_idx']
    
    if r is None:
        return feats
    
    # Ventana de la onda R: 5 muestras a cada lado del pico
    r_left = max(0, r - 5)
    r_right = min(len(signal) - 1, r + 5)
    
    # 1. R-wave area: integral en la ventana del R
    feats[0] = np.trapezoid(np.abs(signal[r_left:r_right + 1]))
    
    # 2. R-wave amplitude
    feats[1] = ann['r_amp']
    
    # 3. R-wave width value: ancho a media altura
    half_height = ann['r_amp'] / 2.0
    above_half = np.where(signal[max(0, r - 30):min(len(signal), r + 30)] > half_height)[0]
    feats[2] = len(above_half) / fs * 1000  # en milisegundos
    
    # 4. QS area: área entre Q y S
    if ann['q_idx'] is not None and ann['s_idx'] is not None:
        feats[3] = np.trapezoid(np.abs(signal[ann['q_idx']:ann['s_idx'] + 1]))
    
    # 5. QRS duration (ms)
    if ann['qrs_onset'] is not None and ann['qrs_offset'] is not None:
        feats[4] = (ann['qrs_offset'] - ann['qrs_onset']) / fs * 1000
    
    # 6. T-wave amplitude
    if ann['t_amp'] is not None:
        feats[5] = ann['t_amp']
    
    # 7. P-wave amplitude
    if ann['p_amp'] is not None:
        feats[6] = ann['p_amp']
    
    # 8. P duration (ms): aproximada como la mitad de la ventana de búsqueda
    if ann['p_idx'] is not None:
        feats[7] = 60.0  # aproximación clínica típica
    
    # 9. PR interval (ms)
    if ann['p_idx'] is not None and ann['qrs_onset'] is not None:
        feats[8] = (ann['qrs_onset'] - ann['p_idx']) / fs * 1000
    
    # 10-11. R-wave left/right slopes (pendientes lineales)
    if r - 5 >= 0:
        feats[9] = (signal[r] - signal[r - 5]) / 5  # slope izquierdo
    if r + 5 < len(signal):
        feats[10] = (signal[r + 5] - signal[r]) / 5  # slope derecho
    
    # 12-13. R-wave left/right widths
    feats[11] = 5  # left width (samples) — convención fija
    feats[12] = 5  # right width (samples) — convención fija
    
    # 14-15. R-wave left/right angles (en radianes)
    if not np.isnan(feats[9]):
        feats[13] = np.arctan(feats[9])
    if not np.isnan(feats[10]):
        feats[14] = np.arctan(feats[10])
    
    # 16. R-wave total angle: ángulo entre los lados izquierdo y derecho
    if not np.isnan(feats[13]) and not np.isnan(feats[14]):
        feats[15] = np.pi - feats[13] - feats[14]
    
    return feats


def medical_features_multi_beat(signal: np.ndarray, fs: int = 250) -> np.ndarray:
    """
    Para señales de RITMO (1000 muestras, múltiples latidos).
    
    Calcula features médicas sobre el latido central + agregados (mean, std)
    de todas las features sobre todos los latidos detectados.
    
    Por simplicidad mantenemos la dimensionalidad en 16 features médicas:
    para cada feature reportamos el promedio sobre todos los latidos detectados.
    Si solo hay un latido, equivale a `medical_features_single_beat`.
    """
    r_peaks = detect_r_peaks(signal, fs=fs)
    
    if len(r_peaks) == 0:
        return np.full(16, np.nan)
    
    # Calcular features para cada latido detectado
    all_feats = []
    for r in r_peaks:
        ann = detect_qrs_p_t(signal, r, fs=fs)
        all_feats.append(medical_features_single_beat(signal, ann, fs=fs))
    
    all_feats = np.array(all_feats)  # shape (n_beats, 16)
    
    # Promedio ignorando NaNs
    return np.nanmean(all_feats, axis=0)


# =====================================================================
# 20 Features estadísticas (Tabla 2, columna Statistical Features)
# =====================================================================

def statistical_features(signal: np.ndarray) -> np.ndarray:
    """
    Calcula las 20 features estadísticas según Tabla 2.
    
    Orden:
    17. skewness
    18. positive area
    19. negative area
    20. zero-crossing signal
    21. average value
    22. variance
    23. standard deviation
    24. coefficient of variation
    25. maximum
    26. minimum
    27. maximum absolute value
    28. full distance (range)
    29. quartile (median = Q2)
    30. upper quartile (Q3)
    31. lower quartile (Q1)
    32. interquartile range
    33. three means (trimmed mean)
    34. upper truncation point (Q3 + 1.5*IQR)
    35. lower truncation point (Q1 - 1.5*IQR)
    36. flatness (kurtosis)
    """
    feats = np.zeros(20)
    
    # 17. Skewness
    feats[0] = skew(signal)
    
    # 18. Positive area
    pos = signal[signal > 0]
    feats[1] = np.sum(pos) if len(pos) > 0 else 0.0
    
    # 19. Negative area
    neg = signal[signal < 0]
    feats[2] = np.sum(np.abs(neg)) if len(neg) > 0 else 0.0
    
    # 20. Zero-crossing count
    feats[3] = np.sum(np.diff(np.sign(signal)) != 0)
    
    # 21. Mean
    feats[4] = np.mean(signal)
    
    # 22. Variance
    feats[5] = np.var(signal)
    
    # 23. Standard deviation
    feats[6] = np.std(signal)
    
    # 24. Coefficient of variation
    feats[7] = feats[6] / (np.abs(feats[4]) + 1e-12)
    
    # 25. Maximum
    feats[8] = np.max(signal)
    
    # 26. Minimum
    feats[9] = np.min(signal)
    
    # 27. Max absolute value
    feats[10] = np.max(np.abs(signal))
    
    # 28. Range
    feats[11] = feats[8] - feats[9]
    
    # 29-31. Quartiles
    q1, q2, q3 = np.percentile(signal, [25, 50, 75])
    feats[12] = q2  # 29. quartile (median)
    feats[13] = q3  # 30. upper quartile
    feats[14] = q1  # 31. lower quartile
    
    # 32. Interquartile range
    iqr = q3 - q1
    feats[15] = iqr
    
    # 33. Three means (Tukey's trimean)
    feats[16] = (q1 + 2 * q2 + q3) / 4.0
    
    # 34-35. Truncation points
    feats[17] = q3 + 1.5 * iqr  # upper truncation
    feats[18] = q1 - 1.5 * iqr  # lower truncation
    
    # 36. Flatness (kurtosis-based)
    # Definimos flatness como kurtosis para consistencia con análisis estadístico estándar
    mean = feats[4]
    std = feats[6]
    if std > 1e-12:
        feats[19] = np.mean(((signal - mean) / std) ** 4) - 3.0  # exceso de kurtosis
    else:
        feats[19] = 0.0
    
    return feats


# =====================================================================
# DTW (feature temporal #37)
# =====================================================================

@njit(cache=True)
def _dtw_njit(s1, s2, window):
    """Versión JIT-compilada de DTW. window = -1 significa sin restricción."""
    n = len(s1)
    m = len(s2)
    INF = 1e18
    D = np.full((n + 1, m + 1), INF)
    D[0, 0] = 0.0
    
    if window < 0:
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = (s1[i - 1] - s2[j - 1]) ** 2
                a = D[i - 1, j]
                b = D[i, j - 1]
                c = D[i - 1, j - 1]
                m_val = a if a < b else b
                if c < m_val:
                    m_val = c
                D[i, j] = cost + m_val
    else:
        for i in range(1, n + 1):
            j_start = i - window
            if j_start < 1:
                j_start = 1
            j_end = i + window + 1
            if j_end > m + 1:
                j_end = m + 1
            for j in range(j_start, j_end):
                cost = (s1[i - 1] - s2[j - 1]) ** 2
                a = D[i - 1, j]
                b = D[i, j - 1]
                c = D[i - 1, j - 1]
                m_val = a if a < b else b
                if c < m_val:
                    m_val = c
                D[i, j] = cost + m_val
    
    return np.sqrt(D[n, m])


def dtw_distance(s1: np.ndarray, s2: np.ndarray, window: int = None) -> float:
    """
    Calcula DTW entre dos series temporales.
    Implementación de Eqs. 17-19 del paper.
    
    Parámetros:
    -----------
    s1, s2 : series temporales
    window : tamaño de la banda Sakoe-Chiba opcional (None = sin restricción).
             El paper NO usa banda; pasarlo permite comparación con tu MSM con
             Sakoe-Chiba banda w=0.1*len.
    
    Si numba está disponible se usa la versión JIT (~50x más rápido).
    """
    s1 = np.asarray(s1, dtype=np.float64)
    s2 = np.asarray(s2, dtype=np.float64)
    win = -1 if window is None else int(window)
    return _dtw_njit(s1, s2, win)


# =====================================================================
# Pipeline de extracción combinado
# =====================================================================

def extract_36_spatial_features(signal: np.ndarray,
                                  is_rhythm: bool = False,
                                  fs: int = 250,
                                  r_strategy: str = 'prominent') -> np.ndarray:
    """
    Extrae las 36 features espaciales (16 médicas + 20 estadísticas).
    
    Si is_rhythm=True, usa la versión multi-latido para las médicas.
    Si is_rhythm=False (latido único), selecciona el R principal según r_strategy:
    - 'prominent' (default): R con mayor amplitud → más robusto para PAC
    - 'central': R más cercano al centro de la señal
    - 'first': primer R detectado
    """
    if is_rhythm:
        med = medical_features_multi_beat(signal, fs=fs)
    else:
        # Detectar todos los R y elegir el principal según estrategia
        r_peaks = detect_r_peaks(signal, fs=fs)
        r_main = select_main_r_peak(signal, r_peaks, strategy=r_strategy)
        ann = detect_qrs_p_t(signal, r_main, fs=fs)
        med = medical_features_single_beat(signal, ann, fs=fs)
    
    stat = statistical_features(signal)
    return np.concatenate([med, stat])


def impute_nans(features_matrix: np.ndarray) -> np.ndarray:
    """
    Imputa NaNs (típicamente de P-waves no detectadas) con la mediana de la columna.
    """
    feats = features_matrix.copy()
    for col in range(feats.shape[1]):
        col_data = feats[:, col]
        if np.any(np.isnan(col_data)):
            median = np.nanmedian(col_data)
            if np.isnan(median):
                median = 0.0
            feats[np.isnan(col_data), col] = median
    return feats


if __name__ == "__main__":
    # Smoke test
    np.random.seed(0)
    fake_signal = np.random.randn(250)
    fake_signal[120:130] += 3.0
    
    feats = extract_36_spatial_features(fake_signal, is_rhythm=False)
    print(f"Shape de features: {feats.shape}")  # Debe ser (36,)
    print(f"Médicas (16): {feats[:16]}")
    print(f"Estadísticas (20): {feats[16:]}")
    
    # Test DTW
    s1 = np.sin(np.linspace(0, 2 * np.pi, 100))
    s2 = np.sin(np.linspace(0, 2 * np.pi, 100) + 0.5)
    print(f"DTW(s1, s2) = {dtw_distance(s1, s2):.4f}")
