"""
modulo_1_clasificador/shapelets.py

Minería de shapelets multi-resolución con dos diccionarios independientes:
  MICRO: extraídos de señales cortas (NORMAL/PAC ~162 muestras)
  MACRO: extraídos de señales largas  (NSR/AFIB ~1000 muestras)

Usa MSM-SC como función de distancia en el Contrast Score.

Referencia Contrast Profile:
    Mercer et al. (2022). "Introducing the Contrast Profile: a novel time
    series primitive that allows real world classification."
    Data Mining and Knowledge Discovery, 36, 877–915.
"""

import random
import numpy as np
from joblib import Parallel, delayed

from config import (
    MSM_C, MSM_WINDOW, N_JOBS,
    NUM_SHAPELETS_MICRO, NUM_SHAPELETS_MACRO,
    NUM_CANDIDATES, NUM_COMPARACIONES,
    UMBRAL_LONGITUD, RANGO_MICRO, RANGO_MACRO,
)
from modulo_1_clasificador.msm_sc import msm_sc


# ─────────────────────────────────────────────────────────────────────────────
# CONTRAST SCORE
# ─────────────────────────────────────────────────────────────────────────────

def contrast_score(candidate: np.ndarray, label: str,
                   X_set: np.ndarray, y_set: np.ndarray) -> float:
    """
    Cuantifica el poder discriminativo de un shapelet candidato.

    C_y(S) = mean_dist(X⁻) − mean_dist(X⁺)

    X⁺ : señales de la misma clase que el candidato
    X⁻ : señales de las demás clases

    Score alto → S está cerca de su clase y lejos de las demás → discriminativo.
    Score bajo o negativo → S no distingue bien entre clases → descartar.

    Usa MSM-SC como función de distancia para que el score refleje
    similitud morfológica real, no diferencias de amplitud.
    """
    dist_pos, dist_neg = [], []
    n_sample = min(NUM_COMPARACIONES, len(X_set))
    indices  = random.sample(range(len(X_set)), n_sample)

    for idx in indices:
        d = msm_sc(candidate, X_set[idx], c=MSM_C, window=MSM_WINDOW)
        if y_set[idx] == label:
            dist_pos.append(d)
        else:
            dist_neg.append(d)

    if not dist_pos or not dist_neg:
        return 0.0

    return float(np.nanmean(dist_neg)) - float(np.nanmean(dist_pos))


# ─────────────────────────────────────────────────────────────────────────────
# EVALUADORES DE CANDIDATOS (llamados por joblib en paralelo)
# ─────────────────────────────────────────────────────────────────────────────

def _evaluar_micro(_, X_short: np.ndarray, y_short: np.ndarray) -> dict:
    """
    Genera y evalúa un candidato MICRO desde señales cortas (NORMAL/PAC).
    Longitud: RANGO_MICRO[0] a min(RANGO_MICRO[1], 80% de la señal fuente).
    """
    if len(X_short) == 0:
        return None
    idx    = random.randint(0, len(X_short) - 1)
    signal = X_short[idx]
    label  = y_short[idx]
    L      = len(signal)
    if L < RANGO_MICRO[0]:
        return None

    max_len = max(RANGO_MICRO[0], int(L * 0.80))
    length  = random.randint(RANGO_MICRO[0], min(RANGO_MICRO[1], max_len))
    start   = random.randint(0, max(0, L - length))
    cand    = signal[start: start + length]
    score   = contrast_score(cand, label, X_short, y_short)

    return {'shapelet': cand, 'score': score, 'clase': label,
            'longitud': length, 'tipo': 'micro'}


def _evaluar_macro(_, X_long: np.ndarray, y_long: np.ndarray) -> dict:
    """
    Genera y evalúa un candidato MACRO desde señales largas (NSR/AFIB).
    Longitud: RANGO_MACRO[0] a min(RANGO_MACRO[1], 70% de la señal fuente).
    """
    if len(X_long) == 0:
        return None
    idx    = random.randint(0, len(X_long) - 1)
    signal = X_long[idx]
    label  = y_long[idx]
    L      = len(signal)
    if L < RANGO_MACRO[0]:
        return None

    max_len = max(RANGO_MACRO[0], int(L * 0.70))
    length  = random.randint(RANGO_MACRO[0], min(RANGO_MACRO[1], max_len))
    start   = random.randint(0, max(0, L - length))
    cand    = signal[start: start + length]
    score   = contrast_score(cand, label, X_long, y_long)

    return {'shapelet': cand, 'score': score, 'clase': label,
            'longitud': length, 'tipo': 'macro'}


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACCIÓN PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def extract_shapelets(X_train: np.ndarray,
                      y_train: np.ndarray) -> tuple:
    """
    Extrae dos diccionarios de shapelets independientes usando paralelismo.

    Separación por longitud de señal
    ---------------------------------
    Las señales del dataset tienen dos grupos naturales:
      NORMAL/PAC  (~162 muestras) → diccionario MICRO
      NSR/AFIB    (~1000 muestras) → diccionario MACRO

    Esto evita que shapelets macro (200-700 muestras) intenten compararse
    contra señales de 162 muestras durante el Contrast Score, lo que
    produciría distancias artificialmente altas y un ranking contaminado.

    Retorna
    -------
    micro_dict : lista de arrays (shapelets micro ordenados por score desc.)
    macro_dict : lista de arrays (shapelets macro ordenados por score desc.)
    micro_meta : lista de dicts con metadata de cada shapelet micro
    macro_meta : lista de dicts con metadata de cada shapelet macro
    """
    mask_short = np.array([len(s) < UMBRAL_LONGITUD for s in X_train])
    X_short    = X_train[mask_short];  y_short = y_train[mask_short]
    X_long     = X_train[~mask_short]; y_long  = y_train[~mask_short]

    print(f"\n[SHAPELETS] Señales cortas (micro): {len(X_short)} "
          f"| Largas (macro): {len(X_long)}")
    print(f"[SHAPELETS] Clases micro: {sorted(set(y_short))} "
          f"| Clases macro: {sorted(set(y_long))}")

    n_micro = NUM_CANDIDATES // 2
    n_macro = NUM_CANDIDATES - n_micro

    # ── Diccionario MICRO ────────────────────────────────────────────────────
    print(f"[SHAPELETS] Evaluando {n_micro} candidatos MICRO con MSM-SC...")
    res_micro = Parallel(n_jobs=N_JOBS, backend='loky', verbose=3)(
        delayed(_evaluar_micro)(i, X_short, y_short)
        for i in range(n_micro)
    )
    micro_cands = sorted(
        [r for r in res_micro if r is not None],
        key=lambda x: x['score'], reverse=True
    )
    micro_dict = [c['shapelet'] for c in micro_cands[:NUM_SHAPELETS_MICRO]]
    micro_meta = micro_cands[:NUM_SHAPELETS_MICRO]

    # ── Diccionario MACRO ────────────────────────────────────────────────────
    print(f"[SHAPELETS] Evaluando {n_macro} candidatos MACRO con MSM-SC...")
    res_macro = Parallel(n_jobs=N_JOBS, backend='loky', verbose=3)(
        delayed(_evaluar_macro)(i, X_long, y_long)
        for i in range(n_macro)
    )
    macro_cands = sorted(
        [r for r in res_macro if r is not None],
        key=lambda x: x['score'], reverse=True
    )
    macro_dict = [c['shapelet'] for c in macro_cands[:NUM_SHAPELETS_MACRO]]
    macro_meta = macro_cands[:NUM_SHAPELETS_MACRO]

    print(f"[SHAPELETS] Diccionario MICRO: {len(micro_dict)} shapelets")
    print(f"[SHAPELETS] Diccionario MACRO: {len(macro_dict)} shapelets")
    return micro_dict, macro_dict, micro_meta, macro_meta
