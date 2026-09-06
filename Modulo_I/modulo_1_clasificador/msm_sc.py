"""
modulo_1_clasificador/msm_sc.py

Implementación de la distancia elástica MSM con restricción de banda
de Sakoe-Chiba y early abandoning.

Referencias
-----------
MSM estándar:
    Stefan, A., Athitsos, V., & Das, G. (2013).
    "The Move-Split-Merge Metric for Time Series."
    IEEE Transactions on Knowledge and Data Engineering, 25(6), 1425–1438.
    DOI: 10.1109/TKDE.2012.88

Banda de Sakoe-Chiba aplicada a MSM:
    Holznigenkemper, J., Siebert, M., & Mutzel, P. (2023).
    "Exact and Heuristic Approaches to Speeding Up the MSM
     Time Series Distance Computation."
    arXiv: 2301.01977
"""

import numpy as np
from config import MSM_C, MSM_WINDOW


def _costo_msm(a: float, b: float, c_val: float, c_param: float) -> float:
    """
    Función de costo C(a, b, c) de MSM para operaciones Split y Merge.

    Determina el costo de insertar o eliminar un punto según la posición
    relativa de b respecto al intervalo [min(a,c), max(a,c)]:

        c_param   si  min(a, c_val) ≤ b ≤ max(a, c_val)
        c_param   si  b == a  ó  b == c_val
        2·c_param en cualquier otro caso

    La lógica es: si el punto b ya está "dentro del rango" entre a y c,
    la operación es barata (costo c). Si está fuera, es costosa (costo 2c).
    Esto hace que MSM sea más selectivo que DTW: dos señales con valores
    similares tienen menor costo de alineación.
    """
    if (min(a, c_val) <= b <= max(a, c_val)) or b == a or b == c_val:
        return c_param
    return 2.0 * c_param


def msm_sc(P: np.ndarray, Q: np.ndarray,
           c: float = None,
           window: float = None,
           best_so_far: float = np.inf) -> float:
    """
    Distancia MSM con banda de Sakoe-Chiba y early abandoning.

    Recurrencia de programación dinámica
    -------------------------------------
    D[0, 0] = 0
    D[i, 0] = D[i-1, 0] + c          (frontera: solo Split)
    D[0, j] = D[0, j-1] + c          (frontera: solo Merge)

    Para i ≥ 1, j ≥ 1  y  |i-j| ≤ w:

      D[i, j] = min(
          D[i-1, j-1] + |P[i] - Q[j]|,              ← Move
          D[i-1, j  ] + C(P[i-1], P[i], Q[j], c),   ← Merge en P
          D[i,   j-1] + C(Q[j-1], P[i], Q[j], c)    ← Split en Q
      )

    Interpretación de las operaciones
    ----------------------------------
    Move   : cambia el valor P[i] por Q[j]  → costo proporcional a |P[i]-Q[j]|
    Merge  : elimina P[i] fusionándolo con P[i-1]  → costo C(P[i-1], P[i], Q[j])
    Split  : inserta Q[j] entre posiciones consecutivas  → costo C(Q[j-1], P[i], Q[j])

    Parámetros
    ----------
    P, Q         : series temporales a comparar (arrays 1D, Z-score aplicado)
    c            : costo de operación Split/Merge (default: MSM_C de config.py)
    window       : fracción de banda Sakoe-Chiba (default: MSM_WINDOW de config.py)
    best_so_far  : umbral para early abandoning (usado en ventana deslizante)

    Retorna
    -------
    Distancia MSM-SC ≥ 0.
    Valor bajo  → morfologías similares.
    Valor alto  → morfologías distintas.
    np.inf      → early abandoning activado (distancia seguramente > best_so_far).

    Propiedades matemáticas de MSM (a diferencia de DTW)
    -----------------------------------------------------
    • Es una métrica verdadera: cumple identidad, simetría y desigualdad triangular.
    • Es invariante a la elección del origen (a diferencia de ERP).
    • El costo variable de Split/Merge hace que señales morfológicamente similares
      tengan menor distancia que en DTW puro con mismo warping.
    """
    if c      is None: c      = MSM_C
    if window is None: window = MSM_WINDOW

    N, M = len(P), len(Q)
    w = max(1, int(max(N, M) * window))

    D = np.full((N + 1, M + 1), np.inf)
    D[0, 0] = 0.0

    # Frontera izquierda: operaciones de Split acumuladas
    for i in range(1, N + 1):
        D[i, 0] = D[i-1, 0] + c

    # Frontera superior: operaciones de Merge acumuladas
    for j in range(1, M + 1):
        D[0, j] = D[0, j-1] + c

    for i in range(1, N + 1):
        fila_min = np.inf
        j_ini    = max(1, i - w)
        j_fin    = min(M + 1, i + w + 1)

        for j in range(j_ini, j_fin):
            # Move
            costo_move = D[i-1, j-1] + abs(P[i-1] - Q[j-1])

            # Merge en P: eliminar P[i], alinearlo con Q[j]
            p_prev     = P[i-2] if i > 1 else P[i-1]
            costo_merge = D[i-1, j] + _costo_msm(p_prev, P[i-1], Q[j-1], c)

            # Split en Q: insertar Q[j], alinearlo con P[i]
            q_prev     = Q[j-2] if j > 1 else Q[j-1]
            costo_split = D[i, j-1] + _costo_msm(q_prev, P[i-1], Q[j-1], c)

            D[i, j] = min(costo_move, costo_merge, costo_split)

            if D[i, j] < fila_min:
                fila_min = D[i, j]

        # Early abandoning: si la fila mínima ya supera best_so_far,
        # ninguna celda futura puede mejorar ese valor → descarta la subsecuencia
        if fila_min >= best_so_far:
            return np.inf

    return float(D[N, M])


def distancia_minima_ventana(signal: np.ndarray,
                              shapelet: np.ndarray,
                              c: float = None,
                              window: float = None) -> float:
    """
    Distancia MSM-SC mínima entre un shapelet y una señal larga,
    usando ventana deslizante con early abandoning.

    Para cada posición de inicio en la señal, extrae una subsecuencia
    de la misma longitud que el shapelet y calcula msm_sc(). El early
    abandoning descarta automáticamente las posiciones que no pueden
    mejorar la mejor distancia encontrada hasta ese momento.

    Parámetros
    ----------
    signal   : señal completa normalizada (puede ser más larga que el shapelet)
    shapelet : fragmento a buscar en la señal
    c, window: parámetros MSM-SC (default: valores de config.py)

    Retorna
    -------
    Mínima distancia MSM-SC encontrada en toda la señal.
    """
    L = len(shapelet)
    if len(signal) < L:
        return msm_sc(shapelet, signal, c, window)

    mejor = np.inf
    for start in range(len(signal) - L + 1):
        d = msm_sc(shapelet, signal[start: start + L],
                   c=c, window=window, best_so_far=mejor)
        if d < mejor:
            mejor = d
    return float(mejor)
