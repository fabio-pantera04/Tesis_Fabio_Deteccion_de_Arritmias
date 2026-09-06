"""
modulo_1_clasificador/lectura_senal.py

Responsabilidad única: leer y normalizar señales ECG.
Soporta CSV y WFDB (.dat + .hea de PhysioNet / Icentia11k).
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import resample

#Función de remuestreo
def remuestrear_senal(senal: np.ndarray, 
                       fs_origen: int, 
                       fs_destino: int) -> np.ndarray:
    """
    Remuestrea una señal ECG de fs_origen Hz a fs_destino Hz.
    Necesario para homogeneizar MIT-BIH (360 Hz) con Icentia11k (250 Hz).
    """
    if fs_origen == fs_destino:
        return senal
    n_muestras_destino = int(len(senal) * fs_destino / fs_origen)
    return resample(senal, n_muestras_destino)


def leer_senal_csv(path: str) -> np.ndarray:
    """Lee una señal ECG desde un CSV de una sola columna."""
    return pd.read_csv(path, header=None).values.flatten().astype(float)


def leer_senal_wfdb(path: str, canal: int = 0) -> np.ndarray:
    """
    Lee una señal ECG desde archivo WFDB (.dat + .hea).

    Parámetros
    ----------
    path  : ruta sin extensión  (ej. 'C:/data/paciente_001')
    canal : índice de derivación (0 = Lead I para Icentia11k)

    Requiere: pip install wfdb
    """
    try:
        import wfdb
    except ImportError:
        raise ImportError(
            "Instala wfdb con:  pip install wfdb\n"
            "Necesario para leer archivos .dat/.hea de PhysioNet."
        )
    record = wfdb.rdrecord(str(path))
    return record.p_signal[:, canal].astype(float)


def leer_senal(path: str, canal: int = 0) -> np.ndarray:
    """
    Detecta el formato por extensión y lee la señal ECG.

    Formatos soportados
    -------------------
    .csv            → leer_senal_csv()
    .dat / .hea     → leer_senal_wfdb() (ruta base sin extensión)
    sin extensión   → leer_senal_wfdb() (asume WFDB)

    Retorna
    -------
    np.ndarray de float64 con los valores de voltaje crudos.
    """
    ext = Path(path).suffix.lower()
    if ext == '.csv':
        return leer_senal_csv(path)
    elif ext in ('.dat', '.hea'):
        base = str(Path(path).with_suffix(''))
        return leer_senal_wfdb(base, canal)
    elif ext == '':
        return leer_senal_wfdb(path, canal)
    else:
        raise ValueError(
            f"Formato '{ext}' no soportado. Use .csv o .dat/.hea (WFDB)."
        )


def normalizar_zscore(senal: np.ndarray) -> np.ndarray:
    """
    Normalización Z-score por señal individual.

    Garantiza que MSM-SC mida similitud morfológica (forma),
    no diferencias de amplitud entre pacientes o equipos.

    zᵢ = (xᵢ − μ) / σ  →  media = 0, desviación estándar = 1

    Si la señal es constante (σ = 0) retorna un array de ceros.
    Los valores no finitos residuales se reemplazan por 0.
    """
    s = senal.std()
    if s == 0 or not np.isfinite(s):
        return np.zeros_like(senal, dtype=float)
    z = (senal - senal.mean()) / s
    return np.where(np.isfinite(z), z, 0.0)


def cargar_y_normalizar(path: str, canal: int = 0) -> np.ndarray:
    """
    Pipeline completo de lectura: leer → normalizar.
    Función de conveniencia para uso en inferencia en tiempo real.

    Retorna
    -------
    np.ndarray normalizado listo para calcular distancias MSM-SC.
    """
    senal_cruda = leer_senal(path, canal)
    return normalizar_zscore(senal_cruda)


def cargar_dataset_balanceado(rutas_bases: list,n_por_clase: int = 100) -> tuple:
    """
    Carga un dataset balanceado por clase desde carpetas CSV.
    Acepta carpetas NRS y NSR en disco como sinónimos de NSR.

    Remuestreo automático:
        MIT-BIH (360 Hz) → 250 Hz si la ruta contiene 'MIT'
        Icentia11k (250 Hz) → sin cambio
    """
    from config import FS_MITBIH, FS_OBJETIVO

    X, y = [], []
    mapa_etiquetas = {
        'NORMAL': 'NORMAL',
        'PAC'   : 'PAC',
        'NRS'   : 'NSR',
        'NSR'   : 'NSR',
        'AFIB'  : 'AFIB',
    }
    ya_cargadas = set()

    for carpeta_disco, etiqueta in mapa_etiquetas.items():
        if etiqueta in ya_cargadas:
            continue
        archivos = []
        for base in rutas_bases:
            if not base:
                continue
            p = Path(base) / carpeta_disco
            if p.exists():
                archivos += list(p.glob('*.csv'))

        import random
        random.shuffle(archivos)
        cargados = 0

        for archivo in archivos[:n_por_clase]:
            try:
                raw = leer_senal_csv(str(archivo))

                # ── Remuestreo automático ─────────────────────────
                # Si la ruta contiene MIT → señal a 360 Hz → remuestrear
                ruta_str = str(archivo).upper()
                es_mitbih = any(k in ruta_str for k in ["MIT BIH", "MIT_BIH", "MITBIH", "MIT-BIH"])
                if es_mitbih:
                    raw = remuestrear_senal(
                        raw,
                        fs_origen  = FS_MITBIH,
                        fs_destino = FS_OBJETIVO
                    )

                senal = normalizar_zscore(raw)
                X.append(senal)
                y.append(etiqueta)
                cargados += 1

            except Exception:
                continue

        if cargados > 0:
            ya_cargadas.add(etiqueta)
        print(f"  [{carpeta_disco:6s} -> {etiqueta:6s}] "
              f"{cargados:4d} señales "
              f"({'MIT-BIH+Icentia' if any('MIT' in str(a).upper() for a in archivos[:5]) else 'Icentia'})")

    return np.array(X, dtype=object), np.array(y)
