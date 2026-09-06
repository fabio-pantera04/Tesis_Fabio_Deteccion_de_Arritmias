"""
main.py
=======
Orquestador del pipeline completo SFP-FACC para una tarea (LATIDO o RITMO).

El código carga señales por PREFIJO DEL NOMBRE DE ARCHIVO desde tus carpetas
actuales sin necesidad de reorganizar nada. Para cada clase puedes apuntar a
múltiples directorios fuente (útil para AFIB que combina Icentia + CinC2017).

Ejemplo de uso desde CLI:
-------------------------

# LATIDO (N vs PAC), todo desde Icentia11k:
python main.py --task latido \
    --class N:D:/Modelo_Tesis/Icentia11k/NORMAL:ICI_NORMAL \
    --class PAC:D:/Modelo_Tesis/Icentia11k/PAC:ICI_PAC

# RITMO (NSR vs AFIB), AFIB combinando Icentia + CinC:
python main.py --task ritmo \
    --class NSR:D:/Modelo_Tesis/Icentia11k/NSR:ICI_NSR \
    --class AFIB:D:/Modelo_Tesis/Icentia11k/AFIB:ICI_AFIB \
    --class AFIB:D:/Modelo_Tesis/CinC2017/AFIB:CIC_AFIB

Formato de cada --class: NOMBRE_CLASE:DIRECTORIO:PREFIJO_ARCHIVO
- Múltiples --class con el mismo NOMBRE_CLASE se concatenan automáticamente.

Alternativa con archivo de configuración (más limpio para tu caso):
-----------------------------------------------------------------
python main.py --config config_ritmo.yaml

Donde config_ritmo.yaml es:

    task: ritmo
    classes:
      NSR:
        - dir: D:/Modelo_Tesis/Icentia11k/NSR
          prefix: ICI_NSR
      AFIB:
        - dir: D:/Modelo_Tesis/Icentia11k/AFIB
          prefix: ICI_AFIB
        - dir: D:/Modelo_Tesis/CinC2017/AFIB
          prefix: CIC_AFIB
"""

import argparse
import time
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, matthews_corrcoef,
    f1_score, confusion_matrix, classification_report
)

from preprocessing import load_signal_from_csv, wavelet_denoise
from features import extract_36_spatial_features, impute_nans
# Imports lazy de lstm_centers (requiere torch) y aco (rápido) — se importan en run_sfp_facc


# =====================================================================
# Configuración de tareas
# =====================================================================

TASK_DEFAULTS = {
    'latido': {
        'is_rhythm': False,
        'signal_length': 250,
        'fs': 250,
        'r_strategy': 'prominent',  # PAC suele tener 2 R-peaks; tomar el más prominente
    },
    'ritmo': {
        'is_rhythm': True,
        'signal_length': 1000,
        'fs': 250,
        'r_strategy': 'prominent',
    },
}


# =====================================================================
# Carga por prefijo desde múltiples directorios
# =====================================================================

def load_by_prefix(directory: Path, prefix: str,
                    expected_length: int = None) -> tuple:
    """
    Carga todos los CSVs en `directory` cuyos nombres comienzan con `prefix`.
    
    Devuelve:
    - signals: array (n, signal_length)
    - filenames: lista de nombres
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directorio no encontrado: {directory}")
    
    csv_files = sorted(directory.glob(f"{prefix}*.csv"))
    if not csv_files:
        raise ValueError(f"No se encontraron archivos '{prefix}*.csv' en {directory}")
    
    signals = []
    filenames = []
    skipped = 0
    
    for csv_file in csv_files:
        try:
            sig = load_signal_from_csv(csv_file)
            if expected_length is not None:
                if len(sig) > expected_length:
                    sig = sig[:expected_length]
                elif len(sig) < expected_length:
                    skipped += 1
                    continue
            signals.append(sig)
            filenames.append(csv_file.name)
        except Exception as e:
            print(f"  ⚠ Error cargando {csv_file.name}: {e}")
            skipped += 1
    
    if skipped > 0:
        print(f"  ⚠ {skipped} archivos omitidos en {directory}")
    
    return np.array(signals), filenames


def load_class_from_sources(class_name: str,
                              sources: list,
                              expected_length: int,
                              max_per_class: int = None,
                              seed: int = 42) -> tuple:
    """
    Carga señales de una clase desde múltiples directorios fuente
    y las concatena.
    """
    all_signals = []
    all_metadata = []
    
    for directory, prefix in sources:
        print(f"  [{class_name}] Cargando '{prefix}*.csv' desde {directory}")
        sigs, fnames = load_by_prefix(Path(directory), prefix, expected_length)
        print(f"    → {len(sigs)} señales cargadas")
        all_signals.append(sigs)
        all_metadata.extend([f"{prefix}:{f}" for f in fnames])
    
    combined = np.vstack(all_signals)
    print(f"  [{class_name}] Total combinado: {len(combined)} señales")
    
    if max_per_class is not None and len(combined) > max_per_class:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(combined), size=max_per_class, replace=False)
        combined = combined[idx]
        all_metadata = [all_metadata[i] for i in idx]
        print(f"  [{class_name}] Submuestreado a {max_per_class} señales")
    
    return combined, all_metadata


# =====================================================================
# Pipeline principal
# =====================================================================

def stratified_split(signals, labels, test_size=0.30, seed=42):
    """Split estratificado 70/30."""
    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, stratify=labels, random_state=seed,
    )
    train_signals = signals[train_idx]
    train_labels = [labels[i] for i in train_idx]
    test_signals = signals[test_idx]
    test_labels = [labels[i] for i in test_idx]
    return train_signals, train_labels, test_signals, test_labels


def extract_features_batch(signals, is_rhythm, fs, r_strategy='prominent', verbose=True):
    """Extrae las 36 features espaciales para un batch de señales."""
    feats = []
    for i, sig in enumerate(signals):
        f = extract_36_spatial_features(sig, is_rhythm=is_rhythm, fs=fs,
                                         r_strategy=r_strategy)
        feats.append(f)
        if verbose and (i + 1) % 200 == 0:
            print(f"  Features extraídas para {i + 1}/{len(signals)}")
    return np.array(feats)


def report_metrics(y_true, y_pred, classes):
    """Reporta métricas completas (kappa, MCC además de accuracy del paper)."""
    print("\n" + "=" * 60)
    print("RESULTADOS DE CLASIFICACIÓN")
    print("=" * 60)
    print(f"Accuracy:      {accuracy_score(y_true, y_pred):.4f}")
    print(f"Cohen's Kappa: {cohen_kappa_score(y_true, y_pred):.4f}")
    print(f"MCC:           {matthews_corrcoef(y_true, y_pred):.4f}")
    print(f"F1 (macro):    {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(f"F1 (weighted): {f1_score(y_true, y_pred, average='weighted'):.4f}")
    
    print("\nMatriz de confusión:")
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    print(f"           {' '.join(f'{c:>8}' for c in classes)}")
    for i, cls in enumerate(classes):
        print(f"  {cls:>6}: {' '.join(f'{v:>8}' for v in cm[i])}")
    
    print("\nReporte por clase:")
    print(classification_report(y_true, y_pred, labels=classes, digits=4))


def run_sfp_facc(task, class_sources, seed=42, test_size=0.30,
                  fit_n_per_class=10, max_per_class=None, verbose=True):
    """Ejecuta el pipeline SFP-FACC completo."""
    # Imports lazy: torch solo se requiere al ejecutar el pipeline
    from lstm_centers import fit_all_cluster_centers, LSTM_CONFIG
    from aco import run_aco_classification, ACO_CONFIG
    
    if task not in TASK_DEFAULTS:
        raise ValueError(f"Tarea desconocida: {task}. Use 'latido' o 'ritmo'.")
    
    cfg = TASK_DEFAULTS[task]
    classes = list(class_sources.keys())
    
    print("\n" + "#" * 60)
    print(f"# SFP-FACC - Tarea: {task.upper()}")
    print(f"# Clases: {classes}")
    print(f"# Longitud de señal: {cfg['signal_length']}")
    print(f"# R-peak strategy: {cfg['r_strategy']}")
    print("#" * 60)
    
    t_total_start = time.time()
    
    # ========== 1. Carga ==========
    print("\n[1/5] Carga de señales por prefijo + wavelet denoising")
    
    per_class_signals = {}
    for cls_name, sources in class_sources.items():
        sigs, _ = load_class_from_sources(
            cls_name, sources, cfg['signal_length'],
            max_per_class=max_per_class, seed=seed,
        )
        print(f"  [{cls_name}] Aplicando wavelet denoising a {len(sigs)} señales...")
        sigs = np.array([wavelet_denoise(s) for s in sigs])
        per_class_signals[cls_name] = sigs
    
    all_signals = np.vstack([per_class_signals[c] for c in classes])
    all_labels = []
    for c in classes:
        all_labels.extend([c] * len(per_class_signals[c]))
    
    print(f"\n  Total: {len(all_signals)} señales en {len(classes)} clases")
    
    # ========== 2. Split ==========
    print("\n[2/5] Split estratificado 70/30")
    train_signals, train_labels, test_signals, test_labels = stratified_split(
        all_signals, all_labels, test_size=test_size, seed=seed
    )
    print(f"  Train: {len(train_signals)} | Test: {len(test_signals)}")
    print(f"  Train por clase: {dict(Counter(train_labels))}")
    print(f"  Test por clase:  {dict(Counter(test_labels))}")
    
    train_per_class = {
        cls: train_signals[np.array(train_labels) == cls] for cls in classes
    }
    
    # ========== 3. LSTM centers ==========
    print("\n[3/5] Ajuste de centros con LSTM")
    t_lstm_start = time.time()
    centers_signals = fit_all_cluster_centers(
        train_per_class,
        fit_n_per_class=fit_n_per_class,
        config=LSTM_CONFIG,
        verbose=verbose,
        seed=seed,
    )
    t_lstm = time.time() - t_lstm_start
    print(f"  Tiempo LSTM: {t_lstm:.2f}s")
    
    # ========== 4. Features ==========
    print("\n[4/5] Extracción de 36 features espaciales")
    print("  Features de los centros...")
    centers_features = {
        cls: extract_36_spatial_features(centers_signals[cls],
                                           is_rhythm=cfg['is_rhythm'],
                                           fs=cfg['fs'],
                                           r_strategy=cfg['r_strategy'])
        for cls in classes
    }
    
    print("  Features de test...")
    t_feat_start = time.time()
    test_features = extract_features_batch(
        test_signals, is_rhythm=cfg['is_rhythm'], fs=cfg['fs'],
        r_strategy=cfg['r_strategy'],
    )
    test_features = impute_nans(test_features)
    
    centers_features_matrix = np.array([centers_features[c] for c in classes])
    centers_features_matrix = impute_nans(centers_features_matrix)
    centers_features = {
        cls: centers_features_matrix[i] for i, cls in enumerate(classes)
    }
    t_feat = time.time() - t_feat_start
    print(f"  Tiempo features: {t_feat:.2f}s")
    
    # ========== 5. ACO ==========
    print("\n[5/5] ACO con feromona dinámica + radix sort")
    t_aco_start = time.time()
    result = run_aco_classification(
        test_signals, test_features, test_labels,
        centers_signals, centers_features,
        config=ACO_CONFIG, verbose=verbose,
    )
    t_aco = time.time() - t_aco_start
    print(f"  Tiempo ACO: {t_aco:.2f}s")
    
    t_total = time.time() - t_total_start
    
    # ========== Reporte ==========
    report_metrics(test_labels, result['predictions'], classes)
    
    print("\n" + "=" * 60)
    print("TIEMPOS")
    print("=" * 60)
    print(f"LSTM (fit + generate): {t_lstm:.2f}s")
    print(f"Feature extraction:    {t_feat:.2f}s")
    print(f"ACO + clasificación:   {t_aco:.2f}s")
    print(f"TOTAL:                 {t_total:.2f}s")
    print(f"Observations/sec:      {len(test_signals) / t_total:.1f}")
    print("=" * 60)
    
    return result


# =====================================================================
# Parsing de argumentos
# =====================================================================

def parse_class_arg(class_arg: str) -> tuple:
    """
    Parsea un argumento --class del formato 'NAME:DIR:PREFIX'.
    
    Estrategia robusta para rutas Windows con `D:/...`:
    - El primer ':' separa NAME del resto.
    - El último ':' separa el PREFIX final.
    - Lo del medio es el directorio (puede contener ':' de unidad Windows).
    """
    first_colon = class_arg.find(':')
    last_colon = class_arg.rfind(':')
    
    if first_colon == -1 or first_colon == last_colon:
        raise ValueError(
            f"Formato inválido: '{class_arg}'. Esperado: 'NAME:DIR:PREFIX'"
        )
    
    name = class_arg[:first_colon]
    directory = class_arg[first_colon + 1:last_colon]
    prefix = class_arg[last_colon + 1:]
    
    return name, directory, prefix


def load_yaml_config(path: Path) -> dict:
    """Carga configuración desde YAML."""
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML no instalado. Instala con: pip install pyyaml")
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="SFP-FACC pipeline (réplica del paper Qin et al. 2024)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--task', choices=['latido', 'ritmo'],
                        help='Tipo de clasificación')
    parser.add_argument('--class', dest='classes', action='append', default=[],
                        metavar='NAME:DIR:PREFIX',
                        help='Fuente de una clase (puede repetirse). '
                             'Ej: AFIB:D:/data/icentia/AFIB:ICI_AFIB')
    parser.add_argument('--config', type=Path,
                        help='Archivo YAML con la configuración (alternativa a --class)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test-size', type=float, default=0.30)
    parser.add_argument('--fit-n-per-class', type=int, default=10,
                        help='Señales por clase para fitting LSTM (paper: 10)')
    parser.add_argument('--max-per-class', type=int, default=None,
                        help='Máximo de señales por clase (default: sin límite)')
    parser.add_argument('--quiet', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.config:
        cfg = load_yaml_config(args.config)
        task = cfg['task']
        class_sources = {}
        for cls_name, sources in cfg['classes'].items():
            class_sources[cls_name] = [(s['dir'], s['prefix']) for s in sources]
        
        run_sfp_facc(
            task=task,
            class_sources=class_sources,
            seed=cfg.get('seed', args.seed),
            test_size=cfg.get('test_size', args.test_size),
            fit_n_per_class=cfg.get('fit_n_per_class', args.fit_n_per_class),
            max_per_class=cfg.get('max_per_class', args.max_per_class),
            verbose=not args.quiet,
        )
    else:
        if not args.task:
            raise ValueError("--task es requerido cuando no se usa --config")
        if not args.classes:
            raise ValueError("Al menos un --class NAME:DIR:PREFIX es requerido "
                             "(o usa --config)")
        
        class_sources = defaultdict(list)
        for class_arg in args.classes:
            name, directory, prefix = parse_class_arg(class_arg)
            class_sources[name].append((directory, prefix))
        
        run_sfp_facc(
            task=args.task,
            class_sources=dict(class_sources),
            seed=args.seed,
            test_size=args.test_size,
            fit_n_per_class=args.fit_n_per_class,
            max_per_class=args.max_per_class,
            verbose=not args.quiet,
        )


if __name__ == "__main__":
    main()
