"""
main.py
=======
Orquestador del pipeline XGBoost-Andotra para una tarea (LATIDO o RITMO).

Pipeline completo:
1. Carga señales desde CSVs (por prefijo desde múltiples directorios)
2. Detección de R-peaks y fiducials P, Q, S, T con NeuroKit2 + fallback
3. Extracción de 17 features (3 RR + 4 intervals + 5 amplitudes + 5 Hermite)
4. Split estratificado 70/30
5. Imputación de NaNs con mediana del train
6. Entrenamiento XGBoost del paper (max_depth=3, n_est=1000, α=0.01, λ=1.0, γ=0.1)
7. Feature selection con SelectFromModel + re-entrenamiento
8. Stratified 5-fold cross-validation sobre el train (paper Sec III-D)
9. Evaluación en test con métricas robustas (accuracy, kappa, MCC, F1, sensitivity)

Uso:
    python main.py --config config_latido.yaml
    python main.py --config config_ritmo.yaml
"""

import argparse
import time
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from sklearn.model_selection import train_test_split

from preprocessing import load_signal_from_csv
from features import extract_17_features, impute_nans_train_test, FEATURE_NAMES
from xgboost_model import (
    train_with_feature_selection,
    stratified_cross_validation,
    compute_full_metrics,
    print_metrics_report,
)


# =====================================================================
# Configuración de tareas
# =====================================================================

TASK_DEFAULTS = {
    'latido': {
        'is_rhythm': False,
        'signal_length': 250,
        'fs': 250,
    },
    'ritmo': {
        'is_rhythm': True,
        'signal_length': 1000,
        'fs': 250,
    },
}


# =====================================================================
# Carga de datos por prefijo (consistente con SFP-FACC)
# =====================================================================

def load_by_prefix(directory: Path, prefix: str,
                    expected_length: int = None) -> tuple:
    """Carga todos los CSVs en `directory` cuyos nombres comienzan con `prefix`."""
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


def load_class_from_sources(class_name: str, sources: list,
                              expected_length: int,
                              max_per_class: int = None,
                              seed: int = 42) -> tuple:
    """Carga y combina señales de múltiples directorios para una clase."""
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
# Extracción de features para batch
# =====================================================================

def extract_features_batch(signals, fs, is_rhythm, verbose=True):
    """Extrae las 17 features para un batch de señales."""
    feats = []
    n_total = len(signals)
    
    for i, sig in enumerate(signals):
        f = extract_17_features(sig, fs=fs, is_rhythm=is_rhythm)
        feats.append(f)
        if verbose and (i + 1) % 200 == 0:
            print(f"    {i + 1}/{n_total} señales procesadas")
    
    return np.array(feats)


# =====================================================================
# Pipeline principal
# =====================================================================

def run_xgboost_andotra(task, class_sources, seed=42, test_size=0.30,
                          max_per_class=1000, run_cv=True, verbose=True):
    """Ejecuta el pipeline XGBoost-Andotra completo."""
    if task not in TASK_DEFAULTS:
        raise ValueError(f"Tarea desconocida: {task}. Use 'latido' o 'ritmo'.")
    
    cfg = TASK_DEFAULTS[task]
    classes = list(class_sources.keys())
    
    print("\n" + "#" * 60)
    print(f"# XGBoost-Andotra - Tarea: {task.upper()}")
    print(f"# Clases: {classes}")
    print(f"# Longitud de señal: {cfg['signal_length']}")
    print("#" * 60)
    
    t_total_start = time.time()
    
    # ========== 1. Carga ==========
    print("\n[1/6] Carga de señales por prefijo")
    
    per_class_signals = {}
    for cls_name, sources in class_sources.items():
        sigs, _ = load_class_from_sources(
            cls_name, sources, cfg['signal_length'],
            max_per_class=max_per_class, seed=seed,
        )
        per_class_signals[cls_name] = sigs
    
    all_signals = np.vstack([per_class_signals[c] for c in classes])
    all_labels_str = []
    for c in classes:
        all_labels_str.extend([c] * len(per_class_signals[c]))
    
    # Encoder de etiquetas (preservando orden de `classes`)
    label_to_idx = {c: i for i, c in enumerate(classes)}
    all_labels = np.array([label_to_idx[l] for l in all_labels_str])
    
    print(f"\n  Total: {len(all_signals)} señales en {len(classes)} clases")
    
    # ========== 2. Split ==========
    print("\n[2/6] Split estratificado 70/30")
    indices = np.arange(len(all_labels))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, stratify=all_labels, random_state=seed,
    )
    
    train_signals = all_signals[train_idx]
    train_labels = all_labels[train_idx]
    test_signals = all_signals[test_idx]
    test_labels = all_labels[test_idx]
    
    print(f"  Train: {len(train_signals)} | Test: {len(test_signals)}")
    print(f"  Train por clase: {dict(Counter([classes[l] for l in train_labels]))}")
    print(f"  Test por clase:  {dict(Counter([classes[l] for l in test_labels]))}")
    
    # ========== 3. Extracción de features ==========
    print("\n[3/6] Extracción de 17 features por señal")
    print("  Features de train...")
    t_feat_start = time.time()
    X_train_raw = extract_features_batch(
        train_signals, fs=cfg['fs'], is_rhythm=cfg['is_rhythm']
    )
    print("  Features de test...")
    X_test_raw = extract_features_batch(
        test_signals, fs=cfg['fs'], is_rhythm=cfg['is_rhythm']
    )
    t_feat = time.time() - t_feat_start
    
    print(f"  Train features shape: {X_train_raw.shape}")
    print(f"  Test features shape:  {X_test_raw.shape}")
    print(f"  NaNs en train: {np.sum(np.isnan(X_train_raw))} "
          f"({100*np.mean(np.isnan(X_train_raw)):.1f}% de las celdas)")
    print(f"  NaNs en test:  {np.sum(np.isnan(X_test_raw))} "
          f"({100*np.mean(np.isnan(X_test_raw)):.1f}% de las celdas)")
    print(f"  Tiempo extracción: {t_feat:.2f}s")
    
    # ========== 4. Imputación de NaNs ==========
    print("\n[4/6] Imputación de NaNs (mediana del train)")
    X_train, X_test, medians = impute_nans_train_test(X_train_raw, X_test_raw)
    print(f"  Train post-imputación: {np.sum(np.isnan(X_train))} NaNs (esperado: 0)")
    print(f"  Test post-imputación:  {np.sum(np.isnan(X_test))} NaNs (esperado: 0)")
    
    # ========== 5. Cross-validation (paper Sec III-D) ==========
    if run_cv:
        print("\n[5/6] Stratified 5-fold cross-validation sobre train")
        t_cv_start = time.time()
        cv_results = stratified_cross_validation(
            X_train, train_labels, n_splits=5, verbose=verbose, seed=seed,
        )
        t_cv = time.time() - t_cv_start
        print(f"\n  CV Accuracy:   {cv_results['accuracy_mean']:.4f} ± {cv_results['accuracy_std']:.4f}")
        print(f"  CV Kappa:      {cv_results['kappa_mean']:.4f} ± {cv_results['kappa_std']:.4f}")
        print(f"  CV MCC:        {cv_results['mcc_mean']:.4f} ± {cv_results['mcc_std']:.4f}")
        print(f"  CV F1 macro:   {cv_results['f1_macro_mean']:.4f} ± {cv_results['f1_macro_std']:.4f}")
        print(f"  Tiempo CV: {t_cv:.2f}s")
    else:
        cv_results = None
        t_cv = 0
    
    # ========== 6. Entrenamiento final + feature selection ==========
    print("\n[6/6] Entrenamiento final con feature selection")
    t_train_start = time.time()
    result = train_with_feature_selection(
        X_train, train_labels, X_test, test_labels,
        feature_names=FEATURE_NAMES, verbose=verbose,
    )
    t_train = time.time() - t_train_start
    print(f"  Tiempo entrenamiento: {t_train:.2f}s")
    
    # ========== Evaluación final en test ==========
    test_metrics = compute_full_metrics(
        test_labels, result['predictions'], class_names=classes
    )
    print_metrics_report(test_metrics, class_names=classes)
    
    # ========== Resumen de tiempos ==========
    t_total = time.time() - t_total_start
    print("\n" + "=" * 60)
    print("TIEMPOS")
    print("=" * 60)
    print(f"Extracción features: {t_feat:.2f}s")
    if run_cv:
        print(f"5-fold CV:           {t_cv:.2f}s")
    print(f"Entrenamiento final: {t_train:.2f}s")
    print(f"TOTAL:               {t_total:.2f}s")
    print(f"Observations/sec:    {len(test_signals) / t_total:.1f}")
    print("=" * 60)
    
    return {
        'test_metrics': test_metrics,
        'cv_results': cv_results,
        'selected_features': result['selected_features'],
        'predictions': result['predictions'],
        'model': result['model_selected'],
    }


# =====================================================================
# CLI / YAML
# =====================================================================

def parse_class_arg(class_arg: str) -> tuple:
    """Parsea 'NAME:DIR:PREFIX' manejando rutas Windows con D:/."""
    first_colon = class_arg.find(':')
    last_colon = class_arg.rfind(':')
    if first_colon == -1 or first_colon == last_colon:
        raise ValueError(f"Formato inválido: '{class_arg}'. Esperado: 'NAME:DIR:PREFIX'")
    return (class_arg[:first_colon],
            class_arg[first_colon+1:last_colon],
            class_arg[last_colon+1:])


def load_yaml_config(path: Path) -> dict:
    """Carga configuración YAML."""
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML no instalado: pip install pyyaml")
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="XGBoost-Andotra pipeline (réplica del paper Andotra & Sunkaria 2024)",
    )
    parser.add_argument('--task', choices=['latido', 'ritmo'],
                        help='Tipo de clasificación')
    parser.add_argument('--class', dest='classes', action='append', default=[],
                        metavar='NAME:DIR:PREFIX',
                        help='Fuente de una clase (puede repetirse)')
    parser.add_argument('--config', type=Path,
                        help='Archivo YAML (alternativa a --class)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test-size', type=float, default=0.30)
    parser.add_argument('--max-per-class', type=int, default=1000)
    parser.add_argument('--no-cv', action='store_true',
                        help='Saltar 5-fold CV (más rápido para pruebas)')
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
        
        run_xgboost_andotra(
            task=task,
            class_sources=class_sources,
            seed=cfg.get('seed', args.seed),
            test_size=cfg.get('test_size', args.test_size),
            max_per_class=cfg.get('max_per_class', args.max_per_class),
            run_cv=not args.no_cv,
            verbose=not args.quiet,
        )
    else:
        if not args.task or not args.classes:
            raise ValueError("Provee --task y --class, o usa --config")
        
        class_sources = defaultdict(list)
        for class_arg in args.classes:
            name, directory, prefix = parse_class_arg(class_arg)
            class_sources[name].append((directory, prefix))
        
        run_xgboost_andotra(
            task=args.task,
            class_sources=dict(class_sources),
            seed=args.seed,
            test_size=args.test_size,
            max_per_class=args.max_per_class,
            run_cv=not args.no_cv,
            verbose=not args.quiet,
        )


if __name__ == "__main__":
    main()
