"""
test_pipeline.py
================
Test integrado del pipeline XGBoost-Andotra con datos sintéticos.
"""

import sys
sys.path.insert(0, '/home/claude/xgboost_andotra')
import numpy as np
from features import extract_17_features, impute_nans_train_test, FEATURE_NAMES
from xgboost_model import (
    train_with_feature_selection,
    stratified_cross_validation,
    compute_full_metrics,
    print_metrics_report,
)


def make_synthetic_ecg(n_per_class=50, signal_length=250, seed=42):
    """Genera ECGs sintéticos con dos morfologías diferenciables."""
    rng = np.random.default_rng(seed)
    
    # Clase N: latido normal
    class_n = []
    for _ in range(n_per_class):
        sig = np.zeros(signal_length)
        sig[70:90] = 0.2 * np.exp(-((np.arange(20) - 10) ** 2) / 30)  # P
        sig[120:130] += 1.5 * np.exp(-((np.arange(10) - 5) ** 2) / 5)  # QRS
        sig[150:180] = 0.3 * np.exp(-((np.arange(30) - 15) ** 2) / 100)  # T
        sig += rng.normal(0, 0.05, signal_length)
        class_n.append(sig)
    
    # Clase PAC: P invertida, QRS estrecho
    class_pac = []
    for _ in range(n_per_class):
        sig = np.zeros(signal_length)
        sig[90:105] = -0.15 * np.exp(-((np.arange(15) - 7) ** 2) / 25)  # P invertida
        sig[120:130] += 1.4 * np.exp(-((np.arange(10) - 5) ** 2) / 5)  # QRS
        sig[150:180] = 0.25 * np.exp(-((np.arange(30) - 15) ** 2) / 100)  # T
        sig += rng.normal(0, 0.05, signal_length)
        class_pac.append(sig)
    
    return np.array(class_n), np.array(class_pac)


def main():
    print("=" * 70)
    print("TEST INTEGRADO: XGBoost-Andotra con datos sintéticos")
    print("=" * 70)
    
    # ===== Datos sintéticos =====
    print("\n[1] Generando 100 N + 100 PAC sintéticos")
    class_n, class_pac = make_synthetic_ecg(100)
    
    # Combinar
    all_signals = np.vstack([class_n, class_pac])
    all_labels = np.array([0] * 100 + [1] * 100)  # 0=N, 1=PAC
    
    # Split 70/30
    rng = np.random.default_rng(42)
    perm = rng.permutation(200)
    train_idx = perm[:140]
    test_idx = perm[140:]
    
    print(f"  Train: {len(train_idx)} | Test: {len(test_idx)}")
    
    # ===== Extracción de features =====
    print("\n[2] Extracción de features (17 por señal)")
    import time
    t0 = time.time()
    
    X_train_raw = np.array([
        extract_17_features(all_signals[i], fs=250, is_rhythm=False)
        for i in train_idx
    ])
    X_test_raw = np.array([
        extract_17_features(all_signals[i], fs=250, is_rhythm=False)
        for i in test_idx
    ])
    
    print(f"  Train shape: {X_train_raw.shape}, NaNs: {np.sum(np.isnan(X_train_raw))}")
    print(f"  Test shape: {X_test_raw.shape}, NaNs: {np.sum(np.isnan(X_test_raw))}")
    print(f"  Tiempo: {time.time()-t0:.2f}s")
    
    # ===== Imputación =====
    print("\n[3] Imputación de NaNs (mediana del train)")
    X_train, X_test, medians = impute_nans_train_test(X_train_raw, X_test_raw)
    print(f"  NaNs post-imputación train: {np.sum(np.isnan(X_train))}")
    print(f"  NaNs post-imputación test: {np.sum(np.isnan(X_test))}")
    
    y_train = all_labels[train_idx]
    y_test = all_labels[test_idx]
    
    # ===== Cross-validation =====
    print("\n[4] Stratified 5-fold CV sobre train")
    cv_results = stratified_cross_validation(
        X_train, y_train, n_splits=5, verbose=True, seed=42,
    )
    print(f"\n  CV Accuracy: {cv_results['accuracy_mean']:.4f} ± {cv_results['accuracy_std']:.4f}")
    print(f"  CV Kappa:    {cv_results['kappa_mean']:.4f} ± {cv_results['kappa_std']:.4f}")
    print(f"  CV MCC:      {cv_results['mcc_mean']:.4f} ± {cv_results['mcc_std']:.4f}")
    
    # ===== Entrenamiento final + feature selection =====
    print("\n[5] Entrenamiento final con feature selection")
    result = train_with_feature_selection(
        X_train, y_train, X_test, y_test,
        feature_names=FEATURE_NAMES,
        verbose=True,
    )
    
    # ===== Métricas finales =====
    test_metrics = compute_full_metrics(
        y_test, result['predictions'], class_names=['N', 'PAC']
    )
    print_metrics_report(test_metrics, class_names=['N', 'PAC'])
    
    print("\n✓ Pipeline end-to-end funcional")


if __name__ == "__main__":
    main()
