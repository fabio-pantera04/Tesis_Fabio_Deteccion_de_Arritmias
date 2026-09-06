"""
xgboost_model.py
================
Implementa el clasificador XGBoost con los hiperparámetros exactos del paper
Andotra & Sunkaria (2024), incluyendo:

1. Hiperparámetros: max_depth=3, n_estimators=1000, α=0.01, λ=1.0, γ=0.1
2. Early stopping con paciencia 10
3. Feature selection con SelectFromModel
4. Re-entrenamiento con features seleccionadas
5. Validación con stratified 5-fold cross-validation

Diferencias con el paper (justificadas):
- Para clasificación binaria, usamos objective='binary:logistic' (no 'multi:softprob')
- Para clasificación binaria, eval_metric='logloss' (no 'mlogloss')
- Reportamos kappa y MCC ADEMÁS de accuracy/F1/sensitivity (paper no las reporta)
"""

import numpy as np
import xgboost as xgb
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, matthews_corrcoef,
    f1_score, recall_score, precision_score,
    confusion_matrix, classification_report,
)


# =====================================================================
# Configuración del paper (Tabla y Sec III-C)
# =====================================================================

PAPER_XGB_PARAMS = {
    'max_depth': 3,
    'n_estimators': 1000,
    'reg_alpha': 0.01,      # L1 (α)
    'reg_lambda': 1.0,      # L2 (λ)
    'gamma': 0.1,           # min loss reduction (γ)
    'learning_rate': 0.1,   # paper no especifica; default razonable
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0,
}

EARLY_STOPPING_ROUNDS = 10


def build_xgb_classifier(n_classes: int = 2, **override_params):
    """
    Construye un XGBClassifier con la configuración del paper.
    Ajusta el objetivo según binario o multi-clase.
    """
    params = PAPER_XGB_PARAMS.copy()
    params.update(override_params)
    
    if n_classes == 2:
        params['objective'] = 'binary:logistic'
        params['eval_metric'] = 'logloss'
    else:
        params['objective'] = 'multi:softprob'
        params['eval_metric'] = 'mlogloss'
        params['num_class'] = n_classes
    
    return xgb.XGBClassifier(**params)


# =====================================================================
# Entrenamiento con early stopping y feature selection
# =====================================================================

def train_with_feature_selection(X_train, y_train, X_test, y_test,
                                   feature_names=None, verbose=True):
    """
    Pipeline de entrenamiento del paper:
    1. Entrenar XGBoost completo con early stopping
    2. Aplicar SelectFromModel para feature selection
    3. Re-entrenar con features seleccionadas
    4. Evaluar en test
    
    Devuelve:
    ---------
    dict con:
    - 'model_full': modelo entrenado con todas las features
    - 'model_selected': modelo entrenado con features seleccionadas
    - 'selected_features': lista de nombres de features seleccionadas
    - 'predictions': predicciones del modelo final en test
    - 'probabilities': probabilidades del modelo final en test
    """
    n_classes = len(np.unique(y_train))
    
    if verbose:
        print(f"  [XGB] Clases: {n_classes}, Train: {len(X_train)}, Test: {len(X_test)}")
    
    # ============ Paso 1: Entrenamiento inicial con early stopping ============
    if verbose:
        print(f"  [XGB] Entrenando modelo completo (early stopping en {EARLY_STOPPING_ROUNDS} rounds)...")
    
    model_full = build_xgb_classifier(
        n_classes=n_classes,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )
    
    model_full.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False,
    )
    
    if verbose:
        print(f"  [XGB] Best iteration: {model_full.best_iteration}")
    
    # ============ Paso 2: Feature selection ============
    if verbose:
        print(f"  [XGB] Aplicando feature selection (SelectFromModel, threshold='median')...")
    
    selector = SelectFromModel(model_full, prefit=True, threshold='median')
    X_train_sel = selector.transform(X_train)
    X_test_sel = selector.transform(X_test)
    
    if feature_names is not None:
        selected_features = [feature_names[i] for i in range(len(feature_names))
                              if selector.get_support()[i]]
    else:
        selected_features = None
    
    if verbose:
        print(f"  [XGB] Features seleccionadas: {X_train_sel.shape[1]}/{X_train.shape[1]}")
        if selected_features:
            print(f"  [XGB] Seleccionadas: {selected_features}")
    
    # ============ Paso 3: Re-entrenamiento con features seleccionadas ============
    if verbose:
        print(f"  [XGB] Re-entrenando con features seleccionadas...")
    
    model_selected = build_xgb_classifier(
        n_classes=n_classes,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )
    
    model_selected.fit(
        X_train_sel, y_train,
        eval_set=[(X_train_sel, y_train), (X_test_sel, y_test)],
        verbose=False,
    )
    
    # ============ Paso 4: Predicciones en test ============
    predictions = model_selected.predict(X_test_sel)
    probabilities = model_selected.predict_proba(X_test_sel)
    
    return {
        'model_full': model_full,
        'model_selected': model_selected,
        'selector': selector,
        'selected_features': selected_features,
        'predictions': predictions,
        'probabilities': probabilities,
        'X_test_sel': X_test_sel,
    }


# =====================================================================
# Stratified 5-fold cross-validation (paper Sec III-D)
# =====================================================================

def stratified_cross_validation(X, y, n_splits: int = 5,
                                  verbose: bool = True, seed: int = 42):
    """
    Stratified K-fold cross-validation usando XGBoost con los mismos hiperparámetros.
    
    Devuelve:
    ---------
    dict con accuracy, kappa, MCC, F1 promedio y desviaciones por fold.
    """
    n_classes = len(np.unique(y))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    fold_metrics = {
        'accuracy': [], 'kappa': [], 'mcc': [], 'f1_macro': [], 'f1_weighted': [],
    }
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        model = build_xgb_classifier(
            n_classes=n_classes,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        
        preds = model.predict(X_val)
        
        acc = accuracy_score(y_val, preds)
        kap = cohen_kappa_score(y_val, preds)
        mcc = matthews_corrcoef(y_val, preds)
        f1_m = f1_score(y_val, preds, average='macro')
        f1_w = f1_score(y_val, preds, average='weighted')
        
        fold_metrics['accuracy'].append(acc)
        fold_metrics['kappa'].append(kap)
        fold_metrics['mcc'].append(mcc)
        fold_metrics['f1_macro'].append(f1_m)
        fold_metrics['f1_weighted'].append(f1_w)
        
        if verbose:
            print(f"    Fold {fold_idx}/{n_splits}: acc={acc:.4f}, "
                  f"κ={kap:.4f}, MCC={mcc:.4f}")
    
    summary = {
        f'{key}_mean': np.mean(vals) for key, vals in fold_metrics.items()
    }
    summary.update({
        f'{key}_std': np.std(vals) for key, vals in fold_metrics.items()
    })
    summary['fold_metrics'] = fold_metrics
    
    return summary


# =====================================================================
# Reporte completo
# =====================================================================

def compute_full_metrics(y_true, y_pred, class_names):
    """
    Calcula todas las métricas: accuracy, kappa, MCC, F1, sensitivity, precision.
    Reporta más métricas que el paper para honestidad metodológica.
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'kappa': cohen_kappa_score(y_true, y_pred),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'sensitivity_per_class': recall_score(y_true, y_pred, average=None, labels=range(len(class_names))),
        'precision_per_class': precision_score(y_true, y_pred, average=None, labels=range(len(class_names))),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=range(len(class_names))),
        'classification_report': classification_report(
            y_true, y_pred, target_names=class_names, digits=4
        ),
    }


def print_metrics_report(metrics, class_names):
    """Imprime el reporte de métricas formateado."""
    print("\n" + "=" * 60)
    print("RESULTADOS DE CLASIFICACIÓN")
    print("=" * 60)
    print(f"Accuracy:      {metrics['accuracy']:.4f}")
    print(f"Cohen's Kappa: {metrics['kappa']:.4f}    (más robusto a desbalance)")
    print(f"MCC:           {metrics['mcc']:.4f}    (más robusto a desbalance)")
    print(f"F1 (macro):    {metrics['f1_macro']:.4f}")
    print(f"F1 (weighted): {metrics['f1_weighted']:.4f}")
    
    print("\nSensitivity por clase (lo que reporta el paper):")
    for cls, sens in zip(class_names, metrics['sensitivity_per_class']):
        print(f"  {cls:>10}: {sens:.4f}")
    
    print("\nMatriz de confusión:")
    cm = metrics['confusion_matrix']
    print(f"           {' '.join(f'{c:>10}' for c in class_names)}")
    for i, cls in enumerate(class_names):
        print(f"  {cls:>8}: {' '.join(f'{v:>10}' for v in cm[i])}")
    
    print("\nReporte por clase:")
    print(metrics['classification_report'])
