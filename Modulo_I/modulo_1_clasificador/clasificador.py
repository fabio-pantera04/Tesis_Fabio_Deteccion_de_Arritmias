"""
modulo_1_clasificador/clasificador.py

Entrenamiento, evaluación completa, serialización e inferencia con XGBoost.

Métricas generadas
------------------
  Accuracy, Precision, Recall, F1 (macro / weighted / por clase)
  Kappa de Cohen          — acuerdo inter-observador ajustado por azar
  MCC (Matthews)          — métrica balanceada para datasets multiclase
  Curvas ROC + AUC-OvR    — por clase y micro-average
  Curva de aprendizaje    — error train vs val (diagnóstico overfitting)
  Ground Truth vs Pred    — comparación visual real vs predicho por clase
  Matriz de confusión     — absoluta y normalizada
  FLOPs de inferencia     — operaciones de punto flotante por predicción
  Tiempos de ejecución    — entrenamiento, embedding, inferencia unitaria
  Error de entrenamiento  — log-loss train vs val por iteración (XGBoost eval)
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.patches import Patch
from pathlib import Path

from sklearn.model_selection import learning_curve
from sklearn.preprocessing   import LabelEncoder, label_binarize
from sklearn.metrics         import (
    accuracy_score, classification_report,
    confusion_matrix, roc_curve, auc,
    cohen_kappa_score, matthews_corrcoef,
    log_loss, precision_recall_fscore_support,
)
from xgboost import XGBClassifier

from config import (
    XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE,
    XGB_REG_ALPHA, XGB_REG_LAMBDA, XGB_RANDOM_STATE, N_JOBS,
    ARCHIVO_XGBOOST_MODELO, ARCHIVO_LABEL_ENCODER, CARPETA_SALIDA,
    MSM_C, MSM_WINDOW, NUM_SHAPELETS_MICRO, NUM_SHAPELETS_MACRO,
    NUM_CANDIDATES, NUM_COMPARACIONES, UMBRAL_CONFIANZA,
)


def _ruta(nombre: str) -> Path:
    return CARPETA_SALIDA / nombre


# ─────────────────────────────────────────────────────────────────────────────
# CONSTRUCCIÓN DEL MODELO
# ─────────────────────────────────────────────────────────────────────────────

def construir_modelo() -> XGBClassifier:
    return XGBClassifier(
        n_estimators      = XGB_N_ESTIMATORS,
        max_depth         = XGB_MAX_DEPTH,
        learning_rate     = XGB_LEARNING_RATE,
        reg_alpha         = XGB_REG_ALPHA,
        reg_lambda        = XGB_REG_LAMBDA,
        objective         = 'multi:softprob',
        eval_metric       = 'mlogloss',
        random_state      = XGB_RANDOM_STATE,
        use_label_encoder = False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENTRENAMIENTO CON REGISTRO DE TIEMPO Y CURVA DE ERROR
# ─────────────────────────────────────────────────────────────────────────────

def entrenar(X_train: pd.DataFrame, y_train: np.ndarray,
             X_val: pd.DataFrame = None, y_val: np.ndarray = None) -> tuple:
    """
    Entrena XGBoost y registra:
      - Tiempo de entrenamiento
      - Log-loss por iteración en train (y val si se provee)
      - Número de árboles efectivos usados

    Parámetros
    ----------
    X_train, y_train : datos de entrenamiento (embedding tabular)
    X_val,   y_val   : opcional — si se proveen, activa eval_set para
                       graficar la curva de error train vs validación

    Retorna
    -------
    modelo, le, clases, tiempo_entrenamiento_seg, evals_result
    """
    le     = LabelEncoder()
    y_enc  = le.fit_transform(y_train)
    clases = list(le.classes_)
    modelo = construir_modelo()

    evals_result = {}
    eval_set     = [(X_train, y_enc)]
    eval_names   = ['train']

    if X_val is not None and y_val is not None:
        y_val_enc = le.transform(y_val)
        eval_set.append((X_val, y_val_enc))
        eval_names.append('validation')

    t0 = time.perf_counter()
    modelo.fit(
        X_train, y_enc,
        eval_set        = eval_set,
        verbose         = False,
    )
    tiempo_train = time.perf_counter() - t0

    # Extraer historial de log-loss por iteración
    evals_result = modelo.evals_result()

    print(f"[XGBOOST] Entrenamiento finalizado en {tiempo_train:.1f}s")
    print(f"[XGBOOST] Clases: {clases}")
    return modelo, le, clases, tiempo_train, evals_result


# ─────────────────────────────────────────────────────────────────────────────
# SERIALIZACIÓN / DESERIALIZACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def guardar_modelo(modelo: XGBClassifier, le: LabelEncoder) -> None:
    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)
    modelo.save_model(str(ARCHIVO_XGBOOST_MODELO))
    np.save(str(ARCHIVO_LABEL_ENCODER), le.classes_, allow_pickle=True)
    print(f"[XGBOOST] Modelo guardado → {ARCHIVO_XGBOOST_MODELO}")


def cargar_modelo() -> tuple:
    modelo = construir_modelo()
    modelo.load_model(str(ARCHIVO_XGBOOST_MODELO))
    clases_arr  = np.load(str(ARCHIVO_LABEL_ENCODER), allow_pickle=True)
    le          = LabelEncoder()
    le.classes_ = clases_arr
    clases      = list(clases_arr)
    print(f"[XGBOOST] Modelo cargado | Clases: {clases}")
    return modelo, le, clases


# ─────────────────────────────────────────────────────────────────────────────
# ESTIMACIÓN DE FLOPs
# ─────────────────────────────────────────────────────────────────────────────

def estimar_flops_inferencia(modelo: XGBClassifier,
                              n_features: int) -> dict:
    """
    Estima las operaciones de punto flotante (FLOPs) para clasificar
    UNA señal nueva a través del embedding + XGBoost.

    Metodología
    -----------
    Embedding MSM-SC:
        Por cada uno de los 60 shapelets se calcula la distancia mínima
        por ventana deslizante. El costo dominante es la matriz DP de MSM.
        Para un shapelet de longitud L_s y una señal de longitud L_sig,
        con banda w = 10%:
            FLOPs_msm ≈ L_s × (2w) × 3  (tres operaciones min por celda)
        Sumado sobre todos los shapelets.

    XGBoost:
        Cada árbol recorre una ruta de profundidad media d_avg.
        En cada nodo se hace 1 comparación (1 FLOP).
        FLOPs_xgb ≈ n_estimadores × d_avg × n_clases

    Retorna
    -------
    dict con FLOPs desglosados y total.
    """
    n_micro = NUM_SHAPELETS_MICRO
    n_macro = NUM_SHAPELETS_MACRO

    # Longitudes representativas
    L_micro_avg = 85      # centro del rango 40-130
    L_macro_avg = 450     # centro del rango 200-700
    L_senal_c   = 162     # señal corta
    L_senal_l   = 1000    # señal larga

    w_micro = max(1, int(L_micro_avg * MSM_WINDOW))
    w_macro = max(1, int(L_macro_avg * MSM_WINDOW))

    # Ventanas deslizantes
    ventanas_micro_corta = max(1, L_senal_c - L_micro_avg + 1)
    ventanas_micro_larga = max(1, L_senal_l - L_micro_avg + 1)
    ventanas_macro_corta = 1   # señal < shapelet → comparación directa
    ventanas_macro_larga = max(1, L_senal_l - L_macro_avg + 1)

    # FLOPs MSM por ventana: L_s × 2w × 3 ops
    flops_msm_micro = (L_micro_avg * 2 * w_micro * 3)
    flops_msm_macro = (L_macro_avg * 2 * w_macro * 3)

    # Total embedding (promedio señal corta + larga)
    flops_emb_micro = n_micro * (
        flops_msm_micro * (ventanas_micro_corta + ventanas_micro_larga) / 2
    )
    flops_emb_macro = n_macro * (
        flops_msm_macro * (ventanas_macro_corta + ventanas_macro_larga) / 2
    )
    flops_embedding = flops_emb_micro + flops_emb_macro

    # FLOPs XGBoost: cada árbol recorre ~max_depth/2 nodos en promedio
    d_avg          = XGB_MAX_DEPTH / 2
    n_clases       = 4
    flops_xgboost  = XGB_N_ESTIMATORS * d_avg * n_clases

    flops_total = flops_embedding + flops_xgboost

    return {
        'flops_embedding_MSM' : int(flops_embedding),
        'flops_embedding_micro': int(flops_emb_micro),
        'flops_embedding_macro': int(flops_emb_macro),
        'flops_xgboost'       : int(flops_xgboost),
        'flops_total'         : int(flops_total),
        'flops_total_M'       : round(flops_total / 1e6, 2),
        'nota': 'Estimacion analitica. Embedding domina el costo computacional.',
    }


# ─────────────────────────────────────────────────────────────────────────────
# TIEMPO DE INFERENCIA UNITARIA
# ─────────────────────────────────────────────────────────────────────────────

def medir_tiempo_inferencia(modelo: XGBClassifier,
                             X_test: pd.DataFrame,
                             n_repeticiones: int = 50) -> dict:
    """
    Mide el tiempo real de inferencia para UNA muestra.
    Repite n_repeticiones veces y reporta media ± std.
    """
    una_fila = X_test.iloc[[0]]
    tiempos  = []
    for _ in range(n_repeticiones):
        t0 = time.perf_counter()
        modelo.predict_proba(una_fila)
        tiempos.append(time.perf_counter() - t0)

    tiempos = np.array(tiempos) * 1000  # convertir a ms
    return {
        'media_ms' : round(float(np.mean(tiempos)),  3),
        'std_ms'   : round(float(np.std(tiempos)),   3),
        'min_ms'   : round(float(np.min(tiempos)),   3),
        'max_ms'   : round(float(np.max(tiempos)),   3),
        'n_reps'   : n_repeticiones,
    }


# ─────────────────────────────────────────────────────────────────────────────
# EVALUACIÓN COMPLETA
# ─────────────────────────────────────────────────────────────────────────────

def evaluar(modelo: XGBClassifier, le: LabelEncoder, clases: list,
            X_test: pd.DataFrame, y_test: np.ndarray) -> dict:
    """
    Evaluación completa sobre el conjunto de test.

    Retorna
    -------
    dict con todas las métricas, predicciones y tiempos.
    """
    y_test_enc = le.transform(y_test)

    t0     = time.perf_counter()
    y_pred = modelo.predict(X_test)
    t_pred = (time.perf_counter() - t0) * 1000  # ms total

    y_proba = modelo.predict_proba(X_test)

    # ── Métricas principales ──────────────────────────────────────────────────
    acc    = accuracy_score(y_test_enc, y_pred)
    kappa  = cohen_kappa_score(y_test_enc, y_pred)
    mcc    = matthews_corrcoef(y_test_enc, y_pred)
    logloss_test = log_loss(y_test_enc, y_proba)

    prec, rec, f1, sup = precision_recall_fscore_support(
        y_test_enc, y_pred, average=None, labels=range(len(clases)))
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test_enc, y_pred, average='macro')
    prec_w,     rec_w,     f1_w,     _ = precision_recall_fscore_support(
        y_test_enc, y_pred, average='weighted')

    print(f"\n{'='*60}")
    print(f"  GLOBAL ACCURACY   : {acc*100:.2f}%")
    print(f"  Kappa de Cohen    : {kappa:.4f}")
    print(f"  MCC               : {mcc:.4f}")
    print(f"  F1-macro          : {f1_macro:.4f}")
    print(f"  F1-weighted       : {f1_w:.4f}")
    print(f"  Log-loss (test)   : {logloss_test:.4f}")
    print(f"  Tiempo predicción : {t_pred:.1f} ms ({len(y_test)} muestras)")
    print(f"{'='*60}\n")
    print(classification_report(y_test_enc, y_pred, target_names=clases))

    return {
        'y_test_enc'   : y_test_enc,
        'y_test_orig'  : y_test,
        'y_pred'       : y_pred,
        'y_proba'      : y_proba,
        'accuracy'     : acc,
        'kappa'        : kappa,
        'mcc'          : mcc,
        'logloss_test' : logloss_test,
        'f1_macro'     : f1_macro,
        'f1_weighted'  : f1_w,
        'precision_por_clase' : dict(zip(clases, prec.tolist())),
        'recall_por_clase'    : dict(zip(clases, rec.tolist())),
        'f1_por_clase'        : dict(zip(clases, f1.tolist())),
        'soporte_por_clase'   : dict(zip(clases, sup.tolist())),
        'clases'       : clases,
        't_pred_ms_total': round(t_pred, 2),
        't_pred_ms_por_muestra': round(t_pred / len(y_test), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZACIONES
# ─────────────────────────────────────────────────────────────────────────────

def plot_shapelets(micro_dict: list, macro_dict: list, top_k: int = 3) -> None:
    n = min(top_k, len(micro_dict)) + min(top_k, len(macro_dict))
    fig, axes = plt.subplots(n, 1, figsize=(11, 3 * n))
    if n == 1: axes = [axes]
    idx = 0
    for i in range(min(top_k, len(micro_dict))):
        axes[idx].plot(micro_dict[i], color='darkred', linewidth=1.8)
        axes[idx].set_title(
            f"Shapelet MICRO #{i+1}  ({len(micro_dict[i])} muestras — latido)  [MSM-SC]",
            fontsize=10)
        axes[idx].set_ylabel("Amplitud (Z)"); axes[idx].grid(alpha=0.3)
        idx += 1
    for i in range(min(top_k, len(macro_dict))):
        axes[idx].plot(macro_dict[i], color='navy', linewidth=1.8)
        axes[idx].set_title(
            f"Shapelet MACRO #{i+1}  ({len(macro_dict[i])} muestras — ritmo)  [MSM-SC]",
            fontsize=10)
        axes[idx].set_ylabel("Amplitud (Z)"); axes[idx].grid(alpha=0.3)
        idx += 1
    plt.tight_layout()
    plt.savefig(_ruta("mejores_shapelets_MSM.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("[PLOT] mejores_shapelets_MSM.png")


def plot_confusion_matrix(metricas: dict) -> None:
    """Matriz de confusión absoluta Y normalizada en el mismo canvas."""
    y_t, y_p = metricas['y_test_enc'], metricas['y_pred']
    clases    = metricas['clases']
    acc       = metricas['accuracy']
    kappa     = metricas['kappa']

    cm      = confusion_matrix(y_t, y_p)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=clases, yticklabels=clases,
                linewidths=0.5, linecolor='gray')
    axes[0].set_title(f'Confusion matrix — counts\nAccuracy: {acc*100:.2f}%  |  Kappa: {kappa:.3f}',
                      fontsize=11)
    axes[0].set_ylabel('True label (cardiologist)')
    axes[0].set_xlabel('Model prediction')

    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=axes[1],
                xticklabels=clases, yticklabels=clases,
                linewidths=0.5, linecolor='gray', vmin=0, vmax=1)
    axes[1].set_title(f'Confusion matrix — normalized (recall per class)\nMCC: {metricas["mcc"]:.3f}',
                      fontsize=11)
    axes[1].set_ylabel('True label (cardiologist)')
    axes[1].set_xlabel('Model prediction')

    plt.tight_layout()
    plt.savefig(_ruta("confusion_matrix_MSM.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("[PLOT] confusion_matrix_MSM.png")


def plot_roc_curves(metricas: dict) -> tuple:
    y_t, y_p  = metricas['y_test_enc'], metricas['y_proba']
    clases     = metricas['clases']
    colores    = ['#e63946', '#457b9d', '#2a9d8f', '#e9c46a']

    plt.figure(figsize=(9, 7))
    aucs  = {}
    y_bin = label_binarize(y_t, classes=range(len(clases)))

    for i, cls in enumerate(clases):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_p[:, i])
        a            = auc(fpr, tpr)
        aucs[cls]    = a
        plt.plot(fpr, tpr, color=colores[i % 4], lw=2,
                 label=f'{cls}  (AUC={a:.3f})')

    fpr_m, tpr_m, _ = roc_curve(y_bin.ravel(), y_p.ravel())
    auc_m            = auc(fpr_m, tpr_m)
    plt.plot(fpr_m, tpr_m, 'k--', lw=2.5,
             label=f'Micro-average  (AUC={auc_m:.3f})')
    plt.plot([0, 1], [0, 1], 'k:', linewidth=1)

    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC curves — MSM+Sakoe-Chiba  (One-vs-Rest)', fontsize=13)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(_ruta("roc_curves_MSM.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("[PLOT] roc_curves_MSM.png")
    return aucs, auc_m


def plot_ground_truth_vs_pred(metricas: dict) -> None:
    """
    Gráfica de barras agrupadas: distribución real vs predicha por clase.
    Permite ver visualmente si el modelo tiene sesgo hacia alguna clase.
    """
    y_t   = metricas['y_test_enc']
    y_p   = metricas['y_pred']
    clases = metricas['clases']
    n      = len(clases)
    x      = np.arange(n)
    ancho  = 0.35

    real  = [np.sum(y_t == i) for i in range(n)]
    pred  = [np.sum(y_p == i) for i in range(n)]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_r = ax.bar(x - ancho/2, real, ancho, label='Ground truth (real)',
                    color='#457b9d', edgecolor='white')
    bars_p = ax.bar(x + ancho/2, pred, ancho, label='Predicted',
                    color='#e63946', edgecolor='white', alpha=0.85)

    ax.bar_label(bars_r, padding=3, fontsize=10)
    ax.bar_label(bars_p, padding=3, fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(clases, fontsize=11)
    ax.set_ylabel('Number of samples', fontsize=11)
    ax.set_title('Ground truth vs predicted — class distribution\n'
                 f'(test set, n={len(y_t)})', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(_ruta("ground_truth_vs_pred_MSM.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("[PLOT] ground_truth_vs_pred_MSM.png")


def plot_error_entrenamiento(evals_result: dict) -> None:
    """
    Curva de error de entrenamiento: log-loss por iteración (árbol).
    Muestra train vs validation para diagnosticar overfitting.
    Si solo hay train (sin val), grafica solo train.
    """
    plt.figure(figsize=(9, 5))

    train_loss = evals_result.get('train',      {}).get('mlogloss', [])
    val_loss   = evals_result.get('validation', {}).get('mlogloss', [])

    iteraciones = range(1, len(train_loss) + 1)

    if train_loss:
        plt.plot(iteraciones, train_loss, color='#457b9d', lw=1.8,
                 label='Training log-loss')
    if val_loss:
        plt.plot(range(1, len(val_loss) + 1), val_loss, color='#e63946',
                 lw=1.8, linestyle='--', label='Validation log-loss')
        # Marcar el mínimo de validación
        min_idx = int(np.argmin(val_loss))
        plt.axvline(min_idx + 1, color='#e63946', lw=0.8,
                    linestyle=':', alpha=0.6)
        plt.scatter([min_idx + 1], [val_loss[min_idx]], color='#e63946',
                    zorder=5, s=40,
                    label=f'Min val loss = {val_loss[min_idx]:.4f} '
                          f'(iter {min_idx+1})')

    plt.xlabel('Boosting iteration (trees)', fontsize=11)
    plt.ylabel('Log-loss', fontsize=11)
    plt.title('Training error curve — XGBoost / MSM+Sakoe-Chiba', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(_ruta("error_entrenamiento_MSM.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("[PLOT] error_entrenamiento_MSM.png")


def plot_feature_importance(modelo: XGBClassifier, columnas: list) -> np.ndarray:
    importancias = modelo.feature_importances_
    idx      = np.argsort(importancias)[::-1]
    colores  = ['#e63946' if columnas[i].startswith('Macro') else '#457b9d'
                for i in idx]
    plt.figure(figsize=(max(12, len(idx) // 3), 5))
    plt.bar(range(len(idx)), importancias[idx] * 100,
            color=colores, edgecolor='white')
    plt.xticks(range(len(idx)), [columnas[i] for i in idx],
               rotation=90, fontsize=7)
    plt.ylabel("Feature importance (%)")
    plt.title("Feature importance — Red=Macro (rhythm), Blue=Micro (beat)  [MSM-SC]",
              fontsize=12)
    plt.legend(handles=[Patch(color='#e63946', label='Macro (rhythm)'),
                         Patch(color='#457b9d', label='Micro (beat)')], fontsize=9)
    plt.tight_layout()
    plt.savefig(_ruta("feature_importance_MSM.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("[PLOT] feature_importance_MSM.png")
    return importancias


def plot_learning_curve(X_train: pd.DataFrame, y_train_enc: np.ndarray) -> None:
    """
    Curva de aprendizaje: accuracy train vs validation en función del
    tamaño del conjunto de entrenamiento.
    Diagnostica si el modelo mejoraría con más datos.
    """
    print("[LEARNING CURVE] Calculando (puede tardar)...")
    modelo_cfg = construir_modelo()
    ts, tr_s, va_s = learning_curve(
        modelo_cfg, X_train, y_train_enc, cv=5, n_jobs=N_JOBS,
        train_sizes=np.linspace(0.1, 1.0, 8), scoring='accuracy')

    plt.figure(figsize=(8, 5))
    plt.plot(ts, tr_s.mean(1) * 100, 'o-', color='#457b9d',
             label='Train accuracy', lw=2)
    plt.fill_between(ts, (tr_s.mean(1) - tr_s.std(1)) * 100,
                     (tr_s.mean(1) + tr_s.std(1)) * 100,
                     alpha=0.15, color='#457b9d')
    plt.plot(ts, va_s.mean(1) * 100, 's--', color='#e63946',
             label='Validation accuracy', lw=2)
    plt.fill_between(ts, (va_s.mean(1) - va_s.std(1)) * 100,
                     (va_s.mean(1) + va_s.std(1)) * 100,
                     alpha=0.15, color='#e63946')
    plt.xlabel("Training set size")
    plt.ylabel("Accuracy (%)")
    plt.title("Learning curve — XGBoost / MSM+Sakoe-Chiba")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(_ruta("learning_curve_MSM.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("[PLOT] learning_curve_MSM.png")


# ─────────────────────────────────────────────────────────────────────────────
# REPORTE COMPLETO EN TEXTO
# ─────────────────────────────────────────────────────────────────────────────

def guardar_reporte(metricas: dict, aucs: dict, auc_m: float,
                    flops: dict, t_train: float, t_inf: dict) -> None:
    """Guarda el reporte completo con todas las métricas en .txt."""
    y_t, y_p = metricas['y_test_enc'], metricas['y_pred']
    clases    = metricas['clases']
    acc       = metricas['accuracy']
    rep       = classification_report(y_t, y_p, target_names=clases)
    cm        = confusion_matrix(y_t, y_p)

    lineas = [
        "=" * 70,
        "REPORTE COMPLETO MODULO I — TESIS CIMAT",
        "Clasificador: XGBoost + MSM con Banda de Sakoe-Chiba",
        "Ref. MSM  : Stefan et al. (2013) IEEE TKDE",
        "Ref. banda: Holznigenkemper et al. (2023) arXiv:2301.01977",
        "=" * 70,

        "\n── METRICAS GLOBALES ──────────────────────────────────────────",
        f"  Accuracy global    : {acc*100:.2f}%",
        f"  Kappa de Cohen     : {metricas['kappa']:.4f}",
        f"  MCC                : {metricas['mcc']:.4f}",
        f"  F1-macro           : {metricas['f1_macro']:.4f}",
        f"  F1-weighted        : {metricas['f1_weighted']:.4f}",
        f"  Log-loss (test)    : {metricas['logloss_test']:.4f}",

        "\n── METRICAS POR CLASE ─────────────────────────────────────────",
        rep,

        "── AUC ROC (One-vs-Rest) ──────────────────────────────────────",
        *[f"  {c:8s}: {v:.4f}" for c, v in aucs.items()],
        f"  Micro-avg: {auc_m:.4f}",

        "\n── MATRIZ DE CONFUSION ────────────────────────────────────────",
        "  (filas = clase real, columnas = clase predicha)",
        "        " + "  ".join(f"{c:7s}" for c in clases),
        *[f"  {clases[i]:6s}  " + "  ".join(f"{v:7d}" for v in fila)
          for i, fila in enumerate(cm)],

        "\n── TIEMPOS DE EJECUCION ───────────────────────────────────────",
        f"  Entrenamiento XGBoost  : {t_train:.2f} s",
        f"  Inferencia por muestra : {t_inf['media_ms']:.3f} ms "
        f"(± {t_inf['std_ms']:.3f} ms, n={t_inf['n_reps']})",
        f"  Inferencia total test  : {metricas['t_pred_ms_total']:.1f} ms "
        f"({len(y_t)} muestras)",

        "\n── FLOPs DE INFERENCIA (estimacion analitica) ─────────────────",
        f"  Embedding MSM-SC total : {flops['flops_embedding_MSM']:,} FLOPs",
        f"    Bloque micro         : {flops['flops_embedding_micro']:,} FLOPs",
        f"    Bloque macro         : {flops['flops_embedding_macro']:,} FLOPs",
        f"  XGBoost predict        : {flops['flops_xgboost']:,} FLOPs",
        f"  TOTAL por señal        : {flops['flops_total']:,} FLOPs "
        f"({flops['flops_total_M']} MFLOPs)",
        f"  Nota: {flops['nota']}",

        "\n── PARAMETROS DEL MODELO ──────────────────────────────────────",
        f"  MSM c (costo split/merge)  : {MSM_C}",
        f"  Banda Sakoe-Chiba          : {MSM_WINDOW} ({int(MSM_WINDOW*100)}%)",
        f"  Shapelets micro={NUM_SHAPELETS_MICRO}, macro={NUM_SHAPELETS_MACRO}",
        f"  Candidatos evaluados       : {NUM_CANDIDATES}",
        f"  Comparaciones por candidato: {NUM_COMPARACIONES}",
        f"  XGBoost n_estimators       : {XGB_N_ESTIMATORS}",
        f"  XGBoost max_depth          : {XGB_MAX_DEPTH}",
        f"  XGBoost learning_rate      : {XGB_LEARNING_RATE}",
        f"  L1={XGB_REG_ALPHA}, L2={XGB_REG_LAMBDA}",
        "=" * 70,
    ]

    with open(_ruta("reporte_metricas_MSM.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lineas))
    print("[REPORTE] reporte_metricas_MSM.txt")


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCIA EN TIEMPO REAL (una señal)
# ─────────────────────────────────────────────────────────────────────────────

def predecir(modelo: XGBClassifier, le: LabelEncoder,
             clases: list, X: pd.DataFrame) -> dict:
    """
    Genera predicción para una o varias señales en tiempo real.
    Retorna dict con clase, probabilidades, gap y alerta.
    """
    y_pred  = modelo.predict(X)
    y_proba = modelo.predict_proba(X)

    resultados = []
    for i in range(len(y_pred)):
        clase = str(le.inverse_transform([y_pred[i]])[0])
        probs = {clases[j]: round(float(y_proba[i, j]), 4)
                 for j in range(len(clases))}
        ps    = sorted(probs.values(), reverse=True)
        gap   = round(ps[0] - ps[1], 4)
        resultados.append({
            'clase_predicha'        : clase,
            'probabilidades'        : probs,
            'confidence_gap'        : gap,
            'alerta_baja_confianza' : bool(gap < UMBRAL_CONFIANZA),
        })

    return resultados[0] if len(resultados) == 1 else resultados
