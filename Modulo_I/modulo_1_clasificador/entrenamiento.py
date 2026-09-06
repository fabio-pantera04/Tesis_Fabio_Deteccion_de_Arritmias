"""
modulo_1_clasificador/entrenamiento.py

Script principal de entrenamiento del Módulo I.
Ejecutar desde la raíz del proyecto:

    python -m modulo_1_clasificador.entrenamiento

Genera en datos/resultados/:
    mejores_shapelets_MSM.png
    confusion_matrix_MSM.png       (absoluta + normalizada)
    roc_curves_MSM.png
    ground_truth_vs_pred_MSM.png
    error_entrenamiento_MSM.png    (log-loss por iteración)
    learning_curve_MSM.png
    feature_importance_MSM.png
    reporte_metricas_MSM.txt       (accuracy, kappa, MCC, FLOPs, tiempos)
    X_train_tabular_MSM.csv
    X_test_tabular_MSM.csv

Genera en datos/:
    shapelets_micro_MSM.npy
    shapelets_macro_MSM.npy
    xgboost_model_MSM.json
    label_encoder_MSM.npy
"""

import sys, time
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    RUTA_ICENTIA, RUTA_MITBIH,
    NUM_MUESTRAS_POR_CLASE, XGB_RANDOM_STATE,
    ARCHIVO_MICRO_SHAPELETS, ARCHIVO_MACRO_SHAPELETS,
    CARPETA_SALIDA,
)
from modulo_1_clasificador.lectura_senal import cargar_dataset_balanceado
from modulo_1_clasificador.shapelets     import extract_shapelets
from modulo_1_clasificador.embedding     import construir_embedding
from modulo_1_clasificador.clasificador  import (
    entrenar, guardar_modelo, evaluar,
    estimar_flops_inferencia, medir_tiempo_inferencia,
    plot_shapelets, plot_confusion_matrix, plot_roc_curves,
    plot_ground_truth_vs_pred, plot_error_entrenamiento,
    plot_feature_importance, plot_learning_curve,
    guardar_reporte,
)


def main():
    tiempos = {}

    print("=" * 70)
    print("  ENTRENAMIENTO MÓDULO I — MSM+Sakoe-Chiba + XGBoost")
    print("=" * 70)

    # ── 1. Carga ──────────────────────────────────────────────────────────────
    bases = [str(p) for p in [RUTA_ICENTIA, RUTA_MITBIH] if str(p)]
    print(f"\n[DATOS] {NUM_MUESTRAS_POR_CLASE} muestras/clase")

    t0 = time.perf_counter()
    X, y = cargar_dataset_balanceado(bases, NUM_MUESTRAS_POR_CLASE)
    tiempos['carga_s'] = round(time.perf_counter() - t0, 2)
    print(f"[DATOS] {len(X)} señales | Clases: {sorted(set(y))} "
          f"| {tiempos['carga_s']}s")

    # ── 2. Split 70/30 ───────────────────────────────────────────────────────
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=XGB_RANDOM_STATE)

    # Para la curva de error usamos 15% del train como validación interna
    X_tr2, X_val, y_tr2, y_val = train_test_split(
        X_tr, y_tr, test_size=0.15, stratify=y_tr,
        random_state=XGB_RANDOM_STATE)

    print(f"[SPLIT] Train={len(X_tr)} | Val interno={len(X_val)} | Test={len(X_te)}")

    # ── 3. Shapelets ─────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    micro_dict, macro_dict, micro_meta, macro_meta = extract_shapelets(X_tr, y_tr)
    tiempos['shapelets_s'] = round(time.perf_counter() - t0, 2)

    np.save(str(ARCHIVO_MICRO_SHAPELETS),
            np.array(micro_dict, dtype=object), allow_pickle=True)
    np.save(str(ARCHIVO_MACRO_SHAPELETS),
            np.array(macro_dict, dtype=object), allow_pickle=True)
    plot_shapelets(micro_dict, macro_dict, top_k=3)
    print(f"[SHAPELETS] {tiempos['shapelets_s']}s")

    # ── 4. Embedding ─────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    print("\n[EMBEDDING] Generando matrices con MSM-SC...")
    df_tr  = construir_embedding(X_tr,  micro_dict, macro_dict, tag="TRAIN")
    df_te  = construir_embedding(X_te,  micro_dict, macro_dict, tag="TEST")
    df_val = construir_embedding(X_val, micro_dict, macro_dict, tag="VAL")
    tiempos['embedding_s'] = round(time.perf_counter() - t0, 2)

    df_tr['Etiqueta'] = list(y_tr)
    df_te['Etiqueta'] = list(y_te)

    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)
    df_tr.to_csv(CARPETA_SALIDA / "X_train_tabular_MSM.csv", index=False)
    df_te.to_csv(CARPETA_SALIDA / "X_test_tabular_MSM.csv",  index=False)
    print(f"[EMBEDDING] {tiempos['embedding_s']}s")

    X_tr_xgb  = df_tr.drop('Etiqueta', axis=1)
    X_te_xgb  = df_te.drop('Etiqueta', axis=1)
    X_val_xgb = df_val
    columnas   = list(X_tr_xgb.columns)

    # ── 5. Entrenamiento con curva de error ───────────────────────────────────
    print("\n[XGBOOST] Entrenando con conjunto de validación interna...")
    modelo, le, clases, t_train, evals_result = entrenar(
        X_tr_xgb, y_tr,
        X_val  = X_val_xgb,
        y_val  = y_val,
    )
    tiempos['entrenamiento_s'] = round(t_train, 2)
    guardar_modelo(modelo, le)

    # ── 6. Evaluación completa ────────────────────────────────────────────────
    metricas = evaluar(modelo, le, clases, X_te_xgb, y_te)

    # ── 7. FLOPs y tiempo de inferencia unitaria ──────────────────────────────
    flops = estimar_flops_inferencia(modelo, len(columnas))
    t_inf = medir_tiempo_inferencia(modelo, X_te_xgb)
    tiempos['inferencia_ms_media'] = t_inf['media_ms']

    print(f"\n[FLOPs] Total por señal: {flops['flops_total_M']} MFLOPs")
    print(f"[TIEMPO] Inferencia unitaria: {t_inf['media_ms']} ms "
          f"(± {t_inf['std_ms']} ms)")

    # ── 8. Gráficas ───────────────────────────────────────────────────────────
    plot_confusion_matrix(metricas)
    aucs, auc_m = plot_roc_curves(metricas)
    plot_ground_truth_vs_pred(metricas)
    plot_error_entrenamiento(evals_result)
    importancias = plot_feature_importance(modelo, columnas)
    plot_learning_curve(X_tr_xgb, le.transform(y_tr))

    # ── 9. Reporte completo ───────────────────────────────────────────────────
    guardar_reporte(metricas, aucs, auc_m, flops, t_train, t_inf)

    # ── 10. Generar vector de entrada para el Módulo II ───────────────────────
    print("\n[VECTOR LLM] Generando vector de entrada para el agente...")
    import json

    vector_salida = []
    for i in range(len(y_te)):
        cr   = str(y_te[i])
        cp   = str(le.inverse_transform([y_pred[i]])[0])
        probs = {clases[j]: round(float(y_proba[i, j]), 4)
             for j in range(len(clases))}
        ps   = sorted(probs.values(), reverse=True)
        gap  = round(ps[0] - ps[1], 4)

        dists = {columnas[int(idx)]: round(float(X_te_xgb.iloc[i, int(idx)]), 4)
             for idx in np.argsort(importancias)[::-1][:5]
             if int(idx) < len(columnas)}

        vector_salida.append({
            'id'                    : int(i),
            'clase_real'            : cr,
            'clase_predicha'        : cp,
            'correcto'              : bool(cr == cp),
            'probabilidades'        : probs,
            'confidence_gap'        : gap,
            'alerta_baja_confianza' : bool(gap < 0.25),
            'distancias_top5_msm'   : dists,
        })

    ruta_vector = CARPETA_SALIDA / "vector_entrada_LLM_MSM.json"
    with open(ruta_vector, 'w', encoding='utf-8') as f:
        json.dump({'version': '4.0', 'predicciones_test': vector_salida},
              f, ensure_ascii=False, indent=2)
    print(f"[VECTOR LLM] Guardado: {ruta_vector} ({len(vector_salida)} muestras)")
    # ── 11. Resumen ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  ENTRENAMIENTO COMPLETO")
    print("=" * 70)
    archivos = [
        "datos/shapelets_micro_MSM.npy",
        "datos/shapelets_macro_MSM.npy",
        "datos/xgboost_model_MSM.json",
        "datos/label_encoder_MSM.npy",
        "datos/resultados/mejores_shapelets_MSM.png",
        "datos/resultados/confusion_matrix_MSM.png",
        "datos/resultados/roc_curves_MSM.png",
        "datos/resultados/ground_truth_vs_pred_MSM.png",
        "datos/resultados/error_entrenamiento_MSM.png",
        "datos/resultados/learning_curve_MSM.png",
        "datos/resultados/feature_importance_MSM.png",
        "datos/resultados/reporte_metricas_MSM.txt",
        "datos/resultados/X_train_tabular_MSM.csv",
        "datos/resultados/X_test_tabular_MSM.csv",
    ]
    for f in archivos:
        print(f"  [OK]  {f}")

    print(f"\n{'─'*50}")
    print(f"  Accuracy          : {metricas['accuracy']*100:.2f}%")
    print(f"  Kappa de Cohen    : {metricas['kappa']:.4f}")
    print(f"  MCC               : {metricas['mcc']:.4f}")
    print(f"  F1-macro          : {metricas['f1_macro']:.4f}")
    print(f"  AUC micro-avg     : {auc_m:.4f}")
    print(f"  FLOPs por señal   : {flops['flops_total_M']} MFLOPs")
    print(f"  T. entrenamiento  : {tiempos['entrenamiento_s']}s")
    print(f"  T. embedding      : {tiempos['embedding_s']}s")
    print(f"  T. inferencia     : {t_inf['media_ms']} ms/señal")
    print("=" * 70)


if __name__ == "__main__":
    main()
