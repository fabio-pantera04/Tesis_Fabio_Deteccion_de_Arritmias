"""
modulo_1_clasificador/entrenamiento_dual.py

Entrenamiento DUAL del Módulo I — Arquitectura separada por escala temporal.

  Modelo LATIDO : NORMAL / PAC  → solo shapelets MICRO (40-130 muestras)
  Modelo RITMO  : NSR   / AFIB  → solo shapelets MACRO (200-700 muestras)

Justificación (AAMI EC57:2012):
  La taxonomía AAMI separa explícitamente la clasificación de latidos
  individuales (clases N, S) de la clasificación de ritmos cardíacos.
  Combinarlos en un único modelo produce contaminación de features:
  shapelets macro dan distancia=0 para señales cortas, causando
  clasificaciones erróneas (PAC → AFIB).

Uso:
    cd C:/Modelo_Tesis/Modulo_I
    python -m modulo_1_clasificador.entrenamiento_dual
    python -m modulo_1_clasificador.entrenamiento_dual --solo latido
    python -m modulo_1_clasificador.entrenamiento_dual --solo ritmo
"""

import sys, time, json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    # Rutas
    CARPETA_SALIDA,
    ARCHIVO_SHAPELETS_LATIDO, ARCHIVO_MODELO_LATIDO, ARCHIVO_ENCODER_LATIDO,
    ARCHIVO_SHAPELETS_RITMO,  ARCHIVO_MODELO_RITMO,  ARCHIVO_ENCODER_RITMO,
    # Clases y fuentes
    CLASES_LATIDO, CLASES_RITMO, FUENTES_POR_CLASE, N_POR_CLASE,
    # MSM
    MSM_C, MSM_WINDOW, N_JOBS,
    # XGB compat
    XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE,
    XGB_REG_ALPHA, XGB_REG_LAMBDA, XGB_RANDOM_STATE,
    # Latido
    NUM_SHAPELETS_LATIDO, NUM_CANDIDATES_LATIDO,
    NUM_COMPARACIONES_LATIDO, RANGO_LATIDO, PASO_VENTANA_LATIDO, XGB_LATIDO,
    # Ritmo
    NUM_SHAPELETS_RITMO, NUM_CANDIDATES_RITMO,
    NUM_COMPARACIONES_RITMO, RANGO_RITMO, PASO_VENTANA_RITMO, XGB_RITMO,
)
from modulo_1_clasificador.shapelets import extract_shapelets
# Importar funciones de graficado de forma segura
try:
    from modulo_1_clasificador.clasificador import (
        plot_shapelets, plot_confusion_matrix, plot_roc_curves,
        plot_ground_truth_vs_pred, plot_feature_importance,
        plot_learning_curve,
    )
    _PLOTS_OK = True
except Exception as _e:
    print(f"[AVISO] Módulo clasificador no disponible: {_e}")
    print("[AVISO] El entrenamiento continuará sin gráficas.")
    _PLOTS_OK = False
from modulo_1_clasificador.lectura_senal import normalizar_zscore
from modulo_1_clasificador.msm_sc import msm_sc


# ─────────────────────────────────────────────────────────────────────────────
# CARGA DE DATOS POR MODELO
# ─────────────────────────────────────────────────────────────────────────────

def cargar_datos_modelo(clases: list) -> tuple:
    """
    Carga señales usando FUENTES_POR_CLASE con límites explícitos por fuente.

    Para AFIB: 306 de Icentia + 694 de CinC = 1000 exactos
    Para NORMAL, PAC, NSR: 1000 de Icentia

    Parámetros
    ----------
    clases : lista de clases a cargar (ej. ['NORMAL','PAC'])

    Retorna
    -------
    X : np.ndarray de señales normalizadas (dtype=object)
    y : np.ndarray de etiquetas
    """
    import random
    X, y = [], []

    for clase in clases:
        fuentes = FUENTES_POR_CLASE.get(clase, [])
        if not fuentes:
            print(f"  [AVISO] Sin fuentes configuradas para clase {clase}")
            continue

        cargados_clase = 0

        for ruta, n_max in fuentes:
            ruta = Path(ruta)
            if not ruta.exists():
                print(f"  [AVISO] Ruta no encontrada: {ruta}")
                continue

            archivos = list(ruta.glob("*.csv"))
            random.shuffle(archivos)
            cargados_fuente = 0

            for archivo in archivos:
                if cargados_fuente >= n_max:
                    break
                try:
                    vals  = pd.read_csv(archivo, header=None).iloc[:, 0]
                    senal = pd.to_numeric(vals, errors="coerce").dropna().values
                    senal = senal.astype(np.float64)
                    if len(senal) < 10:
                        continue
                    senal = normalizar_zscore(senal)
                    X.append(senal)
                    y.append(clase)
                    cargados_fuente += 1
                    cargados_clase  += 1
                except Exception:
                    continue

            print(f"  [{clase:8}] {ruta.name:30}: {cargados_fuente:4d} señales")

        print(f"  [{clase:8}] TOTAL                          : {cargados_clase:4d} señales")

    return np.array(X, dtype=object), np.array(y)


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING ESPECIALIZADO POR TIPO
# ─────────────────────────────────────────────────────────────────────────────

def _dist_min(senal: np.ndarray, shapelet: np.ndarray, paso: int = 1) -> float:
    """Distancia MSM-SC mínima por ventana deslizante con paso variable."""
    L = len(shapelet)
    if len(senal) < L:
        return float(msm_sc(shapelet, senal))
    mejor = np.inf
    for start in range(0, len(senal) - L + 1, paso):
        d = msm_sc(shapelet, senal[start: start + L], best_so_far=mejor)
        if d < mejor:
            mejor = d
    return float(mejor)


def construir_embedding_latido(X: np.ndarray,
                                shapelets: list,
                                tag: str = "") -> pd.DataFrame:
    """
    Embedding para señales de LATIDO.
    Solo usa shapelets MICRO (40-130 muestras) con paso=1.
    """
    from joblib import Parallel, delayed

    print(f"  [{tag}] Embedding LATIDO "
          f"({len(X)} señales × {len(shapelets)} shapelets micro, paso=1)...")

    def fila(sig):
        return [_dist_min(sig, s, paso=PASO_VENTANA_LATIDO) for s in shapelets]

    filas = Parallel(n_jobs=N_JOBS, backend='loky', verbose=3)(
        delayed(fila)(sig) for sig in X)

    df = pd.DataFrame(filas, columns=[f"Micro_S{i+1}" for i in range(len(shapelets))])
    n_inf = int((~np.isfinite(df.values)).sum())
    if n_inf > 0:
        for col in df.columns:
            mask = ~np.isfinite(df[col])
            mx = df[col][np.isfinite(df[col])].max()
            df.loc[mask, col] = float(mx * 2) if np.isfinite(mx) else 0.0
        print(f"    {n_inf} inf reemplazados")
    else:
        print(f"    OK sin inf/nan")
    return df


def construir_embedding_ritmo(X: np.ndarray,
                               shapelets: list,
                               tag: str = "") -> pd.DataFrame:
    """
    Embedding para señales de RITMO.
    Solo usa shapelets MACRO (200-700 muestras) con paso=PASO_VENTANA_RITMO.
    """
    from joblib import Parallel, delayed

    print(f"  [{tag}] Embedding RITMO "
          f"({len(X)} señales × {len(shapelets)} shapelets macro, "
          f"paso={PASO_VENTANA_RITMO})...")

    def fila(sig):
        return [_dist_min(sig, s, paso=PASO_VENTANA_RITMO) for s in shapelets]

    filas = Parallel(n_jobs=N_JOBS, backend='loky', verbose=3)(
        delayed(fila)(sig) for sig in X)

    df = pd.DataFrame(filas, columns=[f"Macro_S{i+1}" for i in range(len(shapelets))])
    n_inf = int((~np.isfinite(df.values)).sum())
    if n_inf > 0:
        for col in df.columns:
            mask = ~np.isfinite(df[col])
            mx = df[col][np.isfinite(df[col])].max()
            df.loc[mask, col] = float(mx * 2) if np.isfinite(mx) else 0.0
        print(f"    {n_inf} inf reemplazados")
    else:
        print(f"    OK sin inf/nan")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# ENTRENAMIENTO DE UN MODELO
# ─────────────────────────────────────────────────────────────────────────────

def entrenar_modelo(
    tipo: str,   # 'latido' o 'ritmo'
    tiempos: dict
) -> dict:
    """
    Entrena un modelo completo (shapelets + XGBoost) para el tipo indicado.

    Retorna dict con métricas del modelo entrenado.
    """
    sep = "=" * 70

    if tipo == 'latido':
        clases           = CLASES_LATIDO
        n_shapelets      = NUM_SHAPELETS_LATIDO
        n_candidates     = NUM_CANDIDATES_LATIDO
        n_comparaciones  = NUM_COMPARACIONES_LATIDO
        rango            = RANGO_LATIDO
        xgb_params       = XGB_LATIDO
        archivo_shap     = ARCHIVO_SHAPELETS_LATIDO
        archivo_modelo   = ARCHIVO_MODELO_LATIDO
        archivo_encoder  = ARCHIVO_ENCODER_LATIDO
        fn_embedding     = construir_embedding_latido
        sufijo           = "LATIDO"
        titulo           = "MODELO LATIDO — NORMAL / PAC — MSM-SC Micro + XGBoost"
    else:
        clases           = CLASES_RITMO
        n_shapelets      = NUM_SHAPELETS_RITMO
        n_candidates     = NUM_CANDIDATES_RITMO
        n_comparaciones  = NUM_COMPARACIONES_RITMO
        rango            = RANGO_RITMO
        xgb_params       = XGB_RITMO
        archivo_shap     = ARCHIVO_SHAPELETS_RITMO
        archivo_modelo   = ARCHIVO_MODELO_RITMO
        archivo_encoder  = ARCHIVO_ENCODER_RITMO
        fn_embedding     = construir_embedding_ritmo
        sufijo           = "RITMO"
        titulo           = "MODELO RITMO — NSR / AFIB — MSM-SC Macro + XGBoost"

    print(f"\n{sep}")
    print(f"  {titulo}")
    print(f"{sep}")

    # ── 1. Carga de datos ─────────────────────────────────────────────────────
    print(f"\n[DATOS] {N_POR_CLASE} señales por clase (balance exacto)")
    t0 = time.perf_counter()
    X, y = cargar_datos_modelo(clases)

    if len(X) == 0:
        print(f"[ERROR] No se cargaron datos para {tipo}. Verifica las rutas.")
        return {}

    print(f"[DATOS] {len(X)} señales | Clases: {sorted(set(y))} | "
          f"{time.perf_counter()-t0:.2f}s")

    # ── 2. Split 70/20/10 ─────────────────────────────────────────────────────
    X_tr_val, X_te, y_tr_val, y_te = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=XGB_RANDOM_STATE)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr_val, y_tr_val, test_size=0.15,
        stratify=y_tr_val, random_state=XGB_RANDOM_STATE)

    print(f"[SPLIT] Train={len(X_tr)} | Val={len(X_val)} | Test={len(X_te)}")
    tiempos['carga_' + tipo] = round(time.perf_counter() - t0, 2)

    # ── 3. Extracción de shapelets ────────────────────────────────────────────
    print(f"\n[SHAPELETS] Extrayendo {n_shapelets} shapelets {sufijo}...")
    t1 = time.perf_counter()

    # extract_shapelets lee parámetros directamente de config
    # Los sobreescribimos en memoria para el tipo correcto
    import config as _cfg
    if tipo == 'latido':
        _cfg.NUM_SHAPELETS_MICRO = n_shapelets
        _cfg.NUM_SHAPELETS_MACRO = 0
        _cfg.NUM_CANDIDATES      = n_candidates
        _cfg.NUM_COMPARACIONES   = n_comparaciones
        _cfg.RANGO_MICRO         = rango
    else:
        _cfg.NUM_SHAPELETS_MICRO = 0
        _cfg.NUM_SHAPELETS_MACRO = n_shapelets
        _cfg.NUM_CANDIDATES      = n_candidates
        _cfg.NUM_COMPARACIONES   = n_comparaciones
        _cfg.RANGO_MACRO         = rango

    micro_dict, macro_dict, micro_meta, macro_meta = extract_shapelets(X_tr, y_tr)

    # El diccionario activo según el tipo
    dict_activo = micro_dict if tipo == 'latido' else macro_dict

    if len(dict_activo) == 0:
        print(f"[ERROR] No se extrajeron shapelets para {tipo}.")
        return {}

    np.save(str(archivo_shap), np.array(dict_activo, dtype=object))
    print(f"[SHAPELETS] {len(dict_activo)} shapelets guardados → {archivo_shap}")
    tiempos['shapelets_' + tipo] = round(time.perf_counter() - t1, 2)

    # Graficar mejores shapelets (solo si el módulo está disponible)
    if _PLOTS_OK:
        try:
            plot_shapelets(
                micro_dict if tipo == 'latido' else [],
                [] if tipo == 'latido' else macro_dict,
                sufijo=f"_{sufijo}_MSM"
            )
        except Exception:
            pass

    # ── 4. Embedding ──────────────────────────────────────────────────────────
    print(f"\n[EMBEDDING] Generando matrices MSM-SC para {sufijo}...")
    t2 = time.perf_counter()

    df_tr  = fn_embedding(X_tr,  dict_activo, tag="TRAIN")
    df_te  = fn_embedding(X_te,  dict_activo, tag="TEST")
    df_val = fn_embedding(X_val, dict_activo, tag="VAL")

    tiempos['embedding_' + tipo] = round(time.perf_counter() - t2, 2)
    print(f"[EMBEDDING] {tiempos['embedding_' + tipo]:.1f}s")

    columnas = list(df_tr.columns)

    # Guardar CSVs de entrenamiento
    df_tr_csv = df_tr.copy()
    df_tr_csv['Etiqueta'] = y_tr
    df_tr_csv.to_csv(CARPETA_SALIDA / f"X_train_{sufijo}_MSM.csv", index=False)

    df_te_csv = df_te.copy()
    df_te_csv['Etiqueta'] = y_te
    df_te_csv.to_csv(CARPETA_SALIDA / f"X_test_{sufijo}_MSM.csv", index=False)

    df_val_csv = df_val.copy()
    df_val_csv['Etiqueta'] = y_val
    df_val_csv.to_csv(CARPETA_SALIDA / f"X_val_{sufijo}_MSM.csv", index=False)
    print(f"[EMBEDDING] CSVs guardados (train/test/val)")

    # ── 5. Entrenamiento XGBoost ──────────────────────────────────────────────
    print(f"\n[XGBOOST_{sufijo}] Entrenando...")
    t3 = time.perf_counter()

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_tr_enc  = le.fit_transform(y_tr)
    y_val_enc = le.transform(y_val)
    y_te_enc  = le.transform(y_te)

    import xgboost as xgb

    # Para 2 clases usamos binary:logistic que es más estable
    n_clases = len(le.classes_)
    if n_clases == 2:
        objetivo   = 'binary:logistic'
        eval_metric = 'logloss'
    else:
        objetivo   = 'multi:softprob'
        eval_metric = 'mlogloss'

    modelo = xgb.XGBClassifier(
        n_estimators          = xgb_params["n_estimators"],
        max_depth             = xgb_params["max_depth"],
        learning_rate         = xgb_params["learning_rate"],
        reg_alpha             = xgb_params["reg_alpha"],
        reg_lambda            = xgb_params["reg_lambda"],
        random_state          = xgb_params["random_state"],
        objective             = objetivo,
        eval_metric           = eval_metric,
        early_stopping_rounds = xgb_params["early_stopping_rounds"],
        n_jobs                = N_JOBS,
    )

    modelo.fit(
        df_tr, y_tr_enc,
        eval_set = [(df_val, y_val_enc)],
        verbose  = 50,
    )

    t_train = round(time.perf_counter() - t3, 2)
    print(f"[XGBOOST_{sufijo}] Entrenado en {t_train}s")

    # Guardar modelo
    np.save(str(archivo_encoder), le.classes_)
    modelo.save_model(str(archivo_modelo))
    print(f"[XGBOOST_{sufijo}] Modelo → {archivo_modelo}")

    # ── 6. Evaluación ─────────────────────────────────────────────────────────
    y_proba = modelo.predict_proba(df_te)
    clases  = list(le.classes_)

    # Para binario predict_proba devuelve (N,2), predict puede devolver (N,2)
    # Usamos argmax sobre probabilidades para garantizar consistencia
    if y_proba.ndim == 2:
        y_pred_enc_final = np.argmax(y_proba, axis=1)
    else:
        y_pred_enc_final = (y_proba >= 0.5).astype(int)

    # Reconstruir nombres de clases para métricas
    y_te_names   = le.inverse_transform(y_te_enc)
    y_pred_names = le.inverse_transform(y_pred_enc_final)

    from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                                  matthews_corrcoef, f1_score,
                                  classification_report, log_loss)

    acc    = accuracy_score(y_te_names, y_pred_names)
    kappa  = cohen_kappa_score(y_te_names, y_pred_names)
    mcc    = matthews_corrcoef(y_te_names, y_pred_names)
    f1mac  = f1_score(y_te_names, y_pred_names, average='macro')
    f1wei  = f1_score(y_te_names, y_pred_names, average='weighted')
    ll     = log_loss(y_te_enc, y_proba)

    print(f"\n{'─'*60}")
    print(f"  MODELO {sufijo}")
    print(f"{'─'*60}")
    print(f"  Accuracy    : {acc*100:.2f}%")
    print(f"  Kappa Cohen : {kappa:.4f}")
    print(f"  MCC         : {mcc:.4f}")
    print(f"  F1-macro    : {f1mac:.4f}")
    print(f"  Log-loss    : {ll:.4f}")
    print(f"\n{classification_report(y_te_names, y_pred_names)}")

    # ── 7. Gráficas ───────────────────────────────────────────────────────────
    metricas = {
        'accuracy': acc, 'kappa': kappa, 'mcc': mcc,
        'f1_macro': f1mac, 'f1_weighted': f1wei, 'log_loss': ll,
        'y_test': y_te_names, 'y_pred': y_pred_names,
        'y_proba': y_proba, 'clases': clases,
        'sufijo': f"_{sufijo}_MSM",
    }

    auc_m = 0.0
    if _PLOTS_OK:
        try:
            plot_confusion_matrix(metricas)
            aucs, auc_m = plot_roc_curves(metricas)
            plot_ground_truth_vs_pred(metricas)
            plot_feature_importance(modelo, columnas, sufijo=f"_{sufijo}_MSM")
            plot_learning_curve(df_tr, y_tr_enc, sufijo=f"_{sufijo}_MSM")
            metricas['auc_micro'] = auc_m
        except Exception as e:
            print(f"[PLOT] Error en gráficas (no crítico): {e}")
    else:
        print("[PLOT] Gráficas omitidas (módulo clasificador no disponible)")

    # ── 8. Vector LLM para el agente ──────────────────────────────────────────
    try:
        imp    = modelo.feature_importances_
        top5   = np.argsort(imp)[::-1][:5]
        vector = []
        for i in range(len(y_te_names)):
            cp    = str(y_pred_names[i])
            pr    = {clases[j]: round(float(y_proba[i, j]), 4)
                     for j in range(len(clases))}
            ps    = sorted(pr.values(), reverse=True)
            gap   = round(ps[0] - ps[1], 4)
            dists = {columnas[int(k)]: round(float(df_te.iloc[i, int(k)]), 4)
                     for k in top5 if int(k) < len(columnas)}
            vector.append({
                'id'                    : int(i),
                'tipo'                  : tipo,
                'clase_real'            : str(y_te_names[i]),
                'clase_predicha'        : cp,
                'correcto'              : bool(str(y_te_names[i]) == cp),
                'probabilidades'        : pr,
                'confidence_gap'        : gap,
                'alerta_baja_confianza' : bool(gap < 0.25),
                'distancias_top5_msm'   : dists,
            })
        ruta_vec = CARPETA_SALIDA / f"vector_LLM_{sufijo}_MSM.json"
        with open(ruta_vec, 'w', encoding='utf-8') as f:
            json.dump({'version': '4.0', 'tipo': tipo,
                       'predicciones_test': vector}, f,
                      ensure_ascii=False, indent=2)
        print(f"[VECTOR LLM] Guardado → {ruta_vec}")
    except Exception as e:
        print(f"[VECTOR LLM] Error: {e}")

    tiempos['train_' + tipo] = t_train

    return {
        'tipo'     : tipo,
        'accuracy' : acc,
        'kappa'    : kappa,
        'mcc'      : mcc,
        'f1_macro' : f1mac,
        'auc_micro': metricas.get('auc_micro', 0.0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Entrenamiento dual MSM-SC: modelo latido y modelo ritmo')
    parser.add_argument('--solo', choices=['latido', 'ritmo'],
                        help='Entrenar solo un modelo (omite el otro)')
    args = parser.parse_args()

    print("=" * 70)
    print("  ENTRENAMIENTO DUAL — MSM-SC + XGBoost")
    print("  Modelo LATIDO (NORMAL/PAC) + Modelo RITMO (NSR/AFIB)")
    print("=" * 70)

    tiempos  = {}
    resultados = {}

    # ── Modelo LATIDO ─────────────────────────────────────────────────────────
    if args.solo is None or args.solo == 'latido':
        res = entrenar_modelo('latido', tiempos)
        if res:
            resultados['latido'] = res

    # ── Modelo RITMO ──────────────────────────────────────────────────────────
    if args.solo is None or args.solo == 'ritmo':
        res = entrenar_modelo('ritmo', tiempos)
        if res:
            resultados['ritmo'] = res

    # ── Resumen final ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  ENTRENAMIENTO DUAL COMPLETO")
    print(f"{'='*70}")

    archivos_ok = [
        ("datos/shapelets_latido_MSM.npy",           ARCHIVO_SHAPELETS_LATIDO),
        ("datos/xgboost_latido_MSM.json",             ARCHIVO_MODELO_LATIDO),
        ("datos/label_encoder_latido_MSM.npy",        ARCHIVO_ENCODER_LATIDO),
        ("datos/shapelets_ritmo_MSM.npy",             ARCHIVO_SHAPELETS_RITMO),
        ("datos/xgboost_ritmo_MSM.json",              ARCHIVO_MODELO_RITMO),
        ("datos/label_encoder_ritmo_MSM.npy",         ARCHIVO_ENCODER_RITMO),
        ("datos/resultados/X_train_LATIDO_MSM.csv",   CARPETA_SALIDA/"X_train_LATIDO_MSM.csv"),
        ("datos/resultados/X_test_LATIDO_MSM.csv",    CARPETA_SALIDA/"X_test_LATIDO_MSM.csv"),
        ("datos/resultados/X_train_RITMO_MSM.csv",    CARPETA_SALIDA/"X_train_RITMO_MSM.csv"),
        ("datos/resultados/X_test_RITMO_MSM.csv",     CARPETA_SALIDA/"X_test_RITMO_MSM.csv"),
        ("datos/resultados/vector_LLM_LATIDO_MSM.json", CARPETA_SALIDA/"vector_LLM_LATIDO_MSM.json"),
        ("datos/resultados/vector_LLM_RITMO_MSM.json",  CARPETA_SALIDA/"vector_LLM_RITMO_MSM.json"),
    ]

    for nombre, ruta in archivos_ok:
        estado = "[OK]" if Path(ruta).exists() else "[--]"
        print(f"  {estado}  {nombre}")

    print(f"\n{'─'*50}")
    for tipo, res in resultados.items():
        print(f"  Modelo {tipo.upper():8}")
        print(f"    Accuracy    : {res['accuracy']*100:.2f}%")
        print(f"    Kappa Cohen : {res['kappa']:.4f}")
        print(f"    MCC         : {res['mcc']:.4f}")
        print(f"    F1-macro    : {res['f1_macro']:.4f}")
        print(f"    AUC micro   : {res['auc_micro']:.4f}")
        print()

    t_total = sum(tiempos.values())
    print(f"  Tiempo total  : {t_total/3600:.1f} horas")
    print("=" * 70)


if __name__ == "__main__":
    main()