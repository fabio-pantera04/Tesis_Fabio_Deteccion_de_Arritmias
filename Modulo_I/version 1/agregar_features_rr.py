"""calcular shapelets ni distancias MSM.

Las features RR capturan el contexto temporal entre latidos, que es
información clínicamente definitoria de la PAC pero ausente en el
embedding morfológico puro:

  rr_prev  : intervalo en ms al latido anterior
  rr_next  : intervalo en ms al latido siguiente
  rr_ratio : rr_prev / rr_next  — PAC < 0.85 por definición (latido prematuro)
  coupling : rr_prev / media_RR_local — índice de acoplamiento

Justificación (AAMI EC57:2012):
  La PAC se caracteriza por un intervalo de acoplamiento corto seguido
  de una pausa compensatoria. Estas features son ortogonales a la
  morfología MSM y se espera una mejora de +5 a +8pp en accuracy.

Tiempo estimado: ~15 minutos (solo XGBoost, sin embedding MSM)

Uso:
    python agregar_features_rr.py

Salida:
    datos/resultados/X_train_LATIDO_RR.csv   — embedding + RR
    datos/resultados/X_test_LATIDO_RR.csv
    datos/resultados/X_val_LATIDO_RR.csv
    datos/xgboost_latido_RR.json             — modelo actualizado
    datos/resultados/reporte_latido_rr.json  — métricas formales
"""

import sys, json, time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                              matthews_corrcoef, f1_score,
                              classification_report, roc_auc_score,
                              precision_score, recall_score)
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    CARPETA_DATOS, CARPETA_SALIDA,
    ARCHIVO_MODELO_LATIDO, ARCHIVO_ENCODER_LATIDO,
    FUENTES_POR_CLASE, XGB_LATIDO, XGB_RANDOM_STATE, N_JOBS,
)

FS = 250.0   # Hz — frecuencia de muestreo Icentia


# ─────────────────────────────────────────────────────────────────────────────
# CÁLCULO DE FEATURES RR
# ─────────────────────────────────────────────────────────────────────────────

def calcular_rr(senal: np.ndarray, fs: float = FS) -> dict:
    """
    Calcula las 4 features de intervalo RR para un latido de 250 muestras.

    La señal de 250 muestras contiene el contexto completo:
      - muestras 0-124  : latido anterior (pausa compensatoria previa)
      - muestra ~125    : pico R del latido actual
      - muestras 125-249: latido siguiente (inicio del próximo ciclo)

    Para señales donde find_peaks no detecta suficientes picos,
    se asignan valores neutros (rr_ratio=1.0, coupling=1.0).
    """
    ms = 1000.0 / fs

    # Detectar picos R en la señal completa de 250 muestras
    # distancia mínima: 0.3s = 75 muestras (200 bpm máximo)
    # altura mínima: 0.5 std (señal normalizada Z-score)
    peaks, _ = find_peaks(senal, distance=int(0.3 * fs), height=0.3)

    n = len(peaks)
    if n < 2:
        # Sin suficientes picos — valores neutros
        return {
            'rr_prev'  : 800.0,
            'rr_next'  : 800.0,
            'rr_ratio' : 1.0,
            'coupling' : 1.0,
        }

    # Pico central: el más cercano al centro de la ventana (muestra 125)
    centro = 125
    idx_central = np.argmin(np.abs(peaks - centro))

    # rr_prev: distancia al pico anterior en ms
    if idx_central > 0:
        rr_prev = (peaks[idx_central] - peaks[idx_central - 1]) * ms
    else:
        rr_prev = 800.0

    # rr_next: distancia al pico siguiente en ms
    if idx_central < n - 1:
        rr_next = (peaks[idx_central + 1] - peaks[idx_central]) * ms
    else:
        rr_next = 800.0

    # Media local RR: promedio de los intervalos vecinos (excluye el actual)
    intervalos = [(peaks[k+1] - peaks[k]) * ms for k in range(n-1)]
    rr_media = np.mean(intervalos) if intervalos else 800.0

    # rr_ratio: < 0.85 indica PAC (latido prematuro + pausa compensatoria)
    rr_ratio = (rr_prev / rr_next) if rr_next > 1.0 else 1.0

    # coupling: < 1.0 indica latido prematuro
    coupling = (rr_prev / rr_media) if rr_media > 1.0 else 1.0

    return {
        'rr_prev'  : round(float(rr_prev),  2),
        'rr_next'  : round(float(rr_next),  2),
        'rr_ratio' : round(float(rr_ratio), 4),
        'coupling' : round(float(coupling), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CARGA ORDENADA DE SEÑALES Y CÁLCULO DE FEATURES RR
# ─────────────────────────────────────────────────────────────────────────────

def cargar_señales_ordenadas(clases: list) -> tuple:
    """
    Carga señales en orden ALFABÉTICO fijo (sin shuffle) para reproducibilidad.
    Retorna señales, etiquetas y features RR en el mismo orden.
    """
    X, y, rr_list = [], [], []

    for clase in clases:
        fuentes = FUENTES_POR_CLASE.get(clase, [])
        for ruta, n_max in fuentes:
            ruta = Path(ruta)
            if not ruta.exists():
                print(f"  [AVISO] Ruta no encontrada: {ruta}")
                continue

            # ORDEN FIJO: sorted() alfabético — sin shuffle
            archivos = sorted(ruta.glob("*.csv"))[:n_max]
            cargados = 0

            for archivo in archivos:
                try:
                    vals = pd.read_csv(archivo, header=None).iloc[:, 0]
                    senal = pd.to_numeric(vals, errors='coerce').dropna().values
                    senal = senal.astype(np.float64)
                    if len(senal) < 50:
                        continue

                    rr = calcular_rr(senal)
                    X.append(senal)
                    y.append(clase)
                    rr_list.append(rr)
                    cargados += 1
                except Exception:
                    continue

            print(f"  [{clase:8}] {ruta.name:35}: {cargados:4d} señales")

    df_rr = pd.DataFrame(rr_list)
    return np.array(X, dtype=object), np.array(y), df_rr


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  FEATURES RR — MODELO LATIDO")
    print("  Agrega rr_prev, rr_next, rr_ratio, coupling al embedding")
    print("=" * 65)

    # ── 1. Cargar señales en orden fijo y calcular features RR ───────────────
    print("\n[1/5] Cargando señales NORMAL y PAC en orden fijo...")
    t0 = time.perf_counter()
    X, y, df_rr = cargar_señales_ordenadas(['NORMAL', 'PAC'])
    print(f"\n  Total señales: {len(X)}")
    print(f"  Features RR calculadas: {len(df_rr)}")
    print(f"  Tiempo: {time.perf_counter()-t0:.1f}s")

    # Diagnóstico: verificar que rr_ratio discrimina las clases
    print("\n  Diagnóstico rr_ratio por clase:")
    for clase in ['NORMAL', 'PAC']:
        mask = y == clase
        vals = df_rr.loc[mask, 'rr_ratio']
        print(f"    {clase:8}: media={vals.mean():.3f}  "
              f"std={vals.std():.3f}  "
              f"<0.85: {(vals < 0.85).sum()} ({(vals < 0.85).mean()*100:.1f}%)")

    # ── 2. Split con misma semilla que el entrenamiento original ─────────────
    print("\n[2/5] Aplicando split estratificado (seed=42)...")
    indices = np.arange(len(X))
    idx_trval, idx_te = train_test_split(
        indices, test_size=0.30, stratify=y,
        random_state=XGB_RANDOM_STATE)
    idx_tr, idx_val = train_test_split(
        idx_trval, test_size=0.15, stratify=y[idx_trval],
        random_state=XGB_RANDOM_STATE)

    print(f"  Train={len(idx_tr)} | Val={len(idx_val)} | Test={len(idx_te)}")

    y_tr  = y[idx_tr]
    y_val = y[idx_val]
    y_te  = y[idx_te]

    rr_tr  = df_rr.iloc[idx_tr].reset_index(drop=True)
    rr_val = df_rr.iloc[idx_val].reset_index(drop=True)
    rr_te  = df_rr.iloc[idx_te].reset_index(drop=True)

    # ── 3. Cargar embedding MSM existente y concatenar features RR ───────────
    print("\n[3/5] Cargando embedding MSM y concatenando features RR...")

    df_tr_msm  = pd.read_csv(CARPETA_SALIDA / 'X_train_LATIDO_MSM.csv')
    df_te_msm  = pd.read_csv(CARPETA_SALIDA / 'X_test_LATIDO_MSM.csv')
    df_val_msm = pd.read_csv(CARPETA_SALIDA / 'X_val_LATIDO_MSM.csv')

    # Verificar que los tamaños coinciden
    assert len(df_tr_msm) == len(rr_tr), \
        f"Desajuste train: MSM={len(df_tr_msm)} RR={len(rr_tr)}"
    assert len(df_te_msm) == len(rr_te), \
        f"Desajuste test: MSM={len(df_te_msm)} RR={len(rr_te)}"
    assert len(df_val_msm) == len(rr_val), \
        f"Desajuste val: MSM={len(df_val_msm)} RR={len(rr_val)}"

    # Separar etiquetas del MSM
    y_tr_msm  = df_tr_msm.pop('Etiqueta').values
    y_te_msm  = df_te_msm.pop('Etiqueta').values
    y_val_msm = df_val_msm.pop('Etiqueta').values

    # Verificar consistencia de etiquetas
    if not np.array_equal(y_tr_msm, y_tr):
        print("  [AVISO] Etiquetas train no coinciden exactamente.")
        print("          Usando etiquetas del split actual.")

    # Concatenar features RR
    X_tr  = pd.concat([df_tr_msm.reset_index(drop=True),  rr_tr],  axis=1)
    X_te  = pd.concat([df_te_msm.reset_index(drop=True),  rr_te],  axis=1)
    X_val = pd.concat([df_val_msm.reset_index(drop=True), rr_val], axis=1)

    print(f"  Features por señal: {X_tr.shape[1]} "
          f"(60 MSM + 4 RR)")

    # Guardar CSVs actualizados
    X_tr_csv = X_tr.copy();  X_tr_csv['Etiqueta']  = y_tr
    X_te_csv = X_te.copy();  X_te_csv['Etiqueta']  = y_te
    X_val_csv = X_val.copy(); X_val_csv['Etiqueta'] = y_val

    X_tr_csv.to_csv(CARPETA_SALIDA / 'X_train_LATIDO_RR.csv', index=False)
    X_te_csv.to_csv(CARPETA_SALIDA / 'X_test_LATIDO_RR.csv',  index=False)
    X_val_csv.to_csv(CARPETA_SALIDA / 'X_val_LATIDO_RR.csv',  index=False)
    print("  CSVs con RR guardados (X_*_LATIDO_RR.csv)")

    # ── 4. Reentrenar XGBoost con las nuevas features ─────────────────────────
    print("\n[4/5] Reentrenando XGBoost con 64 features (60 MSM + 4 RR)...")

    le = LabelEncoder()
    y_tr_enc  = le.fit_transform(y_tr)
    y_val_enc = le.transform(y_val)
    y_te_enc  = le.transform(y_te)

    modelo = xgb.XGBClassifier(
        n_estimators          = XGB_LATIDO["n_estimators"],
        max_depth             = XGB_LATIDO["max_depth"],
        learning_rate         = XGB_LATIDO["learning_rate"],
        reg_alpha             = XGB_LATIDO["reg_alpha"],
        reg_lambda            = XGB_LATIDO["reg_lambda"],
        random_state          = XGB_LATIDO["random_state"],
        objective             = "binary:logistic",
        eval_metric           = "logloss",
        early_stopping_rounds = XGB_LATIDO["early_stopping_rounds"],
        n_jobs                = N_JOBS,
    )

    modelo.fit(X_tr, y_tr_enc,
               eval_set=[(X_val, y_val_enc)],
               verbose=50)

    # Guardar modelo con RR
    ruta_modelo_rr  = CARPETA_DATOS / "xgboost_latido_RR.json"
    ruta_encoder_rr = CARPETA_DATOS / "label_encoder_latido_RR.npy"
    np.save(str(ruta_encoder_rr), le.classes_)
    modelo.save_model(str(ruta_modelo_rr))
    print(f"  Modelo guardado: {ruta_modelo_rr}")

    # ── 5. Evaluación y reporte ───────────────────────────────────────────────
    print("\n[5/5] Evaluando modelo con features RR...")

    y_proba = modelo.predict_proba(X_te)
    idx_pac = list(le.classes_).index('PAC')
    y_pred  = le.inverse_transform(np.argmax(y_proba, axis=1))

    acc   = accuracy_score(y_te, y_pred)
    kappa = cohen_kappa_score(y_te, y_pred)
    mcc   = matthews_corrcoef(y_te, y_pred)
    f1m   = f1_score(y_te, y_pred, average='macro')
    auc   = roc_auc_score(y_te_enc, y_proba[:, idx_pac])

    # Importancia de features RR vs MSM
    imp       = modelo.feature_importances_
    columnas  = list(X_tr.columns)
    rr_cols   = ['rr_prev', 'rr_next', 'rr_ratio', 'coupling']
    imp_rr    = sum(imp[columnas.index(c)] for c in rr_cols if c in columnas)
    imp_msm   = 1.0 - imp_rr

    print(f"\n{'='*55}")
    print(f"  MODELO LATIDO + FEATURES RR — RESULTADOS")
    print(f"{'='*55}")
    print(f"  Accuracy    : {acc*100:.2f}%")
    print(f"  Kappa Cohen : {kappa:.4f}")
    print(f"  MCC         : {mcc:.4f}")
    print(f"  F1-macro    : {f1m:.4f}")
    print(f"  AUC-ROC     : {auc:.4f}")
    print(f"\n  Importancia features:")
    print(f"    MSM (60 distancias) : {imp_msm*100:.1f}%")
    print(f"    RR  (4 features)    : {imp_rr*100:.1f}%")
    print()
    print(classification_report(y_te, y_pred))

    # Comparativa vs modelo sin RR
    print(f"{'─'*55}")
    print(f"  Comparativa: sin RR vs con RR")
    print(f"{'─'*55}")
    print(f"  Accuracy    : 81.17%  →  {acc*100:.2f}%  "
          f"({'+'if acc>0.8117 else ''}{(acc-0.8117)*100:.2f}pp)")
    print(f"  Kappa       : 0.6233  →  {kappa:.4f}  "
          f"({'+'if kappa>0.6233 else ''}{kappa-0.6233:.4f})")
    print(f"  MCC         : 0.6252  →  {mcc:.4f}  "
          f"({'+'if mcc>0.6252 else ''}{mcc-0.6252:.4f})")
    print(f"  AUC-ROC     : 0.8781  →  {auc:.4f}  "
          f"({'+'if auc>0.8781 else ''}{auc-0.8781:.4f})")

    # Guardar reporte JSON
    reporte = {
        "modelo"              : "LATIDO_RR",
        "descripcion"         : "MSM-SC + XGBoost con features RR adicionales",
        "clases"              : ["NORMAL", "PAC"],
        "features"            : {
            "msm_distancias"  : 60,
            "rr_features"     : 4,
            "total"           : 64,
        },
        "features_rr": {
            "rr_prev"  : "intervalo al latido anterior (ms)",
            "rr_next"  : "intervalo al latido siguiente (ms)",
            "rr_ratio" : "rr_prev/rr_next — PAC < 0.85",
            "coupling" : "rr_prev/media_local_RR",
        },
        "importancia_features": {
            "msm_pct" : round(float(imp_msm * 100), 2),
            "rr_pct"  : round(float(imp_rr  * 100), 2),
        },
        "metricas": {
            "accuracy"    : round(acc,   4),
            "kappa_cohen" : round(kappa, 4),
            "mcc"         : round(mcc,   4),
            "f1_macro"    : round(f1m,   4),
            "auc_roc"     : round(auc,   4),
        },
        "por_clase": {
            clase: {
                "precision": round(precision_score(
                    y_te, y_pred, pos_label=clase,
                    average='binary', zero_division=0), 4),
                "recall"   : round(recall_score(
                    y_te, y_pred, pos_label=clase,
                    average='binary', zero_division=0), 4),
                "f1"       : round(f1_score(
                    y_te, y_pred, pos_label=clase,
                    average='binary', zero_division=0), 4),
                "support"  : int((y_te == clase).sum()),
            }
            for clase in ["NORMAL", "PAC"]
        },
        "comparativa_sin_rr": {
            "accuracy_sin_rr" : 0.8117,
            "kappa_sin_rr"    : 0.6233,
            "mcc_sin_rr"      : 0.6252,
            "auc_sin_rr"      : 0.8781,
            "mejora_accuracy" : round(acc   - 0.8117, 4),
            "mejora_kappa"    : round(kappa - 0.6233, 4),
            "mejora_mcc"      : round(mcc   - 0.6252, 4),
            "mejora_auc"      : round(auc   - 0.8781, 4),
        },
        "archivos_generados": {
            "modelo"          : str(ruta_modelo_rr),
            "encoder"         : str(ruta_encoder_rr),
            "train_csv"       : str(CARPETA_SALIDA / 'X_train_LATIDO_RR.csv'),
            "test_csv"        : str(CARPETA_SALIDA / 'X_test_LATIDO_RR.csv'),
            "val_csv"         : str(CARPETA_SALIDA / 'X_val_LATIDO_RR.csv'),
        }
    }

    ruta_reporte = CARPETA_SALIDA / "reporte_latido_rr.json"
    with open(ruta_reporte, 'w', encoding='utf-8') as f:
        json.dump(reporte, f, ensure_ascii=False, indent=2)
    print(f"\n  Reporte guardado: {ruta_reporte}")
    print("=" * 55)


if __name__ == "__main__":
    main()
