import sys, time, json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
    matthews_corrcoef, f1_score, classification_report, log_loss)
import xgboost as xgb
from joblib import Parallel, delayed

sys.path.insert(0, r"C:\Modelo_Tesis\Modulo_I")
from config import (
    CARPETA_SALIDA, ARCHIVO_SHAPELETS_RITMO,
    ARCHIVO_MODELO_RITMO, ARCHIVO_ENCODER_RITMO,
    FUENTES_POR_CLASE, N_JOBS, XGB_RITMO, XGB_RANDOM_STATE,
    PASO_VENTANA_RITMO
)
from modulo_1_clasificador.lectura_senal import normalizar_zscore
from modulo_1_clasificador.msm_sc import msm_sc

import random

def cargar_ritmo():
    X, y = [], []
    for clase in ["NSR", "AFIB"]:
        fuentes = FUENTES_POR_CLASE.get(clase, [])
        cargados = 0
        for ruta, n_max in fuentes:
            ruta = Path(ruta)
            if not ruta.exists():
                print(f"  [AVISO] No existe: {ruta}")
                continue
            archivos = list(ruta.glob("*.csv"))
            random.shuffle(archivos)
            n_fuente = 0
            for archivo in archivos:
                if n_fuente >= n_max:
                    break
                try:
                    vals = pd.read_csv(archivo, header=None).iloc[:,0]
                    senal = pd.to_numeric(vals, errors="coerce").dropna().values.astype(np.float64)
                    if len(senal) < 10:
                        continue
                    senal = normalizar_zscore(senal)
                    X.append(senal)
                    y.append(clase)
                    n_fuente += 1
                    cargados += 1
                except Exception:
                    continue
            print(f"  [{clase:6}] {ruta.name:30}: {n_fuente:4d}")
        print(f"  [{clase:6}] TOTAL: {cargados}")
    return np.array(X, dtype=object), np.array(y)

def dist_min(senal, shapelet, paso):
    L = len(shapelet)
    if len(senal) < L:
        return float(msm_sc(shapelet, senal))
    mejor = np.inf
    for s in range(0, len(senal) - L + 1, paso):
        d = msm_sc(shapelet, senal[s:s+L], best_so_far=mejor)
        if d < mejor:
            mejor = d
    return float(mejor)

def embedding(X, shapelets, tag):
    print(f"  [{tag}] {len(X)} señales x {len(shapelets)} shapelets macro, paso={PASO_VENTANA_RITMO}...")
    def fila(sig):
        return [dist_min(sig, s, PASO_VENTANA_RITMO) for s in shapelets]
    filas = Parallel(n_jobs=N_JOBS, backend="loky", verbose=5)(
        delayed(fila)(sig) for sig in X)
    df = pd.DataFrame(filas, columns=[f"Macro_S{i+1}" for i in range(len(shapelets))])
    n_inf = int((~np.isfinite(df.values)).sum())
    if n_inf > 0:
        for col in df.columns:
            mask = ~np.isfinite(df[col])
            mx = df[col][np.isfinite(df[col])].max()
            df.loc[mask, col] = float(mx*2) if np.isfinite(mx) else 0.0
        print(f"    {n_inf} inf reemplazados")
    else:
        print(f"    OK sin inf/nan")
    return df

print("="*60)
print("  EMBEDDING + XGBOOST — MODELO RITMO (NSR/AFIB)")
print("="*60)

# 1. Cargar shapelets guardados
shapelets = list(np.load(str(ARCHIVO_SHAPELETS_RITMO), allow_pickle=True))
print(f"\nShapelets RITMO cargados: {len(shapelets)}")
print(f"Longitud shapelet 0  : {len(shapelets[0])} muestras")
print(f"Longitud shapelet 30 : {len(shapelets[30])} muestras")

# 2. Cargar datos
print("\n[DATOS] Cargando NSR y AFIB...")
X, y = cargar_ritmo()
print(f"[DATOS] {len(X)} senales | Clases: {sorted(set(y))}")

# 3. Split
X_tr_val, X_te, y_tr_val, y_te = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=XGB_RANDOM_STATE)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_tr_val, y_tr_val, test_size=0.15,
    stratify=y_tr_val, random_state=XGB_RANDOM_STATE)
print(f"[SPLIT] Train={len(X_tr)} | Val={len(X_val)} | Test={len(X_te)}")

# 4. Embedding
print("\n[EMBEDDING] Generando matrices MSM-SC...")
t0 = time.perf_counter()
df_tr  = embedding(X_tr,  shapelets, "TRAIN")
df_te  = embedding(X_te,  shapelets, "TEST")
df_val = embedding(X_val, shapelets, "VAL")
print(f"[EMBEDDING] {(time.perf_counter()-t0)/3600:.2f} horas")

# Guardar CSVs
df_tr_csv = df_tr.copy(); df_tr_csv["Etiqueta"] = y_tr
df_te_csv = df_te.copy(); df_te_csv["Etiqueta"] = y_te
df_val_csv = df_val.copy(); df_val_csv["Etiqueta"] = y_val
df_tr_csv.to_csv(CARPETA_SALIDA / "X_train_RITMO_MSM.csv", index=False)
df_te_csv.to_csv(CARPETA_SALIDA / "X_test_RITMO_MSM.csv",  index=False)
df_val_csv.to_csv(CARPETA_SALIDA / "X_val_RITMO_MSM.csv",  index=False)
print("[EMBEDDING] CSVs guardados")

columnas = list(df_tr.columns)

# 5. XGBoost
print("\n[XGBOOST] Entrenando modelo RITMO...")
le = LabelEncoder()
y_tr_enc  = le.fit_transform(y_tr)
y_val_enc = le.transform(y_val)
y_te_enc  = le.transform(y_te)
print(f"Clases: {list(le.classes_)}")

modelo = xgb.XGBClassifier(
    n_estimators          = XGB_RITMO["n_estimators"],
    max_depth             = XGB_RITMO["max_depth"],
    learning_rate         = XGB_RITMO["learning_rate"],
    reg_alpha             = XGB_RITMO["reg_alpha"],
    reg_lambda            = XGB_RITMO["reg_lambda"],
    random_state          = XGB_RITMO["random_state"],
    objective             = "binary:logistic",
    eval_metric           = "logloss",
    early_stopping_rounds = XGB_RITMO["early_stopping_rounds"],
    n_jobs                = N_JOBS,
)
modelo.fit(df_tr, y_tr_enc, eval_set=[(df_val, y_val_enc)], verbose=50)

np.save(str(ARCHIVO_ENCODER_RITMO), le.classes_)
modelo.save_model(str(ARCHIVO_MODELO_RITMO))
print(f"Modelo guardado -> {ARCHIVO_MODELO_RITMO}")

# 6. Metricas
y_proba = modelo.predict_proba(df_te)
y_pred  = le.inverse_transform(np.argmax(y_proba, axis=1))

print(f"\n{'='*55}")
print(f"  MODELO RITMO — RESULTADOS FINALES")
print(f"{'='*55}")
print(f"  Accuracy  : {accuracy_score(y_te, y_pred)*100:.2f}%")
print(f"  Kappa     : {cohen_kappa_score(y_te, y_pred):.4f}")
print(f"  MCC       : {matthews_corrcoef(y_te, y_pred):.4f}")
print(f"  F1-macro  : {f1_score(y_te, y_pred, average='macro'):.4f}")
print(f"  Log-loss  : {log_loss(y_te_enc, y_proba):.4f}")
print(classification_report(y_te, y_pred))

# 7. Vector LLM
imp  = modelo.feature_importances_
top5 = np.argsort(imp)[::-1][:5]
vector = []
for i in range(len(y_te)):
    pr  = {list(le.classes_)[j]: round(float(y_proba[i,j]),4) for j in range(2)}
    ps  = sorted(pr.values(), reverse=True)
    gap = round(ps[0]-ps[1], 4)
    dists = {columnas[int(k)]: round(float(df_te.iloc[i,int(k)]),4) for k in top5}
    vector.append({
        "id": i, "tipo": "ritmo",
        "clase_real": str(y_te[i]), "clase_predicha": str(y_pred[i]),
        "correcto": bool(y_te[i]==y_pred[i]),
        "probabilidades": pr, "confidence_gap": gap,
        "alerta_baja_confianza": bool(gap < 0.25),
        "distancias_top5_msm": dists
    })
ruta_vec = CARPETA_SALIDA / "vector_LLM_RITMO_MSM.json"
with open(ruta_vec, "w", encoding="utf-8") as f:
    json.dump({"version":"4.0","tipo":"ritmo","predicciones_test":vector},
              f, ensure_ascii=False, indent=2)
print(f"Vector LLM guardado -> {ruta_vec}")
print("\nListo. Modelo RITMO completo.")
