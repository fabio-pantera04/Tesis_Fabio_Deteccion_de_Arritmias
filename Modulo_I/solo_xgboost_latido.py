import sys, numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score, matthews_corrcoef, f1_score, classification_report
import xgboost as xgb

sys.path.insert(0, r"C:\Modelo_Tesis\Modulo_I")
from config import CARPETA_SALIDA, ARCHIVO_MODELO_LATIDO, ARCHIVO_ENCODER_LATIDO, XGB_LATIDO, N_JOBS, XGB_RANDOM_STATE

print("Cargando embeddings LATIDO desde CSV...")
df_tr = pd.read_csv(CARPETA_SALIDA / "X_train_LATIDO_MSM.csv")
df_te = pd.read_csv(CARPETA_SALIDA / "X_test_LATIDO_MSM.csv")

y_tr_all = df_tr["Etiqueta"].values
y_te     = df_te["Etiqueta"].values
X_tr_all = df_tr.drop("Etiqueta", axis=1)
X_te     = df_te.drop("Etiqueta", axis=1)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_tr_all, y_tr_all, test_size=0.15, stratify=y_tr_all, random_state=XGB_RANDOM_STATE)

print(f"Train={len(y_tr)} | Val={len(y_val)} | Test={len(y_te)}")

le = LabelEncoder()
y_tr_enc  = le.fit_transform(y_tr)
y_val_enc = le.transform(y_val)
y_te_enc  = le.transform(y_te)
print(f"Clases: {list(le.classes_)} → {list(range(len(le.classes_)))}")

modelo = xgb.XGBClassifier(
    n_estimators          = XGB_LATIDO["n_estimators"],
    max_depth             = XGB_LATIDO["max_depth"],
    learning_rate         = XGB_LATIDO["learning_rate"],
    reg_alpha             = XGB_LATIDO["reg_alpha"],
    reg_lambda            = XGB_LATIDO["reg_lambda"],
    random_state          = XGB_LATIDO["random_state"],
    early_stopping_rounds = XGB_LATIDO["early_stopping_rounds"],
    n_jobs                = N_JOBS,
    num_class             = len(le.classes_),
    objective             = "multi:softprob",
)

print("Entrenando XGBoost LATIDO...")
modelo.fit(X_tr, y_tr_enc,
           eval_set=[(X_val, y_val_enc)],
           verbose=50)

np.save(str(ARCHIVO_ENCODER_LATIDO), le.classes_)
modelo.save_model(str(ARCHIVO_MODELO_LATIDO))
print(f"Modelo guardado: {ARCHIVO_MODELO_LATIDO}")

y_pred_enc = modelo.predict(X_te)
y_pred     = le.inverse_transform(y_pred_enc)

print(f"\n{'='*55}")
print(f"  MODELO LATIDO — RESULTADOS FINALES")
print(f"{'='*55}")
print(f"  Accuracy  : {accuracy_score(y_te, y_pred)*100:.2f}%")
print(f"  Kappa     : {cohen_kappa_score(y_te, y_pred):.4f}")
print(f"  MCC       : {matthews_corrcoef(y_te, y_pred):.4f}")
print(f"  F1-macro  : {f1_score(y_te, y_pred, average='macro'):.4f}")
print(f"\n{classification_report(y_te, y_pred)}")
