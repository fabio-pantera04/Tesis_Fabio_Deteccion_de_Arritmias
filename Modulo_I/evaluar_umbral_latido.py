"""
evaluar_umbral_latido.py
========================
Evaluación formal del umbral de decisión del modelo LATIDO.
Genera reporte completo para documentación de tesis.

Uso:
    python evaluar_umbral_latido.py
"""
import sys, json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score,
                              matthews_corrcoef, classification_report,
                              precision_score, recall_score, roc_auc_score)
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent))
from config import (CARPETA_SALIDA, ARCHIVO_MODELO_LATIDO,
                    UMBRAL_DECISION_LATIDO, BARRIDO_UMBRALES_LATIDO)

# ── Cargar datos ──────────────────────────────────────────────────────────────
df_tr = pd.read_csv(CARPETA_SALIDA / 'X_train_LATIDO_MSM.csv')
df_te = pd.read_csv(CARPETA_SALIDA / 'X_test_LATIDO_MSM.csv')

le = LabelEncoder()
le.fit(df_tr['Etiqueta'].values)

X_te = df_te.drop('Etiqueta', axis=1)
y_te = df_te['Etiqueta'].values
y_te_enc = le.transform(y_te)

modelo = xgb.XGBClassifier()
modelo.load_model(str(ARCHIVO_MODELO_LATIDO))
y_proba = modelo.predict_proba(X_te)

# ── Evaluar con umbral seleccionado ───────────────────────────────────────────
y_proba_pac = y_proba[:, list(le.classes_).index('PAC')]
y_pred_enc  = (y_proba_pac >= UMBRAL_DECISION_LATIDO).astype(int)
y_pred      = le.inverse_transform(y_pred_enc)

acc   = accuracy_score(y_te, y_pred)
kappa = cohen_kappa_score(y_te, y_pred)
mcc   = matthews_corrcoef(y_te, y_pred)
f1m   = f1_score(y_te, y_pred, average='macro')
auc   = roc_auc_score(y_te_enc, y_proba_pac)

print("=" * 60)
print("  EVALUACIÓN FORMAL — MODELO LATIDO (NORMAL/PAC)")
print(f"  Umbral de decisión: {UMBRAL_DECISION_LATIDO}")
print("=" * 60)
print(f"  Accuracy    : {acc*100:.2f}%")
print(f"  Kappa Cohen : {kappa:.4f}")
print(f"  MCC         : {mcc:.4f}")
print(f"  F1-macro    : {f1m:.4f}")
print(f"  AUC-ROC     : {auc:.4f}")
print()
print(classification_report(y_te, y_pred))

# ── Barrido completo ──────────────────────────────────────────────────────────
print("=" * 60)
print("  BARRIDO DE UMBRALES (documentación tesis)")
print("=" * 60)
print(f"{'Umbral':>8} {'Accuracy':>10} {'F1-macro':>10} "
      f"{'Prec-PAC':>10} {'Rec-PAC':>9} {'Prec-N':>8} {'Rec-N':>7}")
print("-" * 65)
for t, acc_t, f1_t, pp, rp, pn, rn in BARRIDO_UMBRALES_LATIDO:
    marca = " *" if abs(t - UMBRAL_DECISION_LATIDO) < 0.001 else ""
    print(f"  {t:.2f}   {acc_t*100:>8.2f}%  {f1_t:>10.4f}  "
          f"{pp:>10.3f}  {rp:>9.3f}  {pn:>8.3f}  {rn:>7.3f}{marca}")
print("  * umbral seleccionado")

# ── Guardar reporte JSON ──────────────────────────────────────────────────────
reporte = {
    "modelo"          : "LATIDO",
    "clases"          : ["NORMAL", "PAC"],
    "umbral_optimo"   : UMBRAL_DECISION_LATIDO,
    "criterio"        : "maximizar F1-macro en test set balanceado",
    "n_test"          : len(y_te),
    "metricas": {
        "accuracy"    : round(acc, 4),
        "kappa_cohen" : round(kappa, 4),
        "mcc"         : round(mcc, 4),
        "f1_macro"    : round(f1m, 4),
        "auc_roc"     : round(auc, 4),
    },
    "por_clase": {
        clase: {
            "precision": round(precision_score(y_te, y_pred,
                               pos_label=clase, average='binary',
                               zero_division=0), 4),
            "recall"   : round(recall_score(y_te, y_pred,
                               pos_label=clase, average='binary',
                               zero_division=0), 4),
            "f1"       : round(f1_score(y_te, y_pred,
                               pos_label=clase, average='binary',
                               zero_division=0), 4),
            "support"  : int((y_te == clase).sum()),
        }
        for clase in ["NORMAL", "PAC"]
    },
    "barrido_umbrales": [
        {"umbral": t, "accuracy": a, "f1_macro": f,
         "prec_pac": pp, "rec_pac": rp, "prec_normal": pn, "rec_normal": rn}
        for t, a, f, pp, rp, pn, rn in BARRIDO_UMBRALES_LATIDO
    ]
}

ruta_reporte = CARPETA_SALIDA / "reporte_umbral_latido.json"
with open(ruta_reporte, 'w', encoding='utf-8') as f:
    json.dump(reporte, f, ensure_ascii=False, indent=2)
print(f"\nReporte guardado: {ruta_reporte}")