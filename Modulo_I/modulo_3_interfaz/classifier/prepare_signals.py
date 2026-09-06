"""
modulo_3_interfaz/classifier/prepare_signals.py

Procesa una senal ECG cruda (CSV de 1 columna, 60 s a 250 Hz, ya
normalizada con z-score) y emite el JSON que consume la interfaz Flask
de validacion clinica.

Reutiliza la funcion MSM-SC (msm_sc.py) y la sliding-window con paso
adaptativo (_dist_min_msm de embedding.py) del Modulo_I del usuario,
para garantizar consistencia exacta con el espacio aprendido en
entrenamiento.

Esquema del flujo:

    senal_csv (15000 muestras, z-score)
        ├──► segmentacion en 60 ventanas BEAT (250 muestras = 1 s)
        │       └──► _fila_bloque(window_250, shapelets_latido)  ── 60-D
        │              └──► xgboost_LATIDO.predict_proba          ── [P(NORMAL), P(PAC)]
        │                     └──► umbral 0.50 -> NORMAL o PAC
        │
        └──► segmentacion en 15 ventanas RHYTHM (1000 muestras = 4 s)
                └──► _fila_bloque(window_1000, shapelets_ritmo) ── 60-D
                       └──► xgboost_RITMO.predict_proba          ── [P(AFIB), P(NSR)]
                              └──► umbral 0.60 -> NSR o AFIB
                                      (predict NSR si P(NSR) >= 0.60,
                                       equivale a maximizar recall AFIB)

Uso (una senal):
  python prepare_signals.py --input C:/.../senal_001.csv --id sig_001 ^
                            --age 71 --sex M --label "Paroxysmal AFIB onset"

Uso (5 senales en batch, ver `prepare_5_signals.py` adjunto).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import Parallel, delayed

# ---------------------------------------------------------------------
# Rutas y reutilizacion del Modulo_I del usuario
# ---------------------------------------------------------------------
# Este archivo vive en: D:/Modelo_Tesis/Modulo_I/modulo_3_interfaz/classifier/
# El Modulo_I esta en : D:/Modelo_Tesis/Modulo_I/modulo_1_clasificador/
# Subimos 2 niveles desde classifier/ para llegar a Modulo_I/.
HERE = Path(__file__).resolve().parent
INTERFAZ_DIR = HERE.parent
PROJECT_ROOT = INTERFAZ_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Reutilizar los modulos del usuario (NO reimplementar)
from modulo_1_clasificador.embedding import _fila_bloque  # MSM-SC sliding window
from modulo_1_clasificador.lectura_senal import (
    leer_senal_csv, normalizar_zscore,
)

# ---------------------------------------------------------------------
# Constantes operativas
# ---------------------------------------------------------------------
DATA_DIR = INTERFAZ_DIR / "data"
MODELS_DIR = DATA_DIR / "models"
SHAPELETS_DIR = DATA_DIR / "shapelets"
OUTPUT_DIR = DATA_DIR / "signals_processed"

FS = 250                  # sampling rate Hz
BEAT_WINDOW = 250         # samples (1 s)
RHYTHM_WINDOW = 1000      # samples (4 s)

LATIDO_THRESHOLD = 0.50   # predict PAC if proba[PAC] >= 0.50
RITMO_THRESHOLD  = 0.60   # predict NSR if proba[NSR] >= 0.60  (max AFIB recall)
ESCALATION_GAP   = 0.25   # if |proba[1] - proba[0]| < 0.25 -> escalar al medico


# ---------------------------------------------------------------------
# Carga de artefactos (modelos + diccionarios)
# ---------------------------------------------------------------------
def load_models_and_shapelets() -> dict:
    """
    Carga los modelos XGBoost duales, los diccionarios de shapelets,
    los label encoders y los maps de importancia para top-shapelets.
    Devuelve un dict con todo el estado listo para inferencia.
    """
    print("Cargando modelos y diccionarios...")

    beat_clf = xgb.XGBClassifier()
    beat_clf.load_model(str(MODELS_DIR / "xgboost_LATIDO_MSM.json"))

    rhythm_clf = xgb.XGBClassifier()
    rhythm_clf.load_model(str(MODELS_DIR / "xgboost_RITMO_MSM.json"))

    micro_dict = list(np.load(
        str(SHAPELETS_DIR / "shapelets_latido_MSM.npy"),
        allow_pickle=True,
    ))
    macro_dict = list(np.load(
        str(SHAPELETS_DIR / "shapelets_ritmo_MSM.npy"),
        allow_pickle=True,
    ))

    le_beat = list(np.load(
        str(MODELS_DIR / "label_encoder_latido_MSM.npy"),
        allow_pickle=True,
    ))
    le_rhythm = list(np.load(
        str(MODELS_DIR / "label_encoder_ritmo_MSM.npy"),
        allow_pickle=True,
    ))

    with open(MODELS_DIR / "importance_LATIDO.json") as f:
        imp_beat = json.load(f)
    with open(MODELS_DIR / "importance_RITMO.json") as f:
        imp_rhythm = json.load(f)

    print(f"  LATIDO: {len(micro_dict)} shapelets, "
          f"clases {le_beat}")
    print(f"  RITMO : {len(macro_dict)} shapelets, "
          f"clases {le_rhythm}")

    return {
        "beat_clf": beat_clf, "rhythm_clf": rhythm_clf,
        "micro_dict": micro_dict, "macro_dict": macro_dict,
        "le_beat": le_beat, "le_rhythm": le_rhythm,
        "imp_beat": imp_beat, "imp_rhythm": imp_rhythm,
    }


# ---------------------------------------------------------------------
# Segmentacion
# ---------------------------------------------------------------------
def segment_signal(signal: np.ndarray, window_size: int) -> list:
    """Particion sin solapamiento en ventanas de window_size muestras."""
    n_windows = len(signal) // window_size
    return [signal[i * window_size: (i + 1) * window_size]
            for i in range(n_windows)]


# ---------------------------------------------------------------------
# Inferencia por ventana
# ---------------------------------------------------------------------
def predict_window(
    window: np.ndarray,
    clf: xgb.XGBClassifier,
    dictionary: list,
    feature_prefix: str,
    classes: list,
    threshold: float,
    importance_map: dict,
) -> dict:
    """
    Computa el embedding MSM-SC de una ventana y predice con XGBoost.

    Convencion de classes: viene de LabelEncoder().classes_, que es
    orden alfabetico. La probabilidad de classes[1] es la que se
    compara contra threshold (esto reproduce exactamente el codigo
    de entrenamiento del usuario en solo_xgboost.py).

      LATIDO: classes = ['NORMAL', 'PAC']  -> threshold sobre P(PAC)
      RITMO : classes = ['AFIB', 'NSR']    -> threshold sobre P(NSR)

    Devuelve un dict con prediccion, probabilidades, confidence_gap,
    escalation_flag y top-3 shapelets ordenados por importancia.
    """
    # 60 distancias MSM-SC (paso=1 en micro, paso=PASO_VENTANA_MACRO en macro)
    distances = _fila_bloque(window, dictionary)
    feature_names = [f"{feature_prefix}{i + 1}"
                     for i in range(len(dictionary))]
    X = pd.DataFrame([distances], columns=feature_names)

    proba = clf.predict_proba(X)[0]
    # proba[0] = P(classes[0]), proba[1] = P(classes[1])

    if proba[1] >= threshold:
        prediction = str(classes[1])
    else:
        prediction = str(classes[0])

    confidence_gap = abs(float(proba[1]) - float(proba[0]))
    escalation_flag = bool(confidence_gap < ESCALATION_GAP)

    # Top-3 shapelets de mayor importancia segun XGBoost gain
    top3 = sorted(
        [
            (fn, float(distances[i]), float(importance_map.get(fn, 0.0)))
            for i, fn in enumerate(feature_names)
        ],
        key=lambda x: -x[2],
    )[:3]

    return {
        "prediction": prediction,
        "proba_class0": float(proba[0]),
        "proba_class1": float(proba[1]),
        "class0_name": str(classes[0]),
        "class1_name": str(classes[1]),
        "confidence_gap": confidence_gap,
        "escalation_flag": escalation_flag,
        "top_shapelets": [
            {"name": fn,
             "distance": round(d, 4),
             "importance_pct": round(imp, 2)}
            for fn, d, imp in top3
        ],
    }


# ---------------------------------------------------------------------
# Procesar una senal completa -> JSON
# ---------------------------------------------------------------------
def process_signal(
    signal_csv: str,
    signal_id: str,
    age: int,
    sex: str,
    label: str,
    scenario_note: str,
    models: dict,
    verbose: bool = True,
) -> dict:
    """
    Pipeline end-to-end para una senal: cargar CSV, segmentar, predecir
    todas las ventanas, ensamblar el JSON contract.
    """
    t0 = time.perf_counter()

    # 1) Cargar y (re-)normalizar (defensivo)
    raw = leer_senal_csv(signal_csv)
    signal = normalizar_zscore(raw)
    n_samples = len(signal)
    duration_s = n_samples / FS
    if verbose:
        print(f"  Senal cargada: {n_samples} muestras "
              f"({duration_s:.1f} s)")

    # 2) Segmentar
    beat_ws = segment_signal(signal, BEAT_WINDOW)
    rhythm_ws = segment_signal(signal, RHYTHM_WINDOW)
    if verbose:
        print(f"  Ventanas: {len(beat_ws)} BEAT (250 m) | "
              f"{len(rhythm_ws)} RHYTHM (1000 m)")

    # 3) Predecir BEAT — paralelizado sobre las 60 ventanas
    if verbose:
        print(f"  Calculando embeddings BEAT y prediciendo (joblib)...")
    tb = time.perf_counter()

    def _predict_beat(i, w):
        r = predict_window(
            window=w,
            clf=models["beat_clf"],
            dictionary=models["micro_dict"],
            feature_prefix="Micro_S",
            classes=models["le_beat"],
            threshold=LATIDO_THRESHOLD,
            importance_map=models["imp_beat"],
        )
        return (i, r)

    beat_results = Parallel(n_jobs=-1, backend="loky")(
        delayed(_predict_beat)(i, w) for i, w in enumerate(beat_ws)
    )
    beat_windows = []
    for i, r in sorted(beat_results, key=lambda x: x[0]):
        beat_windows.append({
            "window_index": i,
            "start_sample": i * BEAT_WINDOW,
            "end_sample": (i + 1) * BEAT_WINDOW,
            "start_seconds": float(i * BEAT_WINDOW / FS),
            "end_seconds": float((i + 1) * BEAT_WINDOW / FS),
            "prediction": r["prediction"],
            "prob_normal": round(
                r["proba_class0"] if r["class0_name"] == "NORMAL"
                else r["proba_class1"], 4),
            "prob_pac": round(
                r["proba_class0"] if r["class0_name"] == "PAC"
                else r["proba_class1"], 4),
            "confidence_gap": round(r["confidence_gap"], 4),
            "escalation_flag": r["escalation_flag"],
            "top_shapelets": r["top_shapelets"],
        })
    if verbose:
        print(f"    -> {len(beat_windows)} ventanas BEAT "
              f"en {time.perf_counter() - tb:.1f}s")

    # 4) Predecir RHYTHM — paralelizado sobre las 15 ventanas
    if verbose:
        print(f"  Calculando embeddings RHYTHM y prediciendo (joblib)...")
    tr = time.perf_counter()

    def _predict_rhythm(j, w):
        r = predict_window(
            window=w,
            clf=models["rhythm_clf"],
            dictionary=models["macro_dict"],
            feature_prefix="Macro_S",
            classes=models["le_rhythm"],
            threshold=RITMO_THRESHOLD,
            importance_map=models["imp_rhythm"],
        )
        return (j, r)

    rhythm_results = Parallel(n_jobs=-1, backend="loky")(
        delayed(_predict_rhythm)(j, w) for j, w in enumerate(rhythm_ws)
    )
    rhythm_windows = []
    for j, r in sorted(rhythm_results, key=lambda x: x[0]):
        rhythm_windows.append({
            "window_index": j,
            "start_sample": j * RHYTHM_WINDOW,
            "end_sample": (j + 1) * RHYTHM_WINDOW,
            "start_seconds": float(j * RHYTHM_WINDOW / FS),
            "end_seconds": float((j + 1) * RHYTHM_WINDOW / FS),
            "prediction": r["prediction"],
            "prob_nsr": round(
                r["proba_class0"] if r["class0_name"] == "NSR"
                else r["proba_class1"], 4),
            "prob_afib": round(
                r["proba_class0"] if r["class0_name"] == "AFIB"
                else r["proba_class1"], 4),
            "confidence_gap": round(r["confidence_gap"], 4),
            "escalation_flag": r["escalation_flag"],
            "top_shapelets": r["top_shapelets"],
        })
    if verbose:
        print(f"    -> {len(rhythm_windows)} ventanas RHYTHM "
              f"en {time.perf_counter() - tr:.1f}s")

    # 5) Aggregate summary
    beat_preds = [w["prediction"] for w in beat_windows]
    rhythm_preds = [w["prediction"] for w in rhythm_windows]
    beat_dist = (pd.Series(beat_preds).value_counts(normalize=True)
                 * 100).round(2).to_dict() if beat_preds else {}
    rhythm_dist = (pd.Series(rhythm_preds).value_counts(normalize=True)
                   * 100).round(2).to_dict() if rhythm_preds else {}

    anomalies = []
    for w in beat_windows:
        if w["prediction"] == "PAC":
            anomalies.append({
                "type": "PAC",
                "at_second": w["start_seconds"],
                "window_index": w["window_index"],
                "confidence_gap": w["confidence_gap"],
            })
    for w in rhythm_windows:
        if w["prediction"] == "AFIB":
            anomalies.append({
                "type": "AFIB_run",
                "from_second": w["start_seconds"],
                "to_second": w["end_seconds"],
                "window_index": w["window_index"],
                "confidence_gap": w["confidence_gap"],
            })

    output = {
        "signal_metadata": {
            "signal_id": signal_id,
            "patient_id_hash": f"anon_{signal_id}",
            "patient_age_estimate": int(age),
            "patient_sex": str(sex),
            "scenario_note": str(scenario_note),
            "label": str(label),
            "lead": "II",
            "sampling_rate_hz": FS,
            "duration_seconds": round(float(duration_s), 2),
            "n_samples": int(n_samples),
            "preprocessing": "z-score normalisation",
            "processed_at": datetime.now().isoformat(timespec="seconds"),
            "source_csv": str(signal_csv),
        },
        "raw_signal": signal.round(4).tolist(),
        "beat_windows": beat_windows,
        "rhythm_windows": rhythm_windows,
        "aggregate_summary": {
            "predominant_beat_class":
                max(beat_dist, key=beat_dist.get) if beat_dist else "NORMAL",
            "beat_class_distribution_pct":
                {k: float(v) for k, v in beat_dist.items()},
            "predominant_rhythm_class":
                max(rhythm_dist, key=rhythm_dist.get) if rhythm_dist else "NSR",
            "rhythm_class_distribution_pct":
                {k: float(v) for k, v in rhythm_dist.items()},
            "anomalies_detected": anomalies,
            "mean_confidence_gap_beat": round(float(
                np.mean([w["confidence_gap"] for w in beat_windows])
                if beat_windows else 0.0), 4),
            "mean_confidence_gap_rhythm": round(float(
                np.mean([w["confidence_gap"] for w in rhythm_windows])
                if rhythm_windows else 0.0), 4),
            "global_escalation_required": bool(any(
                w["escalation_flag"]
                for w in beat_windows + rhythm_windows
            )),
            "n_beat_windows": len(beat_windows),
            "n_rhythm_windows": len(rhythm_windows),
            "processing_time_seconds": round(time.perf_counter() - t0, 1),
        },
    }
    return output


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def cli_main():
    parser = argparse.ArgumentParser(
        description="Procesa una senal ECG cruda para validacion clinica."
    )
    parser.add_argument("--input", required=True,
                        help="CSV de entrada (1 columna, z-score, ~60 s a 250 Hz).")
    parser.add_argument("--id", required=True,
                        help="Identificador de la senal (ej: sig_001).")
    parser.add_argument("--age", type=int, required=True,
                        help="Edad estimada del paciente.")
    parser.add_argument("--sex", required=True, choices=["M", "F", "X"],
                        help="Sexo del paciente.")
    parser.add_argument("--label", required=True,
                        help="Etiqueta clinica corta (ej: 'Paroxysmal AFIB onset').")
    parser.add_argument("--scenario", default="",
                        help="Nota descriptiva mas larga (opcional).")
    parser.add_argument("--output", default=None,
                        help="Path de salida (default: data/signals_processed/<id>.json).")
    args = parser.parse_args()

    print("=" * 70)
    print(f"  PREPARANDO SENAL  -  {args.id}")
    print("=" * 70)
    print(f"  Input    : {args.input}")
    print(f"  Paciente : ~{args.age} anos, {args.sex}")
    print(f"  Label    : {args.label}")
    print()

    models = load_models_and_shapelets()
    output = process_signal(args.input, args.id, args.age, args.sex,
                            args.label, args.scenario, models)

    out_path = Path(args.output) if args.output else (
        OUTPUT_DIR / f"{args.id}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    s = output["aggregate_summary"]
    print()
    print(f"OK  Guardado: {out_path}  "
          f"({out_path.stat().st_size // 1024} KB)")
    print(f"  Predominante BEAT  : {s['predominant_beat_class']}")
    print(f"  Predominante RHYTHM: {s['predominant_rhythm_class']}")
    print(f"  Anomalias          : {len(s['anomalies_detected'])}")
    print(f"  Global escalation  : {s['global_escalation_required']}")
    print(f"  Tiempo total       : {s['processing_time_seconds']}s")


if __name__ == "__main__":
    cli_main()
