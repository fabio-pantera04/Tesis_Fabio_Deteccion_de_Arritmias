"""
Bootstrap del sistema:
  1. Re-entrena y guarda los dos modelos XGBoost del sistema dual (MSM-only, sin RR)
  2. Genera 5 senales mock pre-procesadas en data/signals_processed/sig_NNN.json
     (estructura JSON Clasificador->LLM definitiva)

Cada senal mock simula 60 segundos de ECG a 250 Hz = 15000 muestras,
con ventanas BEAT cada 250 muestras (1 s) y RHYTHM cada 1000 muestras (4 s),
ambas con stride configurable.

Cuando tengas tus shapelets reales y senales reales, ejecutas tu propio script
'classifier/prepare_signals.py' que produce el mismo formato JSON.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
UPLOADS = Path("/mnt/user-data/uploads")

# ---------------------------------------------------------------
# 1. Entrenar y guardar modelos
# ---------------------------------------------------------------
XGB_PARAMS = dict(
    objective="binary:logistic", eval_metric="logloss",
    max_depth=5, learning_rate=0.05, n_estimators=700,
    reg_alpha=0.05, reg_lambda=1.0,
    random_state=42, early_stopping_rounds=40, n_jobs=-1,
)

MODELS = {
    "LATIDO": {"pos": "PAC",  "neg": "NORMAL", "tau": 0.50, "scale": "beat"},
    "RITMO":  {"pos": "NSR",  "neg": "AFIB",   "tau": 0.60, "scale": "rhythm"},
}


def train_and_save():
    metadata = {}
    for name, cfg in MODELS.items():
        print(f"Training {name}...")
        Xtr = pd.read_csv(UPLOADS / f"X_train_{name}_MSM.csv")
        Xva = pd.read_csv(UPLOADS / f"X_val_{name}_MSM.csv")
        ytr = (Xtr.pop("Etiqueta") == cfg["pos"]).astype(int)
        yva = (Xva.pop("Etiqueta") == cfg["pos"]).astype(int)

        clf = xgb.XGBClassifier(**XGB_PARAMS)
        clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)

        # Save model
        model_path = DATA / "models" / f"xgboost_{name}_MSM.json"
        clf.save_model(str(model_path))

        # Save feature names (the 60 shapelet columns)
        feature_names = list(Xtr.columns)
        with open(DATA / "models" / f"features_{name}.json", "w") as f:
            json.dump(feature_names, f, indent=2)

        # Persist top-shapelet importance lookup for explainability
        gain = clf.get_booster().get_score(importance_type="gain")
        total = sum(gain.values())
        importance = sorted(
            [(fn, gain.get(fn, 0.0) / total * 100) for fn in feature_names],
            key=lambda x: -x[1],
        )
        with open(DATA / "models" / f"importance_{name}.json", "w") as f:
            json.dump({f: round(p, 4) for f, p in importance}, f, indent=2)

        metadata[name] = dict(
            classes=[cfg["neg"], cfg["pos"]],
            pos_class=cfg["pos"], neg_class=cfg["neg"],
            decision_threshold=cfg["tau"], scale=cfg["scale"],
            n_features=len(feature_names),
            best_iteration=int(clf.best_iteration),
            top5_importance=[(f, round(p, 2)) for f, p in importance[:5]],
        )
        print(f"  saved {model_path.name}")

    with open(DATA / "models" / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved.")
    return metadata


# ---------------------------------------------------------------
# 2. Generar 5 senales mock pre-procesadas
# ---------------------------------------------------------------
# Mock signals: 60 seconds at 250 Hz = 15000 samples
# Beat windows every 1 s = 60 windows
# Rhythm windows every 4 s = 15 windows
#
# Each signal has a clinical scenario designed to exhibit the 4 labels
# (NORMAL, PAC, NSR, AFIB) across its windows.

SIGNAL_SCENARIOS = [
    # signal_id, description, beat_pattern, rhythm_pattern
    {
        "id": "sig_001",
        "label": "Predominantly normal with isolated PACs",
        "beat_pattern":  ["NORMAL"]*15 + ["PAC"] + ["NORMAL"]*22 + ["PAC"] + ["NORMAL"]*21,
        "rhythm_pattern": ["NSR"]*15,
        "scenario_note":  "Healthy adult resting baseline, sporadic atrial ectopy",
        "patient_age_estimate": 42, "patient_sex": "M",
    },
    {
        "id": "sig_002",
        "label": "Sustained atrial fibrillation",
        "beat_pattern":  ["NORMAL"]*60,
        "rhythm_pattern": ["AFIB"]*15,
        "scenario_note":  "Confirmed chronic AFIB under anticoagulant therapy",
        "patient_age_estimate": 68, "patient_sex": "F",
    },
    {
        "id": "sig_003",
        "label": "Paroxysmal AFIB onset",
        "beat_pattern":  ["NORMAL"]*30 + ["PAC"]*2 + ["NORMAL"]*28,
        "rhythm_pattern": ["NSR"]*8 + ["AFIB"]*7,
        "scenario_note":  "Sinus rhythm transitioning to AFIB at second 32",
        "patient_age_estimate": 71, "patient_sex": "M",
    },
    {
        "id": "sig_004",
        "label": "Normal sinus rhythm, no anomalies",
        "beat_pattern":  ["NORMAL"]*60,
        "rhythm_pattern": ["NSR"]*15,
        "scenario_note":  "Control baseline from young adult, no findings",
        "patient_age_estimate": 28, "patient_sex": "F",
    },
    {
        "id": "sig_005",
        "label": "Multiple PACs and brief AFIB run",
        "beat_pattern": (
            ["NORMAL"]*10 + ["PAC"] + ["NORMAL"]*9 + ["PAC"] + ["NORMAL"]*9
            + ["PAC"] + ["NORMAL"]*5 + ["NORMAL"]*5 + ["NORMAL"]*5
            + ["NORMAL"]*4 + ["PAC"] + ["NORMAL"]*9
        ),
        "rhythm_pattern": ["NSR"]*6 + ["AFIB"]*2 + ["NSR"]*7,
        "scenario_note":  "Holter excerpt showing burden of supraventricular ectopy",
        "patient_age_estimate": 55, "patient_sex": "M",
    },
]


def synthesize_ecg(duration_s=60, fs=250, beat_pattern=None, rhythm_pattern=None,
                   seed=0):
    """
    Genera una senal ECG sintetica con caracteristicas morfologicas
    distinguibles segun beat_pattern (cada 1 s) y rhythm_pattern (cada 4 s).

    Esta es una senal DEMO. En produccion reemplazas esto leyendo
    el ECG real (WFDB, EDF, CSV) de Icentia11k o tu fuente.
    """
    rng = np.random.default_rng(seed)
    n = duration_s * fs
    t = np.arange(n) / fs

    # Synthesise an HR ~ 75 bpm sinus baseline
    # 1 beat per second; QRS peak modelled as a tall narrow Gaussian
    base_hr = 75.0  # bpm
    rr = 60.0 / base_hr
    signal = np.zeros(n)

    # Beat-by-beat generation
    beat_centers = []
    current_t = 0.5
    while current_t < duration_s:
        center_idx = int(current_t * fs)
        # Decide if this beat is PAC: shortened RR, inverted P
        beat_idx_in_pattern = int(current_t)
        if beat_idx_in_pattern < len(beat_pattern) and beat_pattern[beat_idx_in_pattern] == "PAC":
            # PAC: prematurely fires + compensatory pause
            current_t -= 0.15  # premature
        elif beat_idx_in_pattern < len(rhythm_pattern_per_second(rhythm_pattern)) \
                and rhythm_pattern_per_second(rhythm_pattern)[beat_idx_in_pattern] == "AFIB":
            current_t += rng.normal(0, 0.18)  # irregular RR in AFIB

        center_idx = int(np.clip(current_t * fs, 0, n - 1))
        beat_centers.append(center_idx)

        # QRS complex: narrow Gaussian
        for samp in range(max(0, center_idx - 30), min(n, center_idx + 30)):
            d = samp - center_idx
            signal[samp] += 1.0 * np.exp(-d * d / (2 * 6 * 6))
        # T wave: wider, lower
        for samp in range(max(0, center_idx + 30), min(n, center_idx + 120)):
            d = samp - center_idx - 70
            signal[samp] += 0.25 * np.exp(-d * d / (2 * 25 * 25))
        # P wave (only in NORMAL/NSR)
        if beat_idx_in_pattern < len(beat_pattern) and beat_pattern[beat_idx_in_pattern] != "PAC":
            for samp in range(max(0, center_idx - 60), min(n, center_idx - 20)):
                d = samp - center_idx + 40
                signal[samp] += 0.15 * np.exp(-d * d / (2 * 12 * 12))

        current_t += rr

    # Add baseline wander + Gaussian noise
    signal += 0.05 * np.sin(2 * np.pi * 0.3 * t)
    signal += rng.normal(0, 0.04, n)

    # z-score normalise
    signal = (signal - signal.mean()) / signal.std()
    return signal


def rhythm_pattern_per_second(rhythm_pattern_per_4s):
    """Each rhythm window covers 4 seconds; expand to per-second list."""
    out = []
    for r in rhythm_pattern_per_4s:
        out.extend([r] * 4)
    return out


def build_signal_json(scenario, metadata):
    """Produce the full JSON contract from a synthetic signal scenario."""
    sid = scenario["id"]
    fs = 250
    duration_s = 60
    n = duration_s * fs

    signal = synthesize_ecg(duration_s, fs, scenario["beat_pattern"],
                            scenario["rhythm_pattern"],
                            seed=int(sid.split("_")[1]))

    # Load real models to score classification per window
    beat_clf = xgb.XGBClassifier()
    beat_clf.load_model(str(DATA / "models" / "xgboost_LATIDO_MSM.json"))
    rhythm_clf = xgb.XGBClassifier()
    rhythm_clf.load_model(str(DATA / "models" / "xgboost_RITMO_MSM.json"))

    with open(DATA / "models" / "features_LATIDO.json") as f:
        beat_features = json.load(f)
    with open(DATA / "models" / "features_RITMO.json") as f:
        rhythm_features = json.load(f)
    with open(DATA / "models" / "importance_LATIDO.json") as f:
        beat_importance = json.load(f)
    with open(DATA / "models" / "importance_RITMO.json") as f:
        rhythm_importance = json.load(f)

    # Mock embeddings: synthesize "MSM-SC distances" that yield predictions
    # consistent with scenario['beat_pattern'] / scenario['rhythm_pattern'].
    # For demo, sample from a training row of the right class.
    beat_train = pd.read_csv(UPLOADS / "X_train_LATIDO_MSM.csv")
    rhythm_train = pd.read_csv(UPLOADS / "X_train_RITMO_MSM.csv")

    rng = np.random.default_rng(int(sid.split("_")[1]))

    # --- BEAT windows (every 250 samples = 1 s) ---
    beat_windows = []
    for i in range(60):
        ground_truth = scenario["beat_pattern"][i] if i < len(scenario["beat_pattern"]) else "NORMAL"
        # Sample a training embedding of the same class
        candidates = beat_train[beat_train["Etiqueta"] == ground_truth]
        if len(candidates) == 0:
            candidates = beat_train
        emb = candidates.drop(columns=["Etiqueta"]).sample(1, random_state=int(rng.integers(1e6))).iloc[0]
        emb_vec = emb.values.astype(float)
        proba = beat_clf.predict_proba(emb_vec.reshape(1, -1))[0]
        # proba[1] = PAC, proba[0] = NORMAL  (PAC was encoded as positive)
        pred = "PAC" if proba[1] >= 0.50 else "NORMAL"
        confidence_gap = abs(proba[1] - proba[0])
        top_sh = sorted(
            [(fn, float(emb_vec[k]), float(beat_importance.get(fn, 0.0)))
             for k, fn in enumerate(beat_features)],
            key=lambda x: -x[2],
        )[:3]
        beat_windows.append({
            "window_index": i,
            "start_sample": i * 250,
            "end_sample": (i + 1) * 250,
            "start_seconds": float(i),
            "end_seconds": float(i + 1),
            "prediction": pred,
            "prob_normal": round(float(proba[0]), 4),
            "prob_pac": round(float(proba[1]), 4),
            "confidence_gap": round(float(confidence_gap), 4),
            "escalation_flag": bool(confidence_gap < 0.25),
            "top_shapelets": [
                {"name": fn, "distance": round(d, 4),
                 "importance_pct": round(imp, 2)}
                for fn, d, imp in top_sh
            ],
        })

    # --- RHYTHM windows (every 1000 samples = 4 s) ---
    rhythm_windows = []
    for i in range(15):
        ground_truth = scenario["rhythm_pattern"][i] if i < len(scenario["rhythm_pattern"]) else "NSR"
        candidates = rhythm_train[rhythm_train["Etiqueta"] == ground_truth]
        if len(candidates) == 0:
            candidates = rhythm_train
        emb = candidates.drop(columns=["Etiqueta"]).sample(1, random_state=int(rng.integers(1e6))).iloc[0]
        emb_vec = emb.values.astype(float)
        proba = rhythm_clf.predict_proba(emb_vec.reshape(1, -1))[0]
        # proba[1] = NSR (positive), proba[0] = AFIB
        pred = "NSR" if proba[1] >= 0.60 else "AFIB"
        confidence_gap = abs(proba[1] - proba[0])
        top_sh = sorted(
            [(fn, float(emb_vec[k]), float(rhythm_importance.get(fn, 0.0)))
             for k, fn in enumerate(rhythm_features)],
            key=lambda x: -x[2],
        )[:3]
        rhythm_windows.append({
            "window_index": i,
            "start_sample": i * 1000,
            "end_sample": (i + 1) * 1000,
            "start_seconds": float(i * 4),
            "end_seconds": float((i + 1) * 4),
            "prediction": pred,
            "prob_nsr": round(float(proba[1]), 4),
            "prob_afib": round(float(proba[0]), 4),
            "confidence_gap": round(float(confidence_gap), 4),
            "escalation_flag": bool(confidence_gap < 0.25),
            "top_shapelets": [
                {"name": fn, "distance": round(d, 4),
                 "importance_pct": round(imp, 2)}
                for fn, d, imp in top_sh
            ],
        })

    # --- Aggregate summary ---
    beat_dist = pd.Series([w["prediction"] for w in beat_windows]).value_counts(normalize=True) * 100
    rhythm_dist = pd.Series([w["prediction"] for w in rhythm_windows]).value_counts(normalize=True) * 100
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
            "signal_id": sid,
            "patient_id_hash": f"anon_{sid}",
            "patient_age_estimate": scenario["patient_age_estimate"],
            "patient_sex": scenario["patient_sex"],
            "scenario_note": scenario["scenario_note"],
            "label": scenario["label"],
            "lead": "II",
            "sampling_rate_hz": fs,
            "duration_seconds": duration_s,
            "n_samples": n,
            "preprocessing": "z-score normalisation",
        },
        "raw_signal": signal.round(4).tolist(),
        "beat_windows": beat_windows,
        "rhythm_windows": rhythm_windows,
        "aggregate_summary": {
            "predominant_beat_class": beat_dist.idxmax(),
            "beat_class_distribution_pct": {k: round(float(v), 2) for k, v in beat_dist.items()},
            "predominant_rhythm_class": rhythm_dist.idxmax(),
            "rhythm_class_distribution_pct": {k: round(float(v), 2) for k, v in rhythm_dist.items()},
            "anomalies_detected": anomalies,
            "mean_confidence_gap_beat": round(float(np.mean([w["confidence_gap"] for w in beat_windows])), 4),
            "mean_confidence_gap_rhythm": round(float(np.mean([w["confidence_gap"] for w in rhythm_windows])), 4),
            "global_escalation_required": any(w["escalation_flag"] for w in beat_windows + rhythm_windows),
            "n_beat_windows": 60,
            "n_rhythm_windows": 15,
        },
    }
    return output


def generate_all_signals(metadata):
    for scenario in SIGNAL_SCENARIOS:
        print(f"Generating {scenario['id']}: {scenario['label']}")
        out = build_signal_json(scenario, metadata)
        path = DATA / "signals_processed" / f"{scenario['id']}.json"
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  saved {path.name}  ({path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    metadata = train_and_save()
    generate_all_signals(metadata)
    print("\nBootstrap complete.")
