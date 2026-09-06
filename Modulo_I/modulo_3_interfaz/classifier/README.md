# Scripts de preparación de señales — `modulo_3_interfaz/classifier/`

Convierte señales ECG crudas (CSV de 1 columna, 60 s a 250 Hz, z-score)
en los JSONs que consume la interfaz Flask de validación clínica.

## Dependencias

Estos scripts **reutilizan tu pipeline existente del Módulo I** vía import:

```python
from modulo_1_clasificador.embedding import _fila_bloque
from modulo_1_clasificador.lectura_senal import leer_senal_csv, normalizar_zscore
```

→ No reimplementan MSM-SC ni la lógica de embedding. Si modificas
`modulo_1_clasificador/embedding.py` o `msm_sc.py`, estos scripts heredan
automáticamente los cambios.

## Requisitos previos en `data/`

```
modulo_3_interfaz/data/
├── models/
│   ├── xgboost_LATIDO_MSM.json    ← copia de datos/xgboost_latido_MSM.json
│   ├── xgboost_RITMO_MSM.json     ← copia de datos/xgboost_ritmo_MSM.json
│   ├── importance_LATIDO.json     ← generar con `extract_importance.py` (1 vez)
│   └── importance_RITMO.json
├── shapelets/
│   ├── shapelets_latido_MSM.npy   ← copia de datos/shapelets_latido_MSM.npy
│   ├── shapelets_ritmo_MSM.npy    ← copia de datos/shapelets_ritmo_MSM.npy
│   ├── label_encoder_latido_MSM.npy
│   └── label_encoder_ritmo_MSM.npy
├── signals_raw/                    ← (tus CSVs originales, opcional)
└── signals_processed/              ← donde se guardan los JSONs
```

## Generar los archivos de importancia (1 sola vez)

```python
# scripts/extract_importance.py
import json, xgboost as xgb, pandas as pd
from pathlib import Path

D = Path(__file__).resolve().parent.parent / "data"

for name in ["LATIDO", "RITMO"]:
    clf = xgb.XGBClassifier()
    clf.load_model(str(D / "models" / f"xgboost_{name}_MSM.json"))
    # Read column names from any of your CSVs to keep order consistent
    sample = pd.read_csv(D.parent.parent / "datos" / "resultados" / f"X_train_{name}_MSM.csv", nrows=1)
    cols = [c for c in sample.columns if c != "Etiqueta"]
    gain = clf.get_booster().get_score(importance_type="gain")
    total = sum(gain.values()) or 1
    importance = {c: round(gain.get(c, 0.0) / total * 100, 4) for c in cols}
    with open(D / "models" / f"importance_{name}.json", "w") as f:
        json.dump(importance, f, indent=2)
    print(f"Saved importance_{name}.json")
```

## Uso

### Una sola señal

```bash
cd D:\Modelo_Tesis\Modulo_I\modulo_3_interfaz\classifier
python prepare_signals.py ^
  --input "C:\Users\<TU>\Documents\CIMAT\Senales Medicos\AFIB\senal_002.csv" ^
  --id sig_002 ^
  --age 68 ^
  --sex F ^
  --label "Fibrilacion auricular sostenida" ^
  --scenario "AFIB cronica bajo tratamiento anticoagulante"
```

Output: `data/signals_processed/sig_002.json`

### Las 5 señales en una sola corrida

1. Editar `prepare_5_signals.py` con las rutas reales de tus 5 CSVs
   (subfolders dentro de `Senales Medicos\`).
2. Ejecutar:

   ```bash
   python prepare_5_signals.py
   ```

## Tiempo esperado

Por señal (Ryzen 5 5600G, 6 cores, joblib paralelizado):
- 60 ventanas BEAT × 60 micro-shapelets, paso=1 → ~3-5 minutos
- 15 ventanas RHYTHM × 60 macro-shapelets, paso=5 → ~10-15 minutos
- **Total ≈ 15-20 minutos por señal × 5 señales ≈ 1.5 horas en batch**

Recomendación: ejecuta `prepare_5_signals.py` una vez como tarea overnight.
Los JSONs resultantes son estáticos — la web Flask sólo los lee, no los
recalcula. Solo tienes que regenerar si cambias el modelo o los shapelets.

## Validación de salida

Cada JSON debe pasar las siguientes verificaciones:

```python
import json
with open("data/signals_processed/sig_002.json") as f:
    out = json.load(f)

assert "signal_metadata" in out
assert "raw_signal" in out
assert len(out["raw_signal"]) == 15000           # 60 s × 250 Hz
assert len(out["beat_windows"]) == 60            # 60 × 1 s
assert len(out["rhythm_windows"]) == 15          # 15 × 4 s
assert all(abs(w["prob_normal"] + w["prob_pac"] - 1.0) < 0.01
           for w in out["beat_windows"])         # probas suman 1
assert all(abs(w["prob_nsr"] + w["prob_afib"] - 1.0) < 0.01
           for w in out["rhythm_windows"])
print("OK")
```
