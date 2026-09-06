"""
modulo_3_interfaz/classifier/prepare_5_signals.py

Driver de batch: procesa las 5 senales del experimento de validacion
clinica en una sola corrida.

Solo "id" y "csv" son obligatorios en SENALES. Los demas campos
(age/sex/label/scenario) son opcionales con defaults, porque la UI
nueva ya no los muestra al medico.

Uso:
    python prepare_5_signals.py
"""

import json
import sys
import time
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────
# Forzar que el directorio del script este en sys.path. Esto permite
# que `from prepare_signals import ...` funcione SIN IMPORTAR desde
# que directorio se invoque el script (VS Code, doble click, otro cwd).
# ─────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_signals import load_models_and_shapelets, process_signal, OUTPUT_DIR


# ─────────────────────────────────────────────────────────────────────
# Tus 5 senales reales — ya estan colocadas aqui las rutas correctas.
# ─────────────────────────────────────────────────────────────────────
SENALES = [
    {"id": "sig_001",
     "csv": r"D:/Modelo_Tesis/Modulo_I/modulo_3_interfaz/signals/A00004_AFIB_CinC2017_30s_250Hz.csv"},
    {"id": "sig_002",
     "csv": r"D:/Modelo_Tesis/Modulo_I/modulo_3_interfaz/signals/p00008_s00_NSR_NORMAL_335s_1min.csv"},
    {"id": "sig_003",
     "csv": r"D:/Modelo_Tesis/Modulo_I/modulo_3_interfaz/signals/p00019_s00_PAC_NORMAL_1628s_1min.csv"},
    {"id": "sig_004",
     "csv": r"D:/Modelo_Tesis/Modulo_I/modulo_3_interfaz/signals/p00030_s00_NSR_NORMAL_713s_1min.csv"},
    {"id": "sig_005",
     "csv": r"D:/Modelo_Tesis/Modulo_I/modulo_3_interfaz/signals/p00065_s00_AFIB_733s_1min.csv"},
]


def main():
    print("=" * 70)
    print("  PREPARACION DE LAS 5 SENALES DE VALIDACION")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    missing = [s for s in SENALES if not Path(s["csv"]).exists()]
    if missing:
        print("\nERROR: Los siguientes CSVs no existen:")
        for s in missing:
            print(f"  - {s['id']}: {s['csv']}")
        return

    models = load_models_and_shapelets()

    t_global = time.perf_counter()
    for s in SENALES:
        sig_id = s["id"]
        csv = s["csv"]
        age = s.get("age", 50)
        sex = s.get("sex", "X")
        label = s.get("label", "Senal ECG")
        scenario = s.get("scenario", "")

        print(f"\n{'─' * 70}")
        print(f"  {sig_id}")
        print(f"  CSV: {Path(csv).name}")
        print(f"{'─' * 70}")

        out = process_signal(
            signal_csv=csv,
            signal_id=sig_id,
            age=age,
            sex=sex,
            label=label,
            scenario_note=scenario,
            models=models,
            verbose=True,
        )
        path = OUTPUT_DIR / f"{sig_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        size_kb = path.stat().st_size // 1024
        summary = out["aggregate_summary"]
        print(f"  -> guardado {path.name} ({size_kb} KB)  |  "
              f"BEAT predominante: {summary['predominant_beat_class']}  |  "
              f"RHYTHM predominante: {summary['predominant_rhythm_class']}  |  "
              f"anomalias: {len(summary['anomalies_detected'])}")

    print(f"\n{'=' * 70}")
    print(f"  TODAS LAS SENALES PROCESADAS  -  "
          f"{time.perf_counter() - t_global:.1f}s totales")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
