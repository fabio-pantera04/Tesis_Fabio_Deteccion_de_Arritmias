"""
modulo_3_interfaz/scripts/extract_importance.py

Genera los archivos:
    data/models/importance_LATIDO.json
    data/models/importance_RITMO.json

Cada uno mapea nombre de feature (Micro_S1, Micro_S2, ..., Macro_S60)
al porcentaje de ganancia (gain) que XGBoost le asignó después del
entrenamiento. Este map lo usa `prepare_signals.py` para anotar los
top-3 shapelets en cada ventana del JSON de salida.

Ejecutar una sola vez después de:
  1) Copiar xgboost_LATIDO_MSM.json y xgboost_RITMO_MSM.json a data/models/
  2) Tener los CSVs de entrenamiento accesibles en la ruta original

Uso:
    cd D:\\Modelo_Tesis\\Modulo_I\\modulo_3_interfaz
    python scripts\\extract_importance.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
import xgboost as xgb

HERE = Path(__file__).resolve().parent
INTERFAZ_DIR = HERE.parent
PROJECT_ROOT = INTERFAZ_DIR.parent           # D:\Modelo_Tesis\Modulo_I
RESULTADOS_DIR = PROJECT_ROOT / "datos" / "resultados"
MODELS_DIR = INTERFAZ_DIR / "data" / "models"


def main():
    print("=" * 70)
    print("  EXTRACCIÓN DE FEATURE IMPORTANCE — ambos modelos del sistema dual")
    print("=" * 70)

    for name in ["latido", "ritmo"]:
        model_path = MODELS_DIR / f"xgboost_{name}_MSM.json"
        csv_path = RESULTADOS_DIR / f"X_train_{name}_MSM.csv"

        if not model_path.exists():
            print(f"\n  [{name}] ERROR: No existe {model_path}")
            print(f"           Copia tu xgboost_{name.lower()}_MSM.json de datos/ "
                  f"renombrado en mayúsculas.")
            sys.exit(1)
        if not csv_path.exists():
            print(f"\n  [{name}] ERROR: No existe {csv_path}")
            print(f"           Este script necesita el CSV para leer los nombres "
                  f"de columnas en el orden correcto.")
            sys.exit(1)

        print(f"\n  [{name}] Cargando modelo: {model_path}")
        clf = xgb.XGBClassifier()
        clf.load_model(str(model_path))

        print(f"  [{name}] Leyendo nombres de features de: {csv_path.name}")
        sample = pd.read_csv(csv_path, nrows=1)
        cols = [c for c in sample.columns if c != "Etiqueta"]
        print(f"           {len(cols)} features: {cols[0]} ... {cols[-1]}")

        # XGBoost gain por feature (los que no aparecen en el booster
        # son features que el modelo NUNCA usó — gain = 0.0)
        gain = clf.get_booster().get_score(importance_type="gain")
        total = sum(gain.values()) or 1.0
        importance = {c: round(gain.get(c, 0.0) / total * 100, 4) for c in cols}

        # Resumen visual
        top5 = sorted(importance.items(), key=lambda x: -x[1])[:5]
        n_used = sum(1 for v in importance.values() if v > 0)
        print(f"  [{name}] Features usadas: {n_used} de {len(cols)}")
        print(f"  [{name}] Top-5 por gain:")
        for i, (fn, pct) in enumerate(top5, 1):
            print(f"           {i}. {fn:<14s} {pct:6.2f}%")

        # Guardar
        out_path = MODELS_DIR / f"importance_{name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(importance, f, indent=2)
        print(f"  [{name}] OK guardado: {out_path}")

    print("\n" + "=" * 70)
    print("  Hecho. Ya puedes ejecutar prepare_5_signals.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
