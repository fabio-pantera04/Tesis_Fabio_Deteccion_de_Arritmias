"""
test_robustness.py
==================
Test de robustez para validar si el accuracy del SFP-FACC es 
aprendizaje real o ruido seleccionado por el ACO.

Ejecuta el pipeline con 5 semillas distintas. Si los resultados varían
mucho (>5% std), confirma que el ACO está seleccionando ruido.
"""

import sys
sys.path.insert(0, '.')
import numpy as np
from main import run_sfp_facc, load_yaml_config
from pathlib import Path

# Configurar seeds a probar
SEEDS = [42, 123, 456, 789, 2024]

config_path = Path('config_latido.yaml')  # cambia a config_ritmo.yaml para RITMO

cfg = load_yaml_config(config_path)
class_sources = {}
for cls_name, sources in cfg['classes'].items():
    class_sources[cls_name] = [(s['dir'], s['prefix']) for s in sources]

results = []
for seed in SEEDS:
    print(f"\n\n{'#' * 70}")
    print(f"# SEED = {seed}")
    print(f"{'#' * 70}")
    
    result = run_sfp_facc(
        task=cfg['task'],
        class_sources=class_sources,
        seed=seed,
        test_size=cfg.get('test_size', 0.30),
        fit_n_per_class=cfg.get('fit_n_per_class', 10),
        max_per_class=cfg.get('max_per_class', 1000),
        verbose=False,
    )
    results.append({
        'seed': seed,
        'accuracy': result['accuracy'],
        'base_n_errors': result['base_n_errors'],
        'final_n_errors': result['n_errors'],
    })

print("\n\n" + "=" * 70)
print("RESUMEN DE ROBUSTEZ")
print("=" * 70)
print(f"{'Seed':>6} | {'Base errors':>12} | {'Final errors':>13} | {'Accuracy':>10}")
print("-" * 70)
for r in results:
    base_acc = 1 - r['base_n_errors'] / 600
    print(f"{r['seed']:>6} | {r['base_n_errors']:>12} | {r['final_n_errors']:>13} | "
          f"{r['accuracy']:>10.4f}")

accs = [r['accuracy'] for r in results]
print("-" * 70)
print(f"Mean accuracy: {np.mean(accs):.4f}")
print(f"Std accuracy:  {np.std(accs):.4f}")
print(f"Range:         [{min(accs):.4f}, {max(accs):.4f}]")
print(f"\nInterpretación:")
if np.std(accs) > 0.05:
    print("  ⚠ Std > 5%: el ACO está seleccionando ruido, no aprendizaje")
elif np.std(accs) > 0.02:
    print("  ~ Std moderado: hay algo de aprendizaje pero también mucha varianza")
else:
    print("  ✓ Std < 2%: el modelo es estable, hay aprendizaje real")

# Verificar si predicción base mejora con seeds (debería ser similar siempre)
base_accs = [1 - r['base_n_errors'] / 600 for r in results]
print(f"\nBase accuracy mean: {np.mean(base_accs):.4f}")
print(f"Base accuracy std:  {np.std(base_accs):.4f}")
if np.mean(base_accs) < 0.55:
    print("  ⚠ Base ~50%: la regla Euclidean+DTW NO discrimina las clases")
    print("  → El paper original probablemente sí discrimina porque usa 5 clases muy distintas")
    print("  → En N vs PAC (clases parecidas), la regla falla por features médicas pobres")
