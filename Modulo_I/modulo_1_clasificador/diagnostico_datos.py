# diagnostico_datos.py
import sys, os
sys.path.insert(0, r"C:/Modelo_Tesis/Modulo_I")

from pathlib import Path

RUTA_ICENTIA = Path(r"C:/Modelo_Tesis/Icentia11k")
RUTA_MITBIH  = Path(r"C:/Modelo_Tesis/MIT BIH")
CLASES       = ['NORMAL', 'PAC', 'NSR', 'NRS', 'AFIB']

print("=" * 60)
print("  DIAGNÓSTICO DE SEÑALES DISPONIBLES POR BASE Y CLASE")
print("=" * 60)

total_icentia = 0
total_mitbih  = 0

for clase in CLASES:
    n_ice = 0
    n_mit = 0

    p_ice = RUTA_ICENTIA / clase
    if p_ice.exists():
        n_ice = len(list(p_ice.glob('*.csv')))

    p_mit = RUTA_MITBIH / clase
    if p_mit.exists():
        n_mit = len(list(p_mit.glob('*.csv')))

    if n_ice > 0 or n_mit > 0:
        total = n_ice + n_mit
        print(f"\n  Clase: {clase}")
        print(f"    Icentia11k : {n_ice:4d} archivos")
        print(f"    MIT-BIH    : {n_mit:4d} archivos")
        print(f"    Total      : {total:4d} archivos")
        total_icentia += n_ice
        total_mitbih  += n_mit

print(f"\n{'─'*60}")
print(f"  TOTAL Icentia11k : {total_icentia}")
print(f"  TOTAL MIT-BIH    : {total_mitbih}")
print(f"  TOTAL combinado  : {total_icentia + total_mitbih}")
print(f"\n  Con NUM_MUESTRAS_POR_CLASE=306:")
print(f"  El programa toma 306 del pool combinado por clase")
print(f"  Para tomar 306 de CADA base usar NUM_MUESTRAS_POR_CLASE=612")
print("=" * 60)