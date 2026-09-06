"""
analisis_vector.py
Analiza las 368 predicciones del vector LLM e identifica casos a escalar.

Ejecutar desde C:/Modelo_Tesis/Modulo_I/:
    "C:\Program Files\Python314\python.exe" analisis_vector.py
"""
import sys, json
sys.path.insert(0, r'C:\Modelo_Tesis\Modulo_I')
from config import CARPETA_SALIDA

with open(CARPETA_SALIDA / 'vector_entrada_LLM_MSM.json', encoding='utf-8') as f:
    data = json.load(f)

preds       = data['predicciones_test']
total       = len(preds)
correctas   = sum(1 for p in preds if p['correcto'])
escaladas   = sum(1 for p in preds if p['alerta_baja_confianza'])
incorrectas = total - correctas

print("=" * 60)
print("  ANÁLISIS COMPLETO DEL VECTOR LLM — 368 muestras")
print("=" * 60)
print(f"  Total muestras  : {total}")
print(f"  Correctas       : {correctas} ({correctas/total*100:.1f}%)")
print(f"  Incorrectas     : {incorrectas} ({incorrectas/total*100:.1f}%)")
print(f"  Para escalar    : {escaladas} (gap < 0.25)")
print(f"  Para nota LLM   : {total - escaladas}")

# Casos escalados
casos_escalar = [p for p in preds if p['alerta_baja_confianza']]
if casos_escalar:
    print(f"\n  Casos que se escalarían al médico ({len(casos_escalar)}):")
    print(f"  {'ID':>4}  {'Real':8} {'Predicha':8} {'Gap':6}  {'Correcto'}")
    print(f"  {'-'*46}")
    for p in casos_escalar:
        ok = '✓' if p['correcto'] else '✗ ERROR'
        print(f"  {p['id']:>4}  {p['clase_real']:8} "
              f"{p['clase_predicha']:8} {p['confidence_gap']:.3f}   {ok}")
else:
    print("\n  Ningún caso requiere escalamiento (todos con gap >= 0.25)")

# Casos incorrectos
incorrectos = [p for p in preds if not p['correcto']]
if incorrectos:
    print(f"\n  Casos incorrectos ({len(incorrectos)}):")
    print(f"  {'ID':>4}  {'Real':8} {'Predicha':8} {'Gap':6}  {'Escalado'}")
    print(f"  {'-'*46}")
    for p in incorrectos:
        esc = 'SÍ' if p['alerta_baja_confianza'] else 'NO'
        print(f"  {p['id']:>4}  {p['clase_real']:8} "
              f"{p['clase_predicha']:8} {p['confidence_gap']:.3f}   {esc}")

# Estadística por clase
print(f"\n  Desglose por clase predicha:")
clases = ['AFIB', 'NORMAL', 'NSR', 'PAC']
for c in clases:
    sub  = [p for p in preds if p['clase_predicha'] == c]
    corr = sum(1 for p in sub if p['correcto'])
    esc  = sum(1 for p in sub if p['alerta_baja_confianza'])
    if sub:
        print(f"  {c:8}: {len(sub):3d} predicciones | "
              f"{corr:3d} correctas ({corr/len(sub)*100:.0f}%) | "
              f"{esc} escaladas")

print("=" * 60)
