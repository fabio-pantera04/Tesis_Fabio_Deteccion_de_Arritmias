"""
diagnostico_modelo.py
Diagnostica por qué el modelo clasifica incorrectamente señales PAC/NSR como AFIB.
Ejecutar desde C:/Modelo_Tesis/Modulo_I/:
    "C:\Program Files\Python314\python.exe" diagnostico_modelo.py ruta_al_csv.csv
"""
import sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, r'C:\Modelo_Tesis\Modulo_I')

from config import (CARPETA_SALIDA, UMBRAL_CONFIANZA,
                    ARCHIVO_MICRO_SHAPELETS, ARCHIVO_MACRO_SHAPELETS)
from modulo_1_clasificador.lectura_senal import normalizar_zscore, remuestrear_senal
from modulo_1_clasificador.embedding import embedding_una_senal
from modulo_1_clasificador.clasificador import cargar_modelo

# ── 1. Leer archivo de prueba ─────────────────────────────────────────────────
ruta_csv = sys.argv[1] if len(sys.argv) > 1 else None

if ruta_csv is None:
    print("[DIAG] Uso: python diagnostico_modelo.py ruta_al_csv.csv [fs_origen]")
    print("[DIAG] Usando señal sintética NSR para diagnóstico base...")
    # Señal NSR sintética: onda sinusoidal simple de 1000 muestras
    t = np.linspace(0, 4, 1000)
    senal_raw = np.sin(2 * np.pi * 1.2 * t) * 0.8
    fs_origen = 250
    nombre_archivo = "NSR_sintetico"
else:
    fs_origen = int(sys.argv[2]) if len(sys.argv) > 2 else 250
    nombre_archivo = ruta_csv.split('\\')[-1].split('/')[-1]
    try:
        # Intentar leer el CSV
        df = pd.read_csv(ruta_csv, header=None)
        print(f"\n[DIAG] Archivo: {nombre_archivo}")
        print(f"[DIAG] Shape del CSV: {df.shape} ({df.shape[0]} filas × {df.shape[1]} columnas)")
        print(f"[DIAG] Primeras 3 filas:")
        print(df.head(3).to_string())
        print(f"[DIAG] Tipos de datos: {df.dtypes.tolist()}")

        # Detectar columna numérica
        col_numerica = None
        for c in range(df.shape[1]):
            try:
                vals = pd.to_numeric(df.iloc[:, c], errors='coerce')
                if vals.notna().sum() > 50:
                    col_numerica = c
                    break
            except Exception:
                continue

        if col_numerica is None:
            # Intentar con header
            df2 = pd.read_csv(ruta_csv)
            for c in range(df2.shape[1]):
                try:
                    vals = pd.to_numeric(df2.iloc[:, c], errors='coerce')
                    if vals.notna().sum() > 50:
                        col_numerica = c
                        df = df2
                        break
                except Exception:
                    continue

        print(f"[DIAG] Columna numérica detectada: {col_numerica}")
        senal_raw = pd.to_numeric(df.iloc[:, col_numerica],
                                   errors='coerce').dropna().values
    except Exception as e:
        print(f"[DIAG] Error leyendo CSV: {e}")
        sys.exit(1)

# ── 2. Estadísticas de la señal cruda ────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  DIAGNÓSTICO DE SEÑAL: {nombre_archivo}")
print(f"{'='*60}")
print(f"  Longitud señal cruda    : {len(senal_raw)} muestras")
print(f"  Fs origen               : {fs_origen} Hz")
print(f"  Duración                : {len(senal_raw)/fs_origen:.2f} s")
print(f"  Rango valores           : [{senal_raw.min():.4f}, {senal_raw.max():.4f}]")
print(f"  Media / Std             : {senal_raw.mean():.4f} / {senal_raw.std():.4f}")
print(f"  Contiene NaN/Inf        : {np.any(~np.isfinite(senal_raw))}")

# Tipo de señal esperado por longitud
if len(senal_raw) < 300:
    tipo_esperado = "LATIDO (NORMAL/PAC) — ~162 muestras"
elif len(senal_raw) < 1500:
    tipo_esperado = "RITMO (NSR/AFIB) — ~1000 muestras"
else:
    tipo_esperado = "SEÑAL LARGA — requiere segmentación"
print(f"  Tipo estimado           : {tipo_esperado}")

# ── 3. Remuestreo ─────────────────────────────────────────────────────────────
if fs_origen != 250:
    senal_r = remuestrear_senal(senal_raw, fs_origen=fs_origen, fs_destino=250)
    print(f"\n  Remuestreo {fs_origen}→250 Hz     : {len(senal_raw)} → {len(senal_r)} muestras")
else:
    senal_r = senal_raw.copy()
    print(f"\n  Sin remuestreo (ya a 250 Hz)")

# ── 4. Normalización Z-score ──────────────────────────────────────────────────
senal_n = normalizar_zscore(senal_r)
print(f"  Después de Z-score      : media={senal_n.mean():.4f}, std={senal_n.std():.4f}")
print(f"  Rango normalizado       : [{senal_n.min():.3f}, {senal_n.max():.3f}]")

# ── 5. Comparar con distribución de entrenamiento ────────────────────────────
print(f"\n  Comparando con distribución del entrenamiento...")
try:
    df_train = pd.read_csv(CARPETA_SALIDA / 'X_train_tabular_MSM.csv')
    df_test  = pd.read_csv(CARPETA_SALIDA / 'X_test_tabular_MSM.csv')
    print(f"  Train shape: {df_train.shape} | Test shape: {df_test.shape}")

    # Verificar si la longitud de la señal es compatible
    # Las señales de entrenamiento son ~162 (latidos) o ~1000 (ritmos)
    clases_train = df_train['Etiqueta'].value_counts()
    print(f"  Clases en train: {clases_train.to_dict()}")
except Exception as e:
    print(f"  No se pudo cargar el CSV de entrenamiento: {e}")

# ── 6. Embedding y predicción ─────────────────────────────────────────────────
print(f"\n  Calculando embedding MSM-SC...")
try:
    modelo, le, clases = cargar_modelo()
    micro = list(np.load(str(ARCHIVO_MICRO_SHAPELETS), allow_pickle=True))
    macro = list(np.load(str(ARCHIVO_MACRO_SHAPELETS), allow_pickle=True))

    # Longitudes del diccionario de shapelets
    lens_micro = [len(s) for s in micro]
    lens_macro = [len(s) for s in macro]
    print(f"  Shapelets micro: {len(micro)} ({min(lens_micro)}-{max(lens_micro)} muestras)")
    print(f"  Shapelets macro: {len(macro)} ({min(lens_macro)}-{max(lens_macro)} muestras)")
    print(f"  Señal a clasificar: {len(senal_n)} muestras")

    if len(senal_n) < min(lens_micro):
        print(f"\n  ⚠ PROBLEMA: La señal ({len(senal_n)} m) es más corta que"
              f" el shapelet más pequeño ({min(lens_micro)} m).")
        print(f"  El embedding calculará distancias directas sin ventana deslizante.")
        print(f"  Esto puede dar resultados inválidos.")

    df_emb = embedding_una_senal(senal_n, micro, macro)
    print(f"  Embedding shape: {df_emb.shape}")
    print(f"  NaN/Inf en embedding: {df_emb.isnull().any().any()}")

    # Predicción
    y_pred  = modelo.predict(df_emb)[0]
    y_proba = modelo.predict_proba(df_emb)[0]
    clase   = str(le.inverse_transform([y_pred])[0])
    probs   = {clases[j]: round(float(y_proba[j])*100, 1) for j in range(len(clases))}
    ps      = sorted(y_proba, reverse=True)
    gap     = round(float(ps[0] - ps[1]), 4)

    print(f"\n{'='*60}")
    print(f"  RESULTADO DE CLASIFICACIÓN")
    print(f"{'='*60}")
    print(f"  Clase predicha  : {clase}")
    print(f"  Probabilidades  :")
    for c, p in sorted(probs.items(), key=lambda x: -x[1]):
        bar = '█' * int(p/5)
        print(f"    {c:8}: {p:5.1f}% {bar}")
    print(f"  Confidence gap  : {gap}")
    print(f"  Escalamiento    : {'SÍ' if gap < UMBRAL_CONFIANZA else 'NO'}")

    # ── 7. Diagnóstico del problema ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  DIAGNÓSTICO")
    print(f"{'='*60}")

    if clase == 'AFIB' and len(senal_n) < 400:
        print(f"  ⚠ PROBLEMA DETECTADO: La señal tiene {len(senal_n)} muestras")
        print(f"    pero AFIB solo debería clasificarse en señales de ~1000 muestras.")
        print(f"    Causa probable: el embedding de una señal corta contra shapelets")
        print(f"    macro produce distancias bajas artificialmente (señal < shapelet),")
        print(f"    y XGBoost las interpreta como similares a patrones AFIB.")
        print(f"\n  SOLUCIÓN: Usar solo el bloque micro para señales cortas")
        print(f"    o verificar que la señal cargada tiene la longitud correcta.")

    elif len(senal_n) > 1500:
        print(f"  ⚠ SEÑAL LARGA DETECTADA: {len(senal_n)} muestras = {len(senal_n)/250:.1f}s")
        print(f"    El modelo fue entrenado con ventanas de ~162 o ~1000 muestras.")
        print(f"    Una señal de {len(senal_n)} muestras no es comparable.")
        print(f"\n  SOLUCIÓN: Segmentar la señal antes de clasificar.")
        print(f"    Usar el segmentador.py que se va a implementar.")
    else:
        print(f"  La longitud ({len(senal_n)} muestras) parece correcta.")
        print(f"  Verificar manualmente el contenido del CSV.")

    # Top 5 distancias
    imp  = modelo.feature_importances_
    top5 = np.argsort(imp)[::-1][:5]
    cols = list(df_emb.columns)
    print(f"\n  Top 5 features más importantes:")
    for k in top5:
        print(f"    {cols[int(k)]:20}: {df_emb.iloc[0, int(k)]:.4f}"
              f" (importancia: {imp[int(k)]:.4f})")

except Exception as e:
    import traceback
    print(f"  Error en embedding/predicción: {e}")
    traceback.print_exc()

print(f"\n{'='*60}")
print(f"  Diagnóstico completo.")
print(f"{'='*60}")
