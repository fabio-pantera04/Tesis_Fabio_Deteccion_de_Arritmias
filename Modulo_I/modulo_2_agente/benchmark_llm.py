#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
========================================================================
 BENCHMARK LLM v3 — Notas ancladas al estándar AHA/ACCF/HRS
------------------------------------------------------------------------
 Cambios vs v2:
   - Carga estandar_aha_accf_hrs.json y lo inyecta en el system prompt.
   - El agente ahora SÍ puede usar afirmaciones AHA canónicas (códigos
     20, 23, 32, 53, 1, 2, 3) en lugar de solo describir morfología.
   - Modificadores AHA (frequent, occasional, multifocal, etc.) habilitados.
   - Reglas de mundo cerrado ahora citan códigos AHA específicos.

 REQUISITO: el archivo estandar_aha_accf_hrs.json debe estar en el
 mismo directorio que este script.

 Modelos usados:
   - claude-sonnet-4-6     (Anthropic)
   - gemini-2.5-flash       (Google, vía google-genai)
   - llama-3.3-70b-versatile (Meta, vía Groq)
========================================================================
"""
import json
import os
import sys
import time

# ----------------------------------------------------------------------
# 1) CARGAR EL ESTÁNDAR AHA/ACCF/HRS
# ----------------------------------------------------------------------
RUTA_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "estandar_aha_accf_hrs.json")
if not os.path.exists(RUTA_JSON):
    print(f"ERROR: no encuentro el archivo {RUTA_JSON}")
    print("Coloca estandar_aha_accf_hrs.json junto a este script.")
    sys.exit(1)

with open(RUTA_JSON, "r", encoding="utf-8") as f:
    ESTANDAR = json.load(f)

ESTANDAR_TEXTO = json.dumps(ESTANDAR, indent=2, ensure_ascii=False)
print(f"Estándar cargado: {len(ESTANDAR_TEXTO)} caracteres (~{len(ESTANDAR_TEXTO)//4} tokens estimados)")

# ----------------------------------------------------------------------
# 2) EVIDENCIA DE EJEMPLO (sustituir por salida real del clasificador)
# ----------------------------------------------------------------------
EVIDENCIA_AFIB = {
    "duracion_senal_seg": 60, "frecuencia_cardiaca_lpm": 78,
    "regularidad_RR": "marcadamente irregular",
    "presencia_ondas_P": "ausentes o desorganizadas",
    "ancho_QRS_ms": 92,
    "segmentos_ritmo": [{"inicio_s": 0, "fin_s": 60,
                         "clase_sistema": "AFIB", "prob": 0.91}],
    "segmentos_latido": {"total": 78,
                         "tipo_A_normales": {"n": 78, "prop": 1.0},
                         "tipo_B_anormales": {"n": 0, "prop": 0.0}},
    "confidence_gap": 0.87, "alerta_baja_confianza": False
}

EVIDENCIA_PAC = {
    "duracion_senal_seg": 60, "frecuencia_cardiaca_lpm": 72,
    "regularidad_RR": "ligeramente irregular",
    "presencia_ondas_P": "presentes, algunas adelantadas",
    "ancho_QRS_ms": 88,
    "segmentos_ritmo": [{"inicio_s": 0, "fin_s": 60,
                         "clase_sistema": "NSR", "prob": 0.94}],
    "segmentos_latido": {"total": 93,
                         "tipo_A_normales": {"n": 71, "prop": 0.76},
                         "tipo_B_PAC": {"n": 22, "prop": 0.24}},
    "confidence_gap": 0.70, "alerta_baja_confianza": False
}

# ----------------------------------------------------------------------
# 3) PROMPT v3 — ahora con anclaje al estándar
# ----------------------------------------------------------------------
SYSTEM_PROMPT = f"""Eres un asistente de apoyo informativo para cardiólogos. Tu función es
redactar notas clínicas estructuradas para análisis de ECG en derivación II,
ancladas al estándar AHA/ACCF/HRS para estandarización e interpretación del
electrocardiograma (Partes I, II y III, 2007–2009).

DOCUMENTO DE REFERENCIA — ESTÁNDAR AHA/ACCF/HRS:
A continuación se incluye el extracto operativo del estándar que regula tu
generación de notas. DEBES usar este documento como fuente única de verdad
para terminología diagnóstica, umbrales numéricos, afirmaciones canónicas
permitidas y reglas de mundo cerrado.

```json
{ESTANDAR_TEXTO}
```

REGLAS DE OPERACIÓN — todas obligatorias:

1. ESTRUCTURA: redacta la nota siguiendo los CINCO PILARES definidos en
   `pilares_AHA_ACCF_HRS` del estándar, en este orden:
     1. Frecuencia cardíaca
     2. Ritmo
     3. Eje eléctrico (declarar 'no evaluable' por derivación única)
     4. Intervalos y ondas
     5. Interpretación global y recomendación

2. TERMINOLOGÍA: usa ÚNICAMENTE las afirmaciones AHA Parte II listadas en
   `afirmaciones_AHA_permitidas` para cada pilar. Estas son tus únicas
   afirmaciones diagnósticas válidas:
     - Sinus rhythm (código 20)
     - Sinus arrhythmia (código 23)
     - Atrial premature complex(es) (código 32)
     - Atrial fibrillation (código 53)
     - Sinus tachycardia (código 21)
     - Sinus bradycardia (código 22)
     - Normal ECG (código 1)
     - Otherwise normal ECG (código 2)
     - Abnormal ECG (código 3)

3. MODIFICADORES: para refinar las afirmaciones, usa SOLO los modificadores
   AHA Parte II Tabla 3 listados en `modificadores_AHA_permitidos` (frequent,
   occasional, multiple, in a bigeminal pattern, multifocal, unifocal, with
   a rapid/slow ventricular response, etc.).

4. MUNDO CERRADO — ESTRICTO: NO emitas ninguna afirmación de las categorías
   listadas en `reglas_de_mundo_cerrado.categorias_AHA_prohibidas`. Esto
   incluye: bloqueos AV, bloqueos de rama, hipertrofia, isquemia, infarto,
   alteraciones de ST/T, marcapasos, taquicardias ventriculares, y los
   diagnósticos clínicos secundarios (códigos 200–231). NO uses los términos
   vetados por AHA listados en
   `reglas_de_mundo_cerrado.terminos_explicitamente_no_recomendados_por_AHA`.

5. EJE ELÉCTRICO: SIEMPRE declarar 'no evaluable con la información
   disponible (registro de derivación única)' tal como indica
   `pilares_AHA_ACCF_HRS.3_eje_electrico.restriccion_para_el_agente`.

6. SEGMENTO ST Y ONDA T: SIEMPRE declarar 'no evaluable con la información
   disponible' tal como indica
   `pilares_AHA_ACCF_HRS.4_intervalos_y_ondas.componentes.segmento_ST_y_onda_T`.

7. ANTI-INFERENCIA: si describes propiedades morfológicas NO explícitamente
   cuantificadas en la evidencia (p. ej., homogeneidad morfológica deducida
   de un conteo), marca esa afirmación entre paréntesis como
   '(inferido del conteo de categorías, no medido directamente)'.

8. CIERRE: termina la nota con la afirmación AHA de cierre apropiada
   (código 1, 2 o 3) según los hallazgos. Incluye recordatorio metodológico
   y recomendación clínica según `5_interpretacion_global_y_recomendacion`.

9. NO INVENTAR VALORES: solo reporta valores numéricos presentes en la
   evidencia o derivables de ella; declara 'no evaluable' lo demás.

10. ESCALAMIENTO: si en la evidencia `alerta_baja_confianza` es true, NO
    redactes nota. Responde EXACTAMENTE: 'CASO AMBIGUO — Confianza del
    sistema por debajo del umbral. Se recomienda revisión directa por
    cardiólogo antes de emitir nota clínica.'

IDIOMA: español. TONO: profesional, dirigido a otro médico, conciso.
"""

USER_PROMPT_TEMPLATE = """EVIDENCIA DEL ANÁLISIS AUTOMÁTICO (sistema MSM-SC + XGBoost,
derivación II, 60 segundos):

```json
{evidencia_json}
```

INSTRUCCIÓN:
Redacta la nota clínica siguiendo los CINCO PILARES del estándar AHA/ACCF/HRS,
con las afirmaciones canónicas permitidas y los modificadores apropiados.
Respeta TODAS las reglas del system prompt. Cita el código AHA correspondiente
entre paréntesis al usar una afirmación canónica (p. ej., 'Atrial fibrillation
(AHA 53)' o 'Sinus rhythm with frequent atrial premature complexes
(AHA 20 + 32 + 310)').
"""


def construir_user_prompt(ev):
    return USER_PROMPT_TEMPLATE.format(
        evidencia_json=json.dumps(ev, indent=2, ensure_ascii=False))


# ----------------------------------------------------------------------
# 4) PROVEEDORES (idénticos a v2, sin cambios)
# ----------------------------------------------------------------------
def llamar_claude(ev):
    from anthropic import Anthropic
    cliente = Anthropic()
    t0 = time.time()
    resp = cliente.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": construir_user_prompt(ev)}],
    )
    return ("".join(b.text for b in resp.content if b.type == "text"),
            time.time() - t0)


def llamar_gemini(ev):
    from google import genai
    from google.genai import types
    cliente = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    t0 = time.time()
    resp = cliente.models.generate_content(
        model="gemini-2.5-flash",
        contents=construir_user_prompt(ev),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=2048,
        ),
    )
    return (resp.text, time.time() - t0)


def llamar_groq_llama(ev):
    from groq import Groq
    cliente = Groq(api_key=os.environ["GROQ_API_KEY"])
    t0 = time.time()
    resp = cliente.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=2048,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": construir_user_prompt(ev)},
        ],
    )
    return (resp.choices[0].message.content, time.time() - t0)


PROVEEDORES = [
    ("Claude_Sonnet-4-6", llamar_claude, "ANTHROPIC_API_KEY"),
    ("Gemini_2.5-Flash",  llamar_gemini, "GOOGLE_API_KEY"),
    ("Llama_3.3-70B",     llamar_groq_llama, "GROQ_API_KEY"),
]
CASOS = {"AFIB_simulado": EVIDENCIA_AFIB, "PAC_simulado": EVIDENCIA_PAC}


def main():
    faltan = [k for _, _, k in PROVEEDORES if not os.environ.get(k)]
    if faltan:
        print("ERROR: faltan variables:", ", ".join(faltan)); sys.exit(1)
    resumen = []
    for nombre_caso, ev in CASOS.items():
        print("\n" + "#" * 70)
        print(f"# CASO: {nombre_caso}")
        print("#" * 70)
        for nombre_modelo, fn, _ in PROVEEDORES:
            print(f"\n>>> Llamando a {nombre_modelo}...")
            try:
                nota, dt = fn(ev)
            except Exception as e:
                print(f"  ERROR: {e}"); continue
            print(f"  (tiempo: {dt:.2f}s, longitud: {len(nota)} chars)")
            print("-" * 70); print(nota); print("-" * 70)
            archivo = f"nota_v3_{nombre_caso}__{nombre_modelo}.txt"
            with open(archivo, "w", encoding="utf-8") as f:
                f.write(f"# CASO: {nombre_caso}\n")
                f.write(f"# MODELO: {nombre_modelo}\n")
                f.write(f"# PROMPT: v3 (con estándar AHA/ACCF/HRS inyectado)\n")
                f.write(f"# TIEMPO: {dt:.2f}s\n\n{nota}")
            resumen.append({"caso": nombre_caso, "modelo": nombre_modelo,
                            "tiempo_s": round(dt, 2),
                            "longitud_chars": len(nota), "archivo": archivo})
    print("\n\n" + "=" * 70)
    print(" RESUMEN v3")
    print("=" * 70)
    print(f"{'CASO':<18} {'MODELO':<22} {'TIEMPO(s)':>10} {'LONGITUD':>10}")
    print("-" * 70)
    for r in resumen:
        print(f"{r['caso']:<18} {r['modelo']:<22} "
              f"{r['tiempo_s']:>10.2f} {r['longitud_chars']:>10}")
    with open("resumen_benchmark_v3.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
