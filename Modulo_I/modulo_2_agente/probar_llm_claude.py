#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
========================================================================
 PRUEBA DE LLM — Generación de nota morfológica con Claude API
------------------------------------------------------------------------
 Script mínimo para probar el prompt v2 con la API de Anthropic.
 - Lee evidencia de un JSON (simulando salida del clasificador).
 - Llama a Claude con system + user prompt separados.
 - Guarda la respuesta para inspección.

 PORTABILIDAD: la estructura es la misma para Gemini y Llama; sólo
 cambia el cliente. Al final del archivo dejo el bloque equivalente.
========================================================================
INSTALACIÓN (una vez):
   pip install anthropic

CONFIGURACIÓN DE LA API KEY:
   - Crea cuenta en https://console.anthropic.com
   - Genera una API key
   - En PowerShell:  $env:ANTHROPIC_API_KEY="tu_api_key"
   - En Linux/Mac:   export ANTHROPIC_API_KEY="tu_api_key"
   - NO pongas la key en el código; usa variable de entorno.
========================================================================
USO:
   python probar_llm_claude.py
========================================================================
"""
import json
import os
import sys

try:
    from anthropic import Anthropic
except ImportError:
    print("Instala primero: pip install anthropic")
    sys.exit(1)

# ----------------------------------------------------------------------
# 1) EVIDENCIA DE EJEMPLO (sustituir por la salida real del clasificador)
# ----------------------------------------------------------------------
EVIDENCIA_EJEMPLO_AFIB = {
    "duracion_senal_seg": 60,
    "frecuencia_cardiaca_lpm": 78,
    "regularidad_RR": "marcadamente irregular",
    "presencia_ondas_P": "ausentes o desorganizadas",
    "ancho_QRS_ms": 92,
    "segmentos_ritmo": [
        {"inicio_s": 0, "fin_s": 60,
         "categoria_interna": "ritmo_clase_2", "prob": 0.91}
    ],
    "segmentos_latido": {
        "total": 78,
        "tipo_A": {"n": 78, "prop": 1.0},
        "tipo_B": {"n": 0,  "prop": 0.0}
    },
    "confidence_gap": 0.87,
    "alerta_baja_confianza": False
}

EVIDENCIA_EJEMPLO_PAC = {
    "duracion_senal_seg": 60,
    "frecuencia_cardiaca_lpm": 72,
    "regularidad_RR": "ligeramente irregular",
    "presencia_ondas_P": "presentes, algunas adelantadas",
    "ancho_QRS_ms": 88,
    "segmentos_ritmo": [
        {"inicio_s": 0, "fin_s": 60,
         "categoria_interna": "ritmo_clase_1", "prob": 0.94}
    ],
    "segmentos_latido": {
        "total": 93,
        "tipo_A": {"n": 71, "prop": 0.76},
        "tipo_B": {"n": 22, "prop": 0.24}
    },
    "confidence_gap": 0.70,
    "alerta_baja_confianza": False
}

# ----------------------------------------------------------------------
# 2) PROMPTS  (system fijo + user con evidencia inyectada)
# ----------------------------------------------------------------------
SYSTEM_PROMPT = """Eres un asistente de apoyo informativo para cardiólogos. Tu única función es
describir, en lenguaje médico claro y conciso, los HALLAZGOS MORFOLÓGICOS
observados por un sistema automático de análisis sobre una señal de ECG de
derivación única (Lead II).

ESTILO Y FORMATO:
- Redactas en español, usando la terminología y los criterios de
  interpretación de las guías AHA/ACCF/HRS para electrocardiografía.
- La nota se estructura en los CINCO PILARES estándar de interpretación:
    1. FRECUENCIA CARDÍACA
    2. RITMO
    3. EJE ELÉCTRICO (si la evidencia lo permite; en derivación única
       puede indicarse "no evaluable")
    4. INTERVALOS Y ONDAS (PR, QRS, ondas P, segmento ST, onda T,
       siempre que la evidencia los respalde)
    5. INTERPRETACIÓN GLOBAL Y RECOMENDACIÓN
- Tono: profesional, descriptivo, dirigido a otro médico.
- Si todo es normal, descríbelo como un trazado dentro de límites normales.

REGLAS DURAS — MUNDO CERRADO:
- NO uses los nombres clínicos de las cuatro clases del sistema. NO escribas
  "fibrilación auricular", "ritmo sinusal normal", "contracción auricular
  prematura" ni "latido normal" como diagnósticos nombrados. Describe LO QUE
  SE OBSERVA (p. ej. "intervalos R-R irregulares sin ondas P organizadas").
- NO menciones diagnósticos fuera de las 4 clases del sistema: NO hables de
  infarto, isquemia, hipertrofia, bloqueos de rama, taquicardias ventriculares,
  flutter, pre-excitación, ni cualquier otro hallazgo no cuantificado.
- NO menciones derivaciones distintas de la derivación II.
- NO inventes valores numéricos que no estén en la evidencia.
- Si un pilar no puede evaluarse, escribe "no evaluable con la información
  disponible".
- NO emites diagnóstico definitivo; tu nota es de apoyo informativo.

CONDICIÓN DE ESCALAMIENTO:
Si en la evidencia `alerta_baja_confianza` es true, NO redactes nota.
Responde EXACTAMENTE:
"CASO AMBIGUO — Confianza del sistema por debajo del umbral. Se recomienda
revisión directa por cardiólogo antes de emitir nota clínica."
"""

USER_PROMPT_TEMPLATE = """EVIDENCIA DEL ANÁLISIS AUTOMÁTICO (sistema MSM-SC + XGBoost):

```json
{evidencia_json}
```

INSTRUCCIÓN:
Redacta la nota clínica siguiendo los CINCO PILARES de AHA/ACCF/HRS,
describiendo únicamente los hallazgos morfológicos que la evidencia
respalda. No nombres clases clínicas: describe lo observable. Sigue
todas las reglas del system prompt.
"""

# ----------------------------------------------------------------------
# 3) LLAMADA A LA API
# ----------------------------------------------------------------------
def generar_nota(evidencia: dict, model: str = "claude-opus-4-7") -> str:
    """Llama a Claude con system+user separados y devuelve la nota."""
    cliente = Anthropic()  # toma ANTHROPIC_API_KEY del entorno
    user_prompt = USER_PROMPT_TEMPLATE.format(
        evidencia_json=json.dumps(evidencia, indent=2, ensure_ascii=False)
    )
    respuesta = cliente.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    # respuesta.content es lista de bloques; tomamos el texto
    return "".join(b.text for b in respuesta.content if b.type == "text")


# ----------------------------------------------------------------------
# 4) PRUEBA
# ----------------------------------------------------------------------
if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: define ANTHROPIC_API_KEY como variable de entorno.")
        sys.exit(1)

    casos = {"AFIB_simulado": EVIDENCIA_EJEMPLO_AFIB,
             "PAC_simulado":  EVIDENCIA_EJEMPLO_PAC}

    for nombre, ev in casos.items():
        print("=" * 70)
        print(f" CASO: {nombre}")
        print("=" * 70)
        nota = generar_nota(ev)
        print(nota)
        with open(f"nota_{nombre}.txt", "w", encoding="utf-8") as f:
            f.write(nota)
        print(f"\n[guardado en nota_{nombre}.txt]\n")


# ----------------------------------------------------------------------
# EQUIVALENTES PARA OTROS LLM (para cuando vayas a benchmark)
# ----------------------------------------------------------------------
#
# GEMINI (Google):
#   pip install google-generativeai
#   import google.generativeai as genai
#   genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
#   modelo = genai.GenerativeModel("gemini-1.5-pro",
#                                  system_instruction=SYSTEM_PROMPT)
#   resp = modelo.generate_content(user_prompt,
#            generation_config={"temperature": 0.2, "max_output_tokens": 1024})
#   nota = resp.text
#
# LLAMA local (llama-cpp-python — lo que ya usas):
#   from llama_cpp import Llama
#   llm = Llama(model_path="ruta/llama-3-8b-instruct.Q4_K_M.gguf", n_ctx=4096)
#   resp = llm.create_chat_completion(
#       messages=[{"role":"system","content":SYSTEM_PROMPT},
#                 {"role":"user","content":user_prompt}],
#       temperature=0.2, max_tokens=1024)
#   nota = resp["choices"][0]["message"]["content"]
#
# El SYSTEM_PROMPT y USER_PROMPT son los mismos en los tres casos.
# Esto garantiza que la comparación del benchmark sea justa.
