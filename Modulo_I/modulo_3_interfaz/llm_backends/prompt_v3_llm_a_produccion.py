"""
Prompt v3 del Módulo LLM-A — Versión de producción
────────────────────────────────────────────────────────────────────
Prompt validado empíricamente mediante el experimento LLM-B v2
descrito en la Sección 3.6.4 y Capítulo 4 de la tesis.

Configuración de producción confirmada:
    - Modelo: Gemini 2.5 Flash (SDK google-genai)
    - K = 0 (zero-shot, sin ejemplos few-shot inyectados)
    - Idioma: español clínico formal
    - Derivación: II modificada exclusivamente

Justificación empírica de K=0 en producción:
    - Class Fidelity global: 93.57% con K=0 vs 82.01% con K=8
    - Class Fidelity AFIB: 91% con K=0 vs 59% con K=8
    - Morphological Grounding: 100% en ambas configuraciones
    - Evidence Grounding: 100% en ambas configuraciones
    - Consumo de tokens ~4x menor con K=0
    (véase Tabla 4.7.2 y Sección 4.8.3 de la tesis)

Este archivo se integra en llm_backends/base.py sustituyendo
el prompt anterior. La única entrada externa por llamada es
el JSON del clasificador (contrato de 6 campos formalizado
en la Tabla 3.5.1 de la tesis).
"""

from __future__ import annotations
import json
from typing import Dict, Any


# ═══════════════════════════════════════════════════════════════════
# BLOQUE 1 — ROL CLÍNICO Y CONTEXTO
# ═══════════════════════════════════════════════════════════════════
BLOQUE_1_ROL = """Eres un asistente clínico especializado en la \
interpretación de electrocardiograma en derivación II modificada. \
Tu tarea es generar una nota descriptiva concisa que un cardiólogo \
especialista utilizará como punto de partida para su interpretación \
definitiva del registro.

Redacta la nota en español clínico formal, empleando la primera \
persona plural impersonal característica del informe médico \
("se observa...", "se identifica...", "se aprecia..."). El registro \
lingüístico debe ser sobrio, profesional y preciso, siguiendo la \
convención de reportes cardiológicos estandarizada (SCP-ECG, \
IEC 61754-3:2009).

IMPORTANTE: tu nota NO reemplaza el juicio del cardiólogo. Constituye \
un apoyo informativo que traduce la evidencia geométrica producida \
por el clasificador MSM-SC + XGBoost del Módulo I a lenguaje clínico \
interpretable."""


# ═══════════════════════════════════════════════════════════════════
# BLOQUE 2 — TEMPLATE DE EVIDENCIA DEL CLASIFICADOR
# Se rellena en tiempo de llamada con el JSON real
# ═══════════════════════════════════════════════════════════════════
BLOQUE_2_EVIDENCIA_TEMPLATE = """=== EVIDENCIA DEL CLASIFICADOR ===
{json_clasificador}

INTERPRETACIÓN DE LOS CAMPOS:
- prediccion: clase AAMI EC57:2012 predicha (una de NORMAL, PAC, NSR, AFIB)
- modelo: submodelo del clasificador dual (LATIDO para clases de \
latido, RITMO para clases rítmicas)
- probabilidades: posteriores del clasificador por clase
- confidence_gap: |P(clase_1) - P(clase_0)|; valores cercanos a 1 \
indican alta confianza, valores cercanos a 0 indican baja confianza
- alerta_baja_confianza: True cuando confidence_gap < 0.25 \
(umbral de escalación a especialista)
- distancias_top3_msm: los tres shapelets morfológicos más \
discriminativos con su distancia MSM a la señal analizada, \
ordenados por importancia según feature_importances_ de XGBoost"""


# ═══════════════════════════════════════════════════════════════════
# BLOQUE 3 — GUÍA DE DESCRIPCIÓN MORFOLÓGICA POR CLASE
# Síntesis destilada de las guías AHA/ACCF/HRS
# ═══════════════════════════════════════════════════════════════════
BLOQUE_3_GUIA_MORFOLOGICA = """=== GUÍA DE DESCRIPCIÓN MORFOLÓGICA \
POR CLASE ===
Al describir la señal, incorpora las referencias morfológicas \
apropiadas a la clase predicha en derivación II modificada:

NORMAL (latido normal en derivación II):
- Complejo QRS estrecho (< 120 ms) de morfología normal
- Onda P monofásica de polaridad positiva precediendo el QRS
- Intervalo PR dentro de rangos fisiológicos (120-200 ms)
- Onda T concordante con la deflexión principal del QRS

PAC (contracción auricular prematura en derivación II):
- Latido de aparición prematura respecto al ritmo de base
- Onda P prematura de morfología distinta a las P sinusales
- Complejo QRS asociado estrecho, conducción intraventricular normal
- Pausa compensatoria incompleta característica

NSR (ritmo sinusal normal en derivación II):
- Ondas P monofásicas precediendo cada complejo QRS
- Relación auriculoventricular 1:1 conservada
- Intervalos R-R regulares con variabilidad fisiológica menor al 10%
- Frecuencia cardíaca 60-100 lpm

AFIB (fibrilación auricular en derivación II):
- Ausencia de ondas P discernibles
- Oscilaciones basales irregulares reemplazando la actividad auricular
- Intervalos R-R irregularmente irregulares, sin patrón cíclico
- Respuesta ventricular irregularmente irregular
- Complejos QRS conservan morfología estrecha (conducción normal)"""


# ═══════════════════════════════════════════════════════════════════
# BLOQUE 4 — INSTRUCCIONES DE CITACIÓN EXPLÍCITA (crítico)
# Este es el bloque más restrictivo del prompt y el responsable
# de que Morphological Grounding y Evidence Grounding alcancen 100%
# en el experimento LLM-B v2.
# ═══════════════════════════════════════════════════════════════════
BLOQUE_4_CITACION = """=== INSTRUCCIONES DE CITACIÓN DE EVIDENCIA ===
La nota generada DEBE citar explícitamente los siguientes elementos \
del JSON del clasificador (esto no es opcional):

1. SHAPELET PRINCIPAL: menciona por nombre el shapelet con menor \
distancia MSM del campo distancias_top3_msm. Ejemplo:
   "El shapelet Macro_S54 (distancia MSM = 1.31) constituye la \
evidencia geométrica principal."

2. DISTANCIA MSM: reporta el valor numérico de la distancia del \
shapelet principal. Es la magnitud que respalda la clasificación.

3. CONFIDENCE GAP: menciona el confidence_gap con recomendación \
condicional según su valor:
   - Si confidence_gap < 0.25: "Se identifica confianza baja \
(delta p = X.XX). Se recomienda revisión inmediata del cardiólogo."
   - Si 0.25 <= confidence_gap < 0.40: "Confianza moderada \
(delta p = X.XX). Sugiere confirmación visual."
   - Si confidence_gap >= 0.40: "El confidence gap de X.XX indica \
alta certeza del clasificador."

4. INTERPRETACIÓN MORFOLÓGICA: describe la morfología específica \
observada según la clase predicha (usa la guía del Bloque 3). \
NO utilices frases telegráficas como "ritmo sinusal, ECG normal". \
Elabora describiendo los rasgos morfológicos concretos."""


# ═══════════════════════════════════════════════════════════════════
# BLOQUE 5 — FORMATO DE SALIDA
# ═══════════════════════════════════════════════════════════════════
BLOQUE_5_FORMATO = """=== FORMATO DE SALIDA ===
Genera una nota clínica de 3 a 5 oraciones que incluya:

1. Descripción morfológica temporal de lo observado en derivación II
2. Cita de la evidencia MSM que respalda tus observaciones
3. Reporte del confidence gap con recomendación condicional
4. Recomendación explícita de revisión especialista cuando aplique

Output ÚNICAMENTE el texto de la nota clínica, sin preámbulos, sin \
explicaciones metadata, sin encabezados."""


# ═══════════════════════════════════════════════════════════════════
# BLOQUE 6 — RESTRICCIONES DE MUNDO CERRADO (Safety-by-Design)
# ═══════════════════════════════════════════════════════════════════
BLOQUE_6_RESTRICCIONES = """=== RESTRICCIONES DE SEGURIDAD CLÍNICA ===
Restricciones que debes respetar SIEMPRE:

- No prescribas fármacos, dosis ni procedimientos terapéuticos.
- No emitas un diagnóstico definitivo; usa formulaciones descriptivas \
("compatible con", "se observa un patrón de", "sugestivo de").
- Solo utiliza terminología de las 4 clases AAMI EC57:2012 admitidas: \
NORMAL, PAC, NSR, AFIB. No introduzcas clases diagnósticas fuera de \
este conjunto.
- Si la evidencia es contradictoria o de baja confianza, indica \
explícitamente que requiere revisión humana.
- No uses lenguaje que sugiera certeza absoluta ("se confirma", \
"es seguro que"). Prefiere formulaciones probabilísticas."""


# ═══════════════════════════════════════════════════════════════════
# API PÚBLICA — ensamble del prompt completo
# ═══════════════════════════════════════════════════════════════════

def construir_prompt_llm_a(json_clasificador: Dict[str, Any]) -> str:
    """
    Construye el prompt v3 completo del Módulo LLM-A para una llamada
    individual al agente Gemini 2.5 Flash.

    Parameters
    ----------
    json_clasificador : dict
        Contrato JSON del Módulo I con los 6 campos formalizados:
        prediccion, modelo, probabilidades, confidence_gap,
        alerta_baja_confianza, distancias_top3_msm.

    Returns
    -------
    str
        Prompt completo listo para enviar al modelo Gemini 2.5 Flash
        mediante client.models.generate_content(contents=prompt).

    Notes
    -----
    En producción se usa K=0 (sin ejemplos few-shot inyectados) según
    la recomendación operacional derivada del experimento LLM-B v2
    (véase Sección 4.8 de la tesis).
    """
    bloque_2_relleno = BLOQUE_2_EVIDENCIA_TEMPLATE.format(
        json_clasificador=json.dumps(
            json_clasificador, ensure_ascii=False, indent=2
        )
    )

    prompt = "\n\n".join([
        BLOQUE_1_ROL,
        bloque_2_relleno,
        BLOQUE_3_GUIA_MORFOLOGICA,
        BLOQUE_4_CITACION,
        BLOQUE_5_FORMATO,
        BLOQUE_6_RESTRICCIONES,
    ])
    return prompt


def validar_contrato_json(json_clasificador: Dict[str, Any]) -> None:
    """
    Valida el JSON del clasificador antes de invocar al agente.

    Aplica el principio de Safety-by-Design: cualquier JSON malformado
    o incompleto se rechaza ANTES de la llamada al LLM, garantizando
    que el agente solo reciba evidencia estructuralmente correcta.

    Raises
    ------
    ValueError : si el contrato JSON está incompleto o malformado.
    """
    campos_obligatorios = {
        "prediccion", "modelo", "probabilidades",
        "confidence_gap", "alerta_baja_confianza",
        "distancias_top3_msm",
    }
    faltantes = campos_obligatorios - set(json_clasificador.keys())
    if faltantes:
        raise ValueError(
            f"Contrato JSON incompleto. Campos faltantes: {faltantes}"
        )

    clases_validas = {"NORMAL", "PAC", "NSR", "AFIB"}
    if json_clasificador["prediccion"] not in clases_validas:
        raise ValueError(
            f"Clase inválida: {json_clasificador['prediccion']}. "
            f"Debe ser una de {clases_validas} (AAMI EC57:2012)."
        )

    if json_clasificador["modelo"] not in {"LATIDO", "RITMO"}:
        raise ValueError(
            f"Modelo inválido: {json_clasificador['modelo']}. "
            f"Debe ser LATIDO o RITMO."
        )

    gap = json_clasificador["confidence_gap"]
    if not (0.0 <= gap <= 1.0):
        raise ValueError(
            f"confidence_gap fuera de rango: {gap}. Debe estar en [0, 1]."
        )

    if len(json_clasificador["distancias_top3_msm"]) != 3:
        raise ValueError(
            "distancias_top3_msm debe contener exactamente 3 shapelets."
        )


# ═══════════════════════════════════════════════════════════════════
# INTEGRACIÓN CON base.py — llamada al agente Gemini
# ═══════════════════════════════════════════════════════════════════

def generar_nota_clinica(
    json_clasificador: Dict[str, Any],
    api_key: str = None,
    model_name: str = "gemini-2.5-flash",
) -> str:
    """
    Genera una nota clínica invocando al agente Gemini 2.5 Flash con
    el prompt v3 y el JSON del clasificador validado.

    Parameters
    ----------
    json_clasificador : dict
        Contrato JSON del Módulo I (validado antes de la llamada).
    api_key : str, optional
        Clave del API de Gemini. Si es None, se lee de la variable
        de entorno GEMINI_API_KEY.
    model_name : str
        Nombre del modelo Gemini a utilizar.

    Returns
    -------
    str
        Nota clínica generada por el agente en español clínico formal.
    """
    import os
    from google import genai

    # 1. Validación del contrato JSON (Safety-by-Design)
    validar_contrato_json(json_clasificador)

    # 2. Semáforo — determinar la política de escalación
    #    (información complementaria para el frontend, no cambia el prompt)
    gap = json_clasificador["confidence_gap"]
    if gap < 0.25:
        semaforo = "rojo"    # revisión inmediata
    elif gap < 0.40:
        semaforo = "amarillo"  # confirmación visual
    else:
        semaforo = "verde"    # alta certeza

    # 3. Construcción del prompt v3
    prompt = construir_prompt_llm_a(json_clasificador)

    # 4. Invocación del agente Gemini
    api_key = api_key or os.environ["GEMINI_API_KEY"]
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )

    return response.text.strip() if response.text else ""


# ═══════════════════════════════════════════════════════════════════
# EJEMPLO DE USO
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    ejemplo_json = {
        "prediccion": "AFIB",
        "modelo": "RITMO",
        "probabilidades": {"AFIB": 0.91, "NSR": 0.09},
        "confidence_gap": 0.82,
        "alerta_baja_confianza": False,
        "distancias_top3_msm": {
            "Macro_S54": 1.31,
            "Macro_S56": 2.03,
            "Macro_S22": 6.80,
        },
    }

    print("=" * 70)
    print("PROMPT v3 COMPLETO PARA EL CASO EJEMPLO")
    print("=" * 70)
    print(construir_prompt_llm_a(ejemplo_json))
    print()
    print("=" * 70)
    print("NOTA GENERADA POR GEMINI 2.5 FLASH")
    print("=" * 70)
    try:
        nota = generar_nota_clinica(ejemplo_json)
        print(nota)
    except KeyError:
        print("Error: define GEMINI_API_KEY para probar la generación.")
