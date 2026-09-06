"""
Interfaz comun para todos los backends LLM. Permite intercambiar
Gemini, Claude, Groq, Llama-local o mock sin tocar el resto del sistema.

────────────────────────────────────────────────────────────────────
Prompt v3 de produccion — validado experimentalmente
────────────────────────────────────────────────────────────────────
Este base.py incorpora el prompt v3 de seis bloques disenado y
validado en el experimento LLM-B v2 (véase Capitulo 3.6 y Sec 4.7-4.8
de la tesis), adaptado al contrato agregado por senal completa de
60 segundos que produce el sistema de produccion.

Diferencias respecto al prompt del LLM-B experimental:
    - Contexto: senal completa (60s), no latido individual
    - Evidencia: clase predominante + distribucion por clase +
      anomalias con timestamps, en lugar de top-3 shapelets MSM
    - Longitud: nota de 5 a 8 oraciones (vs 3-5 del LLM-B)
    - Todo lo demas identico: 6 bloques, espanol clinico formal,
      Lead II modificado, Safety-by-Design
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Salida estandar de cualquier backend."""
    text: str
    backend_name: str
    model_id: str
    latency_seconds: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None


# ═══════════════════════════════════════════════════════════════════
# PROMPT V3 DE PRODUCCION — SEIS BLOQUES
# ═══════════════════════════════════════════════════════════════════

BLOQUE_1_ROL = """Eres un asistente clinico especializado en la \
interpretacion de electrocardiograma en derivacion II modificada. \
Tu tarea es generar una nota descriptiva concisa que un cardiologo \
especialista utilizara como punto de partida para su interpretacion \
definitiva del registro completo.

Redacta la nota en espanol clinico formal, empleando la primera \
persona plural impersonal caracteristica del informe medico \
("se observa...", "se identifica...", "se aprecia..."). El registro \
linguistico debe ser sobrio, profesional y preciso, siguiendo la \
convencion de reportes cardiologicos estandarizada (SCP-ECG, \
IEC 61754-3:2009).

IMPORTANTE: tu nota NO reemplaza el juicio del cardiologo. \
Constituye un apoyo informativo que traduce la evidencia del \
clasificador dual MSM-SC + XGBoost del Modulo I a lenguaje clinico \
interpretable."""


BLOQUE_3_GUIA_MORFOLOGICA = """=== GUIA DE DESCRIPCION MORFOLOGICA \
POR CLASE ===
Al describir la senal, incorpora las referencias morfologicas \
apropiadas a la clase predominante y a las anomalias detectadas, \
en derivacion II modificada:

NORMAL (latido normal en derivacion II):
- Complejo QRS estrecho (< 120 ms) de morfologia normal
- Onda P monofasica de polaridad positiva precediendo el QRS
- Intervalo PR dentro de rangos fisiologicos (120-200 ms)
- Onda T concordante con la deflexion principal del QRS

PAC (contraccion auricular prematura en derivacion II):
- Latido de aparicion prematura respecto al ritmo de base
- Onda P prematura de morfologia distinta a las P sinusales
- Complejo QRS asociado estrecho, conduccion intraventricular normal
- Pausa compensatoria incompleta caracteristica

NSR (ritmo sinusal normal en derivacion II):
- Ondas P monofasicas precediendo cada complejo QRS
- Relacion auriculoventricular 1:1 conservada
- Intervalos R-R regulares con variabilidad fisiologica menor al 10%
- Frecuencia cardiaca 60-100 lpm

AFIB (fibrilacion auricular en derivacion II):
- Ausencia de ondas P discernibles
- Oscilaciones basales irregulares reemplazando la actividad auricular
- Intervalos R-R irregularmente irregulares, sin patron ciclico
- Respuesta ventricular irregularmente irregular
- Complejos QRS conservan morfologia estrecha (conduccion normal)"""


BLOQUE_4_CITACION = """=== INSTRUCCIONES DE CITACION DE EVIDENCIA ===
La nota generada DEBE citar explicitamente los siguientes elementos \
de la evidencia del clasificador (esto no es opcional):

1. CLASE PREDOMINANTE: menciona la clase de latido y la clase de \
ritmo predominantes segun el clasificador dual (LATIDO y RITMO).

2. DISTRIBUCION POR CLASE: reporta los porcentajes principales de \
la distribucion por clase cuando sean informativos. Ejemplo: \
"con 85% de latidos NORMAL y 15% de contracciones auriculares \
prematuras" o "ritmo predominantemente sinusal (90%) con episodios \
aislados de fibrilacion auricular (10%)".

3. ANOMALIAS DETECTADAS: cita las anomalias especificas con sus \
timestamps cuando sean relevantes. Ejemplo: "se identifican \
contracciones auriculares prematuras aisladas en los segundos \
12.3 y 47.8".

4. CONFIANZA DEL CLASIFICADOR: reporta las medias de confidence gap \
con recomendacion condicional:
   - Si mean_confidence_gap_beat < 0.25 o mean_confidence_gap_rhythm < 0.25: \
"Se identifica confianza baja del clasificador. Se recomienda \
revision inmediata del cardiologo."
   - Si global_escalation_required es True: la ultima oracion \
DEBE recomendar explicitamente revision cardiologica.
   - En otro caso: menciona que la confianza del clasificador es \
adecuada para uso interpretativo.

5. INTERPRETACION MORFOLOGICA: describe la morfologia esperable segun \
la clase predominante (usa la guia del Bloque 3). NO utilices frases \
telegraficas como "ritmo sinusal, ECG normal". Elabora describiendo \
los rasgos morfologicos concretos observables en la derivacion II \
modificada."""


BLOQUE_5_FORMATO = """=== FORMATO DE SALIDA ===
Genera una nota clinica de 5 a 8 oraciones que incluya:

1. Contexto demografico y del registro (edad, sexo, duracion)
2. Descripcion morfologica de lo observable en derivacion II
3. Cita de la clase predominante y su distribucion
4. Mencion de las anomalias detectadas con timestamps si aplican
5. Reporte de la confianza del clasificador
6. Recomendacion explicita de revision cardiologica si aplica

Output UNICAMENTE el texto de la nota clinica, sin preambulos, sin \
explicaciones metadata, sin encabezados."""


BLOQUE_6_RESTRICCIONES = """=== RESTRICCIONES DE SEGURIDAD CLINICA ===
Restricciones que debes respetar SIEMPRE:

- No prescribas farmacos, dosis ni procedimientos terapeuticos.
- No emitas un diagnostico definitivo; usa formulaciones descriptivas \
("compatible con", "se observa un patron de", "sugestivo de").
- Solo utiliza terminologia de las 4 clases AAMI EC57:2012 admitidas: \
NORMAL, PAC, NSR, AFIB. No introduzcas clases diagnosticas fuera de \
este conjunto.
- Si la evidencia es contradictoria o de baja confianza, indica \
explicitamente que requiere revision humana.
- No uses lenguaje que sugiera certeza absoluta ("se confirma", \
"es seguro que"). Prefiere formulaciones probabilisticas."""


# ═══════════════════════════════════════════════════════════════════
# CONSTRUCCION DEL PROMPT PARA SENAL COMPLETA
# ═══════════════════════════════════════════════════════════════════

def _formatear_anomalias(anomalias: list[dict]) -> str:
    """Formatea las anomalias detectadas con sus timestamps."""
    if not anomalias:
        return "  - Ninguna anomalia especifica detectada por el clasificador.\n"

    lineas = []
    for a in anomalias[:10]:  # limite de 10 para no saturar contexto
        if a["type"] == "PAC":
            lineas.append(
                f"  - PAC en el segundo {a['at_second']:.1f} "
                f"(confidence gap {a['confidence_gap']:.2f})"
            )
        else:
            lineas.append(
                f"  - {a['type']} desde el segundo {a['from_second']:.1f} "
                f"al segundo {a['to_second']:.1f}"
            )
    return "\n".join(lineas) + "\n"


def _construir_bloque_evidencia(signal_json: dict) -> str:
    """Construye el Bloque 2 (evidencia) a partir del JSON agregado."""
    meta = signal_json["signal_metadata"]
    summary = signal_json["aggregate_summary"]

    escalaciones = sum(
        1 for w in signal_json["beat_windows"] + signal_json["rhythm_windows"]
        if w["escalation_flag"]
    )

    anomalias_texto = _formatear_anomalias(summary["anomalies_detected"])

    beat_seq = ",".join(w["prediction"] for w in signal_json["beat_windows"])
    rhythm_seq = ",".join(w["prediction"] for w in signal_json["rhythm_windows"])

    return f"""=== EVIDENCIA DEL CLASIFICADOR DUAL (MSM-SC + XGBoost) ===

Contexto del paciente:
  - Edad estimada: {meta['patient_age_estimate']} anos, sexo: {meta['patient_sex']}
  - Derivacion: {meta['lead']}, muestreo: {meta['sampling_rate_hz']} Hz
  - Duracion del registro: {meta['duration_seconds']} segundos

Salida del clasificador (agregados sobre la senal completa):
  - Clase de latido predominante: {summary['predominant_beat_class']}
  - Distribucion de clases de latido: \
{json.dumps(summary['beat_class_distribution_pct'], ensure_ascii=False)}
  - Clase de ritmo predominante: {summary['predominant_rhythm_class']}
  - Distribucion de clases de ritmo: \
{json.dumps(summary['rhythm_class_distribution_pct'], ensure_ascii=False)}
  - Confidence gap medio (beat / rhythm): \
{summary['mean_confidence_gap_beat']:.2f} / \
{summary['mean_confidence_gap_rhythm']:.2f}
  - Ventanas marcadas para escalacion: {escalaciones}
  - Escalacion global requerida: {summary['global_escalation_required']}

Anomalias detectadas por el clasificador:
{anomalias_texto}
Secuencia de predicciones por segundo (60 ventanas beat, 1s c/u):
  {beat_seq}
Secuencia de predicciones ritmicas (15 ventanas rhythm, 4s c/u):
  {rhythm_seq}

INTERPRETACION DE LOS CAMPOS:
- predominant_beat_class / predominant_rhythm_class: la clase AAMI \
EC57:2012 mas frecuente en la senal (una de NORMAL, PAC, NSR, AFIB)
- beat_class_distribution_pct: distribucion porcentual de las clases \
de latido en las 60 ventanas de 1 segundo
- rhythm_class_distribution_pct: distribucion porcentual de las \
clases de ritmo en las 15 ventanas de 4 segundos
- mean_confidence_gap_beat / mean_confidence_gap_rhythm: media de \
los confidence gaps de las ventanas individuales; valores cercanos a \
1 indican alta confianza, valores cercanos a 0 indican baja confianza
- global_escalation_required: True cuando el sistema recomienda \
escalacion al cardiologo por acumulacion de ventanas de baja confianza
- anomalies_detected: eventos especificos identificados con su \
localizacion temporal en segundos"""


def build_prompt(signal_json: dict) -> str:
    """
    Prompt v3 de produccion. Contrato semantico de seis bloques que
    garantiza descripcion morfologica, citacion de evidencia agregada
    y respeto de las restricciones de seguridad clinica.

    Configuracion: K=0 (zero-shot, sin ejemplos few-shot inyectados)
    segun la recomendacion operacional del experimento LLM-B v2
    (Class Fidelity 93.57% con K=0 vs 82.01% con K=8).

    Parameters
    ----------
    signal_json : dict
        JSON agregado producido por el Modulo I sobre una senal
        completa de 60 segundos. Debe contener las claves
        signal_metadata, aggregate_summary, beat_windows y
        rhythm_windows.

    Returns
    -------
    str
        Prompt completo listo para inyectar en cualquier backend LLM.
    """
    bloque_2_evidencia = _construir_bloque_evidencia(signal_json)

    return "\n\n".join([
        BLOQUE_1_ROL,
        bloque_2_evidencia,
        BLOQUE_3_GUIA_MORFOLOGICA,
        BLOQUE_4_CITACION,
        BLOQUE_5_FORMATO,
        BLOQUE_6_RESTRICCIONES,
    ])


def validar_contrato_signal_json(signal_json: dict) -> None:
    """
    Valida el JSON agregado antes de invocar al agente
    (Safety-by-Design).
    """
    claves_top = {
        "signal_metadata", "aggregate_summary",
        "beat_windows", "rhythm_windows",
    }
    faltantes = claves_top - set(signal_json.keys())
    if faltantes:
        raise ValueError(
            f"JSON incompleto. Claves top-level faltantes: {faltantes}"
        )

    meta_obligatorio = {
        "patient_age_estimate", "patient_sex", "lead",
        "sampling_rate_hz", "duration_seconds",
    }
    faltan_meta = meta_obligatorio - set(signal_json["signal_metadata"].keys())
    if faltan_meta:
        raise ValueError(
            f"signal_metadata incompleto. Faltan: {faltan_meta}"
        )

    summary_obligatorio = {
        "predominant_beat_class", "beat_class_distribution_pct",
        "predominant_rhythm_class", "rhythm_class_distribution_pct",
        "mean_confidence_gap_beat", "mean_confidence_gap_rhythm",
        "anomalies_detected", "global_escalation_required",
    }
    faltan_summary = summary_obligatorio - set(
        signal_json["aggregate_summary"].keys()
    )
    if faltan_summary:
        raise ValueError(
            f"aggregate_summary incompleto. Faltan: {faltan_summary}"
        )

    clases_validas = {"NORMAL", "PAC", "NSR", "AFIB"}
    predominante_beat = signal_json["aggregate_summary"]["predominant_beat_class"]
    predominante_rhythm = signal_json["aggregate_summary"]["predominant_rhythm_class"]
    for etiqueta, clase in (("beat", predominante_beat),
                              ("rhythm", predominante_rhythm)):
        if clase not in clases_validas:
            raise ValueError(
                f"predominant_{etiqueta}_class invalida: {clase}. "
                f"Debe ser una de {clases_validas}."
            )


class LLMBackend(ABC):
    """Interfaz comun de todos los backends."""

    name: str = "abstract"
    model_id: str = "abstract"

    @abstractmethod
    def generate(self, signal_json: dict) -> LLMResponse:
        ...