"""
modulo_2_agente/herramientas.py

Herramientas (Tools) del Agente LLM — Módulo II del sistema de tesis CIMAT.

Arquitectura de agente justificada en:
  - AI Agents in Clinical Medicine (PMC, 2024-2025): agentes con herramientas
    muestran mejora del 53% sobre LLMs sin herramientas en tareas clínicas.
  - NEJM AI (2025): RAG con base de conocimiento local reduce alucinaciones
    en reportes clínicos generados por LLM.
  - Meta LLM Deployment Guide (2024): Llama 3 local elimina riesgo de filtración
    de datos de pacientes fuera del perímetro institucional.

El agente dispone de DOS herramientas:
  1. consultar_guidelines  — RAG sobre base de conocimiento AAMI/AHA local
  2. escalar_medico        — alerta cuando confidence_gap < UMBRAL_CONFIANZA

Estas herramientas implementan Safety-by-Design: el agente no genera una nota
clínica cuando la evidencia geométrica del Módulo I es insuficiente.
"""

import json
import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ARCHIVO_GUIDELINES, UMBRAL_CONFIANZA


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 1 — CONSULTAR GUIDELINES
# ─────────────────────────────────────────────────────────────────────────────

def consultar_guidelines(clase_predicha: str) -> dict:
    """
    Tool 1: Consulta la base de conocimiento médico local (AAMI/AHA).

    Implementa el patrón RAG (Retrieval-Augmented Generation):
    recupera el contexto clínico estructurado para la clase predicha
    antes de que el LLM genere la nota, anclando la generación en
    guías clínicas verificadas y reduciendo el riesgo de alucinación.

    Referencia RAG en cardiología:
        Yu et al. (2023). Zero-shot retrieval-augmented diagnosis technique
        where LLMs retrieve ECG expert knowledge from a curated database.
        Artificial Intelligence Review, Springer (2025).

    Parámetros
    ----------
    clase_predicha : str
        Clase AAMI predicha por el Módulo I (AFIB, NSR, NORMAL, PAC)

    Retorna
    -------
    dict con criterios diagnósticos, relevancia clínica, nivel de alerta
    y guideline de referencia para la clase indicada.
    """
    try:
        with open(ARCHIVO_GUIDELINES, 'r', encoding='utf-8') as f:
            guidelines = json.load(f)
    except FileNotFoundError:
        return {
            'error': f'Base de conocimiento no encontrada en {ARCHIVO_GUIDELINES}',
            'clase': clase_predicha,
            'criterios_diagnosticos_ecg': [],
            'relevancia_clinica': 'No disponible',
            'recomendacion_manejo': 'Consultar guías clínicas manualmente',
            'nivel_alerta': 'DESCONOCIDO',
        }

    clase_upper = clase_predicha.upper()
    if clase_upper not in guidelines['clases']:
        return {
            'error': f'Clase {clase_predicha} no encontrada en guidelines',
            'clases_disponibles': list(guidelines['clases'].keys()),
        }

    info = guidelines['clases'][clase_upper].copy()
    info['clase']   = clase_upper
    info['fuente']  = guidelines.get('fuente', 'AAMI/AHA')
    info['tool']    = 'consultar_guidelines'
    info['status']  = 'OK'
    return info


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 2 — ESCALAR AL MÉDICO
# ─────────────────────────────────────────────────────────────────────────────

def escalar_medico(id_muestra: int,
                   clase_predicha: str,
                   confidence_gap: float,
                   probabilidades: dict,
                   motivo: str = "") -> dict:
    """
    Tool 2: Genera una alerta de escalamiento al médico especialista.

    Se activa cuando confidence_gap < UMBRAL_CONFIANZA (default: 0.25),
    implementando el principio de Safety-by-Design: el sistema se abstiene
    de generar una nota diagnóstica cuando la evidencia es insuficiente.

    Referencia Safety-by-Design en IA médica:
        AI Agents in Clinical Medicine, PMC (2024):
        'Single AI agents manage complete clinical workflows including
         report generation and treatment recommendations.'

    Parámetros
    ----------
    id_muestra      : identificador de la señal ECG
    clase_predicha  : clase con mayor probabilidad del Módulo I
    confidence_gap  : diferencia entre prob_1 y prob_2 (menor = más incertidumbre)
    probabilidades  : dict {clase: probabilidad}
    motivo          : razón adicional del escalamiento (opcional)

    Retorna
    -------
    dict con alerta estructurada lista para mostrar en la interfaz web.
    """
    # Ordenar probabilidades para mostrar las más altas primero
    probs_ordenadas = sorted(
        probabilidades.items(), key=lambda x: x[1], reverse=True)

    # Determinar nivel de urgencia según el gap
    if confidence_gap < 0.10:
        nivel_urgencia = "CRITICO"
        descripcion_urgencia = (
            "El modelo no puede distinguir entre clases. "
            "Revisión manual inmediata requerida."
        )
    elif confidence_gap < 0.25:
        nivel_urgencia = "ALTO"
        descripcion_urgencia = (
            "Confianza insuficiente para generar nota automática. "
            "El médico debe interpretar la señal directamente."
        )
    else:
        nivel_urgencia = "MODERADO"
        descripcion_urgencia = motivo or "Escalamiento preventivo solicitado."

    alerta = {
        'tool'             : 'escalar_medico',
        'status'           : 'ESCALADO',
        'timestamp'        : datetime.datetime.now().isoformat(),
        'id_muestra'       : id_muestra,
        'clase_predicha'   : clase_predicha,
        'confidence_gap'   : round(confidence_gap, 4),
        'umbral_usado'     : UMBRAL_CONFIANZA,
        'nivel_urgencia'   : nivel_urgencia,
        'descripcion'      : descripcion_urgencia,
        'probabilidades'   : dict(probs_ordenadas[:4]),
        'accion_requerida' : (
            f"REVISIÓN MANUAL REQUERIDA — Señal ID {id_muestra}\n"
            f"Clase más probable: {clase_predicha} "
            f"({probabilidades.get(clase_predicha, 0)*100:.1f}%)\n"
            f"Gap de confianza: {confidence_gap:.3f} "
            f"(umbral: {UMBRAL_CONFIANZA})\n"
            f"{descripcion_urgencia}"
        ),
        'nota_sistema'     : (
            "El clasificador geométrico MSM-SC + XGBoost no alcanzó "
            "el umbral de confianza mínimo para generación automática de nota. "
            "Este caso requiere interpretación directa por médico especialista."
        ),
    }
    return alerta


# ─────────────────────────────────────────────────────────────────────────────
# DISPATCHER DE HERRAMIENTAS
# ─────────────────────────────────────────────────────────────────────────────

HERRAMIENTAS_DISPONIBLES = {
    'consultar_guidelines': {
        'funcion'     : consultar_guidelines,
        'descripcion' : (
            "Recupera criterios diagnósticos ECG, relevancia clínica y guía "
            "de referencia para una clase AAMI (AFIB, NSR, NORMAL, PAC). "
            "Usar SIEMPRE antes de generar la nota clínica."
        ),
        'parametros'  : ['clase_predicha'],
    },
    'escalar_medico': {
        'funcion'     : escalar_medico,
        'descripcion' : (
            "Genera alerta de escalamiento cuando confidence_gap < umbral. "
            "Usar cuando la confianza del modelo es insuficiente para "
            "generar una nota automática segura."
        ),
        'parametros'  : ['id_muestra', 'clase_predicha',
                         'confidence_gap', 'probabilidades'],
    },
}


def ejecutar_herramienta(nombre: str, **kwargs) -> dict:
    """
    Dispatcher central de herramientas del agente.
    Valida que la herramienta exista y la ejecuta con los parámetros dados.
    """
    if nombre not in HERRAMIENTAS_DISPONIBLES:
        return {
            'error'                : f"Herramienta '{nombre}' no disponible",
            'herramientas_validas' : list(HERRAMIENTAS_DISPONIBLES.keys()),
        }
    return HERRAMIENTAS_DISPONIBLES[nombre]['funcion'](**kwargs)
