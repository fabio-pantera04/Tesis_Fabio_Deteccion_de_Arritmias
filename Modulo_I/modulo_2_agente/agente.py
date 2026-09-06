"""
modulo_2_agente/agente.py

Agente LLM para generación de notas clínicas ECG.

Flujo de decisión del agente
-----------------------------
1. Recibe el vector de evidencia del Módulo I (clase, probabilidades, distancias)
2. Evalúa el confidence_gap
   - gap < UMBRAL_CONFIANZA → Tool: escalar_medico → devuelve alerta
   - gap >= UMBRAL_CONFIANZA → Tool: consultar_guidelines → genera nota
3. Construye el prompt RAG con la evidencia geométrica + guidelines
4. Llama a Llama 3 local con restricciones hard anti-alucinación
5. Retorna la nota clínica estructurada o la alerta de escalamiento

Justificación de la arquitectura de agente vs LLM puro:
    Un LLM puro siempre genera texto independientemente de la confianza.
    El agente implementa Safety-by-Design: puede abstenerse de generar
    una nota cuando la evidencia es insuficiente (gap < umbral).
    Referencia: AI Agents in Clinical Medicine, PMC (2024).

Dependencias opcionales:
    - llama-cpp-python : para inferencia local con Llama 3 GGUF
      pip install llama-cpp-python
    - Si no está instalado, el agente funciona en modo SIMULADO
      (útil para desarrollo y demostración sin GPU/CPU potente)
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    UMBRAL_CONFIANZA,
    LLAMA3_MODELO_PATH,
    LLAMA3_N_CTX,
    LLAMA3_N_GPU_LAYERS,
    LLAMA3_TEMPERATURA,
    LLAMA3_MAX_TOKENS,
)
from modulo_2_agente.herramientas import (
    ejecutar_herramienta,
    consultar_guidelines,
    escalar_medico,
)


# ─────────────────────────────────────────────────────────────────────────────
# CARGA DEL MODELO LLAMA 3 LOCAL
# ─────────────────────────────────────────────────────────────────────────────

_llama_modelo = None  # singleton — se carga una vez al primer uso

def _cargar_llama3():
    """
    Carga Llama 3 local con llama-cpp-python.
    Si el modelo no está disponible, activa modo SIMULADO.
    """
    global _llama_modelo
    if _llama_modelo is not None:
        return _llama_modelo

    try:
        from llama_cpp import Llama
        ruta = str(LLAMA3_MODELO_PATH)
        if not Path(ruta).exists():
            print(f"[AGENTE] Modelo Llama 3 no encontrado en: {ruta}")
            print("[AGENTE] Activando modo SIMULADO para desarrollo.")
            return None

        print(f"[AGENTE] Cargando Llama 3 desde: {ruta}")
        _llama_modelo = Llama(
            model_path      = ruta,
            n_ctx           = LLAMA3_N_CTX,
            n_gpu_layers    = LLAMA3_N_GPU_LAYERS,
            verbose         = False,
        )
        print("[AGENTE] Llama 3 cargado correctamente.")
        return _llama_modelo

    except ImportError:
        print("[AGENTE] llama-cpp-python no instalado.")
        print("[AGENTE] Para instalarlo: pip install llama-cpp-python")
        print("[AGENTE] Activando modo SIMULADO.")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# CONSTRUCCIÓN DEL PROMPT RAG
# ─────────────────────────────────────────────────────────────────────────────

def _construir_prompt_rag(evidencia: dict, guidelines: dict) -> str:
    """
    Construye el prompt con patrón RAG (Retrieval-Augmented Generation).

    El prompt tiene tres secciones:
    1. SYSTEM: rol del asistente + restricciones hard anti-alucinación
    2. CONTEXTO RECUPERADO: guidelines clínicos de la base de conocimiento
    3. EVIDENCIA GEOMÉTRICA: datos numéricos del Módulo I

    El LLM SOLO puede usar la información de las secciones 2 y 3.
    Esto implementa el grounding clínico documentado en:
        NEJM AI (2025): RAG reduces hallucination in clinical LLMs.
        MDPI AI Review (2025): RAG aligns patient data with clinical guidelines.
    """
    clase     = evidencia.get('clase_predicha', '?')
    prob      = evidencia.get('probabilidades', {}).get(clase, 0)
    gap       = evidencia.get('confidence_gap', 0)
    dists     = evidencia.get('distancias_top5_msm', {})
    alerta    = evidencia.get('alerta_baja_confianza', False)

    criterios = "\n".join(
        f"  - {c}" for c in guidelines.get('criterios_diagnosticos_ecg', []))
    recomendacion = guidelines.get('recomendacion_manejo', '')
    nivel_alerta  = guidelines.get('nivel_alerta', 'DESCONOCIDO')
    referencia    = guidelines.get('guideline_referencia', '')
    cie10         = guidelines.get('codigo_cie10', '')

    dists_str = "\n".join(
        f"  - {k}: {v:.4f}" for k, v in list(dists.items())[:5])

    probs_str = "\n".join(
        f"  - {c}: {p*100:.1f}%"
        for c, p in sorted(
            evidencia.get('probabilidades', {}).items(),
            key=lambda x: x[1], reverse=True))

    advertencia_confianza = (
        "\n⚠️  ADVERTENCIA: El nivel de confianza del modelo es bajo "
        f"(gap={gap:.3f} < {UMBRAL_CONFIANZA}). "
        "Indicar esta limitación prominentemente en la nota.\n"
        if alerta else ""
    )

    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Eres un asistente de apoyo diagnostico cardiologico especializado en ECG.
Tu unica funcion es generar notas clinicas estructuradas a partir de evidencia
geometrica cuantitativa provista por un clasificador MSM-SC + XGBoost.

REGLAS ESTRICTAS — INCUMPLIRLAS INVALIDA LA NOTA:
1. SOLO puedes afirmar hallazgos respaldados por las distancias MSM-SC provistas.
2. NUNCA inventes hallazgos no presentes en los datos numericos.
3. NUNCA uses conocimiento propio no anclado en el contexto recuperado.
4. Si el gap de confianza es bajo, DEBES indicarlo como advertencia prominente.
5. Termina SIEMPRE con el aviso estandar de responsabilidad medica.
6. La nota debe seguir exactamente la estructura de 5 secciones indicada.
<|eot_id|><|start_header_id|>user<|end_header_id|>

=== CONTEXTO RECUPERADO — BASE DE CONOCIMIENTO CLINICO (AAMI/AHA) ===
Diagnostico probable : {guidelines.get('nombre_completo', clase)} ({clase})
Codigo CIE-10        : {cie10}
Nivel de alerta      : {nivel_alerta}
Referencia clinica   : {referencia}

Criterios diagnosticos ECG para {clase}:
{criterios}

Recomendacion de manejo segun guidelines:
  {recomendacion}

=== EVIDENCIA GEOMETRICA — MODULO I (MSM-SC + XGBoost) ===
{advertencia_confianza}
Clase predicha       : {clase} — {guidelines.get('nombre_completo', '')}
Probabilidad         : {prob*100:.1f}%
Confidence gap       : {gap:.4f} (umbral: {UMBRAL_CONFIANZA})

Distribucion de probabilidades por clase:
{probs_str}

Distancias MSM-SC a shapelets discriminativos (menor = mayor similitud):
{dists_str}

=== INSTRUCCION ===
Genera una nota clinica estructurada con exactamente estas 5 secciones:

1. DIAGNOSTICO PRINCIPAL
2. EVIDENCIA MORFOLOGICA (basada EXCLUSIVAMENTE en las distancias MSM-SC)
3. NIVEL DE CONFIANZA DEL SISTEMA
4. RECOMENDACION CLINICA (segun guideline recuperado)
5. AVISO DE RESPONSABILIDAD

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    return prompt


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCIA CON LLAMA 3 / MODO SIMULADO
# ─────────────────────────────────────────────────────────────────────────────

def _inferencia_llama3(prompt: str, modelo) -> str:
    """Ejecuta inferencia con Llama 3 local."""
    respuesta = modelo(
        prompt,
        max_tokens  = LLAMA3_MAX_TOKENS,
        temperature = LLAMA3_TEMPERATURA,
        stop        = ["<|eot_id|>", "<|end_of_text|>"],
        echo        = False,
    )
    return respuesta['choices'][0]['text'].strip()


def _nota_simulada(evidencia: dict, guidelines: dict) -> str:
    """
    Genera una nota clínica simulada (modo desarrollo sin Llama 3 instalado).
    Útil para demostrar la interfaz web sin requerir el modelo completo.
    """
    clase  = evidencia.get('clase_predicha', '?')
    prob   = evidencia.get('probabilidades', {}).get(clase, 0)
    gap    = evidencia.get('confidence_gap', 0)
    nombre = guidelines.get('nombre_completo', clase)
    cie10  = guidelines.get('codigo_cie10', '')
    nivel  = guidelines.get('nivel_alerta', '')
    recom  = guidelines.get('recomendacion_manejo', '')
    ref    = guidelines.get('guideline_referencia', '')
    dists  = evidencia.get('distancias_top5_msm', {})

    criterios = guidelines.get('criterios_diagnosticos_ecg', [])
    hallazgos = []
    for i, (shapelet, dist) in enumerate(list(dists.items())[:3]):
        tipo = "macro (ritmo)" if "Macro" in shapelet else "micro (latido)"
        nivel_sim = "alta" if dist < 30 else "moderada" if dist < 60 else "baja"
        hallazgos.append(
            f"  • Shapelet {tipo} '{shapelet}': distancia MSM-SC = {dist:.4f} "
            f"→ similitud {nivel_sim} con patron {clase}")

    advertencia = (
        f"\n⚠️  CONFIANZA BAJA: gap={gap:.3f} < {UMBRAL_CONFIANZA}. "
        "Resultados con incertidumbre elevada.\n"
        if gap < UMBRAL_CONFIANZA else ""
    )

    nota = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  NOTA CLÍNICA DE APOYO — SISTEMA MSM-SC + AGENTE LLM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{advertencia}
1. DIAGNÓSTICO PRINCIPAL
   {nombre} ({clase}) — CIE-10: {cie10}
   Probabilidad del clasificador: {prob*100:.1f}%
   Nivel de alerta clínica: {nivel}

2. EVIDENCIA MORFOLÓGICA (Clasificador geométrico MSM-SC)
   Las siguientes distancias elásticas MSM-SC sustentan el diagnóstico:
{chr(10).join(hallazgos)}

   Criterios diagnósticos ECG compatibles (según guidelines {ref}):
{chr(10).join(f'  ✓ {c}' for c in criterios[:3])}

3. NIVEL DE CONFIANZA DEL SISTEMA
   Probabilidad clase predicha : {prob*100:.1f}%
   Confidence gap              : {gap:.4f}
   Umbral de confianza mínimo  : {UMBRAL_CONFIANZA}
   Estado                      : {'⚠️  BAJA CONFIANZA' if gap < UMBRAL_CONFIANZA else '✓ CONFIANZA ACEPTABLE'}

4. RECOMENDACIÓN CLÍNICA
   {recom}
   Referencia: {ref}

5. AVISO DE RESPONSABILIDAD
   Este reporte fue generado automáticamente por el Sistema de Asistencia
   Informativa para Etiquetado de Arritmias (Tesis CIMAT, 2026).
   Es una herramienta de apoyo informativo basada en evidencia geométrica
   cuantificable. La interpretación diagnóstica final y las decisiones
   terapéuticas son responsabilidad exclusiva del médico especialista.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    return nota.strip()


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN PRINCIPAL DEL AGENTE
# ─────────────────────────────────────────────────────────────────────────────

def ejecutar_agente(evidencia: dict) -> dict:
    """
    Función principal del agente. Recibe el vector de evidencia del Módulo I
    y retorna la nota clínica o la alerta de escalamiento.

    Flujo de decisión
    -----------------
    1. Evalúa confidence_gap
       - gap < UMBRAL → Tool: escalar_medico → retorna alerta
       - gap >= UMBRAL → Tool: consultar_guidelines → genera nota
    2. Construye prompt RAG con evidencia + guidelines
    3. Llama a Llama 3 local (o modo simulado)
    4. Retorna resultado estructurado

    Parámetros
    ----------
    evidencia : dict
        Vector de evidencia del Módulo I con campos:
        id_muestra, clase_predicha, probabilidades, confidence_gap,
        alerta_baja_confianza, distancias_top5_msm

    Retorna
    -------
    dict con:
        status          : 'NOTA_GENERADA' | 'ESCALADO' | 'ERROR'
        herramientas_usadas : lista de tools llamadas
        resultado       : nota clínica (str) o alerta (dict)
        tiempo_ms       : tiempo de procesamiento
        modo            : 'llama3_local' | 'simulado'
    """
    t0 = time.perf_counter()
    herramientas_usadas = []

    clase         = evidencia.get('clase_predicha', 'DESCONOCIDO')
    gap           = evidencia.get('confidence_gap', 0.0)
    id_muestra    = evidencia.get('id_muestra', -1)
    probabilidades = evidencia.get('probabilidades', {})

    # ── DECISIÓN DEL ROUTER ───────────────────────────────────────────────────
    if gap < UMBRAL_CONFIANZA:
        # Confianza insuficiente → escalar al médico
        print(f"[AGENTE] Muestra {id_muestra}: gap={gap:.3f} < {UMBRAL_CONFIANZA}"
              f" → ESCALANDO al médico")
        alerta = ejecutar_herramienta(
            'escalar_medico',
            id_muestra     = id_muestra,
            clase_predicha = clase,
            confidence_gap = gap,
            probabilidades = probabilidades,
        )
        herramientas_usadas.append('escalar_medico')

        return {
            'status'              : 'ESCALADO',
            'herramientas_usadas' : herramientas_usadas,
            'resultado'           : alerta,
            'tiempo_ms'           : round((time.perf_counter() - t0) * 1000, 1),
            'modo'                : 'regla_determinista',
        }

    # ── CONFIANZA ACEPTABLE → GENERAR NOTA ───────────────────────────────────
    print(f"[AGENTE] Muestra {id_muestra}: gap={gap:.3f} >= {UMBRAL_CONFIANZA}"
          f" → generando nota para {clase}")

    # Tool 1: Recuperar guidelines (RAG)
    guidelines = ejecutar_herramienta(
        'consultar_guidelines',
        clase_predicha = clase,
    )
    herramientas_usadas.append('consultar_guidelines')

    if 'error' in guidelines:
        return {
            'status'              : 'ERROR',
            'herramientas_usadas' : herramientas_usadas,
            'resultado'           : guidelines['error'],
            'tiempo_ms'           : round((time.perf_counter() - t0) * 1000, 1),
            'modo'                : 'error',
        }

    # Tool 2: Generar nota con Llama 3 o modo simulado
    modelo = _cargar_llama3()
    if modelo is not None:
        prompt = _construir_prompt_rag(evidencia, guidelines)
        nota   = _inferencia_llama3(prompt, modelo)
        modo   = 'llama3_local'
    else:
        nota = _nota_simulada(evidencia, guidelines)
        modo = 'simulado'

    tiempo_ms = round((time.perf_counter() - t0) * 1000, 1)
    print(f"[AGENTE] Nota generada en {tiempo_ms} ms (modo: {modo})")

    return {
        'status'              : 'NOTA_GENERADA',
        'herramientas_usadas' : herramientas_usadas,
        'resultado'           : nota,
        'tiempo_ms'           : tiempo_ms,
        'modo'                : modo,
        'clase_diagnosticada' : clase,
        'confianza_pct'       : round(probabilidades.get(clase, 0) * 100, 1),
        'nivel_alerta'        : guidelines.get('nivel_alerta', ''),
        'guideline_referencia': guidelines.get('guideline_referencia', ''),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PROCESAR LOTE DE PREDICCIONES (del vector_entrada_LLM_MSM.json)
# ─────────────────────────────────────────────────────────────────────────────

def procesar_vector_llm(ruta_json: str,
                        max_muestras: int = None,
                        solo_incorrectas: bool = False) -> list:
    """
    Procesa el vector_entrada_LLM_MSM.json generado por el Módulo I
    y ejecuta el agente para cada predicción.

    Parámetros
    ----------
    ruta_json        : ruta al archivo JSON del Módulo I
    max_muestras     : límite de muestras a procesar (None = todas)
    solo_incorrectas : si True, solo procesa las predicciones incorrectas

    Retorna
    -------
    Lista de resultados del agente por muestra.
    """
    with open(ruta_json, 'r', encoding='utf-8') as f:
        datos = json.load(f)

    predicciones = datos.get('predicciones_test', [])
    if solo_incorrectas:
        predicciones = [p for p in predicciones if not p.get('correcto', True)]
    if max_muestras:
        predicciones = predicciones[:max_muestras]

    print(f"\n[AGENTE] Procesando {len(predicciones)} predicciones...")
    resultados = []

    for i, pred in enumerate(predicciones):
        print(f"  [{i+1}/{len(predicciones)}] ID={pred['id']} "
              f"clase={pred['clase_predicha']} gap={pred['confidence_gap']:.3f}")
        resultado = ejecutar_agente(pred)
        resultado['id_original'] = pred['id']
        resultado['clase_real']  = pred.get('clase_real', '?')
        resultados.append(resultado)

    escalados = sum(1 for r in resultados if r['status'] == 'ESCALADO')
    notas     = sum(1 for r in resultados if r['status'] == 'NOTA_GENERADA')

    print(f"\n[AGENTE] Resumen:")
    print(f"  Notas generadas  : {notas}")
    print(f"  Casos escalados  : {escalados}")
    print(f"  Total procesados : {len(resultados)}")

    return resultados
