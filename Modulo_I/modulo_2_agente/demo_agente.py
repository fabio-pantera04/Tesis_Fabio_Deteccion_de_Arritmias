"""
modulo_2_agente/demo_agente.py

Script de demostración del agente para mostrar avances de tesis.
Funciona en modo SIMULADO sin necesitar Llama 3 instalado.

Muestra:
  - Nota clínica para una predicción con alta confianza (NOTA_GENERADA)
  - Alerta de escalamiento para una predicción con baja confianza (ESCALADO)
  - Resumen del vector del Módulo I

Ejecutar desde C:/Modelo_Tesis/Modulo_I/:
    python -m modulo_2_agente.demo_agente

O con el vector real del entrenamiento:
    python -m modulo_2_agente.demo_agente --vector ruta/al/vector_LLM.json
"""

import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from modulo_2_agente.agente import ejecutar_agente


def demo_manual():
    """Demostración con datos de ejemplo sin necesitar el vector real."""

    print("=" * 70)
    print("  DEMOSTRACIÓN MÓDULO II — AGENTE LLM (modo simulado)")
    print("  Tesis CIMAT — Sistema de Asistencia para Etiquetado de Arritmias")
    print("=" * 70)

    # ── Caso 1: AFIB con alta confianza → debe generar nota ──────────────────
    print("\n" + "─" * 70)
    print("  CASO 1 — AFIB con alta confianza (debe generar nota clínica)")
    print("─" * 70)

    evidencia_afib = {
        'id_muestra'            : 42,
        'clase_real'            : 'AFIB',
        'clase_predicha'        : 'AFIB',
        'correcto'              : True,
        'probabilidades'        : {
            'AFIB': 0.9124, 'NSR': 0.0531,
            'NORMAL': 0.0231, 'PAC': 0.0114
        },
        'confidence_gap'        : 0.8593,   # gap alto → nota generada
        'alerta_baja_confianza' : False,
        'distancias_top5_msm'   : {
            'Macro_S3' : 11.24,   # distancia baja = alta similitud con AFIB
            'Macro_S17': 14.87,
            'Micro_S8' : 89.43,   # distancia alta = no parece latido normal
            'Macro_S1' : 18.92,
            'Micro_S22': 76.31,
        },
    }

    resultado_1 = ejecutar_agente(evidencia_afib)
    print(f"\n  Status           : {resultado_1['status']}")
    print(f"  Herramientas     : {resultado_1['herramientas_usadas']}")
    print(f"  Modo             : {resultado_1['modo']}")
    print(f"  Tiempo           : {resultado_1['tiempo_ms']} ms")
    print(f"\n{resultado_1['resultado']}")

    # ── Caso 2: PAC con baja confianza → debe escalar ────────────────────────
    print("\n" + "─" * 70)
    print("  CASO 2 — PAC con baja confianza (debe escalar al médico)")
    print("─" * 70)

    evidencia_pac_baja = {
        'id_muestra'            : 87,
        'clase_real'            : 'NORMAL',   # predicción incorrecta
        'clase_predicha'        : 'PAC',
        'correcto'              : False,
        'probabilidades'        : {
            'PAC': 0.3812, 'NORMAL': 0.3541,
            'AFIB': 0.1492, 'NSR': 0.1155
        },
        'confidence_gap'        : 0.0271,   # gap bajo → escalar
        'alerta_baja_confianza' : True,
        'distancias_top5_msm'   : {
            'Micro_S8' : 45.12,
            'Micro_S22': 52.87,
            'Macro_S3' : 78.43,
            'Macro_S17': 81.92,
            'Micro_S14': 49.31,
        },
    }

    resultado_2 = ejecutar_agente(evidencia_pac_baja)
    print(f"\n  Status           : {resultado_2['status']}")
    print(f"  Herramientas     : {resultado_2['herramientas_usadas']}")
    print(f"  Modo             : {resultado_2['modo']}")
    print(f"  Tiempo           : {resultado_2['tiempo_ms']} ms")
    print(f"\n  ALERTA GENERADA:")
    alerta = resultado_2['resultado']
    print(f"  Nivel urgencia   : {alerta['nivel_urgencia']}")
    print(f"  Confidence gap   : {alerta['confidence_gap']}")
    print(f"\n  {alerta['accion_requerida']}")

    # ── Caso 3: NSR correcto ──────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  CASO 3 — NSR con confianza moderada (genera nota con advertencia)")
    print("─" * 70)

    evidencia_nsr = {
        'id_muestra'            : 103,
        'clase_real'            : 'NSR',
        'clase_predicha'        : 'NSR',
        'correcto'              : True,
        'probabilidades'        : {
            'NSR': 0.7234, 'AFIB': 0.1891,
            'NORMAL': 0.0541, 'PAC': 0.0334
        },
        'confidence_gap'        : 0.5343,
        'alerta_baja_confianza' : False,
        'distancias_top5_msm'   : {
            'Macro_S1' : 9.87,
            'Macro_S11': 13.42,
            'Micro_S5' : 67.23,
            'Macro_S7' : 16.91,
            'Micro_S19': 72.44,
        },
    }

    resultado_3 = ejecutar_agente(evidencia_nsr)
    print(f"\n  Status           : {resultado_3['status']}")
    print(f"  Herramientas     : {resultado_3['herramientas_usadas']}")
    print(f"  Confianza        : {resultado_3.get('confianza_pct', '')}%")
    print(f"  Nivel alerta     : {resultado_3.get('nivel_alerta', '')}")
    print(f"\n{resultado_3['resultado']}")

    # ── Resumen ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RESUMEN DE LA DEMOSTRACIÓN")
    print("=" * 70)
    casos = [resultado_1, resultado_2, resultado_3]
    for i, r in enumerate(casos, 1):
        print(f"  Caso {i}: {r['status']:15s} | "
              f"herramientas={r['herramientas_usadas']} | "
              f"tiempo={r['tiempo_ms']}ms")
    print()
    print("  El agente demostró:")
    print("  ✓ Generación de nota clínica estructurada con alta confianza")
    print("  ✓ Escalamiento automático cuando confianza < 0.25")
    print("  ✓ Recuperación de guidelines médicos (RAG pattern)")
    print("  ✓ Nota con advertencia cuando confianza es moderada")
    print("=" * 70)


def demo_con_vector(ruta_json: str, n: int = 5):
    """Procesa las primeras N predicciones del vector real del Módulo I."""
    from modulo_2_agente.agente import procesar_vector_llm

    print(f"\n[DEMO] Procesando {n} muestras del vector: {ruta_json}")
    resultados = procesar_vector_llm(ruta_json, max_muestras=n)

    for r in resultados:
        print(f"\n  ID={r['id_original']} | real={r['clase_real']} | "
              f"status={r['status']} | modo={r['modo']}")
        if r['status'] == 'NOTA_GENERADA':
            print(r['resultado'][:300] + "...")
        else:
            alerta = r['resultado']
            print(f"  ESCALADO: {alerta.get('nivel_urgencia', '')} — "
                  f"{alerta.get('descripcion', '')}")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == '--vector':
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        demo_con_vector(sys.argv[2], n)
    else:
        demo_manual()
