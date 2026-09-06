"""
Script de prueba — Prompt v3 de produccion end-to-end
────────────────────────────────────────────────────────────────────
Prueba el nuevo build_prompt() con una senal real del sistema:
  1. Carga un JSON de senal precargada
  2. Valida el contrato
  3. Construye el prompt v3
  4. Invoca a Gemini 2.5 Flash
  5. Imprime la nota clinica generada
  6. Realiza verificaciones cualitativas automaticas

Uso:
    Poner GEMINI_API_KEY en el entorno y ejecutar:
    python test_prompt_v3_produccion.py --json ruta/al/senal.json

    Sin --json usa un JSON de ejemplo hardcodeado.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ajusta el import segun donde tengas base.py:
# Opcion A: base.py en el mismo directorio
from base import build_prompt, validar_contrato_signal_json

# Opcion B: si base.py esta en llm_backends/
# sys.path.insert(0, r"D:\Modelo_Tesis\Modulo_I\modulo_3_interfaz")
# from llm_backends.base import build_prompt, validar_contrato_signal_json


# ─────────────────────────────────────────────────────────────────
# JSON DE EJEMPLO (usar cuando no hay ruta --json)
# Representa una senal AFIB con confidence gap alto
# ─────────────────────────────────────────────────────────────────
JSON_EJEMPLO = {
    "signal_metadata": {
        "patient_age_estimate": 68,
        "patient_sex": "M",
        "lead": "II-modified",
        "sampling_rate_hz": 250,
        "duration_seconds": 60,
    },
    "aggregate_summary": {
        "predominant_beat_class": "NORMAL",
        "beat_class_distribution_pct": {"NORMAL": 88.3, "PAC": 11.7},
        "predominant_rhythm_class": "AFIB",
        "rhythm_class_distribution_pct": {"AFIB": 93.3, "NSR": 6.7},
        "mean_confidence_gap_beat": 0.72,
        "mean_confidence_gap_rhythm": 0.85,
        "global_escalation_required": False,
        "anomalies_detected": [
            {"type": "PAC", "at_second": 12.3, "confidence_gap": 0.68},
            {"type": "PAC", "at_second": 34.7, "confidence_gap": 0.71},
            {"type": "AFIB", "from_second": 0.0, "to_second": 60.0,
             "confidence_gap": 0.85},
        ],
    },
    "beat_windows": (
        [{"prediction": "NORMAL", "escalation_flag": False}] * 53
        + [{"prediction": "PAC", "escalation_flag": False}] * 7
    ),
    "rhythm_windows": (
        [{"prediction": "AFIB", "escalation_flag": False}] * 14
        + [{"prediction": "NSR", "escalation_flag": False}] * 1
    ),
}


# ─────────────────────────────────────────────────────────────────
# VERIFICACIONES CUALITATIVAS
# ─────────────────────────────────────────────────────────────────
def verificar_nota(nota: str, signal_json: dict) -> dict:
    """
    Verifica cualitativamente que la nota generada cumpla los
    requisitos del prompt v3 de produccion.
    """
    import re

    nota_lower = nota.lower()
    checks = {}

    # 1. En espanol
    palabras_espanol = ["se observa", "se identifica", "se aprecia",
                          "compatible", "sugestivo", "morfologia",
                          "ritmo", "onda", "complejo"]
    matches_es = sum(1 for p in palabras_espanol if p in nota_lower)
    checks["idioma_espanol"] = matches_es >= 3

    # 2. Menciona derivacion II
    checks["menciona_lead_II"] = bool(
        re.search(r"derivaci[oó]n\s*ii|lead\s*ii|dii\b", nota_lower)
    )

    # 3. Describe morfologia
    morfologia = ["qrs", "onda p", "onda t", "r-r", "pr",
                    "fibrilat", "ectopic", "prematur"]
    matches_morf = sum(1 for m in morfologia if m in nota_lower)
    checks["describe_morfologia"] = matches_morf >= 2

    # 4. Menciona clase predominante
    predom_beat = signal_json["aggregate_summary"]["predominant_beat_class"].lower()
    predom_rhythm = signal_json["aggregate_summary"]["predominant_rhythm_class"].lower()
    checks["menciona_clase_beat"] = predom_beat in nota_lower
    checks["menciona_clase_rhythm"] = (
        predom_rhythm in nota_lower
        or ("fibrilacion" in nota_lower and predom_rhythm == "afib")
        or ("sinusal" in nota_lower and predom_rhythm == "nsr")
    )

    # 5. Menciona confianza del clasificador
    checks["menciona_confianza"] = bool(
        re.search(r"confidence\s*gap|confianza|certeza|clasificador",
                    nota_lower)
    )

    # 6. Longitud apropiada (aprox 5-8 oraciones)
    num_oraciones = len(re.findall(r"[.!?]+\s", nota + " "))
    checks["longitud_apropiada"] = 4 <= num_oraciones <= 10

    # 7. No usa lenguaje de certeza absoluta
    prohibidas = ["se confirma", "es seguro que", "definitivamente",
                    "sin duda"]
    checks["sin_certeza_absoluta"] = not any(
        p in nota_lower for p in prohibidas
    )

    # 8. No prescribe farmacos ni dosis
    checks["sin_prescripcion"] = not bool(
        re.search(r"prescrib|dosis|\bmg\b|\bmiligramo", nota_lower)
    )

    return checks


def imprimir_checks(checks: dict) -> None:
    print("\n" + "=" * 70)
    print("VERIFICACIONES CUALITATIVAS AUTOMATICAS")
    print("=" * 70)
    for nombre, ok in checks.items():
        simbolo = "OK" if ok else "FAIL"
        print(f"  [{simbolo}] {nombre}")
    total = len(checks)
    pass_ = sum(1 for v in checks.values() if v)
    print(f"\n  Resultado: {pass_}/{total} verificaciones aprobadas")


# ─────────────────────────────────────────────────────────────────
# LLAMADA A GEMINI
# ─────────────────────────────────────────────────────────────────
def llamar_gemini(prompt: str, model: str = "gemini-2.5-flash") -> tuple[str, float]:
    """Invoca a Gemini con el prompt y retorna (texto, latencia_segundos)."""
    from google import genai

    if "GEMINI_API_KEY" not in os.environ:
        sys.exit("ERROR: define GEMINI_API_KEY en el entorno primero.")

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    t0 = time.time()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )
    latencia = time.time() - t0
    return (response.text.strip() if response.text else ""), latencia


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, default=None,
                          help="Ruta a un JSON de senal real; "
                               "si no se pasa, usa el ejemplo AFIB.")
    parser.add_argument("--mostrar-prompt", action="store_true",
                          help="Imprime el prompt completo antes "
                               "de llamar a Gemini.")
    args = parser.parse_args()

    # 1. Cargar JSON
    if args.json:
        print(f"Cargando senal desde: {args.json}")
        signal_json = json.loads(args.json.read_text(encoding="utf-8"))
    else:
        print("Usando JSON de ejemplo (AFIB con PACs)")
        signal_json = JSON_EJEMPLO

    # 2. Validar contrato
    print("\nValidando contrato JSON...")
    try:
        validar_contrato_signal_json(signal_json)
        print("  OK: JSON valido")
    except ValueError as e:
        sys.exit(f"  ERROR: {e}")

    # 3. Construir prompt
    print("\nConstruyendo prompt v3...")
    prompt = build_prompt(signal_json)
    print(f"  Longitud del prompt: {len(prompt):,} caracteres "
           f"(~{len(prompt) // 4:,} tokens estimados)")

    if args.mostrar_prompt:
        print("\n" + "=" * 70)
        print("PROMPT COMPLETO")
        print("=" * 70)
        print(prompt)

    # 4. Llamar a Gemini
    print("\nInvocando a Gemini 2.5 Flash...")
    try:
        nota, latencia = llamar_gemini(prompt)
        print(f"  Latencia: {latencia:.2f}s")
    except Exception as e:
        sys.exit(f"  ERROR al llamar Gemini: {type(e).__name__}: {e}")

    # 5. Mostrar nota
    print("\n" + "=" * 70)
    print("NOTA CLINICA GENERADA POR EL MODULO LLM-A")
    print("=" * 70)
    print(nota)

    # 6. Verificaciones cualitativas
    checks = verificar_nota(nota, signal_json)
    imprimir_checks(checks)


if __name__ == "__main__":
    main()
