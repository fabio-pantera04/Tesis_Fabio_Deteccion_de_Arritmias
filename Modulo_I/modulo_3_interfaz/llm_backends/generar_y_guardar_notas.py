"""
generar_y_guardar_notas.py
────────────────────────────────────────────────────────────────────
Genera notas clinicas del Modulo LLM-A para todas las senales
precargadas y las guarda persistentemente en JSON estructurado.

Las notas guardadas son leidas por la interfaz Flask del Modulo III
para mostrarlas al cardiologo sin regenerar en cada consulta.

Uso:
    # Generar/regenerar TODAS las notas (5 llamadas al API):
    python generar_y_guardar_notas.py

    # Regenerar solo una senal especifica:
    python generar_y_guardar_notas.py --senal sig_001

    # Regenerar sobrescribiendo notas existentes (default: skip):
    python generar_y_guardar_notas.py --force

Salida:
    data/notas_llm_a/nota_sig_001.json ... nota_sig_005.json
    data/notas_llm_a/log_generacion.json  (historial de generaciones)
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# Ajusta el path segun tu proyecto
BASE_DIR    = Path(r"E:\Modelo_Tesis\Modulo_I\modulo_3_interfaz")
SIGNALS_DIR = BASE_DIR / "data" / "signals_processed"
NOTAS_DIR   = BASE_DIR / "data" / "notas_llm_a"  # se crea si no existe

sys.path.insert(0, str(BASE_DIR / "llm_backends"))
from base import build_prompt, validar_contrato_signal_json


MODELO = "gemini-2.5-flash"
PROMPT_VERSION = "v3"
MAX_REINTENTOS = 3
ESPERA_INICIAL_S = 5


def verificar_nota(nota: str, signal_json: dict) -> dict:
    """Aplica las 9 verificaciones cualitativas al texto generado."""
    nota_lower = nota.lower()
    checks = {}

    palabras_es = ["se observa", "se identifica", "se aprecia",
                     "compatible", "sugestivo", "morfologia",
                     "ritmo", "onda", "complejo"]
    checks["idioma_espanol"] = sum(
        1 for p in palabras_es if p in nota_lower
    ) >= 3

    checks["menciona_lead_II"] = bool(
        re.search(r"derivaci[oó]n\s*ii|lead\s*ii|dii\b", nota_lower)
    )

    morfologia = ["qrs", "onda p", "onda t", "r-r", "pr",
                    "fibrilat", "ectopic", "prematur"]
    checks["describe_morfologia"] = sum(
        1 for m in morfologia if m in nota_lower
    ) >= 2

    predom_beat = signal_json["aggregate_summary"][
        "predominant_beat_class"].lower()
    predom_rhythm = signal_json["aggregate_summary"][
        "predominant_rhythm_class"].lower()
    checks["menciona_clase_beat"] = predom_beat in nota_lower
    checks["menciona_clase_rhythm"] = (
        predom_rhythm in nota_lower
        or ("fibrilacion" in nota_lower and predom_rhythm == "afib")
        or ("sinusal" in nota_lower and predom_rhythm == "nsr")
    )

    checks["menciona_confianza"] = bool(re.search(
        r"confidence\s*gap|confianza|certeza|clasificador", nota_lower
    ))

    num_oraciones = len(re.findall(r"[.!?]+\s", nota + " "))
    checks["longitud_apropiada"] = 4 <= num_oraciones <= 10

    prohibidas = ["se confirma", "es seguro que", "definitivamente",
                    "sin duda"]
    checks["sin_certeza_absoluta"] = not any(
        p in nota_lower for p in prohibidas
    )

    checks["sin_prescripcion"] = not bool(re.search(
        r"prescrib|dosis|\bmg\b|\bmiligramo", nota_lower
    ))
    return checks


def llamar_gemini_con_reintentos(prompt: str) -> tuple[str, float]:
    """Llama a Gemini con backoff exponencial ante errores 503/429."""
    from google import genai

    if "GEMINI_API_KEY" not in os.environ:
        sys.exit("ERROR: define GEMINI_API_KEY en el entorno primero.")

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    espera = ESPERA_INICIAL_S

    for intento in range(1, MAX_REINTENTOS + 1):
        try:
            t0 = time.time()
            resp = client.models.generate_content(
                model=MODELO, contents=prompt,
            )
            latencia = time.time() - t0
            return (resp.text.strip() if resp.text else ""), latencia
        except Exception as e:
            msg = str(e)
            recuperable = ("503" in msg or "429" in msg
                            or "UNAVAILABLE" in msg
                            or "RESOURCE_EXHAUSTED" in msg)
            if recuperable and intento < MAX_REINTENTOS:
                print(f"    Reintento {intento} en {espera}s "
                       f"({type(e).__name__})...")
                time.sleep(espera)
                espera *= 2
                continue
            raise


def generar_nota_para_senal(signal_path: Path,
                              force: bool = False) -> dict | None:
    """
    Genera y guarda la nota para una senal individual.

    Returns None si la nota ya existe y force=False, o el dict guardado
    si se genero exitosamente.
    """
    signal_id = signal_path.stem  # sig_001
    nota_path = NOTAS_DIR / f"nota_{signal_id}.json"

    if nota_path.exists() and not force:
        print(f"  [SKIP] {signal_id}: nota ya existe. "
               f"Usa --force para regenerar.")
        return None

    signal_json = json.loads(signal_path.read_text(encoding="utf-8"))

    try:
        validar_contrato_signal_json(signal_json)
    except ValueError as e:
        print(f"  [ERROR] {signal_id}: contrato invalido -- {e}")
        return None

    prompt = build_prompt(signal_json)

    try:
        nota_texto, latencia = llamar_gemini_con_reintentos(prompt)
        if not nota_texto:
            print(f"  [ERROR] {signal_id}: respuesta vacia")
            return None
    except Exception as e:
        print(f"  [ERROR] {signal_id}: {type(e).__name__}: {e}")
        return None

    checks = verificar_nota(nota_texto, signal_json)
    n_ok = sum(1 for v in checks.values() if v)

    registro = {
        "signal_id":         signal_id,
        "generated_at":      datetime.utcnow().isoformat() + "Z",
        "model":             MODELO,
        "prompt_version":    PROMPT_VERSION,
        "latency_seconds":   round(latencia, 2),
        "signal_metadata":   signal_json["signal_metadata"],
        "aggregate_summary": signal_json["aggregate_summary"],
        "clinical_note":     nota_texto,
        "quality_checks":    checks,
        "checks_passed":     n_ok,
        "checks_total":      len(checks),
    }

    nota_path.write_text(
        json.dumps(registro, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"  [OK] {signal_id}: {n_ok}/{len(checks)} checks, "
           f"{latencia:.1f}s, guardado en {nota_path.name}")
    return registro


def actualizar_log(registros: list[dict]) -> None:
    """Registra el historial de generaciones para auditoria."""
    log_path = NOTAS_DIR / "log_generacion.json"
    log = []
    if log_path.exists():
        log = json.loads(log_path.read_text(encoding="utf-8"))
    log.append({
        "timestamp":    datetime.utcnow().isoformat() + "Z",
        "model":        MODELO,
        "prompt_ver":   PROMPT_VERSION,
        "n_generadas":  len(registros),
        "signal_ids":   [r["signal_id"] for r in registros if r],
    })
    log_path.write_text(
        json.dumps(log, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--senal", type=str, default=None,
                          help="Un signal_id especifico (ej: sig_001). "
                               "Si se omite, procesa todas.")
    parser.add_argument("--force", action="store_true",
                          help="Regenera aunque ya exista.")
    args = parser.parse_args()

    NOTAS_DIR.mkdir(parents=True, exist_ok=True)

    if args.senal:
        signals = [SIGNALS_DIR / f"{args.senal}.json"]
    else:
        signals = sorted(SIGNALS_DIR.glob("sig_*.json"))

    if not signals:
        sys.exit(f"No hay senales en {SIGNALS_DIR}")

    print(f"Procesando {len(signals)} senal(es)...")
    print(f"Salida: {NOTAS_DIR}\n")

    registros = []
    for signal_path in signals:
        if not signal_path.exists():
            print(f"  [MISS] {signal_path.name} no existe")
            continue
        r = generar_nota_para_senal(signal_path, force=args.force)
        if r:
            registros.append(r)
            time.sleep(3)  # throttling entre llamadas

    if registros:
        actualizar_log(registros)

    print(f"\n{len(registros)} nota(s) generadas exitosamente.")


if __name__ == "__main__":
    main()
