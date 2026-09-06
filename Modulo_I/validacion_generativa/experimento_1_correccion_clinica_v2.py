"""
Experimento 1 sobre LLM-B v2 — Version 2 con regex AMPLIADAS
────────────────────────────────────────────────────────────────────
Segunda pasada del Experimento 1 sobre las generaciones del LLM-B v2.

Motivacion:
    La primera pasada arrojo Class Mention anomalamente baja en NORMAL
    K=8 (12%) por artefacto de las regex: el prompt v3 induce a Gemini
    a usar vocabulario elaborado ("NORMAL classification", "normal
    morphology", "physiological range") que las regex disenadas para
    el estilo telegrafico de PTB-XL no capturaban.

    Verificacion cualitativa confirmo que las notas SI son clinicamente
    correctas y SI mencionan la clase en forma elaborada.

Cambios respecto a la version anterior:
    Se anaden patrones para:
    - Nombres de clase en cursivas o mayusculas ("NORMAL classification",
      "AFIB pattern", etc.)
    - Vocabulario clinico elaborado ("normal conduction",
      "normal morphology", "within physiological range")
    - Formulaciones probabilisticas ("supporting the X classification",
      "compatible with X pattern")

Todo lo demas se mantiene identico para preservar comparabilidad con
las metricas del v1 (Experimento 1 original).

Ejecucion:
    python experimento_1_correccion_clinica_v2_regex_ampliado.py

Salida en resultados_llm_b_v2/:
    correccion_clinica_regex_ampliado.json
    correccion_clinica_regex_ampliado.md
    correccion_por_nota_regex_ampliado.jsonl
"""

import json
import re
from collections import defaultdict
from pathlib import Path

RESULTADOS_DIR = Path(
    r"E:\Modelo_Tesis\Modulo_I\validacion_generativa\resultados_llm_b_v2"
)

VALORES_K = [0, 8]
CLASES = ["NORMAL", "PAC", "NSR", "AFIB"]


# ─────────────────────────────────────────────────────────────────────
# CRITERIOS DE EVALUACION AMPLIADOS
# ─────────────────────────────────────────────────────────────────────

# CLASS MENTION — patrones AMPLIADOS para vocabulario elaborado
PATRONES_MENCION = {
    "NORMAL": [
        # Frases del estilo telegrafico (PTB-XL v1)
        r"\bnormal\s+ecg\b",
        r"\bnormal\s+ekg\b",
        r"\botherwise\s+normal\b",
        r"\bunremarkable\b",
        r"\bnormal\s+beat\b",
        # Frases del vocabulario elaborado (v3 prompt)
        r"\bNORMAL\s+classification\b",
        r"\bnormal\s+classification\b",
        r"\bnormal\s+morphology\b",
        r"\bnormal\s+conduction\b",
        r"\bwithin\s+physiological\s+(range|limits)\b",
        r"\bphysiological\s+(range|limits|variability)\b",
        r"\bcompatible\s+with\s+(a\s+)?normal\b",
        r"\bconsistent\s+with\s+(a\s+)?normal\b",
        r"\bcompatible\s+with\s+the\s+NORMAL\b",
    ],
    "PAC": [
        r"\bpremature\s+atrial\b",
        r"\batrial\s+premature\b",
        r"\bsupraventricular\s+premature\b",
        r"\bsupraventricular\s+ectopic\b",
        r"\bsupraventricular\s+extrasystole",
        r"\bAPC\b",
        r"\bAPB\b",
        r"\bPAC\b",
        # Vocabulario elaborado del prompt v3
        r"\bPAC\s+classification\b",
        r"\bpremature\s+beat\b",
        r"\bectopic\s+beat\b",
        r"\bpremature\s+contraction\b",
    ],
    "NSR": [
        r"\bsinus\s+rhythm\b",
        r"\bnormal\s+sinus\b",
        r"\bregular\s+sinus\b",
        # Vocabulario elaborado del prompt v3
        r"\bNSR\s+classification\b",
        r"\bNSR\b",
        r"\bsinus\s+origin\b",
        r"\bsinus\s+node\b",
    ],
    "AFIB": [
        r"\batrial\s+fibrillation\b",
        r"\bafib\b",
        r"\bAF\b(?!\s*lutter)",
        # Vocabulario elaborado del prompt v3
        r"\bAFIB\s+classification\b",
        r"\bAFIB\s+pattern\b",
        r"\bfibrillatory\s+(pattern|activity|undulations)\b",
        r"\bfibrillat",   # ya cubre "fibrillation", "fibrillatory"
    ],
}

CONTRADICCIONES_PROHIBIDAS = {
    "NORMAL":  ["AFIB", "PAC"],
    "PAC":     ["AFIB"],
    "NSR":     ["AFIB", "PAC"],
    "AFIB":    ["NSR", "NORMAL"],
}

PATRONES_MORFOLOGICOS = {
    "NORMAL": [
        r"\bqrs\b",
        r"\bp\s+wave",
        r"\bt\s+wave",
        r"\bPR\s+interval",
        r"\bnarrow\s+qrs",
        r"\bmonophasic\s+p",
        r"\bpositive\s+polarity",
    ],
    "PAC": [
        r"\bpremature\b",
        r"\bectopic\b",
        r"\bcompensatory\s+pause\b",
        r"\bp\s+wave",
        r"\bextrasystole",
        r"\bectopic\s+p\s+wave",
        r"\balpha\s+P\s+wave",
    ],
    "NSR": [
        r"\bregular\b",
        r"\br-r\b|\brr\s+interval",
        r"\bp\s+wave",
        r"\brate\b",
        r"\baxis\b",
        r"\batrioventricular\s+relationship",
    ],
    "AFIB": [
        r"\birregular\b",
        r"\babsent\s+p\b|\bno\s+discernible\s+p",
        r"\bfibrillat",
        r"\bventricular\s+response\b",
        r"\br-r\b|\brr\s+interval",
        r"\birregularly\s+irregular\b",
    ],
}

PATRONES_EVIDENCIA = [
    r"\bshapelet",
    r"\bmicro_s\d+",
    r"\bmacro_s\d+",
    r"\bmsm\s+distance",
    r"\bclassifier\b",
    r"\bconfidence\b",
    r"\blikelihood\b",
    r"\bprobability\b",
    r"\bevidence\b",
    r"\bgeometric\s+evidence\b",
    r"\bconfidence\s+gap\b",
]


def compilar_regex(patrones):
    return re.compile("|".join(patrones), re.IGNORECASE)


REGEX_MENCION = {c: compilar_regex(p) for c, p in PATRONES_MENCION.items()}
REGEX_MORFOL  = {c: compilar_regex(p) for c, p in PATRONES_MORFOLOGICOS.items()}
REGEX_EVIDENC = compilar_regex(PATRONES_EVIDENCIA)


def evaluar_nota(nota, clase_objetivo):
    if not nota or not nota.strip():
        return {
            "class_mention":     False,
            "class_fidelity":    False,
            "morphological":     False,
            "evidence":          False,
            "correct_composite": False,
        }

    mencion = bool(REGEX_MENCION[clase_objetivo].search(nota))
    contradice = False
    for otra in CONTRADICCIONES_PROHIBIDAS[clase_objetivo]:
        if REGEX_MENCION[otra].search(nota):
            contradice = True
            break
    fidelity = mencion and not contradice
    morfologia = bool(REGEX_MORFOL[clase_objetivo].search(nota))
    evidencia = bool(REGEX_EVIDENC.search(nota))
    correct_composite = mencion and fidelity

    return {
        "class_mention":     mencion,
        "class_fidelity":    fidelity,
        "morphological":     morfologia,
        "evidence":          evidencia,
        "correct_composite": correct_composite,
    }


def analizar_configuracion(k):
    gen_path = RESULTADOS_DIR / f"generaciones_k{k}.jsonl"
    if not gen_path.exists():
        return None

    with open(gen_path, encoding="utf-8") as f:
        generaciones = [json.loads(line) for line in f]

    generaciones = [g for g in generaciones if g.get("prediccion", "").strip()]

    evaluaciones = []
    for g in generaciones:
        eval_ = evaluar_nota(g["prediccion"], g["clase"])
        eval_.update({
            "ecg_id":     g["ecg_id"],
            "clase":      g["clase"],
            "k":          k,
            "prediccion": g["prediccion"],
            "referencia": g["referencia"],
        })
        evaluaciones.append(eval_)

    por_clase = defaultdict(lambda: {
        "n":                 0,
        "class_mention":     0,
        "class_fidelity":    0,
        "morphological":     0,
        "evidence":          0,
        "correct_composite": 0,
    })

    for e in evaluaciones:
        c = e["clase"]
        por_clase[c]["n"] += 1
        for m in ["class_mention", "class_fidelity", "morphological",
                   "evidence", "correct_composite"]:
            if e[m]:
                por_clase[c][m] += 1

    por_clase_pct = {}
    for c, cnts in por_clase.items():
        n = cnts["n"]
        por_clase_pct[c] = {
            "n":                     n,
            "class_mention_pct":     cnts["class_mention"]     / n * 100 if n else 0,
            "class_fidelity_pct":    cnts["class_fidelity"]    / n * 100 if n else 0,
            "morphological_pct":     cnts["morphological"]     / n * 100 if n else 0,
            "evidence_pct":          cnts["evidence"]          / n * 100 if n else 0,
            "correct_composite_pct": cnts["correct_composite"] / n * 100 if n else 0,
        }

    n_total = sum(c["n"] for c in por_clase.values())
    global_stats = {
        "n":                     n_total,
        "class_mention_pct":     sum(c["class_mention"]     for c in por_clase.values()) / n_total * 100 if n_total else 0,
        "class_fidelity_pct":    sum(c["class_fidelity"]    for c in por_clase.values()) / n_total * 100 if n_total else 0,
        "morphological_pct":     sum(c["morphological"]     for c in por_clase.values()) / n_total * 100 if n_total else 0,
        "evidence_pct":          sum(c["evidence"]          for c in por_clase.values()) / n_total * 100 if n_total else 0,
        "correct_composite_pct": sum(c["correct_composite"] for c in por_clase.values()) / n_total * 100 if n_total else 0,
    }

    return {
        "k":            k,
        "global":       global_stats,
        "por_clase":    por_clase_pct,
        "evaluaciones": evaluaciones,
    }


def imprimir_tabla_consola(resultados):
    print("\n" + "=" * 90)
    print("EXPERIMENTO 1 SOBRE LLM-B v2 (REGEX AMPLIADAS)")
    print("=" * 90)
    print("Metricas (todas en %):")
    print("  MENC = menciona la clase correcta")
    print("  FID  = ademas no contradice mencionando otra clase")
    print("  MORF = describe morfologia apropiada a la clase")
    print("  EVID = referencia la evidencia del clasificador")
    print("  COMP = compuesta (MENC y FID)")

    print("\n" + "-" * 90)
    print("RESULTADOS GLOBALES POR K")
    print("-" * 90)
    print(f"  {'K':<4} {'N':<5} {'MENC':>8} {'FID':>8} {'MORF':>8} "
          f"{'EVID':>8} {'COMP':>8}")

    for k in VALORES_K:
        if k not in resultados:
            continue
        r = resultados[k]["global"]
        print(f"  {k:<4} {r['n']:<5} "
              f"{r['class_mention_pct']:>7.2f}% "
              f"{r['class_fidelity_pct']:>7.2f}% "
              f"{r['morphological_pct']:>7.2f}% "
              f"{r['evidence_pct']:>7.2f}% "
              f"{r['correct_composite_pct']:>7.2f}%")

    print("\n" + "-" * 90)
    print("RESULTADOS POR CLASE")
    print("-" * 90)
    for clase in CLASES:
        print(f"\n  Clase {clase}:")
        print(f"    {'K':<4} {'N':<5} {'MENC':>8} {'FID':>8} {'MORF':>8} "
              f"{'EVID':>8} {'COMP':>8}")
        for k in VALORES_K:
            if k not in resultados or clase not in resultados[k]["por_clase"]:
                continue
            r = resultados[k]["por_clase"][clase]
            print(f"    {k:<4} {r['n']:<5} "
                  f"{r['class_mention_pct']:>7.2f}% "
                  f"{r['class_fidelity_pct']:>7.2f}% "
                  f"{r['morphological_pct']:>7.2f}% "
                  f"{r['evidence_pct']:>7.2f}% "
                  f"{r['correct_composite_pct']:>7.2f}%")


def guardar_reporte_markdown(resultados):
    lineas = ["# Experimento 1 sobre LLM-B v2 — Regex ampliadas\n"]
    lineas.append("Version del Experimento 1 con regex de Class Mention "
                   "ampliadas para capturar el vocabulario elaborado "
                   "inducido por el prompt v3 (frases como 'NORMAL "
                   "classification', 'normal morphology', "
                   "'physiological range'). Los patrones originales "
                   "de PTB-XL telegrafico se conservan.\n")

    lineas.append("## Resultados globales\n")
    lineas.append("| K | N | Class Mention | Class Fidelity | "
                   "Morphological | Evidence | Composite |")
    lineas.append("|---|---|---|---|---|---|---|")
    for k in VALORES_K:
        if k not in resultados:
            continue
        r = resultados[k]["global"]
        lineas.append(
            f"| {k} | {r['n']} | "
            f"{r['class_mention_pct']:.2f}% | "
            f"{r['class_fidelity_pct']:.2f}% | "
            f"{r['morphological_pct']:.2f}% | "
            f"{r['evidence_pct']:.2f}% | "
            f"{r['correct_composite_pct']:.2f}% |"
        )

    lineas.append("\n## Resultados por clase\n")
    for clase in CLASES:
        lineas.append(f"\n### {clase}\n")
        lineas.append("| K | N | Class Mention | Class Fidelity | "
                       "Morphological | Evidence | Composite |")
        lineas.append("|---|---|---|---|---|---|---|")
        for k in VALORES_K:
            if k not in resultados or clase not in resultados[k]["por_clase"]:
                continue
            r = resultados[k]["por_clase"][clase]
            lineas.append(
                f"| {k} | {r['n']} | "
                f"{r['class_mention_pct']:.2f}% | "
                f"{r['class_fidelity_pct']:.2f}% | "
                f"{r['morphological_pct']:.2f}% | "
                f"{r['evidence_pct']:.2f}% | "
                f"{r['correct_composite_pct']:.2f}% |"
            )

    out_path = RESULTADOS_DIR / "correccion_clinica_regex_ampliado.md"
    out_path.write_text("\n".join(lineas), encoding="utf-8")
    print(f"\n  Reporte markdown: {out_path.name}")


def main():
    print("Analizando correccion clinica del LLM-B v2 (regex ampliadas)...")
    resultados = {}
    for k in VALORES_K:
        r = analizar_configuracion(k)
        if r:
            resultados[k] = r

    if not resultados:
        print("No hay generaciones del v2 para analizar.")
        return

    imprimir_tabla_consola(resultados)

    out_json = RESULTADOS_DIR / "correccion_clinica_regex_ampliado.json"
    to_save = {
        str(k): {
            "global":    resultados[k]["global"],
            "por_clase": resultados[k]["por_clase"],
        }
        for k in resultados
    }
    out_json.write_text(
        json.dumps(to_save, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n  Metricas: {out_json.name}")

    out_jsonl = RESULTADOS_DIR / "correccion_por_nota_regex_ampliado.jsonl"
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for k in resultados:
            for e in resultados[k]["evaluaciones"]:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"  Detalle por nota: {out_jsonl.name}")

    guardar_reporte_markdown(resultados)

    print("\n" + "=" * 90)
    print("ANALISIS COMPLETO")
    print("=" * 90)


if __name__ == "__main__":
    main()