"""
Experimento 1 — Analisis de correccion clinica de las notas LLM-B
────────────────────────────────────────────────────────────────────
Complementa las metricas de similitud (BLEU/ROUGE/SBERT) con metricas
de correccion clinica que responden la pregunta:

    "Gemini realmente interpreta la evidencia del clasificador,
     o solo copia el estilo SCP-ECG?"

Analiza los archivos generaciones_k{0,10,20}.jsonl que ya tienes y
computa 4 metricas complementarias por nota generada:

    1. Class Mention Accuracy: menciona la clase correcta?
    2. Class Fidelity:         no menciona otra clase como
                               diagnostico principal?
    3. Morphological Grounding: describe morfologia apropiada a
                                la clase (ondas P, QRS, R-R)?
    4. Evidence Grounding:     referencia la evidencia del
                               clasificador (shapelets, MSM,
                               probabilidades)?

Ejecucion:
    python experimento_1_correccion_clinica.py

Salida en resultados_llm_b/:
    correccion_clinica.json      metricas agregadas
    correccion_clinica.md        reporte legible
    correccion_por_nota.jsonl    detalle por nota individual
"""

import json
import re
from collections import defaultdict
from pathlib import Path

RESULTADOS_DIR = Path(
    r"E:\Modelo_Tesis\Modulo_I\validacion_generativa\resultados_llm_b"
)

VALORES_K = [0, 10, 20]
CLASES = ["NORMAL", "PAC", "NSR", "AFIB"]


# ─────────────────────────────────────────────────────────────────────
# CRITERIOS DE EVALUACION POR CLASE
# ─────────────────────────────────────────────────────────────────────

# Metrica 1: patrones que indican mencion correcta de cada clase
PATRONES_MENCION = {
    "NORMAL": [
        r"\bnormal\s+ecg\b",
        r"\bnormal\s+ekg\b",
        r"\botherwise\s+normal\b",
        r"\bunremarkable\b",
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
    ],
    "NSR": [
        r"\bsinus\s+rhythm\b",
        r"\bnormal\s+sinus\b",
        r"\bregular\s+sinus\b",
    ],
    "AFIB": [
        r"\batrial\s+fibrillation\b",
        r"\bafib\b",
        r"\bAF\b(?!\s*lutter)",   # AF pero no AFlutter
    ],
}

# Metrica 2: patrones que indican contradiccion (nota dice otra clase
# como diagnostico principal en lugar de la clase objetivo)
# Nota: NORMAL y NSR pueden coexistir (uno de latido, otro de ritmo).
# PAC y AFIB pueden coexistir con NSR de fondo. Consideramos
# contradiccion solo cuando aparece un rimico NO compatible.
CONTRADICCIONES_PROHIBIDAS = {
    "NORMAL":  ["AFIB", "PAC"],   # NORMAL no debe mencionar AFIB o PAC principal
    "PAC":     ["AFIB"],           # PAC no debe mencionar AFIB
    "NSR":     ["AFIB", "PAC"],   # NSR puro no debe mencionar AFIB o PAC principal
    "AFIB":    ["NSR", "NORMAL"], # AFIB no debe decir "normal" o "sinus rhythm"
}

# Metrica 3: keywords morfologicas apropiadas por clase
# El sistema debe demostrar comprension de que morfologia esperar
PATRONES_MORFOLOGICOS = {
    "NORMAL": [
        r"\bqrs\b",
        r"\bp\s+wave",
        r"\bt\s+wave",
        r"\bPR\s+interval",
        r"\bnarrow\s+qrs",
    ],
    "PAC": [
        r"\bpremature\b",
        r"\bectopic\b",
        r"\bcompensatory\s+pause\b",
        r"\bp\s+wave",   # P prematura
        r"\bextrasystole",
    ],
    "NSR": [
        r"\bregular\b",
        r"\br-r\b|\brr\s+interval",
        r"\bp\s+wave",
        r"\brate\b",
        r"\baxis\b",
    ],
    "AFIB": [
        r"\birregular\b",
        r"\babsent\s+p\b|\bno\s+discernible\s+p",
        r"\bfibrillat",
        r"\bventricular\s+response\b",
        r"\br-r\b|\brr\s+interval",
    ],
}

# Metrica 4: keywords que indican referencia a la evidencia
# del clasificador (shapelets, MSM, probabilidades)
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
]


def compilar_regex(patrones: list) -> re.Pattern:
    """Compila una lista de patrones a una sola regex OR."""
    return re.compile("|".join(patrones), re.IGNORECASE)


REGEX_MENCION = {c: compilar_regex(p) for c, p in PATRONES_MENCION.items()}
REGEX_MORFOL  = {c: compilar_regex(p) for c, p in PATRONES_MORFOLOGICOS.items()}
REGEX_EVIDENC = compilar_regex(PATRONES_EVIDENCIA)


# ─────────────────────────────────────────────────────────────────────
# EVALUACION POR NOTA
# ─────────────────────────────────────────────────────────────────────

def evaluar_nota(nota: str, clase_objetivo: str) -> dict:
    """Evalua las 4 dimensiones de correccion clinica de una nota."""
    if not nota or not nota.strip():
        return {
            "class_mention":       False,
            "class_fidelity":      False,
            "morphological":       False,
            "evidence":            False,
            "correct_composite":   False,
        }

    # Metrica 1: menciona la clase correcta?
    mencion = bool(REGEX_MENCION[clase_objetivo].search(nota))

    # Metrica 2: no menciona una clase contradictoria?
    contradice = False
    for otra_clase in CONTRADICCIONES_PROHIBIDAS[clase_objetivo]:
        if REGEX_MENCION[otra_clase].search(nota):
            contradice = True
            break
    fidelity = mencion and not contradice

    # Metrica 3: describe morfologia apropiada a la clase?
    morfologia = bool(REGEX_MORFOL[clase_objetivo].search(nota))

    # Metrica 4: referencia la evidencia del clasificador?
    evidencia = bool(REGEX_EVIDENC.search(nota))

    # Metrica compuesta: cumple mencion Y fidelity (las dos esenciales)
    correct_composite = mencion and fidelity

    return {
        "class_mention":     mencion,
        "class_fidelity":    fidelity,
        "morphological":     morfologia,
        "evidence":          evidencia,
        "correct_composite": correct_composite,
    }


# ─────────────────────────────────────────────────────────────────────
# ANALISIS AGREGADO
# ─────────────────────────────────────────────────────────────────────

def analizar_configuracion(k: int) -> dict:
    """Analiza todas las notas de una configuracion K."""
    gen_path = RESULTADOS_DIR / f"generaciones_k{k}.jsonl"
    if not gen_path.exists():
        return None

    with open(gen_path, encoding="utf-8") as f:
        generaciones = [json.loads(line) for line in f]

    # Filtrar solo las con prediccion no vacia
    generaciones = [g for g in generaciones if g.get("prediccion", "").strip()]

    # Evaluar cada nota
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

    # Agregar por clase
    por_clase = defaultdict(lambda: {
        "n":                    0,
        "class_mention":        0,
        "class_fidelity":       0,
        "morphological":        0,
        "evidence":             0,
        "correct_composite":    0,
    })

    for e in evaluaciones:
        clase = e["clase"]
        por_clase[clase]["n"] += 1
        for metrica in ["class_mention", "class_fidelity", "morphological",
                         "evidence", "correct_composite"]:
            if e[metrica]:
                por_clase[clase][metrica] += 1

    # Convertir a porcentajes
    por_clase_pct = {}
    for clase, cnts in por_clase.items():
        n = cnts["n"]
        por_clase_pct[clase] = {
            "n":                        n,
            "class_mention_pct":        cnts["class_mention"]     / n * 100 if n else 0,
            "class_fidelity_pct":       cnts["class_fidelity"]    / n * 100 if n else 0,
            "morphological_pct":        cnts["morphological"]     / n * 100 if n else 0,
            "evidence_pct":             cnts["evidence"]          / n * 100 if n else 0,
            "correct_composite_pct":    cnts["correct_composite"] / n * 100 if n else 0,
        }

    # Agregado global
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
        "k":              k,
        "global":         global_stats,
        "por_clase":      por_clase_pct,
        "evaluaciones":   evaluaciones,
    }


# ─────────────────────────────────────────────────────────────────────
# REPORTE
# ─────────────────────────────────────────────────────────────────────

def imprimir_tabla_consola(resultados: dict) -> None:
    print("\n" + "=" * 90)
    print("EXPERIMENTO 1 — CORRECCION CLINICA DE LAS NOTAS GENERADAS")
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


def guardar_reporte_markdown(resultados: dict) -> None:
    lineas = ["# Experimento 1 — Correccion clinica de notas LLM-B\n"]
    lineas.append("**Objetivo**: verificar que Gemini interpreta "
                   "correctamente la evidencia del clasificador, no "
                   "solo reproduce el estilo SCP-ECG.\n")
    lineas.append("## Metricas evaluadas\n")
    lineas.append("| Metrica | Descripcion |")
    lineas.append("|---|---|")
    lineas.append("| **Class Mention** | La nota menciona la clase "
                   "correcta que dice el clasificador |")
    lineas.append("| **Class Fidelity** | Ademas no menciona una clase "
                   "contradictoria como diagnostico principal |")
    lineas.append("| **Morphological** | Describe morfologia apropiada "
                   "a la clase (ondas P, QRS, R-R, etc.) |")
    lineas.append("| **Evidence** | Referencia explicitamente la "
                   "evidencia del clasificador (shapelets, MSM, "
                   "confidence) |")
    lineas.append("| **Composite** | Cumple Mention Y Fidelity |")

    lineas.append("\n## Resultados globales\n")
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

    out_path = RESULTADOS_DIR / "correccion_clinica.md"
    out_path.write_text("\n".join(lineas), encoding="utf-8")
    print(f"\n  Reporte markdown: {out_path.name}")


def main() -> None:
    print("Analizando correccion clinica de las 3 configuraciones...")
    resultados = {}
    for k in VALORES_K:
        r = analizar_configuracion(k)
        if r:
            resultados[k] = r

    if not resultados:
        print("No hay generaciones para analizar.")
        return

    imprimir_tabla_consola(resultados)

    # Guardar JSON con metricas agregadas
    out_json = RESULTADOS_DIR / "correccion_clinica.json"
    to_save = {
        str(k): {
            "global":    resultados[k]["global"],
            "por_clase": resultados[k]["por_clase"],
        }
        for k in resultados
    }
    out_json.write_text(json.dumps(to_save, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    print(f"\n  Metricas agregadas: {out_json.name}")

    # Guardar detalle por nota (para inspeccion)
    out_jsonl = RESULTADOS_DIR / "correccion_por_nota.jsonl"
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
