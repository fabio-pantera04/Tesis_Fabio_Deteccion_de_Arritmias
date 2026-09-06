"""
Modulo LLM-B v2 — Validacion de generacion clinica elaborada
────────────────────────────────────────────────────────────────────
Segundo experimento del Modulo LLM-B con objetivo COMPLEMENTARIO al v1:

  v1: valida que Gemini aprende estilo SCP-ECG con few-shot puro
      (hallazgo: mimicry, K=10 sweet spot para BLEU/SBERT)

  v2: valida que Gemini genera notas CLINICAMENTE ELABORADAS con
      instrucciones explicitas + ejemplos artificiales elaborados
      (hipotesis: Morphological Grounding y Evidence Grounding
       aumentan significativamente vs v1)

Los dos experimentos coexisten en la tesis (Sec 4.6.1 vs 4.6.2) y
su comparacion justifica el diseno del prompt v3 del LLM-A de
produccion.

Diferencias tecnicas vs v1:
  1. Prompt v3 con 6 bloques explicitos (rol + evidencia + guia
     morfologica + INSTRUCCIONES CITACION + formato + restricciones)
  2. Ejemplos few-shot ARTIFICIALES elaborados (redactados, no
     extraidos de PTB-XL) desde ejemplos_fewshot_elaborados.json
  3. Configuraciones: K=0 y K=8 (2 por clase balanceado)
  4. Todo lo demas identico (400 notas de test, throttling 5s,
     checkpointing, mismo modelo Gemini 2.5 Flash)

Prerrequisitos:
  - Los conjuntos test/few-shot ya extraidos (v1)
  - ejemplos_fewshot_elaborados.json en el mismo directorio
  - GEMINI_API_KEY con billing activado

Ejecucion:
  python modulo_llm_b_v2_validacion.py --k 0
  python modulo_llm_b_v2_validacion.py --k 8

  Opcional --reanudar, --limite N, --clase CLASE

Salida:
  resultados_llm_b_v2/
    generaciones_k{K}.jsonl
    checkpoint_k{K}.json
    metricas_k{K}.json
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

os.environ["HF_HOME"] = r"E:\huggingface_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ─────────────────────────────────────────────────────────────────────
# CONFIGURACION
# ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(r"E:\Modelo_Tesis\Modulo_I\validacion_generativa")
CONJUNTOS_DIR = BASE_DIR / "conjuntos_ptbxl"
EJEMPLOS_JSON = BASE_DIR / "ejemplos_fewshot_elaborados.json"
RESULTADOS_DIR = BASE_DIR / "resultados_llm_b_v2"
RESULTADOS_DIR.mkdir(exist_ok=True, parents=True)

CLASES = ["NORMAL", "PAC", "NSR", "AFIB"]
MODELO_GEMINI = "gemini-2.5-flash"
SEG_ENTRE_LLAMADAS = 5.0
CHECKPOINT_CADA = 20

random.seed(42)


# ─────────────────────────────────────────────────────────────────────
# PROMPT V3 ADAPTADO A INGLES
# ─────────────────────────────────────────────────────────────────────
BLOQUE_1_ROL = """You are a clinical assistant specialized in ECG \
interpretation on modified Lead II. Your task is to generate a \
descriptive note that a specialist cardiologist will use as a \
starting point for their definitive interpretation.

Write in formal clinical English, using impersonal plural first person \
("is observed...", "is identified..."). The linguistic register must \
be sober, professional, and precise.

IMPORTANT: your note does NOT replace the cardiologist's judgment. It \
is an informative support that translates the classifier's geometric \
evidence into interpretable clinical language."""

BLOQUE_2_EVIDENCIA_TEMPLATE = """=== CLASSIFIER EVIDENCE (MSM-SC + XGBoost) ===
{json_clasificador}

FIELD INTERPRETATION:
- prediction: AAMI EC57:2012 class predicted by the classifier
- probabilities: classifier posteriors per class
- confidence_gap: |P(class_1) - P(class_0)|; values near 1 indicate \
high confidence, values near 0 indicate low confidence
- distancias_top3_msm: the three most discriminative morphological \
shapelets with their MSM distance to the analyzed signal
- alerta_baja_confianza: True when confidence_gap < 0.25 (specialist \
escalation threshold)"""

BLOQUE_3_GUIA_MORFOLOGICA = """=== MORPHOLOGICAL DESCRIPTION GUIDE BY CLASS ===
When describing the signal, incorporate the following morphological \
references for the predicted class:

NORMAL (normal beat on Lead II):
- Narrow QRS complex (< 120 ms) of normal morphology
- Monophasic P wave of positive polarity preceding QRS
- PR interval within physiological range (120-200 ms)
- T wave concordant with the main QRS deflection

PAC (premature atrial contraction on Lead II):
- Beat with premature onset relative to base rhythm
- Premature P wave of morphology distinct from sinus P waves
- Associated narrow QRS complex, normal intraventricular conduction
- Characteristic incomplete compensatory pause

NSR (normal sinus rhythm on Lead II):
- Monophasic P waves preceding each QRS complex
- Preserved 1:1 atrioventricular relationship
- Regular R-R intervals with physiological variability under 10%
- Heart rate 60-100 bpm

AFIB (atrial fibrillation on Lead II):
- Absent discernible P waves
- Irregular baseline fibrillatory undulations replacing atrial activity
- Irregularly irregular R-R intervals, no cyclic pattern
- Irregularly irregular ventricular response
- QRS complexes retain narrow morphology (normal conduction)"""

BLOQUE_4_CITACION = """=== EVIDENCE CITATION INSTRUCTIONS ===
The generated note MUST explicitly cite the following elements from \
the classifier JSON (this is not optional):

1. PRINCIPAL SHAPELET: mention by name the shapelet with the lowest \
MSM distance from the distancias_top3_msm field. Example: \
"The shapelet Macro_S54 (MSM distance = 1.31) constitutes the \
principal geometric evidence."

2. MSM DISTANCE: report the numerical distance value of the principal \
shapelet. This is the magnitude supporting the classification.

3. CONFIDENCE GAP: mention the confidence_gap only if it is less than \
0.40. In that case, the note must include an explicit review \
recommendation:
   - If confidence_gap < 0.25: "Confidence gap = X.XX (< 0.25) indicates \
low classifier certainty. Immediate cardiologist review is recommended."
   - If 0.25 <= confidence_gap < 0.40: "Confidence gap = X.XX indicates \
moderate certainty. Visual confirmation is suggested."
   - If confidence_gap >= 0.40: mention only if high certainty adds \
value ("Confidence gap = X.XX indicates high classifier certainty").

4. MORPHOLOGICAL INTERPRETATION: describe the specific morphology \
observed according to the predicted class (use the guide from the \
previous block). Do NOT use telegraphic phrases like "sinus rhythm. \
normal ecg". Elaborate describing the morphological features."""

BLOQUE_5_FORMATO = """=== OUTPUT FORMAT ===
Generate a clinical note of 3 to 5 sentences including:

1. Morphological description of what is observed on Lead II
2. Citation of the MSM evidence supporting your observations
3. Confidence gap report following the rules above

Output ONLY the clinical note text, with no preambles, no metadata \
explanations, no headers."""

BLOQUE_6_RESTRICCIONES = """=== CLINICAL SAFETY RESTRICTIONS ===
Restrictions you must ALWAYS respect:
- Do not prescribe drugs, doses, or therapeutic procedures.
- Do not issue a definitive diagnosis; use descriptive formulations \
("compatible with", "a pattern of X is observed", "suggestive of").
- Use only terminology of the 4 AAMI EC57:2012 classes: NORMAL, PAC, \
NSR, AFIB.
- If evidence is contradictory or low-confidence, explicitly indicate \
that human review is required.
- Do not use language suggesting absolute certainty ("it is confirmed", \
"it is certain"). Prefer probabilistic formulations."""


def construir_prompt_v3(json_clasificador: dict,
                          ejemplos_few_shot: List[dict]) -> str:
    """Construye el prompt v3 completo con o sin ejemplos few-shot."""
    bloque_2_relleno = BLOQUE_2_EVIDENCIA_TEMPLATE.format(
        json_clasificador=json.dumps(
            json_clasificador, ensure_ascii=False, indent=2
        )
    )

    partes = [
        BLOQUE_1_ROL,
        "",
        bloque_2_relleno,
        "",
        BLOQUE_3_GUIA_MORFOLOGICA,
        "",
        BLOQUE_4_CITACION,
        "",
    ]

    # Si hay ejemplos, insertarlos antes del formato
    if ejemplos_few_shot:
        partes.append("=== ELABORATED REFERENCE EXAMPLES ===")
        partes.append(f"Below are {len(ejemplos_few_shot)} elaborated "
                       "clinical notes paired with their classifier "
                       "evidence, illustrating the expected style.")
        partes.append("")

        for i, ej in enumerate(ejemplos_few_shot, start=1):
            partes.append(f"--- Example {i} (class {ej['clase']}, "
                           f"{ej['escenario']}) ---")
            partes.append("Classifier evidence:")
            partes.append(json.dumps(
                ej["json_clasificador"], ensure_ascii=False, indent=2
            ))
            partes.append(f"Clinical note: {ej['nota_clinica']}")
            partes.append("")

        partes.append("=== END OF EXAMPLES ===")
        partes.append("")

    partes.append(BLOQUE_5_FORMATO)
    partes.append("")
    partes.append(BLOQUE_6_RESTRICCIONES)
    partes.append("")
    partes.append("=== CURRENT CASE — GENERATE NOTE NOW ===")

    return "\n".join(partes)


# ─────────────────────────────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────────────────────────────
def cargar_conjunto(prefijo: str) -> Dict[str, List[dict]]:
    return {
        clase: json.loads(
            (CONJUNTOS_DIR / f"{prefijo}_{clase}.json").read_text(encoding="utf-8")
        )
        for clase in CLASES
    }


def cargar_ejemplos_elaborados() -> List[dict]:
    """Carga los 12 ejemplos artificiales elaborados."""
    if not EJEMPLOS_JSON.exists():
        sys.exit(f"ERROR: no encuentro {EJEMPLOS_JSON}. Colocalo en "
                 f"{BASE_DIR}")
    data = json.loads(EJEMPLOS_JSON.read_text(encoding="utf-8"))
    return data["ejemplos"]


def muestrear_ejemplos_balanceados(pool: List[dict], k: int) -> List[dict]:
    """Muestrea K ejemplos balanceados por clase desde el pool artificial."""
    if k == 0:
        return []
    n_por_clase = k // 4
    resto = k % 4
    seleccionados = []
    for i, clase in enumerate(CLASES):
        pool_clase = [e for e in pool if e["clase"] == clase]
        n_aqui = n_por_clase + (1 if i < resto else 0)
        n_aqui = min(n_aqui, len(pool_clase))
        seleccionados.extend(random.sample(pool_clase, n_aqui))
    random.shuffle(seleccionados)
    return seleccionados


# ─────────────────────────────────────────────────────────────────────
# RECONSTRUCCION DEL JSON DEL CLASIFICADOR
# (identica al LLM-B v1 para permitir comparabilidad)
# ─────────────────────────────────────────────────────────────────────
SHAPELETS_TIPICOS = {
    "NORMAL": {"Micro_S3":  1.87, "Micro_S52": 3.02, "Micro_S59": 2.65},
    "PAC":    {"Micro_S59": 1.42, "Micro_S52": 1.65, "Micro_S3":  3.10},
    "NSR":    {"Macro_S22": 2.31, "Macro_S54": 8.45, "Macro_S56": 7.20},
    "AFIB":   {"Macro_S54": 1.31, "Macro_S56": 2.03, "Macro_S22": 6.80},
}
CLASE_OPUESTA = {"NORMAL": "PAC", "PAC": "NORMAL",
                 "NSR": "AFIB",   "AFIB": "NSR"}


def simular_json_clasificador(registro_ptbxl: dict) -> dict:
    clase = registro_ptbxl["clase_aami"]
    likelihood = registro_ptbxl["likelihood_principal"]
    if likelihood == 0.0:
        prob_positive = 0.90
    else:
        prob_positive = min(0.99, max(0.51, likelihood / 100.0))
    prob_negative = 1.0 - prob_positive
    confidence_gap = abs(prob_positive - prob_negative)
    es_latido = clase in ("NORMAL", "PAC")
    return {
        "prediccion": clase,
        "modelo": "LATIDO" if es_latido else "RITMO",
        "probabilidades": {
            clase: round(prob_positive, 3),
            CLASE_OPUESTA[clase]: round(prob_negative, 3),
        },
        "confidence_gap": round(confidence_gap, 3),
        "alerta_baja_confianza": confidence_gap < 0.25,
        "distancias_top3_msm": SHAPELETS_TIPICOS[clase],
    }


# ─────────────────────────────────────────────────────────────────────
# LLAMADA A GEMINI CON REINTENTOS
# ─────────────────────────────────────────────────────────────────────
def crear_cliente():
    from google import genai
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def llamar_gemini(cliente, prompt: str, max_reintentos: int = 5) -> str:
    espera = 5
    for intento in range(1, max_reintentos + 1):
        try:
            respuesta = cliente.models.generate_content(
                model=MODELO_GEMINI,
                contents=prompt,
            )
            return respuesta.text.strip() if respuesta.text else ""
        except Exception as e:
            msg = str(e)
            recuperable = ("503" in msg or "429" in msg
                            or "UNAVAILABLE" in msg
                            or "RESOURCE_EXHAUSTED" in msg)
            if recuperable and intento < max_reintentos:
                print(f"    [reintento {intento}] esperando {espera}s")
                time.sleep(espera)
                espera *= 2
                continue
            raise
    return ""


# ─────────────────────────────────────────────────────────────────────
# METRICAS (identicas al v1)
# ─────────────────────────────────────────────────────────────────────
def calcular_bleu(referencias, predicciones):
    import sacrebleu
    return sacrebleu.corpus_bleu(predicciones, [referencias]).score


def calcular_rouge(referencias, predicciones):
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    scores = [scorer.score(r, p) for r, p in zip(referencias, predicciones) if p]
    if not scores:
        return {"rouge1_f": 0.0, "rouge2_f": 0.0, "rougeL_f": 0.0}
    return {
        "rouge1_f": sum(s["rouge1"].fmeasure for s in scores) / len(scores),
        "rouge2_f": sum(s["rouge2"].fmeasure for s in scores) / len(scores),
        "rougeL_f": sum(s["rougeL"].fmeasure for s in scores) / len(scores),
    }


def calcular_sbert(referencias, predicciones):
    from sentence_transformers import SentenceTransformer, util
    modelo = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    pares = [(r, p) for r, p in zip(referencias, predicciones) if p]
    if not pares:
        return 0.0
    refs = [r for r, _ in pares]
    preds = [p for _, p in pares]
    emb_ref = modelo.encode(refs, convert_to_tensor=True)
    emb_pred = modelo.encode(preds, convert_to_tensor=True)
    cosinos = util.cos_sim(emb_ref, emb_pred).diagonal()
    return float(cosinos.mean())


# ─────────────────────────────────────────────────────────────────────
# CHECKPOINTING
# ─────────────────────────────────────────────────────────────────────
def cargar_checkpoint(k):
    ruta = RESULTADOS_DIR / f"checkpoint_k{k}.json"
    if ruta.exists():
        return json.loads(ruta.read_text(encoding="utf-8"))
    return {"ecg_ids_procesados": [], "n_completados": 0}


def guardar_checkpoint(k, ids, n):
    ruta = RESULTADOS_DIR / f"checkpoint_k{k}.json"
    ruta.write_text(
        json.dumps({"ecg_ids_procesados": ids, "n_completados": n}, indent=2),
        encoding="utf-8",
    )


# ─────────────────────────────────────────────────────────────────────
# EXPERIMENTO PRINCIPAL
# ─────────────────────────────────────────────────────────────────────
def ejecutar_experimento(k, limite=None, clase_filtro=None, reanudar=False):
    print(f"\n{'=' * 70}")
    print(f"LLM-B v2 — EXPERIMENTO CON K = {k} EJEMPLOS ELABORADOS")
    print(f"{'=' * 70}")

    print("\nCargando datos...")
    test = cargar_conjunto("test")
    ejemplos_pool = cargar_ejemplos_elaborados()
    print(f"  Pool de ejemplos elaborados: {len(ejemplos_pool)}")

    test_flat = []
    for clase in CLASES:
        if clase_filtro and clase != clase_filtro:
            continue
        test_flat.extend(test[clase])
    if limite:
        test_flat = test_flat[:limite]
    print(f"  Notas de test: {len(test_flat)}")

    checkpoint = cargar_checkpoint(k) if reanudar else {
        "ecg_ids_procesados": [], "n_completados": 0
    }
    ids_ya = set(checkpoint["ecg_ids_procesados"])

    print("\nInicializando cliente Gemini...")
    cliente = crear_cliente()

    modo = "a" if reanudar and ids_ya else "w"
    out_jsonl = RESULTADOS_DIR / f"generaciones_k{k}.jsonl"

    n_procesados = checkpoint["n_completados"]
    n_errores = 0
    tiempo_inicio = time.time()

    print(f"\nProcesando (throttling {SEG_ENTRE_LLAMADAS}s)...\n")

    with open(out_jsonl, modo, encoding="utf-8") as fout:
        for i, registro in enumerate(test_flat, start=1):
            ecg_id = registro["ecg_id_ptbxl"]
            if ecg_id in ids_ya:
                continue

            ejemplos = muestrear_ejemplos_balanceados(ejemplos_pool, k)
            json_clf = simular_json_clasificador(registro)
            prompt = construir_prompt_v3(json_clf, ejemplos)

            try:
                t0 = time.time()
                nota = llamar_gemini(cliente, prompt)
                latencia_ms = int((time.time() - t0) * 1000)
                if not nota:
                    raise ValueError("respuesta vacia")
            except Exception as e:
                print(f"  [{i}/{len(test_flat)}] ERROR ecg_id "
                       f"{ecg_id}: {type(e).__name__}")
                nota = ""
                latencia_ms = -1
                n_errores += 1

            fout.write(json.dumps({
                "ecg_id":      ecg_id,
                "clase":       registro["clase_aami"],
                "referencia":  registro["report_ptbxl"],
                "prediccion":  nota,
                "k":           k,
                "latencia_ms": latencia_ms,
            }, ensure_ascii=False) + "\n")
            fout.flush()

            n_procesados += 1
            ids_ya.add(ecg_id)

            if i % 10 == 0 or i == len(test_flat):
                elapsed = time.time() - tiempo_inicio
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(test_flat) - i) / rate if rate > 0 else 0
                print(f"  [{i:3d}/{len(test_flat)}] "
                       f"errores={n_errores}, "
                       f"ritmo={rate*60:.1f}/min, "
                       f"ETA={eta/60:.1f} min")

            if i % CHECKPOINT_CADA == 0:
                guardar_checkpoint(k, list(ids_ya), n_procesados)

            time.sleep(SEG_ENTRE_LLAMADAS)

    guardar_checkpoint(k, list(ids_ya), n_procesados)
    print(f"\n  Generaciones: {out_jsonl}")
    print(f"  Total: {n_procesados}, errores: {n_errores}")

    print("\nCalculando metricas...")
    with open(out_jsonl, encoding="utf-8") as f:
        generaciones = [json.loads(line) for line in f]

    referencias = [g["referencia"] for g in generaciones]
    predicciones = [g["prediccion"] for g in generaciones]

    print("  BLEU...")
    bleu = calcular_bleu(referencias, predicciones)
    print("  ROUGE...")
    rouge = calcular_rouge(referencias, predicciones)
    print("  SBERT...")
    sbert = calcular_sbert(referencias, predicciones)

    metricas_clase = {}
    for clase in CLASES:
        gen_c = [g for g in generaciones
                  if g["clase"] == clase and g["prediccion"]]
        if not gen_c:
            continue
        refs = [g["referencia"] for g in gen_c]
        preds = [g["prediccion"] for g in gen_c]
        metricas_clase[clase] = {
            "n":         len(gen_c),
            "bleu":      calcular_bleu(refs, preds),
            **calcular_rouge(refs, preds),
            "sbert_cos": calcular_sbert(refs, preds),
        }

    metricas = {
        "k": k, "n_test": len(generaciones), "n_errores": n_errores,
        "global": {"bleu": bleu, **rouge, "sbert_cos": sbert},
        "por_clase": metricas_clase,
    }
    out_met = RESULTADOS_DIR / f"metricas_k{k}.json"
    out_met.write_text(
        json.dumps(metricas, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  Metricas: {out_met}")

    print(f"\n{'-' * 70}")
    print(f"RESULTADOS LLM-B v2 K = {k}")
    print(f"{'-' * 70}")
    print(f"  BLEU:       {bleu:7.3f}")
    print(f"  ROUGE-1 F:  {rouge['rouge1_f']:7.4f}")
    print(f"  ROUGE-2 F:  {rouge['rouge2_f']:7.4f}")
    print(f"  ROUGE-L F:  {rouge['rougeL_f']:7.4f}")
    print(f"  SBERT cos:  {sbert:7.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, required=True, choices=[0, 4, 8])
    parser.add_argument("--limite", type=int, default=None)
    parser.add_argument("--clase", type=str, default=None, choices=CLASES)
    parser.add_argument("--reanudar", action="store_true")
    args = parser.parse_args()

    if "GEMINI_API_KEY" not in os.environ:
        sys.exit("ERROR: define GEMINI_API_KEY primero.")

    ejecutar_experimento(
        k=args.k, limite=args.limite,
        clase_filtro=args.clase, reanudar=args.reanudar,
    )
