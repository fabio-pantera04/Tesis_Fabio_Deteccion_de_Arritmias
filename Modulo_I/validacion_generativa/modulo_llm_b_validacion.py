"""
Modulo LLM-B — Experimento de validacion de generacion con few-shot
────────────────────────────────────────────────────────────────────
Proposito:
    Valida cuantitativamente el efecto del numero de ejemplos few-shot
    (K = 0, 10, 20) sobre la calidad de las notas clinicas generadas
    por Gemini 2.5 Flash. Para cada nota de test:
        1. Reconstruye sinteticamente el JSON que produciria el
           clasificador del Modulo I a partir del codigo SCP conocido.
        2. Construye un prompt con K ejemplos few-shot muestreados
           estratificadamente por clase desde el conjunto few-shot.
        3. Llama al API de Gemini 2.5 Flash con el prompt completo.
        4. Compara la nota generada contra el reporte real de PTB-XL
           usando BLEU, ROUGE-1/2/L y SBERT (multilingue).

    Este modulo es INDEPENDIENTE del sistema de produccion (Modulo
    LLM-A). Comparten solo la API key de Gemini. La nota experimental
    se genera en INGLES (para comparar contra el reporte PTB-XL); la
    nota de produccion se genera en espanol clinico (validacion por
    cardiologos).

Prerrequisitos:
    - Extraccion previa hecha con extraer_fewshot_y_test.py
    - Variable de entorno GEMINI_API_KEY definida
    - Paquetes: google-genai, sacrebleu, rouge-score,
                sentence-transformers, tqdm

Ejecucion:
    python modulo_llm_b_validacion.py --k 0
    python modulo_llm_b_validacion.py --k 10
    python modulo_llm_b_validacion.py --k 20

    Opcionalmente:
    --limite N       Procesa solo N notas (para debug rapido)
    --clase CLASE    Procesa solo una clase (NORMAL, PAC, NSR, AFIB)
    --reanudar       Retoma desde el ultimo checkpoint si existe

Salida:
    resultados_llm_b/
        generaciones_k{K}.jsonl       una linea por nota generada
        checkpoint_k{K}.json          para reanudar si se interrumpe
        metricas_k{K}.json            BLEU, ROUGE, SBERT agregadas
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
# CONFIGURACION — RUTAS Y CONSTANTES
# ─────────────────────────────────────────────────────────────────────
CONJUNTOS_DIR = Path(
    r"E:\Modelo_Tesis\Modulo_I\validacion_generativa\conjuntos_ptbxl"
)
RESULTADOS_DIR = Path(
    r"E:\Modelo_Tesis\Modulo_I\validacion_generativa\resultados_llm_b"
)
RESULTADOS_DIR.mkdir(exist_ok=True, parents=True)

CLASES = ["NORMAL", "PAC", "NSR", "AFIB"]
MODELO_GEMINI = "gemini-2.5-flash"

# Throttling: free tier permite 15 RPM. Usamos 12 RPM para tener margen.
SEG_ENTRE_LLAMADAS = 5.0

# Checkpointing: cada N notas guardamos el estado
CHECKPOINT_CADA = 20

random.seed(42)


# ─────────────────────────────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────────────────────────────
def cargar_conjunto(prefijo: str) -> Dict[str, List[dict]]:
    """Carga few_shot_XXX.json o test_XXX.json de las 4 clases."""
    return {
        clase: json.loads(
            (CONJUNTOS_DIR / f"{prefijo}_{clase}.json").read_text(encoding="utf-8")
        )
        for clase in CLASES
    }


# ─────────────────────────────────────────────────────────────────────
# RECONSTRUCCION SINTETICA DEL JSON DEL CLASIFICADOR
# El LLM-B recibe el MISMO formato de datos que recibira en produccion.
# Como no ejecutamos el clasificador real sobre PTB-XL, reconstruimos
# el JSON a partir de los codigos SCP conocidos.
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
    """Reconstruye el contrato JSON del Modulo I a partir del registro
    PTB-XL. La probabilidad se deriva del likelihood SCP cuando esta
    disponible; cuando es 0 (no calibrado) asumimos alta confianza."""
    clase = registro_ptbxl["clase_aami"]
    likelihood = registro_ptbxl["likelihood_principal"]

    # Cuando likelihood = 0 (no calibrado en PTB-XL) tratamos como si
    # fuera 90 (alta confianza por defecto); en caso contrario usamos
    # el valor normalizado.
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
            clase:            round(prob_positive, 3),
            CLASE_OPUESTA[clase]: round(prob_negative, 3),
        },
        "confidence_gap": round(confidence_gap, 3),
        "alerta_baja_confianza": confidence_gap < 0.25,
        "distancias_top3_msm": SHAPELETS_TIPICOS[clase],
    }


# ─────────────────────────────────────────────────────────────────────
# CONSTRUCCION DEL PROMPT
# ─────────────────────────────────────────────────────────────────────
PROMPT_SISTEMA = """You are a clinical assistant specialized in ECG \
interpretation following the SCP-ECG standard (Standard Communication \
Protocol for Computer-Assisted Electrocardiography). You receive \
quantitative evidence from a dual MSM-SC + XGBoost classifier and must \
generate a short cardiologist-style report note.

The classifier outputs:
  - prediction: the predicted AAMI EC57:2012 class (NORMAL, PAC, NSR, AFIB)
  - probabilities per class and confidence_gap
  - distances to the top-3 most discriminative MSM shapelets

Generate the note in ENGLISH following the SCP-ECG report style \
observed in PTB-XL: concise, clinical, morphology-oriented, no \
therapeutic recommendations. Use lowercase like the reference reports.
"""


def construir_prompt(json_clasificador: dict,
                      ejemplos_few_shot: List[dict]) -> str:
    partes = [PROMPT_SISTEMA, ""]

    if ejemplos_few_shot:
        partes.append("=== REFERENCE EXAMPLES FROM PTB-XL ===")
        partes.append(f"Below are {len(ejemplos_few_shot)} real "
                       "cardiologist reports paired with the classifier "
                       "evidence that would be produced for each case. "
                       "Follow this exact style.")
        partes.append("")

        for i, ej in enumerate(ejemplos_few_shot, start=1):
            json_ej = simular_json_clasificador(ej)
            partes.append(f"--- Example {i} (class {ej['clase_aami']}) ---")
            partes.append("Classifier evidence:")
            partes.append(json.dumps(json_ej, ensure_ascii=False, indent=2))
            partes.append(f"Cardiologist report: {ej['report_ptbxl']}")
            partes.append("")

        partes.append("=== END OF EXAMPLES ===")
        partes.append("")

    partes.append("=== CURRENT CASE ===")
    partes.append("Classifier evidence for the case to interpret:")
    partes.append(json.dumps(json_clasificador, ensure_ascii=False, indent=2))
    partes.append("")
    partes.append("Generate the cardiologist report note for this case, "
                   "in the same style as the reference examples "
                   "(if any were provided). Output ONLY the note text, "
                   "no preamble or explanation.")

    return "\n".join(partes)


# ─────────────────────────────────────────────────────────────────────
# LLAMADA A GEMINI (nueva SDK google-genai)
# ─────────────────────────────────────────────────────────────────────
def crear_cliente():
    """Inicializa el cliente de Gemini una sola vez."""
    from google import genai
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def llamar_gemini(cliente, prompt: str, max_reintentos: int = 5) -> str:
    """Llama a Gemini 2.5 Flash con reintentos automaticos ante errores
    503 (servidor saturado) y 429 (rate limit). Usa backoff exponencial:
    espera 5, 10, 20, 40, 80 segundos entre reintentos."""
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
            # 503 = servidor saturado; 429 = rate limit
            es_recuperable = ("503" in msg or "429" in msg
                              or "UNAVAILABLE" in msg
                              or "RESOURCE_EXHAUSTED" in msg)
            if es_recuperable and intento < max_reintentos:
                print(f"    [reintento {intento}/{max_reintentos - 1}] "
                       f"esperando {espera}s por: "
                       f"{type(e).__name__}")
                time.sleep(espera)
                espera *= 2   # backoff exponencial: 5, 10, 20, 40, 80
                continue
            # No es recuperable o ya agotamos reintentos
            raise
    return ""


# ─────────────────────────────────────────────────────────────────────
# METRICAS
# ─────────────────────────────────────────────────────────────────────
def calcular_bleu(referencias: List[str], predicciones: List[str]) -> float:
    """BLEU corpus-level con sacrebleu."""
    import sacrebleu
    return sacrebleu.corpus_bleu(predicciones, [referencias]).score


def calcular_rouge(referencias: List[str],
                    predicciones: List[str]) -> dict:
    """ROUGE-1, ROUGE-2, ROUGE-L promediados sobre el conjunto."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    scores = [
        scorer.score(ref, pred)
        for ref, pred in zip(referencias, predicciones)
        if pred  # solo notas generadas correctamente
    ]
    if not scores:
        return {"rouge1_f": 0.0, "rouge2_f": 0.0, "rougeL_f": 0.0}
    return {
        "rouge1_f": sum(s["rouge1"].fmeasure for s in scores) / len(scores),
        "rouge2_f": sum(s["rouge2"].fmeasure for s in scores) / len(scores),
        "rougeL_f": sum(s["rougeL"].fmeasure for s in scores) / len(scores),
    }


def calcular_sbert(referencias: List[str],
                    predicciones: List[str]) -> float:
    """Similitud coseno promedio con embeddings SBERT multilingue."""
    from sentence_transformers import SentenceTransformer, util
    modelo = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    # Filtrar los que fallaron
    pares = [(r, p) for r, p in zip(referencias, predicciones) if p]
    if not pares:
        return 0.0
    refs = [r for r, _ in pares]
    preds = [p for _, p in pares]
    emb_ref  = modelo.encode(refs, convert_to_tensor=True)
    emb_pred = modelo.encode(preds, convert_to_tensor=True)
    cosinos = util.cos_sim(emb_ref, emb_pred).diagonal()
    return float(cosinos.mean())


# ─────────────────────────────────────────────────────────────────────
# CHECKPOINTING
# ─────────────────────────────────────────────────────────────────────
def cargar_checkpoint(k: int) -> dict:
    ruta = RESULTADOS_DIR / f"checkpoint_k{k}.json"
    if ruta.exists():
        return json.loads(ruta.read_text(encoding="utf-8"))
    return {"ecg_ids_procesados": [], "n_completados": 0}


def guardar_checkpoint(k: int, ecg_ids_procesados: list,
                        n_completados: int) -> None:
    ruta = RESULTADOS_DIR / f"checkpoint_k{k}.json"
    ruta.write_text(
        json.dumps({
            "ecg_ids_procesados": ecg_ids_procesados,
            "n_completados": n_completados,
        }, indent=2),
        encoding="utf-8",
    )


# ─────────────────────────────────────────────────────────────────────
# EXPERIMENTO PRINCIPAL
# ─────────────────────────────────────────────────────────────────────
def muestrear_few_shot(few_shot: Dict[str, List[dict]], k: int) -> List[dict]:
    """Muestrea K ejemplos few-shot balanceados por clase."""
    if k == 0:
        return []
    n_por_clase = k // 4
    resto = k % 4
    ejemplos = []
    for j, clase in enumerate(CLASES):
        n_aqui = n_por_clase + (1 if j < resto else 0)
        n_disponibles = min(n_aqui, len(few_shot[clase]))
        ejemplos.extend(random.sample(few_shot[clase], n_disponibles))
    random.shuffle(ejemplos)
    return ejemplos


def ejecutar_experimento(k: int, limite: int = None,
                          clase_filtro: str = None,
                          reanudar: bool = False) -> None:
    print(f"\n{'=' * 70}")
    print(f"EXPERIMENTO LLM-B con K = {k} ejemplos few-shot")
    print(f"{'=' * 70}")

    print("\nCargando conjuntos...")
    few_shot = cargar_conjunto("few_shot")
    test = cargar_conjunto("test")

    # Aplanar test a lista unica y filtrar si se especifico una clase
    test_flat = []
    for clase in CLASES:
        if clase_filtro and clase != clase_filtro:
            continue
        test_flat.extend(test[clase])

    if limite:
        test_flat = test_flat[:limite]

    total_disponible = len(test_flat)
    print(f"  Few-shot pool: {sum(len(v) for v in few_shot.values())} notas")
    print(f"  Test a procesar: {total_disponible} notas")
    if clase_filtro:
        print(f"  Filtro de clase: {clase_filtro}")

    # Cargar checkpoint si aplica
    checkpoint = cargar_checkpoint(k) if reanudar else {
        "ecg_ids_procesados": [], "n_completados": 0
    }
    ids_ya_hechos = set(checkpoint["ecg_ids_procesados"])
    print(f"  Ya procesados (checkpoint): {len(ids_ya_hechos)}")

    # Cliente de Gemini
    print("\nInicializando cliente Gemini...")
    cliente = crear_cliente()

    # Preparar archivo JSONL de salida (append mode si reanudar)
    modo = "a" if reanudar and ids_ya_hechos else "w"
    out_jsonl = RESULTADOS_DIR / f"generaciones_k{k}.jsonl"

    n_procesados = checkpoint["n_completados"]
    n_errores = 0
    tiempo_inicio = time.time()

    print(f"\nProcesando (throttling {SEG_ENTRE_LLAMADAS}s entre "
           "llamadas)...\n")

    with open(out_jsonl, modo, encoding="utf-8") as fout:
        for i, registro in enumerate(test_flat, start=1):
            ecg_id = registro["ecg_id_ptbxl"]

            # Saltar si ya se proceso en checkpoint
            if ecg_id in ids_ya_hechos:
                continue

            # Muestrear K ejemplos few-shot balanceados
            ejemplos = muestrear_few_shot(few_shot, k)

            # Simular JSON del clasificador y construir prompt
            json_clf = simular_json_clasificador(registro)
            prompt = construir_prompt(json_clf, ejemplos)

            # Llamar Gemini con manejo de errores
            try:
                t0 = time.time()
                nota_generada = llamar_gemini(cliente, prompt)
                latencia_ms = int((time.time() - t0) * 1000)
                if not nota_generada:
                    raise ValueError("respuesta vacia de Gemini")
            except Exception as e:
                print(f"  [{i}/{total_disponible}] ERROR ecg_id "
                       f"{ecg_id}: {type(e).__name__}: {e}")
                nota_generada = ""
                latencia_ms = -1
                n_errores += 1

            # Guardar en JSONL
            fout.write(json.dumps({
                "ecg_id":     ecg_id,
                "clase":      registro["clase_aami"],
                "referencia": registro["report_ptbxl"],
                "prediccion": nota_generada,
                "k":          k,
                "latencia_ms": latencia_ms,
            }, ensure_ascii=False) + "\n")
            fout.flush()

            n_procesados += 1
            ids_ya_hechos.add(ecg_id)

            # Log de progreso cada 10
            if i % 10 == 0 or i == total_disponible:
                elapsed = time.time() - tiempo_inicio
                rate = i / elapsed if elapsed > 0 else 0
                eta = (total_disponible - i) / rate if rate > 0 else 0
                print(f"  [{i:3d}/{total_disponible}] "
                       f"errores={n_errores}, "
                       f"ritmo={rate*60:.1f}/min, "
                       f"ETA={eta/60:.1f} min")

            # Checkpoint periodico
            if i % CHECKPOINT_CADA == 0:
                guardar_checkpoint(k, list(ids_ya_hechos), n_procesados)

            time.sleep(SEG_ENTRE_LLAMADAS)

    # Checkpoint final
    guardar_checkpoint(k, list(ids_ya_hechos), n_procesados)
    print(f"\n  Generaciones guardadas: {out_jsonl}")
    print(f"  Total procesadas: {n_procesados}, errores: {n_errores}")

    # Calcular metricas sobre lo generado
    print("\nCalculando metricas...")
    with open(out_jsonl, encoding="utf-8") as f:
        generaciones = [json.loads(line) for line in f]

    referencias  = [g["referencia"] for g in generaciones]
    predicciones = [g["prediccion"] for g in generaciones]

    print("  BLEU corpus...")
    bleu = calcular_bleu(referencias, predicciones)

    print("  ROUGE 1/2/L...")
    rouge = calcular_rouge(referencias, predicciones)

    print("  SBERT multilingue (descarga modelo la primera vez)...")
    sbert = calcular_sbert(referencias, predicciones)

    # Metricas por clase tambien
    metricas_por_clase = {}
    for clase in CLASES:
        gen_clase = [g for g in generaciones if g["clase"] == clase and g["prediccion"]]
        if not gen_clase:
            continue
        refs = [g["referencia"] for g in gen_clase]
        preds = [g["prediccion"] for g in gen_clase]
        metricas_por_clase[clase] = {
            "n": len(gen_clase),
            "bleu": calcular_bleu(refs, preds),
            **calcular_rouge(refs, preds),
            "sbert_cos": calcular_sbert(refs, preds),
        }

    metricas = {
        "k":            k,
        "n_test":       len(generaciones),
        "n_errores":    n_errores,
        "global": {
            "bleu":     bleu,
            **rouge,
            "sbert_cos": sbert,
        },
        "por_clase": metricas_por_clase,
    }

    out_met = RESULTADOS_DIR / f"metricas_k{k}.json"
    out_met.write_text(
        json.dumps(metricas, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"  Metricas guardadas: {out_met}")

    print(f"\n{'-' * 70}")
    print(f"RESULTADOS K = {k}")
    print(f"{'-' * 70}")
    print(f"  BLEU:       {bleu:7.3f}")
    print(f"  ROUGE-1 F:  {rouge['rouge1_f']:7.4f}")
    print(f"  ROUGE-2 F:  {rouge['rouge2_f']:7.4f}")
    print(f"  ROUGE-L F:  {rouge['rougeL_f']:7.4f}")
    print(f"  SBERT cos:  {sbert:7.4f}")


# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, required=True, choices=[0, 10, 20])
    parser.add_argument("--limite", type=int, default=None,
                        help="Procesar solo N notas (para debug rapido)")
    parser.add_argument("--clase", type=str, default=None,
                        choices=CLASES,
                        help="Procesar solo una clase")
    parser.add_argument("--reanudar", action="store_true",
                        help="Reanudar desde ultimo checkpoint")
    args = parser.parse_args()

    if "GEMINI_API_KEY" not in os.environ:
        sys.exit("ERROR: define la variable GEMINI_API_KEY primero.")

    ejecutar_experimento(
        k=args.k,
        limite=args.limite,
        clase_filtro=args.clase,
        reanudar=args.reanudar,
    )