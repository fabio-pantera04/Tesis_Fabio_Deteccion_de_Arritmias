"""
Pipeline de Evaluación LLM — Módulo RITMO (v2)
==============================================
Tesis: Clasificación de Arritmias ECG y Generación de Notas Clínicas
CIMAT Zacatecas — Fabio — Defensa Junio 2026

CAMBIOS PRINCIPALES vs v1:
  • Filtro NORM puro corregido vía mapeo a superclases (scp_statements.csv)
  • Detector QRS real (wfdb.processing.xqrs_detect), no cruce de umbral
  • Descarga con wfdb.dl_files (portable, no depende de wget)
  • Confianza NO simulada: modo "agente aislado" (assume-perfect = 1.00)
    o modo "end-to-end" (flag MODO_EVALUACION, requiere XGBoost integrado)
  • Manejo robusto de NaN en reports
  • Retry exponencial para llamadas a API
  • Sin input() bloqueante
  • Métricas: concept recall (principal) + BERTScore multilingüe + BLEU + ROUGE + METEOR
  • Traducción opcional EN→ES de referencias PTB-XL para comparabilidad léxica
  • Reproducibilidad con SEED=42; CSV con muestra y métricas por registro

Uso:
  pip install wfdb nltk rouge-score sacrebleu bert-score pandas numpy anthropic tqdm
  export ANTHROPIC_API_KEY="sk-ant-..."
  python ptbxl_evaluation_pipeline_v2.py
"""

import os
import ast
import json
import random
import time
import re
import urllib.request
import urllib.error
import shutil
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import wfdb
import wfdb.processing as wp
import anthropic
from anthropic import APIError, RateLimitError, APIConnectionError
from tqdm import tqdm

# ── Métricas NLP ───────────────────────────────────────────────────────────────
import nltk
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import sacrebleu
from bert_score import score as bertscore_score

for paquete in ["wordnet", "omw-1.4", "punkt", "punkt_tab"]:
    try:
        nltk.download(paquete, quiet=True)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════

PTBXL_DIR       = Path("./ptb-xl-1.0.3")
OUTPUT_DIR      = Path("./resultados_evaluacion")
OUTPUT_DIR.mkdir(exist_ok=True)

N_PER_CLASS     = 4
SEED            = 42
SAMPLING_RATE   = 500            # records500/ → 500 Hz, 10s = 5000 muestras
LEAD_II_IDX     = 1              # I=0, II=1

# Configuración PhysioNet — URL base directa para descarga de archivos individuales
# (más estable que wfdb.dl_files, que tiene bugs de duplicación de versión en wfdb >= 4.x)
PTBXL_VERSION    = "1.0.3"
PHYSIONET_BASE   = f"https://physionet.org/files/ptb-xl/{PTBXL_VERSION}/"
# Para wfdb.rdsamp(..., pn_dir=PN_DIR): formato 'db_name/version'
PN_DIR           = f"ptb-xl/{PTBXL_VERSION}"

# Modo de evaluación
#   "agente_aislado"  → asume clasificación perfecta (clase real, confianza=1.00)
#                       Evalúa la calidad de generación textual del LLM.
#                       Defendible si se reporta junto con métricas de XGBoost separadas.
#   "end_to_end"      → requiere integración con XGBoost (no implementado aquí;
#                       requiere pipeline MSM-SC shapelets para features Macro_S*).
MODO_EVALUACION = "agente_aislado"

# API Anthropic
ANTHROPIC_MODEL = "claude-sonnet-4-5"   # ajusta al modelo que uses en producción
MAX_TOKENS_OUT  = 512
MAX_REINTENTOS  = 3

# Idioma de evaluación
#   PTB-XL reports están en alemán/inglés telegráfico.
#   Para reducir penalización léxica injusta, evaluamos en INGLÉS:
#   pedimos al agente generar la nota EN INGLÉS para esta comparación.
#   (En producción tu agente genera en español; aquí cambiamos idioma SOLO
#   para esta evaluación de comparabilidad. Documentar esto en el Cap. 4.)
IDIOMA_EVALUACION = "en"   # "en" o "es"

# BERTScore: modelo multilingüe para soportar mezcla de idiomas
BERTSCORE_MODEL = "xlm-roberta-large"
BERTSCORE_LANG  = "en"   # usado para baseline rescaling

# ── Prompt del agente ─────────────────────────────────────────────────────────
SYSTEM_PROMPT_ES = """Eres un asistente médico especializado en electrocardiografía.
Recibirás el resultado de un clasificador de ECG de una sola derivación (Lead II)
junto con métricas de la señal. Tu tarea es generar una nota clínica estructurada
en español que incluya:
1. Diagnóstico de ritmo
2. Hallazgos relevantes de la señal
3. Recomendación clínica según guías AAMI/AHA

Sé conciso pero completo. No inventes datos no proporcionados."""

SYSTEM_PROMPT_EN = """You are a medical assistant specialized in electrocardiography.
You will receive the output of a single-lead (Lead II) ECG classifier together with
signal metrics. Your task is to generate a structured clinical note in English with:
1. Rhythm diagnosis
2. Relevant signal findings
3. Clinical recommendation per AAMI/AHA guidelines

Be concise but complete. Do not invent data that was not provided."""

SYSTEM_PROMPT = SYSTEM_PROMPT_EN if IDIOMA_EVALUACION == "en" else SYSTEM_PROMPT_ES


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 — DESCARGA DE PTB-XL
# ══════════════════════════════════════════════════════════════════════════════

def _descargar_url(url: str, destino: Path, intentos: int = 3):
    """Descarga un archivo desde una URL a una ruta local con reintentos."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    ultima_excepcion = None
    for n in range(intentos):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (research pipeline)"}
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                if resp.status != 200:
                    raise urllib.error.HTTPError(
                        url, resp.status, "non-200", resp.headers, None
                    )
                tmp = destino.with_suffix(destino.suffix + ".tmp")
                with open(tmp, "wb") as f:
                    shutil.copyfileobj(resp, f)
                tmp.replace(destino)
            return
        except Exception as e:
            ultima_excepcion = e
            if n < intentos - 1:
                time.sleep(2 ** n)
    raise RuntimeError(f"Falló descarga de {url}: {ultima_excepcion}")


def descargar_metadatos_ptbxl():
    """Descarga ptbxl_database.csv y scp_statements.csv si no existen."""
    PTBXL_DIR.mkdir(parents=True, exist_ok=True)

    archivos = ["ptbxl_database.csv", "scp_statements.csv"]
    faltantes = [a for a in archivos if not (PTBXL_DIR / a).exists()]

    if not faltantes:
        print("[FASE 1] Metadatos PTB-XL ya presentes. Saltando descarga.")
        return

    print(f"[FASE 1] Descargando metadatos: {faltantes}")
    for archivo in faltantes:
        url = PHYSIONET_BASE + archivo
        destino = PTBXL_DIR / archivo
        try:
            _descargar_url(url, destino)
            print(f"  ✓ {archivo}")
        except Exception as e:
            raise RuntimeError(
                f"No se pudo descargar {archivo} desde {url}: {e}\n"
                f"Descarga manual: {PHYSIONET_BASE}"
            )


def descargar_registro_individual(filename_hr: str):
    """
    Descarga .dat + .hea de un registro WFDB si no existen localmente.
    filename_hr viene como 'records500/00000/00001_hr' desde el CSV.
    """
    for ext in [".dat", ".hea"]:
        destino = PTBXL_DIR / (filename_hr + ext)
        if destino.exists() and destino.stat().st_size > 0:
            continue
        url = PHYSIONET_BASE + filename_hr + ext
        _descargar_url(url, destino)


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 — FILTRADO Y MUESTREO BALANCEADO
# ══════════════════════════════════════════════════════════════════════════════

def cargar_metadata() -> pd.DataFrame:
    """Carga y parsea ptbxl_database.csv."""
    df = pd.read_csv(PTBXL_DIR / "ptbxl_database.csv", index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
    return df


def cargar_mapeo_superclases() -> dict:
    """
    Mapea cada SCP code diagnóstico a su superclase (NORM, MI, STTC, HYP, CD).
    Solo incluye statements diagnósticos (diagnostic == 1).
    """
    df_scp = pd.read_csv(PTBXL_DIR / "scp_statements.csv", index_col=0)
    df_scp = df_scp[df_scp["diagnostic"] == 1]
    return df_scp["diagnostic_class"].to_dict()


def superclases_diagnosticas(scp_dict: dict, mapeo: dict) -> set:
    """Conjunto de superclases diagnósticas presentes en el registro."""
    return {mapeo[c] for c in scp_dict if c in mapeo}


def es_norm_puro(scp_dict: dict, mapeo: dict) -> bool:
    """
    NORM puro = única superclase diagnóstica es NORM y NO co-ocurre con
    statements de arritmia (AFIB, AFLT, SVPB, PVC, etc.) ni con formas
    de pre-excitación o bloqueos relevantes.
    """
    supers = superclases_diagnosticas(scp_dict, mapeo)
    if supers != {"NORM"}:
        return False

    # Exclusiones adicionales por statement específico (no diagnóstico pero clínicamente relevantes)
    excluir = {"AFIB", "AFLT", "SVPB", "PVC", "PSVT", "WPW", "TRIGU",
               "BIGU", "PACE", "SVTAC", "VTAC"}
    return len(set(scp_dict.keys()) & excluir) == 0


def es_afib(scp_dict: dict) -> bool:
    """
    AFIB presente y AFLT ausente (queremos fibrilación, no flutter).
    Likelihood: en PTB-XL para AFIB, presencia se considera válida incluso
    con likelihood=0 (convención no documentada formalmente).
    """
    if "AFLT" in scp_dict:
        return False
    return "AFIB" in scp_dict


def seleccionar_muestra(df: pd.DataFrame, mapeo: dict) -> pd.DataFrame:
    """100 NORM puro + 100 AFIB con semilla fija."""
    random.seed(SEED)
    np.random.seed(SEED)

    mask_norm = df["scp_codes"].apply(lambda x: es_norm_puro(x, mapeo))
    mask_afib = df["scp_codes"].apply(es_afib)

    df_norm = df[mask_norm].copy()
    df_afib = df[mask_afib].copy()

    print(f"[FASE 2] Registros NORM puros disponibles: {len(df_norm)}")
    print(f"[FASE 2] Registros AFIB disponibles:       {len(df_afib)}")

    if len(df_norm) < N_PER_CLASS or len(df_afib) < N_PER_CLASS:
        raise ValueError(
            f"Insuficientes registros. NORM={len(df_norm)}, "
            f"AFIB={len(df_afib)}, requerido={N_PER_CLASS}"
        )

    muestra_norm = df_norm.sample(n=N_PER_CLASS, random_state=SEED)
    muestra_norm["clase_tesis"] = "NSR"

    muestra_afib = df_afib.sample(n=N_PER_CLASS, random_state=SEED)
    muestra_afib["clase_tesis"] = "AFIB"

    muestra = pd.concat([muestra_norm, muestra_afib]).sample(
        frac=1, random_state=SEED
    )
    print(f"[FASE 2] Muestra final: {len(muestra)} registros "
          f"({N_PER_CLASS} NSR + {N_PER_CLASS} AFIB)")
    return muestra


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3 — EXTRACCIÓN DE SEÑAL
# ══════════════════════════════════════════════════════════════════════════════

def extraer_lead_ii(filename_hr: str) -> np.ndarray:
    """Carga registro WFDB y retorna Lead II (5000 muestras a 500 Hz)."""
    ruta = str(PTBXL_DIR / filename_hr)
    signal, _ = wfdb.rdsamp(ruta)        # (5000, 12)
    return signal[:, LEAD_II_IDX]


def estimar_fc_real(lead_ii: np.ndarray, fs: int = SAMPLING_RATE) -> int:
    """FC vía detector QRS Pan-Tompkins (xqrs). Retorna -1 si falla."""
    try:
        qrs = wp.xqrs_detect(lead_ii, fs=fs, verbose=False)
        if len(qrs) < 3:
            return -1
        rr_samples = np.diff(qrs)
        fc = float(60 * fs / np.mean(rr_samples))
        return int(round(fc)) if 30 < fc < 250 else -1
    except Exception:
        return -1


def calcular_metricas_senal(lead_ii: np.ndarray, fs: int = SAMPLING_RATE) -> dict:
    """Métricas básicas que tu sistema real ya calcula antes de invocar al agente."""
    amplitud_mv = float(np.ptp(lead_ii)) / 1000.0   # PTB-XL en µV
    fc = estimar_fc_real(lead_ii, fs)
    return {
        "amplitud_mv":     round(amplitud_mv, 3),
        "fc_estimada_bpm": fc,
        "duracion_s":      len(lead_ii) / fs,
        "fs_hz":           fs,
    }


# ══════════════════════════════════════════════════════════════════════════════
# FASE 4 — GENERACIÓN DE NOTAS VÍA AGENTE
# ══════════════════════════════════════════════════════════════════════════════

def construir_user_prompt(clase: str, confianza: float, metricas: dict,
                          row: pd.Series, idioma: str = "en") -> str:
    edad = row.get("age", "N/D")
    sex_val = row.get("sex", 0)
    sexo = "M" if sex_val == 0 else "F"   # PTB-XL: 0=masculino, 1=femenino

    if idioma == "en":
        return f"""Rhythm classifier output:
- Predicted class: {clase}
- Model confidence: {confianza:.1%}
- Estimated HR: {metricas['fc_estimada_bpm']} bpm
- Lead II amplitude: {metricas['amplitud_mv']:.3f} mV
- Segment duration: {metricas['duracion_s']} s
- Patient: age={edad}, sex={sexo}

Generate the corresponding clinical note."""
    else:
        return f"""Resultado del clasificador de ritmo:
- Clase predicha: {clase}
- Confianza del modelo: {confianza:.1%}
- FC estimada: {metricas['fc_estimada_bpm']} bpm
- Amplitud Lead II: {metricas['amplitud_mv']:.3f} mV
- Duración del segmento: {metricas['duracion_s']} s
- Datos del paciente: edad={edad}, sexo={sexo}

Genera la nota clínica correspondiente."""


def generar_nota_agente(cliente: anthropic.Anthropic, clase: str, confianza: float,
                        metricas: dict, row: pd.Series) -> str:
    """Llama al agente con reintentos exponenciales."""
    user_prompt = construir_user_prompt(
        clase, confianza, metricas, row, idioma=IDIOMA_EVALUACION
    )

    ultimo_error = None
    for intento in range(MAX_REINTENTOS):
        try:
            respuesta = cliente.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=MAX_TOKENS_OUT,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return respuesta.content[0].text.strip()
        except (RateLimitError, APIConnectionError) as e:
            ultimo_error = e
            espera = 2 ** intento + random.random()
            time.sleep(espera)
        except APIError as e:
            ultimo_error = e
            if intento == MAX_REINTENTOS - 1:
                raise
            time.sleep(2 ** intento)

    raise RuntimeError(f"Fallaron {MAX_REINTENTOS} reintentos: {ultimo_error}")


# ══════════════════════════════════════════════════════════════════════════════
# FASE 5 — MÉTRICAS DE EVALUACIÓN
# ══════════════════════════════════════════════════════════════════════════════

# ── 5.1 Concept Recall ────────────────────────────────────────────────────────
#
# Para cada clase, definimos un conjunto de "conceptos clínicos esperados" tanto
# en inglés como en español (variantes léxicas y abreviaturas). El recall mide
# qué fracción de conceptos del reporte de referencia aparecen en la nota
# generada. Es la métrica MÁS DEFENDIBLE para texto clínico.

CONCEPTOS_POR_CLASE = {
    "NSR": {
        "sinus_rhythm":     ["sinus rhythm", "ritmo sinusal", "sinusrhythmus",
                             "normal sinus", "nsr"],
        "normal_ecg":       ["normal ecg", "normales ekg", "unauffälliges ekg",
                             "electrocardiograma normal", "ekg normal"],
        "regular_rhythm":   ["regular rhythm", "ritmo regular", "regelmäßig"],
    },
    "AFIB": {
        "atrial_fibrillation": ["atrial fibrillation", "fibrilación auricular",
                                "vorhofflimmern", "afib", "a-fib", "af",
                                "fibrillation"],
        "irregular_rhythm":    ["irregular rhythm", "ritmo irregular",
                                "irregularly irregular", "arrhythmia",
                                "arritmia", "unregelmäßig"],
        "absent_p_waves":      ["absent p waves", "no p waves", "ausencia de ondas p",
                                "ondas p ausentes", "fehlen", "p-welle"],
    },
}


def normalizar(texto: str) -> str:
    """Lowercase + espacios colapsados + sin puntuación marginal."""
    texto = texto.lower()
    texto = re.sub(r"[^\w\sáéíóúñü-]", " ", texto)
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()


def concept_recall(nota: str, clase: str) -> dict:
    """
    Para la clase predicha, ¿cuántos de los conceptos clínicos clave aparecen
    en la nota? Retorna recall (0-1) + conceptos detectados.
    """
    nota_norm = normalizar(nota)
    conceptos = CONCEPTOS_POR_CLASE.get(clase, {})
    if not conceptos:
        return {"recall": float("nan"), "encontrados": [], "esperados": []}

    encontrados = []
    for concepto_id, variantes in conceptos.items():
        if any(normalizar(v) in nota_norm for v in variantes):
            encontrados.append(concepto_id)

    recall = len(encontrados) / len(conceptos)
    return {
        "recall":      round(recall, 4),
        "encontrados": encontrados,
        "esperados":   list(conceptos.keys()),
    }


def concept_recall_referencia(nota: str, referencia: str, clase: str) -> dict:
    """
    Variante más estricta: solo cuenta como esperados los conceptos
    que efectivamente aparecen en el REPORTE DE REFERENCIA.
    Esto evita penalizar a la nota por conceptos que la referencia ni mencionaba.
    """
    ref_norm = normalizar(referencia)
    nota_norm = normalizar(nota)
    conceptos = CONCEPTOS_POR_CLASE.get(clase, {})

    esperados_en_ref = []
    encontrados_en_nota = []

    for concepto_id, variantes in conceptos.items():
        en_ref  = any(normalizar(v) in ref_norm  for v in variantes)
        en_nota = any(normalizar(v) in nota_norm for v in variantes)
        if en_ref:
            esperados_en_ref.append(concepto_id)
            if en_nota:
                encontrados_en_nota.append(concepto_id)

    if not esperados_en_ref:
        return {"recall_vs_ref": float("nan"),
                "n_esperados_ref": 0,
                "n_encontrados": 0}

    return {
        "recall_vs_ref":    round(len(encontrados_en_nota) / len(esperados_en_ref), 4),
        "n_esperados_ref":  len(esperados_en_ref),
        "n_encontrados":    len(encontrados_en_nota),
    }


# ── 5.2 BLEU / ROUGE / METEOR ─────────────────────────────────────────────────

def calcular_bleu(hipotesis: str, referencia: str) -> float:
    """BLEU-4 con sacrebleu (más robusto que nltk para sentencias cortas)."""
    if not referencia.strip() or not hipotesis.strip():
        return 0.0
    bleu = sacrebleu.sentence_bleu(hipotesis, [referencia])
    return round(bleu.score / 100.0, 4)   # sacrebleu retorna 0-100


def calcular_meteor(hipotesis: str, referencia: str) -> float:
    tokens_hip = hipotesis.lower().split()
    tokens_ref = referencia.lower().split()
    if not tokens_ref or not tokens_hip:
        return 0.0
    try:
        return round(meteor_score([tokens_ref], tokens_hip), 4)
    except Exception:
        return 0.0


def calcular_rouge(hipotesis: str, referencia: str) -> dict:
    if not referencia.strip() or not hipotesis.strip():
        return {"rouge1_f": 0.0, "rougeL_f": 0.0}
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    s = scorer.score(referencia, hipotesis)
    return {
        "rouge1_f": round(s["rouge1"].fmeasure, 4),
        "rougeL_f": round(s["rougeL"].fmeasure, 4),
    }


# ── 5.3 BERTScore multilingüe ─────────────────────────────────────────────────
#
# BERTScore en batch al final (es 100x más rápido que registro por registro).

def calcular_bertscore_batch(hipotesis: list, referencias: list) -> dict:
    """
    Aplica BERTScore en batch. Filtra pares vacíos (los marca como NaN).
    Retorna listas paralelas de P/R/F1 con NaN donde no aplica.
    """
    print(f"\n[FASE 5] Calculando BERTScore con {BERTSCORE_MODEL} (batch)...")

    # Identificar índices válidos (no vacíos)
    idx_validos = [i for i, (h, r) in enumerate(zip(hipotesis, referencias))
                   if h.strip() and r.strip()]

    if not idx_validos:
        n = len(hipotesis)
        return {"P": [float("nan")] * n, "R": [float("nan")] * n,
                "F1": [float("nan")] * n}

    h_val = [hipotesis[i]   for i in idx_validos]
    r_val = [referencias[i] for i in idx_validos]

    P, R, F1 = bertscore_score(
        h_val, r_val,
        model_type=BERTSCORE_MODEL,
        lang=BERTSCORE_LANG,
        rescale_with_baseline=False,   # baseline no disponible para xlm-roberta-large
        verbose=False,
    )

    # Reinsertar en posiciones originales
    P_out  = [float("nan")] * len(hipotesis)
    R_out  = [float("nan")] * len(hipotesis)
    F1_out = [float("nan")] * len(hipotesis)
    for k, i in enumerate(idx_validos):
        P_out[i]  = round(float(P[k]),  4)
        R_out[i]  = round(float(R[k]),  4)
        F1_out[i] = round(float(F1[k]), 4)

    return {"P": P_out, "R": R_out, "F1": F1_out}


# ══════════════════════════════════════════════════════════════════════════════
# UTILIDADES
# ══════════════════════════════════════════════════════════════════════════════

def limpiar_referencia(report_raw) -> str:
    """Limpia y normaliza el campo 'report' de PTB-XL."""
    if pd.isna(report_raw) or not isinstance(report_raw, str):
        return ""
    return report_raw.strip()


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("Pipeline Evaluación LLM — Módulo RITMO (v2)")
    print(f"Modo: {MODO_EVALUACION}  |  Idioma evaluación: {IDIOMA_EVALUACION}")
    print("=" * 70)

    # ── Fase 1: descarga de metadatos ─────────────────────────────────────
    descargar_metadatos_ptbxl()

    # ── Fase 2: filtrado y muestreo ───────────────────────────────────────
    df = cargar_metadata()
    mapeo = cargar_mapeo_superclases()
    muestra = seleccionar_muestra(df, mapeo)

    muestra[["clase_tesis", "report", "scp_codes"]].to_csv(
        OUTPUT_DIR / "muestra_indices.csv"
    )
    print(f"[FASE 2] Índice guardado en {OUTPUT_DIR}/muestra_indices.csv")

    # ── API ───────────────────────────────────────────────────────────────
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "Falta ANTHROPIC_API_KEY. Configura tu clave de API:\n"
            "  PowerShell (sesión actual):\n"
            "    $env:ANTHROPIC_API_KEY = 'sk-ant-...'\n"
            "  PowerShell (permanente):\n"
            "    [System.Environment]::SetEnvironmentVariable("
            "'ANTHROPIC_API_KEY','sk-ant-...','User')\n"
            "  Linux/Mac:\n"
            "    export ANTHROPIC_API_KEY='sk-ant-...'"
        )

    # Detección de placeholder común (evita gastar 200 llamadas si la clave es inválida)
    placeholders = {"sk-ant-tu-clave-aqui", "sk-ant-...", "sk-ant-xxxx",
                    "tu-clave-aqui", "your-key-here"}
    if api_key.strip() in placeholders or not api_key.startswith("sk-ant-"):
        raise EnvironmentError(
            f"ANTHROPIC_API_KEY parece ser un placeholder o tiene formato inválido.\n"
            f"  Valor actual: {api_key[:20]}...\n"
            f"  Obtén tu clave real en: https://console.anthropic.com/settings/keys\n"
            f"  Una clave válida empieza con 'sk-ant-api03-' seguido de caracteres alfanuméricos."
        )

    cliente = anthropic.Anthropic(api_key=api_key)

    # Validación de autenticación con 1 llamada mínima ANTES de procesar los 200 registros
    print("\n[FASE 3] Validando autenticación con Anthropic API...")
    try:
        _test = cliente.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=10,
            messages=[{"role": "user", "content": "ping"}],
        )
        print("  ✓ Autenticación OK.")
    except anthropic.AuthenticationError as e:
        raise EnvironmentError(
            f"Autenticación rechazada por Anthropic API: {e}\n"
            f"Verifica que tu ANTHROPIC_API_KEY sea correcta y esté activa en\n"
            f"https://console.anthropic.com/settings/keys"
        )
    except Exception as e:
        print(f"  ⚠ Advertencia en validación inicial: {e}")
        print("  Continuando de todos modos (puede ser un problema transitorio)...")

    # ── Fase 3+4: descarga señal, extracción, generación ──────────────────
    resultados = []
    print(f"\n[FASE 3+4] Procesando {len(muestra)} registros...\n")

    for ecg_id, row in tqdm(muestra.iterrows(), total=len(muestra)):
        try:
            descargar_registro_individual(row["filename_hr"])
            lead_ii = extraer_lead_ii(row["filename_hr"])
            metricas = calcular_metricas_senal(lead_ii)

            # Modo "agente aislado": clase verdadera + confianza=1.00
            # En end-to-end aquí iría: clase, conf = xgboost.predict(features_msm)
            if MODO_EVALUACION == "agente_aislado":
                clase_input = row["clase_tesis"]
                confianza   = 1.00
            else:
                raise NotImplementedError(
                    "Modo 'end_to_end' requiere integración con pipeline MSM-SC "
                    "+ XGBoost. Implementar en módulo separado."
                )

            nota_generada = generar_nota_agente(
                cliente, clase_input, confianza, metricas, row
            )

            reporte_ref = limpiar_referencia(row.get("report"))

            # Métricas léxicas
            bleu   = calcular_bleu(nota_generada,   reporte_ref)
            meteor = calcular_meteor(nota_generada, reporte_ref)
            rouge  = calcular_rouge(nota_generada,  reporte_ref)

            # Concept recall
            cr_clase = concept_recall(nota_generada, row["clase_tesis"])
            cr_ref   = concept_recall_referencia(
                nota_generada, reporte_ref, row["clase_tesis"]
            )

            resultados.append({
                "ecg_id":            ecg_id,
                "clase_tesis":       row["clase_tesis"],
                "reporte_ptbxl":     reporte_ref,
                "nota_generada":     nota_generada,
                "concept_recall":             cr_clase["recall"],
                "concept_recall_vs_ref":      cr_ref["recall_vs_ref"],
                "conceptos_encontrados":      ";".join(cr_clase["encontrados"]),
                "conceptos_n_esperados_ref":  cr_ref["n_esperados_ref"],
                "bleu":              bleu,
                "meteor":            meteor,
                "rouge1_f":          rouge["rouge1_f"],
                "rougeL_f":          rouge["rougeL_f"],
                "fc_estimada_bpm":   metricas["fc_estimada_bpm"],
                "amplitud_mv":       metricas["amplitud_mv"],
                "edad":              row.get("age", "N/D"),
                "sexo":              "M" if row.get("sex", 0) == 0 else "F",
                "error":             None,
            })

            time.sleep(0.3)   # respeta rate limits

        except Exception as e:
            print(f"\n  [ERROR] ecg_id={ecg_id}: {e}")
            resultados.append({
                "ecg_id": ecg_id,
                "clase_tesis": row.get("clase_tesis", "?"),
                "error": str(e),
            })

    df_res = pd.DataFrame(resultados)

    # ── BERTScore en batch al final ───────────────────────────────────────
    df_ok = df_res[df_res["error"].isna()].copy()
    if len(df_ok) > 0:
        bs = calcular_bertscore_batch(
            df_ok["nota_generada"].tolist(),
            df_ok["reporte_ptbxl"].tolist(),
        )
        df_ok["bertscore_P"]  = bs["P"]
        df_ok["bertscore_R"]  = bs["R"]
        df_ok["bertscore_F1"] = bs["F1"]
        # Merge BERTScore al df principal
        for col in ["bertscore_P", "bertscore_R", "bertscore_F1"]:
            df_res[col] = df_ok[col]

    # ── Exportar resultados ───────────────────────────────────────────────
    ruta_csv = OUTPUT_DIR / "resultados_evaluacion.csv"
    df_res.to_csv(ruta_csv, index=False)
    print(f"\n[FASE 5] Resultados guardados en {ruta_csv}")

    # ── Resumen ────────────────────────────────────────────────────────────
    df_ok = df_res[df_res["error"].isna()].copy()

    print("\n" + "=" * 70)
    print("RESUMEN DE MÉTRICAS — Módulo RITMO (NSR vs AFIB)")
    print("=" * 70)

    cols_metricas = ["concept_recall", "concept_recall_vs_ref",
                     "bleu", "meteor", "rouge1_f", "rougeL_f",
                     "bertscore_F1"]

    def imprimir_resumen(sub, titulo):
        print(f"\n  {titulo}  (n={len(sub)})")
        for c in cols_metricas:
            if c not in sub.columns:
                continue
            vals = pd.to_numeric(sub[c], errors="coerce").dropna()
            if len(vals) == 0:
                print(f"    {c:<25}: (sin datos)")
            else:
                print(f"    {c:<25}: {vals.mean():.4f} ± {vals.std():.4f}")

    for clase in ["NSR", "AFIB"]:
        imprimir_resumen(df_ok[df_ok["clase_tesis"] == clase], f"Clase: {clase}")
    imprimir_resumen(df_ok, "GLOBAL")

    # ── Resumen JSON ──────────────────────────────────────────────────────
    def stats_dict(sub):
        out = {"n": int(len(sub))}
        for c in cols_metricas:
            if c not in sub.columns:
                continue
            vals = pd.to_numeric(sub[c], errors="coerce").dropna()
            if len(vals) > 0:
                out[f"{c}_mean"] = round(float(vals.mean()), 4)
                out[f"{c}_std"]  = round(float(vals.std()),  4)
        return out

    resumen = {
        "metodologia": {
            "dataset_referencia":    "PTB-XL v1.0.3 (PhysioNet)",
            "modo_evaluacion":       MODO_EVALUACION,
            "idioma_evaluacion":     IDIOMA_EVALUACION,
            "n_por_clase":           N_PER_CLASS,
            "clases":                ["NSR", "AFIB"],
            "semilla":               SEED,
            "modelo_llm":            ANTHROPIC_MODEL,
            "bertscore_model":       BERTSCORE_MODEL,
            "metrica_principal":     "concept_recall_vs_ref",
            "metricas_secundarias":  ["bertscore_F1", "meteor", "rouge1_f",
                                      "rougeL_f", "bleu"],
            "nota_metodologica": (
                "Las métricas léxicas (BLEU/ROUGE/METEOR) se reportan por "
                "completitud, pero penalizan parafraseo y diferencias de "
                "estilo entre la nota estructurada del agente y los reportes "
                "telegráficos de PTB-XL. La métrica principal es concept "
                "recall vs referencia, que mide si los conceptos clínicos del "
                "reporte aparecen en la nota generada. BERTScore complementa "
                "con similitud semántica multilingüe."
            ),
            "nota_pac": (
                "PAC excluido: PTB-XL no contiene etiqueta PAC pura. "
                "El módulo LATIDO se evalúa con 5 médicos (validación humana)."
            ),
        },
        "global":      stats_dict(df_ok),
        "por_clase":   {c: stats_dict(df_ok[df_ok["clase_tesis"] == c])
                        for c in ["NSR", "AFIB"]},
        "n_errores":   int(df_res["error"].notna().sum()),
    }

    ruta_json = OUTPUT_DIR / "resumen_metricas.json"
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)
    print(f"\n[FASE 5] Resumen JSON guardado en {ruta_json}")
    print("\n  Pipeline completado.")


if __name__ == "__main__":
    main()
