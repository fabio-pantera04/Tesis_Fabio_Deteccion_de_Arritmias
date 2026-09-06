"""
Script — Extraccion de conjuntos few-shot y test desde PTB-XL
────────────────────────────────────────────────────────────────────
VERSION v4 — Anade filtros de calidad semantica.

Problemas detectados en v3:
    1. Reportes en aleman e ingles mezclados (PTB-XL es de PTB Alemania)
    2. Etiquetas AAMI inconsistentes con el texto del reporte
       (un registro puede tener 5 codigos SCP simultaneos)
    3. Edad = 300 en algunos casos (convencion PTB-XL para no calibrada)

Solucion en v4:
    1. Filtro de idioma: solo reportes en INGLES (heuristica de palabras
       clave alemanas y caracteres especiales)
    2. Filtro de consistencia semantica: el reporte debe mencionar
       palabras clave de la clase objetivo
    3. Filtro de dominancia: la clase objetivo debe ser el hallazgo
       principal (menos de N codigos rimicos concurrentes)
    4. Filtro de edad: descartar edad = 300 (marcador de no calibrada)

Ejecucion:
    python extraer_fewshot_y_test.py
"""

import ast
import json
import re
from pathlib import Path
from typing import Optional, Set

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────
# CONFIGURACION
# ─────────────────────────────────────────────────────────────────────
PTBXL_ROOT = Path(
    r"E:\Modelo_Tesis\one_shot_ptbxl\datos_ptbxl"
)
OUTPUT_DIR = Path(r"E:\Modelo_Tesis\Modulo_I\validacion_generativa\conjuntos_ptbxl")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

N_FEWSHOT_POR_CLASE = 20
N_TEST_POR_CLASE    = 100

MAPA_CLASES = {
    "NORMAL": "NORM",
    "PAC":    "PAC",
    "NSR":    "SR",
    "AFIB":   "AFIB",
}

# Codigos SCP excluyentes de comorbilidad (para criterio estricto)
CODIGOS_EXCLUIR_ESTRICTO = {
    "IMI", "AMI", "LMI", "PMI", "LVH", "RVH",
    "LAFB", "LPFB", "CRBBB", "CLBBB", "IRBBB", "ILBBB",
    "WPW", "PACE", "1AVB", "2AVB", "3AVB",
}

# ─────────────────────────────────────────────────────────────────────
# FILTROS DE CALIDAD SEMANTICA
# ─────────────────────────────────────────────────────────────────────

# 1) Palabras y patrones que indican reporte en ALEMAN (deben excluirse)
PATRONES_ALEMAN = [
    r"\bsinusrhythmus\b",       # sinus rhythm
    r"\bvorhof",                # atrial (Vorhofflimmern = AFIB, Vorhofflattern = flutter)
    r"\bkammer",                # ventricular
    r"\btachykard",             # tachycardic
    r"\bbradykard",             # bradycardic
    r"\blagetyp\b",             # axis type
    r"\blinkstyp\b",            # left type
    r"\brechtstyp\b",           # right type
    r"\bschenkelblock\b",       # bundle branch block
    r"\bnormales\s+ekg\b",      # normal ECG (alemán)
    r"\bpathologisches\s+ekg\b",# pathological ECG
    r"\bextrasystole",          # extrasystoles (variante alemana)
    r"\bsupraventrikul",        # supraventricular (alemán)
    r"\bunauffällig\b",         # inconspicuous
    r"\bniederspannung\b",      # low voltage
    r"[äöüßÄÖÜ]",               # caracteres especiales alemanes
]
REGEX_ALEMAN = re.compile("|".join(PATRONES_ALEMAN), re.IGNORECASE)

# 2) Palabras clave que deben aparecer en el reporte para cada clase
#    (validacion de consistencia semantica)
PATRONES_INGLES_POR_CLASE = {
    "NORMAL": [
        r"\bnormal\s+ecg\b",
        r"\bnormal\s+ekg\b",
        r"\botherwise\s+normal\b",
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
    ],
    "AFIB": [
        r"\batrial\s+fibrillation\b",
        r"\bafib\b",
        r"\bAF\b",
    ],
}
REGEX_POR_CLASE = {
    clase: re.compile("|".join(patrones), re.IGNORECASE)
    for clase, patrones in PATRONES_INGLES_POR_CLASE.items()
}

# 3) Codigos ritmicos concurrentes que hacen la clasificacion ambigua
#    Si un registro tiene mas de 1 de estos codigos, la etiqueta es ambigua
CODIGOS_RITMICOS = {"SR", "AFIB", "AFLT", "SARRH", "SBRAD", "STACH", "SVARR", "PSVT"}


def es_reporte_en_ingles(reporte: str) -> bool:
    """True si el reporte NO contiene patrones alemanes."""
    if not reporte:
        return False
    return not bool(REGEX_ALEMAN.search(reporte))


def reporte_menciona_clase(reporte: str, clase: str) -> bool:
    """True si el reporte contiene al menos una keyword de la clase."""
    if not reporte or clase not in REGEX_POR_CLASE:
        return False
    return bool(REGEX_POR_CLASE[clase].search(reporte))


def clase_es_dominante(scp_codes: dict, codigo_objetivo: str) -> bool:
    """
    True si el codigo objetivo es el rimico dominante.
    Regla: no debe haber mas codigos ritmicos concurrentes distintos
    del objetivo (excepto SR que es fisiologico basal).
    """
    codigos_ritmicos_presentes = set(scp_codes.keys()) & CODIGOS_RITMICOS
    otros = codigos_ritmicos_presentes - {codigo_objetivo}
    # Permitimos SR concurrente para clases de latido (NORM, PAC) porque
    # son a nivel de latido y coexisten fisiologicamente con SR de fondo
    if codigo_objetivo in {"NORM", "PAC"}:
        otros = otros - {"SR"}
    return len(otros) == 0


def edad_valida(edad) -> bool:
    """True si la edad es un valor humano razonable (0-120)."""
    if pd.isna(edad):
        return False
    return 0 < float(edad) < 120


# ─────────────────────────────────────────────────────────────────────
def cargar_metadatos() -> pd.DataFrame:
    csv_path = PTBXL_ROOT / "ptbxl_database.csv"
    print(f"  Leyendo {csv_path.name} ...")
    df = pd.read_csv(csv_path, index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
    return df


def diagnostico_calidad(df: pd.DataFrame) -> None:
    """Cuenta cuantos registros pasan cada nivel de filtro por clase."""
    print("\n" + "=" * 70)
    print("DIAGNOSTICO DE CALIDAD POR CLASE (candidatos que pasan filtros)")
    print("=" * 70)
    print(f"  {'Clase':<10} {'Presente':<12} {'+Ingles':<12} "
          f"{'+Menciona':<12} {'+Dominante':<12} {'+Edad OK':<10}")
    print("  " + "-" * 68)

    for nombre_clase, codigo in MAPA_CLASES.items():
        n_presente = 0
        n_ingles = 0
        n_menciona = 0
        n_dominante = 0
        n_edad_ok = 0

        for ecg_id, row in df.iterrows():
            scp = row["scp_codes"]
            reporte = str(row["report"]) if pd.notna(row["report"]) else ""
            if codigo not in scp:
                continue
            n_presente += 1

            if not es_reporte_en_ingles(reporte):
                continue
            n_ingles += 1

            if not reporte_menciona_clase(reporte, nombre_clase):
                continue
            n_menciona += 1

            if not clase_es_dominante(scp, codigo):
                continue
            n_dominante += 1

            if not edad_valida(row.get("age")):
                continue
            n_edad_ok += 1

        print(f"  {nombre_clase:<10} {n_presente:<12,} {n_ingles:<12,} "
              f"{n_menciona:<12,} {n_dominante:<12,} {n_edad_ok:<10,}")


def registro_a_dict(ecg_id: int, row: pd.Series,
                    codigo_scp: str) -> dict:
    scp_codes = row["scp_codes"]
    return {
        "ecg_id_ptbxl":         int(ecg_id),
        "patient_id":           int(row["patient_id"]) if pd.notna(row["patient_id"]) else None,
        "edad":                 float(row["age"]) if pd.notna(row["age"]) else None,
        "sexo":                 int(row["sex"])   if pd.notna(row["sex"])   else None,
        "clase_aami":           None,
        "codigo_scp_principal": codigo_scp,
        "likelihood_principal": float(scp_codes.get(codigo_scp, 0)),
        "todos_scp_codes":      {k: float(v) for k, v in scp_codes.items()},
        "report_ptbxl":         str(row["report"]).strip(),
        "validated_by_human":   bool(row.get("validated_by_human", False)),
        "filename_lr":          str(row["filename_lr"]),
        "filename_hr":          str(row["filename_hr"]),
        "fuente":               "PTB-XL v1.0.3 (Wagner et al. 2020)",
    }


# ─────────────────────────────────────────────────────────────────────
# NUCLEO
# ─────────────────────────────────────────────────────────────────────
def filtrar_candidatos(
    df: pd.DataFrame,
    codigo_scp: str,
    nombre_clase: str,
    aplicar_filtro_comorbilidades: bool,
    requerir_validacion_humana: bool,
    excluir_ecg_ids: Optional[Set[int]] = None,
) -> pd.DataFrame:
    """Filtro completo con todas las capas de calidad semantica."""
    dfw = df.copy()

    dfw["_presente"] = dfw["scp_codes"].apply(
        lambda d: codigo_scp in d
    )
    dfw["_likelihood"] = dfw["scp_codes"].apply(
        lambda d: d.get(codigo_scp, -1.0)
    )
    dfw["_report_str"] = dfw["report"].apply(
        lambda x: str(x) if pd.notna(x) else ""
    )
    dfw["_ingles"] = dfw["_report_str"].apply(es_reporte_en_ingles)
    dfw["_menciona"] = dfw["_report_str"].apply(
        lambda r: reporte_menciona_clase(r, nombre_clase)
    )
    dfw["_dominante"] = dfw["scp_codes"].apply(
        lambda d: clase_es_dominante(d, codigo_scp)
    )
    dfw["_edad_ok"] = dfw["age"].apply(edad_valida)
    dfw["_report_no_vacio"] = dfw["_report_str"].str.strip() != ""

    if aplicar_filtro_comorbilidades:
        dfw["_sin_comorbilidad"] = dfw["scp_codes"].apply(
            lambda d: not bool(set(d.keys()) & CODIGOS_EXCLUIR_ESTRICTO)
        )
    else:
        dfw["_sin_comorbilidad"] = True

    mask = (
        dfw["_presente"]
        & dfw["_report_no_vacio"]
        & dfw["_ingles"]
        & dfw["_menciona"]
        & dfw["_dominante"]
        & dfw["_edad_ok"]
        & dfw["_sin_comorbilidad"]
    )
    if requerir_validacion_humana:
        mask = mask & (dfw["validated_by_human"] == True)
    if excluir_ecg_ids:
        mask = mask & (~dfw.index.isin(excluir_ecg_ids))

    return dfw.loc[mask].sort_values("_likelihood", ascending=False)


def extraer_con_fallback(
    df: pd.DataFrame,
    codigo_scp: str,
    nombre_clase: str,
    n_objetivo: int,
    cascada: list,
    excluir_ecg_ids: Optional[Set[int]] = None,
) -> list:
    candidatos = pd.DataFrame()
    criterio_usado = None

    for i, (con_comorb, req_valid) in enumerate(cascada, start=1):
        candidatos = filtrar_candidatos(
            df=df, codigo_scp=codigo_scp, nombre_clase=nombre_clase,
            aplicar_filtro_comorbilidades=con_comorb,
            requerir_validacion_humana=req_valid,
            excluir_ecg_ids=excluir_ecg_ids,
        )
        criterio_usado = (i, con_comorb, req_valid)
        if len(candidatos) >= n_objetivo:
            break

    if len(candidatos) > n_objetivo:
        top_half = candidatos.head(n_objetivo // 2)
        resto = candidatos.iloc[n_objetivo // 2:]
        sample_half = resto.sample(
            n_objetivo - n_objetivo // 2, random_state=42
        )
        seleccion = pd.concat([top_half, sample_half])
    else:
        seleccion = candidatos.head(n_objetivo)

    nivel, sin_comorb, validado = criterio_usado
    etiq_c = "SIN comorbilidades" if sin_comorb else "comorbilidades OK"
    etiq_v = "validado humano" if validado else "cualquier fuente"
    print(f"    · Criterio nivel {nivel}: {etiq_c}, {etiq_v}")
    print(f"    · Candidatos disponibles: {len(candidatos):,}, "
          f"seleccionados: {len(seleccion)}")

    resultado = []
    for ecg_id, row in seleccion.iterrows():
        d = registro_a_dict(ecg_id, row, codigo_scp)
        d["clase_aami"] = nombre_clase
        resultado.append(d)

    return resultado


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
def main() -> None:
    print("Cargando metadatos PTB-XL...")
    df = cargar_metadatos()
    print(f"  Total registros: {len(df):,}")

    diagnostico_calidad(df)

    CASCADA_FEWSHOT = [
        (True,  True),    # Nivel 1: sin comorbilidades + validado
        (True,  False),   # Nivel 2: sin comorbilidades
        (False, True),    # Nivel 3: con comorbilidades OK, validado
        (False, False),   # Nivel 4: sin restricciones adicionales
    ]
    CASCADA_TEST = [
        (False, True),
        (False, False),
    ]

    print("\n" + "=" * 70)
    print("FASE 1 — EXTRACCION DE CONJUNTO FEW-SHOT")
    print(f"  Objetivo: {N_FEWSHOT_POR_CLASE} registros por clase")
    print("=" * 70)

    ids_fewshot_todos = set()
    for nombre_clase, codigo_scp in MAPA_CLASES.items():
        print(f"\n  Procesando {nombre_clase} (codigo SCP: {codigo_scp})...")
        registros = extraer_con_fallback(
            df=df, codigo_scp=codigo_scp, nombre_clase=nombre_clase,
            n_objetivo=N_FEWSHOT_POR_CLASE,
            cascada=CASCADA_FEWSHOT, excluir_ecg_ids=None,
        )
        ids_fewshot_todos.update(r["ecg_id_ptbxl"] for r in registros)
        out_path = OUTPUT_DIR / f"few_shot_{nombre_clase}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(registros, f, ensure_ascii=False, indent=2)
        print(f"    Guardado: {out_path.name}")

    print(f"\n  Total ecg_ids en few-shot: {len(ids_fewshot_todos)}")

    print("\n" + "=" * 70)
    print("FASE 2 — EXTRACCION DE CONJUNTO TEST")
    print(f"  Objetivo: {N_TEST_POR_CLASE} registros por clase")
    print("=" * 70)

    for nombre_clase, codigo_scp in MAPA_CLASES.items():
        print(f"\n  Procesando {nombre_clase} (codigo SCP: {codigo_scp})...")
        registros = extraer_con_fallback(
            df=df, codigo_scp=codigo_scp, nombre_clase=nombre_clase,
            n_objetivo=N_TEST_POR_CLASE,
            cascada=CASCADA_TEST, excluir_ecg_ids=ids_fewshot_todos,
        )
        out_path = OUTPUT_DIR / f"test_{nombre_clase}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(registros, f, ensure_ascii=False, indent=2)
        print(f"    Guardado: {out_path.name}")

    print("\n" + "=" * 70)
    print("VERIFICACION FINAL")
    print("=" * 70)
    all_test_ids = set()
    for clase in MAPA_CLASES:
        with open(OUTPUT_DIR / f"few_shot_{clase}.json", encoding="utf-8") as f:
            n_fs = len(json.load(f))
        with open(OUTPUT_DIR / f"test_{clase}.json", encoding="utf-8") as f:
            registros = json.load(f)
            n_ts = len(registros)
            all_test_ids.update(r["ecg_id_ptbxl"] for r in registros)
        print(f"    {clase:8s}: few-shot={n_fs:3d}  test={n_ts:3d}")

    overlap = ids_fewshot_todos & all_test_ids
    if not overlap:
        print("\n  CONFIRMADO: no hay ecg_ids duplicados entre few-shot y test")
    else:
        print(f"\n  ADVERTENCIA: {len(overlap)} ecg_ids duplicados")

    print(f"\n  Directorio de salida: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()