"""
Separa las notas persistidas en dos partes: nota clinica + nota del clasificador.
Version 2 con muchos mas marcadores + modo interactivo si no encuentra ninguno.

Uso:
  cd E:\\Modelo_Tesis\\Modulo_I\\modulo_3_interfaz
  python scripts\\separar_notas_v2.py

Restaura desde los backups (nota_sig_XXX_bkp_v1.json) antes de aplicar.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTES_DIR = ROOT / "data" / "notas_llm_a"

# Lista extendida de marcadores (todos en minusculas para comparacion)
CLASSIFIER_MARKERS = [
    "la confianza media del clasificador",
    "la confianza global del clasificador",
    "la confianza del clasificador",
    "el nivel de certeza del clasificador",
    "el nivel de confianza del clasificador",
    "el nivel de confianza global",
    "en cuanto a la confianza",
    "en terminos de confianza",
    "en términos de confianza",
    "con respecto a la confianza",
    "respecto a la confianza",
    "sobre la confianza del clasificador",
    "el clasificador presenta una confianza",
    "el clasificador muestra una confianza",
    "el clasificador reporta",
    "el sistema reporta una confianza",
    "es importante destacar que el clasificador",
    "cabe destacar que el clasificador",
    "cabe senalar que",
    "cabe señalar que",
    "se debe destacar que el clasificador",
    "se destaca que",
    "el sistema ha determinado",
    "dado que se ha determinado",
    "por lo anterior",
    "por lo tanto se recomienda",
    "se recomienda encarecidamente",
    "se sugiere la revision",
    "se sugiere la revisión",
    "considerando la confianza",
    "la escalacion global",
    "la escalación global",
]


def renombrar_confidence_gap(texto: str) -> str:
    """Renombra 'confidence gap' por 'nivel de certeza' en todas sus variantes."""
    reemplazos = [
        (r"\bconfidence[\s_-]*gap\b", "nivel de certeza"),
        (r"\bConfidence[\s_-]*gap\b", "Nivel de certeza"),
        (r"\bCONFIDENCE[\s_-]*GAP\b", "NIVEL DE CERTEZA"),
        (r"\bconfianza[- ]gap\b", "nivel de certeza"),
        (r"\bgap de confianza\b", "nivel de certeza"),
    ]
    for pattern, replacement in reemplazos:
        texto = re.sub(pattern, replacement, texto)
    return texto


def encontrar_corte(texto: str) -> int | None:
    """Devuelve el indice del primer marcador que aparezca (o None)."""
    texto_lower = texto.lower()
    mejor = None
    for marker in CLASSIFIER_MARKERS:
        idx = texto_lower.find(marker)
        if idx != -1 and (mejor is None or idx < mejor):
            mejor = idx
    return mejor


def dividir_nota_auto(texto_completo: str) -> tuple[str, str, bool]:
    """
    Divide la nota en (clinical_note, classifier_note, exito).
    Si no encuentra marcador, devuelve (texto_completo, "", False).
    """
    texto = renombrar_confidence_gap(texto_completo)
    corte = encontrar_corte(texto)
    if corte is None:
        return texto.strip(), "", False
    return texto[:corte].strip(), texto[corte:].strip(), True


def dividir_por_ultimas_oraciones(texto: str, n_oraciones: int = 2) -> tuple[str, str]:
    """
    Fallback: divide asumiendo que las ultimas N oraciones son la nota del clasificador.
    Util cuando ningun marcador funciona.
    """
    # Split por ". " para separar oraciones
    partes = re.split(r'(?<=[.!?])\s+', texto.strip())
    if len(partes) <= n_oraciones:
        return texto.strip(), ""
    clinical = " ".join(partes[:-n_oraciones]).strip()
    classifier = " ".join(partes[-n_oraciones:]).strip()
    return clinical, classifier


def procesar_todas_las_notas():
    if not NOTES_DIR.exists():
        print(f"ERROR: no existe la carpeta {NOTES_DIR}")
        return

    archivos = sorted(NOTES_DIR.glob("nota_sig_*.json"))
    # Excluir los backups
    archivos = [f for f in archivos if "_bkp_" not in f.name]

    print(f"Procesando {len(archivos)} notas en {NOTES_DIR}\n")
    problemas = []

    for f in archivos:
        print(f"  - {f.name}")

        # Intentar cargar el backup primero para trabajar con el texto ORIGINAL
        backup = f.parent / f"{f.stem}_bkp_v1.json"
        if backup.exists():
            with open(backup, encoding="utf-8") as fb:
                bkp_data = json.load(fb)
            texto_original = bkp_data.get("clinical_note_original", "")
        else:
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
            texto_original = data.get("clinical_note", "")
            # Guardar backup
            with open(backup, "w", encoding="utf-8") as fb:
                json.dump({"clinical_note_original": texto_original}, fb, ensure_ascii=False, indent=2)

        if not texto_original:
            print(f"    [!] Sin texto original, saltando.")
            continue

        clinical, classifier, exito = dividir_nota_auto(texto_original)

        if not exito:
            # Fallback: usar las ultimas 2 oraciones
            print(f"    [!] Sin marcador. Usando fallback: ultimas 2 oraciones.")
            clinical, classifier = dividir_por_ultimas_oraciones(
                renombrar_confidence_gap(texto_original), n_oraciones=2
            )
            problemas.append(f.name)

        # Actualizar el JSON
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        data["clinical_note"] = clinical
        data["classifier_note"] = classifier

        with open(f, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)

        print(f"    Nota clinica     : {len(clinical)} chars")
        print(f"    Nota clasificador: {len(classifier)} chars")

    if problemas:
        print(f"\n[!] Notas que usaron fallback (ultimas 2 oraciones): {problemas}")
        print("    Verifica visualmente que la separacion sea correcta.")
        print("    Si no, edita manualmente el JSON o mandame el texto original.")

    print("\nOK. Notas procesadas.")


if __name__ == "__main__":
    procesar_todas_las_notas()
