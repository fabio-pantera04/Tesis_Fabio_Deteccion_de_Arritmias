"""
Separa las notas persistidas en dos partes:
  - clinical_note      : diagnostico y morfologia
  - classifier_note    : nivel de certeza del clasificador y escalacion

Ademas renombra "confidence gap" a "nivel de certeza" en el texto,
para que sea comprensible para los medicos evaluadores.

Uso:
  cd E:\\Modelo_Tesis\\Modulo_I\\modulo_3_interfaz
  python scripts\\separar_notas.py
"""
import json
import re
from pathlib import Path

# Ajusta esta ruta si tu carpeta de notas esta en otro lugar
ROOT = Path(__file__).resolve().parent.parent
NOTES_DIR = ROOT / "data" / "notas_llm_a"

# Marcadores que indican el inicio de la seccion del clasificador
CLASSIFIER_MARKERS = [
    "La confianza media del clasificador",
    "La confianza global del clasificador",
    "El nivel de certeza del clasificador",
    "Confianza del clasificador",
]


def renombrar_confidence_gap(texto: str) -> str:
    """Renombra 'confidence gap' por 'nivel de certeza' en todas sus variantes."""
    reemplazos = [
        (r"\bconfidence[\s_-]*gap\b", "nivel de certeza"),
        (r"\bConfidence[\s_-]*gap\b", "Nivel de certeza"),
        (r"\bCONFIDENCE[\s_-]*GAP\b", "NIVEL DE CERTEZA"),
        (r"\bconfianza-gap\b", "nivel de certeza"),
        (r"\bgap de confianza\b", "nivel de certeza"),
    ]
    for pattern, replacement in reemplazos:
        texto = re.sub(pattern, replacement, texto)
    return texto


def dividir_nota(texto_completo: str) -> tuple[str, str]:
    """
    Divide el texto completo en (clinical_note, classifier_note).
    Busca el marcador de inicio de la seccion del clasificador.
    Si no encuentra ningun marcador, devuelve (todo, "").
    """
    texto = renombrar_confidence_gap(texto_completo)

    # Buscar el primer marcador que aparezca
    corte = None
    for marker in CLASSIFIER_MARKERS:
        idx = texto.find(marker)
        if idx != -1:
            if corte is None or idx < corte:
                corte = idx

    if corte is None:
        # No se encontro marcador - devolvemos todo como clinical_note
        print(f"    [!] No se encontro marcador de clasificador. Se deja todo como nota clinica.")
        return texto.strip(), ""

    clinical = texto[:corte].strip()
    classifier = texto[corte:].strip()
    return clinical, classifier


def procesar_todas_las_notas():
    if not NOTES_DIR.exists():
        print(f"ERROR: no existe la carpeta {NOTES_DIR}")
        return
    archivos = sorted(NOTES_DIR.glob("nota_sig_*.json"))
    print(f"Procesando {len(archivos)} notas en {NOTES_DIR}")

    for f in archivos:
        print(f"  - {f.name}")
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)

        texto = data.get("clinical_note", "")
        if not texto:
            print(f"    [!] Sin campo clinical_note, saltando.")
            continue

        clinical, classifier = dividir_nota(texto)

        # Actualizar el JSON con las dos notas separadas
        data["clinical_note"] = clinical
        data["classifier_note"] = classifier

        # Backup del original si aun no existe
        backup = f.parent / f"{f.stem}_bkp_v1.json"
        if not backup.exists():
            with open(backup, "w", encoding="utf-8") as fb:
                json.dump({"clinical_note_original": texto}, fb, ensure_ascii=False, indent=2)

        with open(f, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)

        print(f"    Nota clinica: {len(clinical)} chars")
        print(f"    Nota clasificador: {len(classifier)} chars")

    print("\nOK. Las notas fueron divididas y persistidas.")
    print("Backups originales guardados como nota_sig_XXX_bkp_v1.json en la misma carpeta.")


if __name__ == "__main__":
    procesar_todas_las_notas()
