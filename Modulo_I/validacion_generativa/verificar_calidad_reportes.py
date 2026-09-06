"""
Script rapido — Inspeccion de calidad de los reportes extraidos
────────────────────────────────────────────────────────────────
Muestra 3 ejemplos de reporte por cada clase, tanto del few-shot
como del test, para verificar:
    - Que los reportes esten en ingles (esperado: si, PTB-XL v1.0.3)
    - Que la longitud sea razonable
    - Que la clase mencionada en el reporte coincida con la etiqueta
    - Que no haya reportes truncados o corruptos

Ejecucion:
    python verificar_calidad_reportes.py
"""

import json
from pathlib import Path

CONJUNTOS_DIR = Path(r"E:\Modelo_Tesis\Modulo_I\validacion_generativa\conjuntos_ptbxl")
CLASES = ["NORMAL", "PAC", "NSR", "AFIB"]


def imprimir_ejemplos(prefijo: str, n_ejemplos: int = 3) -> None:
    print("\n" + "=" * 70)
    print(f"CONJUNTO: {prefijo.upper()}")
    print("=" * 70)

    for clase in CLASES:
        archivo = CONJUNTOS_DIR / f"{prefijo}_{clase}.json"
        with open(archivo, encoding="utf-8") as f:
            registros = json.load(f)

        # Estadisticas de longitud del reporte
        longitudes = [len(r["report_ptbxl"]) for r in registros]
        avg_len = sum(longitudes) / len(longitudes) if longitudes else 0
        min_len = min(longitudes) if longitudes else 0
        max_len = max(longitudes) if longitudes else 0

        print(f"\n  ─── {clase} ({len(registros)} registros) ───")
        print(f"    Longitud reporte: min={min_len}, media={avg_len:.0f}, "
              f"max={max_len} caracteres")

        print(f"\n    Primeros {n_ejemplos} ejemplos:")
        for i, r in enumerate(registros[:n_ejemplos], start=1):
            reporte = r["report_ptbxl"]
            if len(reporte) > 200:
                reporte = reporte[:200] + "..."
            print(f"    [{i}] ecg_id {r['ecg_id_ptbxl']:5d}  "
                  f"edad {r['edad']}  sexo {r['sexo']}  "
                  f"lik {r['likelihood_principal']:.0f}")
            print(f"        Codigos SCP: {list(r['todos_scp_codes'].keys())}")
            print(f"        Reporte: \"{reporte}\"")
            print()


if __name__ == "__main__":
    imprimir_ejemplos("few_shot", n_ejemplos=3)
    imprimir_ejemplos("test", n_ejemplos=2)

    print("\n" + "=" * 70)
    print("VERIFICACION LISTA")
    print("=" * 70)
    print("""
  Que revisar en los ejemplos de arriba:
    1. Los reportes estan en ingles (esperado)
    2. La clase AAMI mencionada en el reporte coincide con la etiqueta
       - NORMAL: debe decir "normal ECG" o similar
       - PAC:    debe mencionar "atrial premature" o "APB" o similar
       - NSR:    debe decir "sinus rhythm"
       - AFIB:   debe decir "atrial fibrillation" o "AF"
    3. Los reportes no estan vacios ni truncados de forma anomala
    4. La longitud media esta entre 30 y 300 caracteres aproximadamente
""")
