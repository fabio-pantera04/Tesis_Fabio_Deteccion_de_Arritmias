"""
Script — Consolidacion de resultados del Modulo LLM-B (v2)
────────────────────────────────────────────────────────────────────
Version mejorada que ademas de las metricas globales muestra el
desglose POR CLASE, revelando la heterogeneidad de dificultad entre
NORMAL/PAC/NSR/AFIB (hallazgo clave del experimento).

Ejecucion:
    python analizar_resultados_llm_b.py

Salida en resultados_llm_b/:
    tabla_latex_llm_b.tex
    grafico_llm_b_global.png
    grafico_llm_b_por_clase.png
    ejemplos_cualitativos.md
"""

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt

RESULTADOS_DIR = Path(
    r"E:\Modelo_Tesis\Modulo_I\validacion_generativa\resultados_llm_b"
)

VALORES_K = [0, 10, 20]
CLASES = ["NORMAL", "PAC", "NSR", "AFIB"]

CREAM      = "#FBF5E5"
INK_DEEP   = "#3A2A24"
INK_SOFT   = "#5C453A"
ECG_RED    = "#B22222"
AZUL       = "#3B5B72"
VERDE      = "#4A7C4E"
AMARILLO   = "#C29A37"

COLOR_CLASE = {
    "NORMAL": VERDE,
    "PAC":    AMARILLO,
    "NSR":    AZUL,
    "AFIB":   ECG_RED,
}


def cargar_todas_las_metricas() -> dict:
    metricas_por_k = {}
    for k in VALORES_K:
        ruta = RESULTADOS_DIR / f"metricas_k{k}.json"
        if not ruta.exists():
            print(f"  ADVERTENCIA: falta {ruta.name}")
            continue
        metricas_por_k[k] = json.loads(ruta.read_text(encoding="utf-8"))
    return metricas_por_k


def imprimir_tablas_consola(metricas: dict) -> None:
    print("\n" + "=" * 76)
    print("RESULTADOS GLOBALES DEL EXPERIMENTO LLM-B")
    print("=" * 76)
    print(f"  {'K':<4} {'N':<5} {'BLEU':>8} {'ROUGE-1':>10} "
          f"{'ROUGE-2':>10} {'ROUGE-L':>10} {'SBERT':>10}")
    print("  " + "-" * 66)

    for k in VALORES_K:
        if k not in metricas:
            continue
        m = metricas[k]
        g = m["global"]
        print(f"  {k:<4} {m['n_test']:<5} "
              f"{g['bleu']:>8.3f} {g['rouge1_f']:>10.4f} "
              f"{g['rouge2_f']:>10.4f} {g['rougeL_f']:>10.4f} "
              f"{g['sbert_cos']:>10.4f}")

    print("\n" + "=" * 76)
    print("RESULTADOS POR CLASE (revelan heterogeneidad de dificultad)")
    print("=" * 76)
    for clase in CLASES:
        print(f"\n  Clase {clase}:")
        print(f"    {'K':<4} {'N':<5} {'BLEU':>8} {'ROUGE-1':>10} "
              f"{'ROUGE-2':>10} {'ROUGE-L':>10} {'SBERT':>10}")
        for k in VALORES_K:
            if k not in metricas or clase not in metricas[k]["por_clase"]:
                continue
            mc = metricas[k]["por_clase"][clase]
            print(f"    {k:<4} {mc['n']:<5} "
                  f"{mc['bleu']:>8.3f} {mc['rouge1_f']:>10.4f} "
                  f"{mc['rouge2_f']:>10.4f} {mc['rougeL_f']:>10.4f} "
                  f"{mc['sbert_cos']:>10.4f}")


def generar_tabla_latex(metricas: dict) -> str:
    lineas = [
        "% Tablas del Modulo LLM-B para Cap 4.6",
        "",
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Resultados globales del experimento de validacion "
        "generativa (Modulo LLM-B) sobre PTB-XL.}",
        "\\label{tab:llm-b-global}",
        "\\begin{tabular}{cccccc}",
        "\\hline",
        "\\textbf{K} & \\textbf{BLEU} & \\textbf{ROUGE-1} & "
        "\\textbf{ROUGE-2} & \\textbf{ROUGE-L} & \\textbf{SBERT} \\\\",
        "\\hline",
    ]
    for k in VALORES_K:
        if k not in metricas:
            continue
        g = metricas[k]["global"]
        lineas.append(
            f"{k} & {g['bleu']:.3f} & {g['rouge1_f']:.4f} & "
            f"{g['rouge2_f']:.4f} & {g['rougeL_f']:.4f} & "
            f"{g['sbert_cos']:.4f} \\\\"
        )
    lineas.extend([
        "\\hline",
        "\\end{tabular}",
        "\\end{table}",
        "",
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{BLEU por clase AAMI. NORMAL y NSR presentan "
        "los reportes SCP-ECG mas homogeneos y alcanzan valores "
        "sustancialmente mas altos que las clases con mayor "
        "variabilidad clinica (AFIB, PAC).}",
        "\\label{tab:llm-b-bleu-clase}",
        "\\begin{tabular}{lccc}",
        "\\hline",
        "\\textbf{Clase} & \\textbf{K=0} & \\textbf{K=10} & "
        "\\textbf{K=20} \\\\",
        "\\hline",
    ])
    for clase in CLASES:
        vals = []
        for k in VALORES_K:
            if k in metricas and clase in metricas[k]["por_clase"]:
                vals.append(f"{metricas[k]['por_clase'][clase]['bleu']:.2f}")
            else:
                vals.append("---")
        lineas.append(f"{clase} & {vals[0]} & {vals[1]} & {vals[2]} \\\\")
    lineas.extend([
        "\\hline",
        "\\end{tabular}",
        "\\end{table}",
        "",
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Similitud semantica SBERT por clase. Muestra que "
        "Gemini comprende el contenido clinico con alta fidelidad en "
        "todas las clases incluso desde zero-shot.}",
        "\\label{tab:llm-b-sbert-clase}",
        "\\begin{tabular}{lccc}",
        "\\hline",
        "\\textbf{Clase} & \\textbf{K=0} & \\textbf{K=10} & "
        "\\textbf{K=20} \\\\",
        "\\hline",
    ])
    for clase in CLASES:
        vals = []
        for k in VALORES_K:
            if k in metricas and clase in metricas[k]["por_clase"]:
                vals.append(f"{metricas[k]['por_clase'][clase]['sbert_cos']:.4f}")
            else:
                vals.append("---")
        lineas.append(f"{clase} & {vals[0]} & {vals[1]} & {vals[2]} \\\\")
    lineas.extend([
        "\\hline",
        "\\end{tabular}",
        "\\end{table}",
    ])
    return "\n".join(lineas)


def grafico_global(metricas: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=180, facecolor=CREAM)
    ks_disp = [k for k in VALORES_K if k in metricas]

    ax = axes[0]
    ax.set_facecolor(CREAM)
    bleu_vals = [metricas[k]["global"]["bleu"] for k in ks_disp]
    r1_vals = [metricas[k]["global"]["rouge1_f"] * 100 for k in ks_disp]
    r2_vals = [metricas[k]["global"]["rouge2_f"] * 100 for k in ks_disp]
    rL_vals = [metricas[k]["global"]["rougeL_f"] * 100 for k in ks_disp]

    ax.plot(ks_disp, bleu_vals, marker="o", markersize=10, linewidth=2.5,
            color=ECG_RED, label="BLEU (0-100)")
    ax.plot(ks_disp, r1_vals, marker="s", markersize=10, linewidth=2.5,
            color=AZUL, label="ROUGE-1 F1 (x100)")
    ax.plot(ks_disp, r2_vals, marker="^", markersize=10, linewidth=2.5,
            color=VERDE, label="ROUGE-2 F1 (x100)")
    ax.plot(ks_disp, rL_vals, marker="D", markersize=10, linewidth=2.5,
            color=AMARILLO, label="ROUGE-L F1 (x100)")

    ax.set_xlabel("Numero de ejemplos few-shot (K)", fontsize=12, color=INK_DEEP)
    ax.set_ylabel("Puntuacion", fontsize=12, color=INK_DEEP)
    ax.set_title("BLEU y ROUGE globales en funcion de K",
                 fontsize=13, fontweight="bold", color=INK_DEEP, pad=10)
    ax.set_xticks(ks_disp)
    ax.legend(loc="lower right", fontsize=9, facecolor=CREAM)
    ax.grid(True, alpha=0.25, color=INK_SOFT, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=INK_SOFT)

    ax = axes[1]
    ax.set_facecolor(CREAM)
    sbert_vals = [metricas[k]["global"]["sbert_cos"] for k in ks_disp]
    ax.plot(ks_disp, sbert_vals, marker="D", markersize=14, linewidth=2.5,
            color=AZUL, label="SBERT global")

    for clase in CLASES:
        vals = []
        for k in ks_disp:
            if clase in metricas[k]["por_clase"]:
                vals.append(metricas[k]["por_clase"][clase]["sbert_cos"])
            else:
                vals.append(None)
        ax.plot(ks_disp, vals, marker="o", markersize=7, linewidth=1.5,
                color=COLOR_CLASE[clase], alpha=0.7, linestyle="--",
                label=f"SBERT {clase}")

    ax.set_xlabel("Numero de ejemplos few-shot (K)", fontsize=12, color=INK_DEEP)
    ax.set_ylabel("Similitud SBERT (coseno)", fontsize=12, color=INK_DEEP)
    ax.set_title("SBERT global y por clase",
                 fontsize=13, fontweight="bold", color=INK_DEEP, pad=10)
    ax.set_xticks(ks_disp)
    ax.legend(loc="lower right", fontsize=9, facecolor=CREAM)
    ax.grid(True, alpha=0.25, color=INK_SOFT, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=INK_SOFT)
    ax.set_ylim([0.6, 1.0])

    fig.suptitle("Modulo LLM-B: efecto del numero de ejemplos few-shot",
                 fontsize=14, fontweight="bold", color=INK_DEEP, y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTADOS_DIR / "grafico_llm_b_global.png",
                dpi=200, facecolor=CREAM,
                bbox_inches="tight", pad_inches=0.3)
    plt.close()
    print(f"  Grafico global: grafico_llm_b_global.png")


def grafico_por_clase(metricas: dict) -> None:
    fig, ax = plt.subplots(figsize=(12, 6), dpi=180, facecolor=CREAM)
    ax.set_facecolor(CREAM)

    ks_disp = [k for k in VALORES_K if k in metricas]
    n_clases = len(CLASES)
    x_base = list(range(len(ks_disp)))
    ancho = 0.2

    for i, clase in enumerate(CLASES):
        vals = []
        for k in ks_disp:
            if clase in metricas[k]["por_clase"]:
                vals.append(metricas[k]["por_clase"][clase]["bleu"])
            else:
                vals.append(0)
        offset = (i - (n_clases - 1) / 2) * ancho
        pos = [x + offset for x in x_base]
        ax.bar(pos, vals, width=ancho,
               color=COLOR_CLASE[clase], alpha=0.85,
               edgecolor=INK_DEEP, linewidth=0.7,
               label=clase)
        for x, v in zip(pos, vals):
            ax.text(x, v + 1.5, f"{v:.1f}",
                    ha="center", fontsize=8.5, color=INK_DEEP)

    ax.set_xticks(x_base)
    ax.set_xticklabels([f"K = {k}" for k in ks_disp], fontsize=11)
    ax.set_xlabel("Configuracion de few-shot", fontsize=12, color=INK_DEEP)
    ax.set_ylabel("BLEU corpus", fontsize=12, color=INK_DEEP)
    ax.set_title("BLEU por clase AAMI y configuracion K",
                 fontsize=13, fontweight="bold", color=INK_DEEP, pad=10)
    ax.legend(loc="upper right", fontsize=10, facecolor=CREAM,
              title="Clase")
    ax.grid(True, alpha=0.25, color=INK_SOFT, linestyle=":", axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=INK_SOFT)

    plt.tight_layout()
    plt.savefig(RESULTADOS_DIR / "grafico_llm_b_por_clase.png",
                dpi=200, facecolor=CREAM,
                bbox_inches="tight", pad_inches=0.3)
    plt.close()
    print(f"  Grafico por clase: grafico_llm_b_por_clase.png")


def generar_ejemplos_cualitativos() -> None:
    lineas = ["# Ejemplos cualitativos del Modulo LLM-B\n"]
    lineas.append("Muestra de generaciones vs referencia PTB-XL, "
                   "3 casos por clase para cada K.\n")

    for k in VALORES_K:
        gen_path = RESULTADOS_DIR / f"generaciones_k{k}.jsonl"
        if not gen_path.exists():
            continue
        with open(gen_path, encoding="utf-8") as f:
            generaciones = [json.loads(line) for line in f]

        lineas.append(f"\n## K = {k}\n")
        for clase in CLASES:
            gens_clase = [g for g in generaciones
                          if g["clase"] == clase and g["prediccion"]]
            if not gens_clase:
                continue
            muestra = random.sample(gens_clase, min(3, len(gens_clase)))
            lineas.append(f"\n### {clase} (K={k})\n")
            for i, g in enumerate(muestra, start=1):
                lineas.append(f"**Ejemplo {i}** (ecg_id {g['ecg_id']}):\n")
                lineas.append(f"- **Referencia PTB-XL**: {g['referencia']}\n")
                lineas.append(f"- **Generada por LLM-B**: {g['prediccion']}\n")

    out_path = RESULTADOS_DIR / "ejemplos_cualitativos.md"
    out_path.write_text("\n".join(lineas), encoding="utf-8")
    print(f"  Ejemplos cualitativos: {out_path.name}")


def main() -> None:
    print("Cargando resultados de los 3 experimentos...")
    metricas = cargar_todas_las_metricas()

    if not metricas:
        print("No hay resultados.")
        return

    imprimir_tablas_consola(metricas)

    print("\nGenerando tabla LaTeX...")
    latex = generar_tabla_latex(metricas)
    out_tex = RESULTADOS_DIR / "tabla_latex_llm_b.tex"
    out_tex.write_text(latex, encoding="utf-8")
    print(f"  Tablas LaTeX: {out_tex.name}")

    print("\nGenerando graficos...")
    grafico_global(metricas)
    grafico_por_clase(metricas)

    print("\nExtrayendo ejemplos cualitativos...")
    random.seed(42)
    generar_ejemplos_cualitativos()

    print("\n" + "=" * 76)
    print(f"ANALISIS COMPLETO en {RESULTADOS_DIR}")
    print("=" * 76)


if __name__ == "__main__":
    main()
