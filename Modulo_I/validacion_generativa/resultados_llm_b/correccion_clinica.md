# Experimento 1 — Correccion clinica de notas LLM-B

**Objetivo**: verificar que Gemini interpreta correctamente la evidencia del clasificador, no solo reproduce el estilo SCP-ECG.

## Metricas evaluadas

| Metrica | Descripcion |
|---|---|
| **Class Mention** | La nota menciona la clase correcta que dice el clasificador |
| **Class Fidelity** | Ademas no menciona una clase contradictoria como diagnostico principal |
| **Morphological** | Describe morfologia apropiada a la clase (ondas P, QRS, R-R, etc.) |
| **Evidence** | Referencia explicitamente la evidencia del clasificador (shapelets, MSM, confidence) |
| **Composite** | Cumple Mention Y Fidelity |

## Resultados globales

| K | N | Class Mention | Class Fidelity | Morphological | Evidence | Composite |
|---|---|---|---|---|---|---|
| 0 | 389 | 96.14% | 95.63% | 57.07% | 2.31% | 95.63% |
| 10 | 389 | 100.00% | 98.46% | 51.41% | 0.00% | 98.46% |
| 20 | 389 | 100.00% | 97.94% | 51.41% | 0.00% | 97.94% |

## Resultados por clase


### NORMAL

| K | N | Class Mention | Class Fidelity | Morphological | Evidence | Composite |
|---|---|---|---|---|---|---|
| 0 | 100 | 87.00% | 86.00% | 12.00% | 2.00% | 86.00% |
| 10 | 100 | 100.00% | 100.00% | 0.00% | 0.00% | 100.00% |
| 20 | 100 | 100.00% | 100.00% | 0.00% | 0.00% | 100.00% |

### PAC

| K | N | Class Mention | Class Fidelity | Morphological | Evidence | Composite |
|---|---|---|---|---|---|---|
| 0 | 100 | 99.00% | 99.00% | 98.00% | 2.00% | 99.00% |
| 10 | 100 | 100.00% | 100.00% | 100.00% | 0.00% | 100.00% |
| 20 | 100 | 100.00% | 100.00% | 100.00% | 0.00% | 100.00% |

### NSR

| K | N | Class Mention | Class Fidelity | Morphological | Evidence | Composite |
|---|---|---|---|---|---|---|
| 0 | 89 | 98.88% | 97.75% | 15.73% | 2.25% | 97.75% |
| 10 | 89 | 100.00% | 100.00% | 0.00% | 0.00% | 100.00% |
| 20 | 89 | 100.00% | 100.00% | 0.00% | 0.00% | 100.00% |

### AFIB

| K | N | Class Mention | Class Fidelity | Morphological | Evidence | Composite |
|---|---|---|---|---|---|---|
| 0 | 100 | 100.00% | 100.00% | 98.00% | 3.00% | 100.00% |
| 10 | 100 | 100.00% | 94.00% | 100.00% | 0.00% | 94.00% |
| 20 | 100 | 100.00% | 92.00% | 100.00% | 0.00% | 92.00% |