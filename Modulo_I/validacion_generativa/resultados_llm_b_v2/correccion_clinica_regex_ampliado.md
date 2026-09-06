# Experimento 1 sobre LLM-B v2 — Regex ampliadas

Version del Experimento 1 con regex de Class Mention ampliadas para capturar el vocabulario elaborado inducido por el prompt v3 (frases como 'NORMAL classification', 'normal morphology', 'physiological range'). Los patrones originales de PTB-XL telegrafico se conservan.

## Resultados globales

| K | N | Class Mention | Class Fidelity | Morphological | Evidence | Composite |
|---|---|---|---|---|---|---|
| 0 | 389 | 96.14% | 93.57% | 100.00% | 100.00% | 93.57% |
| 8 | 389 | 92.54% | 82.01% | 100.00% | 100.00% | 82.01% |

## Resultados por clase


### NORMAL

| K | N | Class Mention | Class Fidelity | Morphological | Evidence | Composite |
|---|---|---|---|---|---|---|
| 0 | 100 | 100.00% | 99.00% | 100.00% | 100.00% | 99.00% |
| 8 | 100 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |

### PAC

| K | N | Class Mention | Class Fidelity | Morphological | Evidence | Composite |
|---|---|---|---|---|---|---|
| 0 | 100 | 85.00% | 85.00% | 100.00% | 100.00% | 85.00% |
| 8 | 100 | 77.00% | 77.00% | 100.00% | 100.00% | 77.00% |

### NSR

| K | N | Class Mention | Class Fidelity | Morphological | Evidence | Composite |
|---|---|---|---|---|---|---|
| 0 | 89 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| 8 | 89 | 93.26% | 93.26% | 100.00% | 100.00% | 93.26% |

### AFIB

| K | N | Class Mention | Class Fidelity | Morphological | Evidence | Composite |
|---|---|---|---|---|---|---|
| 0 | 100 | 100.00% | 91.00% | 100.00% | 100.00% | 91.00% |
| 8 | 100 | 100.00% | 59.00% | 100.00% | 100.00% | 59.00% |