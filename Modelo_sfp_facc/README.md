# Réplica SFP-FACC para comparación con sistema dual LATIDO/RITMO

Réplica del método propuesto por Qin et al. (2024) "ECG arrhythmia classification based on the fast ant colony clustering algorithm with improved spatiotemporal feature perception ability", Heliyon 10, e37111.

## Adaptación al setup de tesis Fabio

El paper original clasifica 5 clases (N, L, R, A, V) en un solo modelo unificado. Para hacer comparación justa con la arquitectura dual de la tesis, replicamos SFP-FACC **dos veces independientemente**:

- **SFP-FACC-LATIDO**: 2 clusters (N, PAC) sobre señales de 250 muestras
- **SFP-FACC-RITMO**: 2 clusters (NSR, AFIB) sobre señales de 1000 muestras

Esto evita repetir el bug del modelo unificado descartado en iteraciones previas de la tesis.

## Estructura del proyecto

```
sfp_facc/
├── README.md                # este archivo
├── preprocessing.py         # carga de CSVs, wavelet denoising, detección R/QRS/P/T
├── features.py              # extracción de las 37 features
├── lstm_centers.py          # ajuste de centros con LSTM (Sec. 2.5.1)
├── aco.py                   # ACO + radix sort (Sec. 2.5.3-2.5.4)
├── main.py                  # orquestador con CLI y soporte YAML
├── config_latido.yaml       # configuración para LATIDO
├── config_ritmo.yaml        # configuración para RITMO
└── test_pipeline_no_lstm.py # smoke test del pipeline sin LSTM
```

## Instalación de dependencias

```bash
pip install numpy pandas scipy scikit-learn pywavelets torch numba pyyaml
```

`numba` es opcional pero **muy recomendado**: acelera DTW ~50× (crítico para RITMO).

## Uso

### Modo recomendado: con archivo YAML

Edita `config_latido.yaml` o `config_ritmo.yaml` con tus rutas reales y ejecuta:

```bash
python main.py --config config_latido.yaml
python main.py --config config_ritmo.yaml
```

### Modo CLI alternativo

Para LATIDO desde Icentia:

```bash
python main.py --task latido \
    --class N:D:/Modelo_Tesis/Icentia11k/NORMAL:ICI_NORMAL \
    --class PAC:D:/Modelo_Tesis/Icentia11k/PAC:ICI_PAC
```

Para RITMO con AFIB combinando dos fuentes:

```bash
python main.py --task ritmo \
    --class NSR:D:/Modelo_Tesis/Icentia11k/NSR:ICI_NSR \
    --class AFIB:D:/Modelo_Tesis/Icentia11k/AFIB:ICI_AFIB \
    --class AFIB:D:/Modelo_Tesis/CinC2017/AFIB:CIC_AFIB
```

Formato `--class`: `NOMBRE_CLASE:DIRECTORIO:PREFIJO_ARCHIVO`. Múltiples `--class` con el mismo nombre se concatenan.

### Estructura de datos esperada

```
D:/Modelo_Tesis/
├── Icentia11k/
│   ├── NORMAL/   ICI_NORMAL_p*_s*_beat*.csv (250 muestras)
│   ├── PAC/      ICI_PAC_p*_s*_beat*.csv    (250 muestras)
│   ├── NSR/      ICI_NSR_p*_s*_seg*.csv     (1000 muestras)
│   └── AFIB/     ICI_AFIB_p*_s*_seg*.csv    (1000 muestras)
└── CinC2017/
    └── AFIB/     CIC_AFIB_A*.csv            (1000 muestras)
```

Cada CSV: una columna, normalizada, sin header.

## Decisiones de implementación

| Aspecto | Decisión | Justificación |
|---|---|---|
| Wavelet madre | db6, 6 niveles | Estándar ECG; paper no lo especifica |
| Umbral | Soft + universal (Donoho con MAD) | Robusto |
| Detección R | `scipy.signal.find_peaks` | Señales ya normalizadas |
| **R en latidos** | **Estrategia `prominent` (mayor amplitud)** | **PAC suele tener 2 R; el principal es el más alto** |
| Features médicas RITMO | Promedio sobre todos los latidos detectados | Confirmado |
| LSTM target | Predicción autoregresiva del siguiente sample | Lectura natural del paper |
| Split | 70/30 estratificado, max 1000 por clase | Más robusto que los 466 del paper |
| ρ dinámico | Eq. 22 del paper | Réplica fiel |
| Empate Euclidean vs DTW | Penalización ad-hoc del paper | Réplica fiel pese a ser heurística |

## Limitaciones de la réplica

1. P-wave: detectada por ventanas; imputación con mediana si falla.
2. Slopes/angles: convención lineal de 5 puntos.
3. LSTM: el paper usa solo 10 señales por clase para fitting (controlado por `fit_n_per_class`).

## Tiempos estimados (Ryzen 5 5600G, 16GB)

- LATIDO (~600 test × 250): ~10-15 min total
- RITMO (~600 test × 1000): ~25-30 min total

Sin numba el RITMO puede tardar horas. **Instala numba**.
