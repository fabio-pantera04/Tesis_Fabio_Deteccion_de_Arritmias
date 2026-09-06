# Réplica XGBoost-Andotra para comparación con sistema LATIDO/RITMO

Réplica fiel del método propuesto por **Andotra & Sunkaria (2024)** "Early Detection of Arrhythmia Using XGBoost Model", 15th ICCCNT, IEEE, junio 2024.

## Adaptación al setup de la tesis Fabio

**Diferencias clave con el paper original:**

1. **Single-lead vs dual-lead**: el paper usa Lead II + V5 (34 features). Tus CSVs son single-lead, así que reducimos a **17 features (solo Lead II equivalente)**.

2. **Sin upsampling por duplicación**: el paper duplica muestras de clases minoritarias para igualar la mayoritaria (técnica metodológicamente defectuosa que causa data leakage). Usamos **balance natural a 1000 por clase** consistente con tu setup.

3. **Clasificación binaria por sub-modelo**: el paper hace 5-clases (N/S/V/F/Q) sobre un solo modelo. Para comparación justa con tu arquitectura dual, replicamos XGBoost dos veces:
   - **XGBoost-LATIDO**: N vs PAC (donde PAC = clase S del paper) sobre señales de 250 muestras
   - **XGBoost-RITMO**: NSR vs AFIB sobre señales de 1000 muestras

4. **Split 70/30** estratificado (vs 80/20 del paper) para comparabilidad con tu sistema.

## Decisiones de implementación donde el paper es ambiguo

| Aspecto | Lo que dice el paper | Decisión tomada | Razón |
|---|---|---|---|
| Detección de P, Q, R, S, T peaks | NO especifica algoritmo | NeuroKit2 (`nk.ecg_delineate` + `nk.ecg_peaks`) | Estándar de facto en investigación ECG |
| Features morfológicas `m_0` a `m_4` | "Vector representing various shape attributes of QRS" — SIN definir | **5 coeficientes de la expansión de Hermite del QRS** | Hermite es el estándar para describir morfología QRS (Lagerholm et al. 2000) |
| Resampling | No discutido | Asumimos 250 Hz (consistente con tus datos) | Tus CSVs ya están a 250 Hz |
| Manejo de NaN en P-wave (común en PAC) | No discutido | Imputación con mediana de la columna en train | Default robusto |
| Random seed | No reportado | 42 (consistente con resto de tesis) | Reproducibilidad |
| Feature selection threshold | `SelectFromModel` sin threshold explícito | `threshold='median'` (default de sklearn) | Default de sklearn |

## Estructura del proyecto

```
xgboost_andotra/
├── README.md                # este archivo
├── preprocessing.py         # carga CSVs + detección de fiducials (P,Q,R,S,T)
├── features.py              # 17 features (3 RR + 4 intervals + 5 amplitudes + 5 morphological)
├── xgboost_model.py         # configuración XGBoost del paper + feature selection
├── main.py                  # orquestador con config YAML
├── config_latido.yaml       # configuración para LATIDO
└── config_ritmo.yaml        # configuración para RITMO
```

## Instalación

```bash
pip install numpy pandas scipy scikit-learn xgboost neurokit2 pyyaml pywavelets
```

## Uso

```bash
python main.py --config config_latido.yaml
python main.py --config config_ritmo.yaml
```

## Hiperparámetros del XGBoost (idénticos al paper)

| Parámetro | Valor |
|---|---|
| max_depth | 3 |
| n_estimators | 1000 |
| reg_alpha (L1, α) | 0.01 |
| reg_lambda (L2, λ) | 1.0 |
| gamma (γ) | 0.1 |
| objective | multi:softprob (binary:logistic para 2 clases) |
| eval_metric | mlogloss (logloss para 2 clases) |
| early_stopping_rounds | 10 |

## Features (17 por lead = 17 total para single-lead)

### RR Intervals (3 features)
1. RR (tiempo entre R-peaks consecutivos)
2. Average RR
3. Post RR (RR del latido siguiente)

### Heartbeat Interval Features (4 features)
4. PQ Interval = t(Q) - t(P)
5. QRS Interval = t(S) - t(Q)
6. QT Interval = t(T) - t(Q)
7. ST Interval = t(T) - t(S)

### Heartbeat Amplitude Features (5 features)
8-12. Amplitudes de P, Q, R, S, T peaks

### Morphological Features (5 features) — REINTERPRETADAS
13-17. **Coeficientes de Hermite de orden 0-4** sobre el complejo QRS (paper original no los define)

## Limitaciones conocidas de la réplica

1. **Features morfológicas reinterpretadas**: el paper no define qué son `m_0` a `m_4`. Usamos coeficientes de Hermite que es la convención estándar en literatura ECG, pero estrictamente NO sabemos qué usó el paper original.

2. **NeuroKit2 puede fallar en señales muy irregulares (AFIB)**: cuando esto ocurre usamos fallback custom con `scipy.signal.find_peaks` y ventanas heurísticas para P, Q, S, T.

3. **Single-lead**: el paper usa 2 leads (II + V5), nosotros 1 lead. Esto representa una réplica fiel del sub-conjunto disponible en nuestros datos.

4. **Las features dependen de detección correcta de ondas**: PAC con onda P invertida/ausente puede generar features ruidosas. Usamos imputación con mediana.

## Comparación esperada con tu sistema

Tu sistema (shapelets + RR + XGBoost) probablemente supere a este (handcrafted + XGBoost) porque:
- Shapelets aprenden patrones discriminativos automáticamente
- Las 17 features handcrafted dependen de detección perfecta de cada onda
- En PAC la onda P puede estar invertida o ausente → features ruidosas

Esto es **una historia muy buena para tu Capítulo 5**: misma arquitectura de clasificador, distinto feature engineering, demuestra valor de los shapelets.
