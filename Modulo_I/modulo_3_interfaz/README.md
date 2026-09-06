# Módulo III · Validación clínica human-in-the-loop

Interfaz web Flask para que cardiólogos evalúen las predicciones del sistema
dual MSM-SC + XGBoost y las notas clínicas generadas por el agente LLM.
Forma parte del Módulo III de la tesis CIMAT.

---

## Arquitectura

```
modulo_3_validacion/
├── app.py                     # Flask app con todas las rutas
├── config.py                  # Backend LLM activo, paths, tema
├── requirements.txt
├── README.md
│
├── classifier/                # (Opcional) script local para procesar señales reales
│
├── llm_backends/              # Abstracción multi-backend
│   ├── base.py                # Interfaz común + builder del prompt
│   ├── claude_backend.py      # Anthropic Claude Sonnet 4.6
│   ├── gemini_backend.py      # Google Gemini 2.5 Flash
│   ├── groq_backend.py        # Groq / Llama 3.3 70B
│   └── mock_backend.py        # Mock determinista para demos sin API
│
├── database/
│   ├── schema.sql             # Esquema SQLite
│   ├── db.py                  # Helpers de acceso
│   └── evaluaciones.db        # Se crea al primer arranque
│
├── data/
│   ├── models/                # Modelos XGBoost + feature names + importance
│   │   ├── xgboost_LATIDO_MSM.json
│   │   ├── xgboost_RITMO_MSM.json
│   │   └── ...
│   ├── shapelets/             # (Reservado) tus shapelets reales
│   ├── signals_raw/           # (Reservado) ECG crudo
│   └── signals_processed/     # 5 JSONs precomputados (input al sistema)
│
├── scripts/
│   └── bootstrap.py           # Genera modelos + 5 señales mock
│
├── static/
│   ├── css/medico.css         # Tema crema/rojo (papel ECG)
│   └── js/
│       ├── evaluar.js
│       └── analytics.js
│
└── templates/
    ├── base.html
    ├── registro.html
    ├── seleccion.html
    ├── evaluar.html
    └── analytics.html
```

---

## Primer arranque

```bash
# 1. Crear entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Bootstrap: entrena los dos XGBoost desde tus CSVs y crea 5 señales mock
#    (requiere los CSVs en /mnt/user-data/uploads/ o ajustar UPLOADS en scripts/bootstrap.py)
python scripts/bootstrap.py

# 4. Arrancar Flask
python app.py
```

Luego abrir en navegador: `http://127.0.0.1:5000`

---

## Flujo del cardiólogo

1. **Registro** — formulario inicial con nombre, especialidad, años de experiencia, auto-evaluación de habilidad ECG (1–5).
2. **Selección** — elige una de las 5 señales disponibles.
3. **Evaluación**:
   - ECG graficado a 25 mm/s 10 mm/mV en rojo sobre cuadrícula crema
   - Bajo el ECG, dos pistas con las etiquetas del clasificador: una por ventana de **LATIDO** (1 s) y una por ventana de **RITMO** (4 s)
   - Bajo cada etiqueta, semáforo 🟢🟡🔴 para calificar la concordancia
   - Más abajo, la nota clínica generada por el LLM activo
   - Semáforo global de la nota + tres ejes Likert 1–5 (exactitud, coherencia, utilidad)
   - Campo de comentarios libres
4. **Métricas** — dashboard agregado en `/analytics`

---

## Cambiar el backend LLM

En `config.py`:

```python
LLM_BACKEND = "claude"   # o "gemini", "groq", "mock"
```

Y exportar la API key correspondiente:

```bash
export ANTHROPIC_API_KEY=sk-ant-...        # para claude
export GEMINI_API_KEY=AIza...              # para gemini
export GROQ_API_KEY=gsk_...                # para groq
```

Para comparar varios backends sobre las mismas señales, basta cambiar el valor de `LLM_BACKEND` y pedir a los mismos cardiólogos que evalúen las 5 señales con cada uno. El dashboard agrega resultados por backend automáticamente.

---

## Reemplazar las señales mock por señales reales

Las 5 señales mock fueron generadas por `scripts/bootstrap.py` con datos sintéticos. Para usar señales reales:

1. Reemplazar los JSONs en `data/signals_processed/sig_NNN.json`
2. Cada JSON debe seguir el contrato definido en `llm_backends/base.py:build_prompt()`:
   - `signal_metadata` (id, derivación, paciente)
   - `raw_signal` (array de muestras z-score)
   - `beat_windows[]` (predicciones por ventana de 250 muestras)
   - `rhythm_windows[]` (predicciones por ventana de 1000 muestras)
   - `aggregate_summary` (distribuciones y banderas)

Lo que se requiere localmente para producir esos JSONs:
- Los 60 shapelets MICRO + 60 shapelets MACRO en formato .npy o .json
- La señal ECG cruda (WFDB, EDF, CSV) de Icentia11k o tu fuente
- El cálculo MSM-SC con banda Sakoe-Chiba (puedes reutilizar el código de tu pipeline original)

---

## Tablas en la base de datos

`medicos`, `evaluaciones`, `respuestas_ventana`. Detalle en `database/schema.sql`. Para exportar los datos a CSV:

```bash
sqlite3 database/evaluaciones.db \
  ".headers on" ".mode csv" ".output evaluaciones.csv" \
  "SELECT * FROM evaluaciones"
```

---

## Salidas estadísticas

Disponibles vía dashboard en `/analytics` y vía API en `/api/analytics_data`:

- **% verde/amarillo/rojo por escala** — concordancia BEAT vs RHYTHM
- **% por clase predicha** — desglose por NORMAL, PAC, NSR, AFIB
- **Comparativa entre backends** — promedios Likert + distribución semáforo de la nota
- **Listado de cardiólogos** — perfil completo

Estos datos alimentan directamente la **Sección 4.3 (Validación clínica humana)** de la tesis.
