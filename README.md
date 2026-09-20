# Sistema de Asistencia Informativa para Etiquetado de Arritmias en ECG

**Clasificación con Medidas de Distancia Elástica y LLMs Adaptados al Conocimiento Médico**

Trabajo de tesis de maestría en Ingeniería de Software del Centro de Investigación en Matemáticas (CIMAT), Unidad Zacatecas.

---

## Estado del proyecto

Fase de validación clínica activa. Sistema desplegado en un entorno controlado para evaluación por cardiólogos.

Manuscrito derivado en preparación para revista JCR Q1 (objetivo: *Biomedical Signal Processing and Control*).

---

## Descripción

Este proyecto implementa un sistema de asistencia informativa para el etiquetado de arritmias en electrocardiogramas de una derivación. El sistema no reemplaza al cardiólogo: asiste al proceso de anotación clínica proponiendo predicciones interpretables acompañadas de explicaciones en lenguaje natural.

El sistema aborda tres problemas simultáneamente:

1. **Interpretabilidad**: el clasificador basa sus decisiones en subsecuencias representativas (shapelets) extraídas de las señales, permitiendo trazar cada predicción hasta patrones morfológicos concretos.
2. **Seguridad narrativa**: un agente basado en modelo de lenguaje genera notas clínicas explicables sobre las predicciones, con validaciones automáticas que impiden la emisión de aseveraciones absolutas o prescripciones.
3. **Human-in-the-loop**: una interfaz web permite a cardiólogos evaluadores validar tanto las predicciones del clasificador como las notas generadas, calculando métricas de concordancia inter-evaluador.

---

## Arquitectura del sistema

El sistema consta de **tres módulos** que trabajan en cascada:

```
[Señal ECG cruda]
        |
        v
+---------------------+
| Modulo I            |  Extraccion de shapelets (MSM + SC)
| Clasificador dual   |  XGBoost LATIDO + XGBoost RITMO
+---------------------+
        |
        v
+---------------------+
| Modulo II           |  Prompt engineering adaptado al dominio clinico
| Agente LLM          |  Verificaciones automaticas (9 checks)
+---------------------+
        |
        v
+---------------------+
| Modulo III          |  Flask + SQLite
| Interfaz de         |  Vista medico + panel administrativo
| validacion clinica  |  Deployado en PythonAnywhere
+---------------------+
        |
        v
[Metricas de concordancia inter-evaluador]
```

### Módulo I — Clasificador dual con shapelets multi-escala

Implementación de un clasificador dual basado en:

- **Move-Split-Merge (MSM) distance**: distancia elástica robusta a desplazamientos temporales, superior a Dynamic Time Warping para clasificación de series temporales biomédicas.
- **Shapelet Coverage (SC)**: reducción dimensional que representa cada señal como un vector de similitudes con shapelets representativas extraídas del conjunto de entrenamiento.
- **XGBoost**: dos clasificadores independientes entrenados sobre los vectores SC.

Dos escalas de predicción:

- **LATIDO** (ventanas de 1 segundo): NORMAL vs PAC. Umbral de confianza τ = 0.50.
- **RITMO** (ventanas de 4 segundos): NSR vs AFIB. Umbral de confianza τ = 0.60.

Cuando el nivel de certeza cae por debajo del umbral, la ventana se marca con bandera de escalación para revisión especialista.

### Módulo II — Agente basado en LLM

Genera dos notas para cada señal analizada:

1. **Nota clínica**: interpretación diagnóstica narrativa que describe la morfología, ritmo, presencia de arritmias y ubicación temporal de eventos.
2. **Nota del clasificador**: nota breve sobre el nivel de certeza del modelo y necesidad de escalación.

Backend actual: **Gemini 2.5 Flash** (Google).

Verificaciones automáticas por nota (9 checks):

- Idioma español correcto
- Mención explícita de la derivación
- Descripción de la morfología
- Mención de las clases predichas (latido y ritmo)
- Mención del nivel de certeza
- Longitud apropiada
- Ausencia de certezas absolutas
- Ausencia de prescripciones farmacológicas

Las notas son **reproducibles**: se persisten en JSON con metadatos de generación (latencia, versión del prompt, modelo).

### Módulo III — Interfaz de validación clínica

Aplicación web para la evaluación humana del sistema:

**Vista del cardiólogo**:
- Trazado ECG en tiras de 10 segundos (formato clínico estándar 25 mm/s, 10 mm/mV)
- Predicciones del clasificador debajo de cada tira con semáforo (verde/amarillo/rojo) por ventana
- Notas del LLM con subrayado por segmentos y veredicto global
- Puntuación Likert 1-7 en tres dimensiones: exactitud, coherencia, utilidad
- Guardado automático en localStorage del navegador
- Comentarios libres opcionales

**Panel administrativo**:
- Dashboard con métricas globales
- Detalle por cardiólogo y por evaluación
- Cálculo de kappa de Cohen inter-evaluador
- Vista comparativa tri-fuente: etiqueta real de la base de datos vs predicción del clasificador vs veredictos de médicos
- Exportación a CSV
- Todas las gráficas incluyen sus fórmulas en LaTeX renderizadas con MathJax

---

## Datasets

El sistema fue entrenado y evaluado con tres bases de datos públicas de PhysioNet:

- **PTB-XL** (Wagner et al., 2020): 21,837 registros ECG de 10 segundos, 12 derivaciones, anotados por cardiólogos. Utilizado para entrenamiento.
- **Icentia11k** (Tan et al., 2019): 11,000 pacientes con anotaciones beat-a-beat en derivación única. Utilizado para entrenamiento y validación.
- **PhysioNet CinC Challenge 2017** (Clifford et al., 2017): 8,528 registros ECG cortos (9-60s) anotados a nivel de registro. Utilizado para validación externa.

Cinco señales representativas fueron seleccionadas para el estudio de validación clínica (una de CinC2017 y cuatro de Icentia11k), cubriendo casos normales, fibrilación auricular y contracciones auriculares prematuras.

---

## Métricas actuales del clasificador

Resultados sobre el conjunto de prueba independiente:

| Escala | Kappa de Cohen | Coeficiente MCC | AUC-ROC |
|--------|:--------------:|:---------------:|:-------:|
| LATIDO | 0.623          | 0.625           | 0.878   |
| RITMO  | 0.747          | 0.748           | 0.938   |

Cobertura de la política de escalación (H4): 34% de las ventanas con Δp < 0.25.

Estado de las hipótesis del trabajo:

- **H1** — Interpretabilidad del clasificador: verificada
- **H2** — κ y MCC ≥ 0.60: verificada
- **H3** — Seguridad narrativa del LLM: parcial (Likert de cardiólogos en curso)
- **H4** — Seguridad clínica (política de escalación): parcial

---

## Estructura del repositorio

```
Tesis_Fabio_Deteccion_de_Arritmias/
├── Modulo_I/
│   ├── modulo_1_clasificador/           Modulo I: entrenamiento y clasificador
│   │   ├── data/                        Datasets procesados
│   │   ├── artifacts/                   Modelos entrenados (.json de XGBoost)
│   │   ├── entrenamiento_dual.py        Script principal de entrenamiento
│   │   ├── msm_sc.py                    Implementacion MSM + SC
│   │   ├── shapelets.py                 Extraccion de shapelets
│   │   ├── clasificador.py              Wrapper del clasificador dual
│   │   └── reportes/                    Metricas y curvas ROC
│   │
│   ├── modulo_2_agente/                 Modulo II: agente LLM
│   │   ├── agente_notas.py              Generacion de notas
│   │   ├── prompts/                     Templates de prompt engineering
│   │   ├── verificaciones.py            9 checks automaticos
│   │   └── backends/                    Adaptadores para diferentes LLMs
│   │
│   └── modulo_3_interfaz/               Modulo III: interfaz web
│       ├── app.py                       Aplicacion Flask
│       ├── database/
│       │   ├── db.py                    Capa de acceso SQLite
│       │   ├── schema.sql               Esquema de la BD
│       │   └── evaluaciones.db          Base de datos (no versionada)
│       ├── templates/                   HTML Jinja2
│       ├── static/                      CSS y JavaScript
│       ├── data/
│       │   ├── signals_processed/       Senales ECG procesadas (JSON)
│       │   ├── notas_llm_a/             Notas generadas por el LLM
│       │   └── signal_labels.json       Ground truth de las senales
│       └── scripts/                     Utilidades (separacion de notas, etc.)
│
├── docs/                                Documentacion adicional
│   ├── consentimiento_informado.pdf     Documento etico
│   └── plan_de_trabajo.pdf              Cronograma
│
├── README.md                            Este archivo
└── requirements.txt                     Dependencias Python
```

---

## Requisitos

- **Python** 3.10 o superior
- **Sistema operativo**: Windows, Linux o macOS
- **Memoria RAM**: 8 GB mínimo (16 GB recomendado para reentrenar el clasificador)
- **Espacio en disco**: 3 GB para el proyecto completo con datasets

### Dependencias principales

```
numpy>=1.24
pandas>=2.0
scipy>=1.11
xgboost>=2.0
scikit-learn>=1.3
matplotlib>=3.7
plotly>=5.17
flask>=3.0
google-generativeai>=0.3   # Para Gemini
tqdm>=4.66
```

Ver `requirements.txt` para la lista completa.

---

## Instalación

```bash
# 1. Clonar el repositorio (requiere acceso institucional)
git clone https://github.com/fabio-pantera04/Tesis_Fabio_Deteccion_de_Arritmias.git
cd Tesis_Fabio_Deteccion_de_Arritmias

# 2. Crear entorno virtual
python -m venv venv
# En Windows:
venv\Scripts\activate
# En Linux/macOS:
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. (Opcional) Configurar API key de Gemini para regenerar notas
# Crear archivo .env en la raiz con:
# GEMINI_API_KEY=tu_api_key_aqui
```

---

## Uso

### Módulo I — Entrenar el clasificador desde cero

```bash
cd Modulo_I/modulo_1_clasificador
python entrenamiento_dual.py
```

Genera los artefactos en `artifacts/`: `xgboost_latido_MSM.json`, `xgboost_ritmo_MSM.json`, matrices de confusión y curvas ROC.

### Módulo II — Generar notas clínicas para las señales

```bash
cd Modulo_I/modulo_2_agente
python generar_notas.py --backend llm_a --signals sig_001 sig_002 sig_003 sig_004 sig_005
```

Las notas se guardan en `Modulo_I/modulo_3_interfaz/data/notas_llm_a/`.

### Módulo III — Levantar la interfaz de validación

```bash
cd Modulo_I/modulo_3_interfaz
python app.py
```

Abre `http://localhost:5000` en el navegador.

- Vista médico: registro, selección de señal, evaluación.
- Panel administrativo: `http://localhost:5000/admin/login` (contraseña definida en variable de entorno `ADMIN_PASSWORD`).

---

## Deploy en producción (PythonAnywhere)

El sistema está desplegado en **https://fabiovelasco.pythonanywhere.com** para el estudio de validación clínica.

Flujo de actualización desde local:

```bash
# En la máquina local
git add .
git commit -m "Descripción del cambio"
git push origin main

# En consola bash de PythonAnywhere
cd ~/app-cimat
git pull
# Recargar el web app desde el panel Web de PythonAnywhere
```

Configuración WSGI apunta a `/home/fabiovelasco/app-cimat/Modulo_I/modulo_3_interfaz`, entorno virtual en `/home/fabiovelasco/.virtualenvs/cimat-env` (Python 3.10). La aplicación usa sparse-checkout del subdirectorio `Modulo_I/modulo_3_interfaz` para minimizar espacio en disco del hosting.

---

## Marco ético y datos

El estudio de validación clínica se apega a:

- Declaración de Helsinki (versión revisada)
- Ley Federal de Protección de Datos Personales en Posesión de los Particulares (LFPDPPP, México)
- Reglamento de la Ley General de Salud en Materia de Investigación para la Salud, artículo 17 fracción I (categoría: **sin riesgo**)
- Guías **TRIPOD-AI** (2024) para el reporte de modelos de predicción clínica basados en inteligencia artificial

Las señales evaluadas provienen exclusivamente de datasets públicos anonimizados. No se procesan datos de pacientes reales bajo cuidado de los cardiólogos evaluadores.

---

## Referencias bibliográficas clave

- Wagner, P., et al. (2020). PTB-XL: A large publicly available electrocardiography dataset. *Scientific Data*, 7(1), 154.
- Tan, S., et al. (2019). Icentia11k: An Unsupervised Representation Learning Dataset for Arrhythmia Subtype Discovery. *arXiv:1910.09570*.
- Clifford, G. D., et al. (2017). AF Classification from a Short Single Lead ECG Recording: The PhysioNet Computing in Cardiology Challenge 2017. *Computing in Cardiology*, 44.
- Stefan, A., Athitsos, V., & Das, G. (2013). The Move-Split-Merge Metric for Time Series. *IEEE Transactions on Knowledge and Data Engineering*, 25(6), 1425-1438.
- Ye, L., & Keogh, E. (2009). Time series shapelets: A new primitive for data mining. *ACM SIGKDD*.
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *ACM SIGKDD*.
- Cohen, J. (1960). A Coefficient of Agreement for Nominal Scales. *Educational and Psychological Measurement*, 20(1), 37-46.
- Landis, J. R., & Koch, G. G. (1977). The Measurement of Observer Agreement for Categorical Data. *Biometrics*, 33(1), 159-174.
- Collins, G. S., et al. (2024). TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ*, 385:e078378.

---

## Autoría y créditos

**Autor principal**:
- Fabio Misael Velasco Dorado — Estudiante de maestría en Ingeniería de Software, CIMAT Unidad Zacatecas

**Dirección académica**:
- Dr. Hugo Mitre-Hernández — Director de tesis, CIMAT Unidad Zacatecas
- Dr. Fernando Sánchez-Vega — Co-director de tesis, CIMAT Unidad Zacatecas

**Colaboración clínica**:
- Dra. Aranxa F. Jiménez Galván — Cardióloga, ISSSTE

**Institución**:
Centro de Investigación en Matemáticas, A.C. (CIMAT), Unidad Zacatecas.

---

## Contacto

Para consultas sobre el proyecto:

- **Datos y participación en el estudio**: hmitre@cimat.mx
- **Ejercicio de derechos ARCO** (participantes del estudio): hmitre@cimat.mx

---

## Licencia y difusión

Trabajo académico de tesis de maestría. Los derechos de reproducción y difusión están sujetos a los lineamientos del CIMAT.

Los datos anonimizados y agregados del estudio de validación clínica serán depositados en repositorios de acceso abierto (Zenodo, PhysioNet) tras la publicación del manuscrito derivado.

---

*Última actualización del README: septiembre 2026.*
