-- Esquema de la base de datos de validacion clinica
-- Diseno: tres tablas normalizadas para evaluaciones por ventana y por nota.

PRAGMA foreign_keys = ON;

-- Cardiologos registrados (perfil capturado una vez por sesion)
CREATE TABLE IF NOT EXISTS medicos (
    medico_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    nombre_completo      TEXT NOT NULL,
    edad                 INTEGER,
    sexo                 TEXT,
    institucion          TEXT,
    especialidad         TEXT,            -- e.g. 'Cardiologia general', 'Electrofisiologia'
    sub_especialidad     TEXT,
    anos_experiencia     INTEGER,         -- clinical experience total
    anos_experiencia_ecg INTEGER,         -- specific ECG/arrhythmia experience
    auto_eval_habilidad  INTEGER,         -- 1-5 Likert self-assessed ECG skill
    email                TEXT,
    fecha_registro       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Una evaluacion = un cardiologo evalua una senal
-- Captura la nota clinica del LLM evaluada, los semaforos globales,
-- y las marcaciones de subrayado por color hechas sobre el texto.
CREATE TABLE IF NOT EXISTS evaluaciones (
    evaluacion_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    medico_id            INTEGER NOT NULL,
    signal_id            TEXT    NOT NULL,           -- 'sig_001', etc.
    llm_backend          TEXT    NOT NULL,           -- 'claude' / 'gemini' / 'groq' / 'mock'
    llm_model_id         TEXT,                       -- specific model snapshot
    nota_clinica_texto   TEXT    NOT NULL,           -- the LLM-generated note exactly as shown
    nota_semaforo_global TEXT,                       -- 'verde' / 'amarillo' / 'rojo'
    likert_exactitud     INTEGER,                    -- 1..5
    likert_coherencia    INTEGER,                    -- 1..5
    likert_utilidad      INTEGER,                    -- 1..5
    marcaciones_nota     TEXT,                       -- JSON: {marcaciones:[{start,end,color}], pct_verde, pct_amarillo, pct_rojo, pct_sin_marcar, total_caracteres}
    comentarios_libres   TEXT,
    duracion_segundos    REAL,                       -- time the physician spent
    fecha_evaluacion     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (medico_id) REFERENCES medicos(medico_id)
);

-- Validacion por ventana (semaforos por window BEAT o RHYTHM)
CREATE TABLE IF NOT EXISTS respuestas_ventana (
    respuesta_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    evaluacion_id    INTEGER NOT NULL,
    escala           TEXT    NOT NULL,        -- 'beat' or 'rhythm'
    window_index     INTEGER NOT NULL,
    prediccion_modelo TEXT   NOT NULL,        -- the label the classifier produced
    semaforo         TEXT    NOT NULL,        -- 'verde' / 'amarillo' / 'rojo'
    confidence_gap   REAL,                    -- captured for analysis
    FOREIGN KEY (evaluacion_id) REFERENCES evaluaciones(evaluacion_id)
);

-- Indexes for analytics queries
CREATE INDEX IF NOT EXISTS idx_eval_medico ON evaluaciones(medico_id);
CREATE INDEX IF NOT EXISTS idx_eval_signal ON evaluaciones(signal_id);
CREATE INDEX IF NOT EXISTS idx_eval_backend ON evaluaciones(llm_backend);
CREATE INDEX IF NOT EXISTS idx_resp_eval ON respuestas_ventana(evaluacion_id);
CREATE INDEX IF NOT EXISTS idx_resp_escala ON respuestas_ventana(escala);
