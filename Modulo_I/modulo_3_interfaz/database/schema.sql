-- ================================================================
-- Schema principal. Aplicado por db.init_db() al arrancar.
-- Todas las nuevas columnas se agregan tambien por PRAGMA
-- table_info en db.py, para migracion idempotente en BDs existentes.
-- ================================================================

CREATE TABLE IF NOT EXISTS medicos (
    medico_id             INTEGER PRIMARY KEY AUTOINCREMENT,
    nombre_completo       TEXT NOT NULL,
    edad                  TEXT,                -- Ahora es rango (string): "<30", "30-39", etc.
    sexo                  TEXT,
    institucion           TEXT,
    especialidad          TEXT,
    sub_especialidad      TEXT,
    anos_experiencia      INTEGER,
    anos_experiencia_ecg  INTEGER,
    auto_eval_habilidad   INTEGER,
    email                 TEXT,
    acepto_terminos       INTEGER DEFAULT 0,    -- 1 si el medico marco el checkbox
    fecha_aceptacion      TEXT,                 -- ISO timestamp de la aceptacion
    fecha_registro        TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS evaluaciones (
    evaluacion_id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    medico_id                      INTEGER NOT NULL,
    signal_id                      TEXT NOT NULL,
    llm_backend                    TEXT,
    llm_model_id                   TEXT,

    -- Nota clinica (interpretacion diagnostica)
    nota_clinica_texto             TEXT,
    nota_semaforo_global           TEXT,       -- verde/amarillo/rojo sobre la nota clinica
    marcaciones_nota               TEXT,       -- JSON con subrayado + porcentajes
    likert_exactitud               INTEGER,
    likert_coherencia              INTEGER,
    likert_utilidad                INTEGER,
    comentarios_nota               TEXT,       -- antes: comentarios_libres

    -- Nota del clasificador (nivel de certeza / escalacion)
    nota_clasificador_texto        TEXT,
    nota_clasificador_semaforo     TEXT,       -- verde/amarillo/rojo sobre la nota clasificador
    nota_clasificador_marcaciones  TEXT,       -- JSON con subrayado + porcentajes
    comentarios_clasificador       TEXT,

    duracion_segundos              REAL,
    fecha_evaluacion               TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (medico_id) REFERENCES medicos (medico_id)
);

CREATE TABLE IF NOT EXISTS respuestas_ventana (
    respuesta_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    evaluacion_id      INTEGER NOT NULL,
    escala             TEXT NOT NULL,   -- 'beat' | 'rhythm'
    window_index       INTEGER NOT NULL,
    prediccion_modelo  TEXT NOT NULL,
    semaforo           TEXT NOT NULL,   -- 'verde' | 'amarillo' | 'rojo'
    confidence_gap     REAL,
    FOREIGN KEY (evaluacion_id) REFERENCES evaluaciones (evaluacion_id)
);
