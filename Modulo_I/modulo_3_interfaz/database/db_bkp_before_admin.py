"""
Capa fina de acceso a la base SQLite. Toda la logica de queries vive aqui.
"""
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "evaluaciones.db"
SCHEMA_PATH = ROOT / "schema.sql"


def init_db():
    """Crea las tablas si no existen y aplica migraciones idempotentes."""
    with open(SCHEMA_PATH) as f:
        schema = f.read()
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(schema)
        # Migracion segura: anade columna marcaciones_nota si no existe.
        # SQLite no soporta "ALTER TABLE ... ADD COLUMN IF NOT EXISTS",
        # asi que lo verificamos con PRAGMA table_info.
        cols = [r[1] for r in conn.execute("PRAGMA table_info(evaluaciones)")]
        if "marcaciones_nota" not in cols:
            conn.execute(
                "ALTER TABLE evaluaciones ADD COLUMN marcaciones_nota TEXT"
            )
        conn.commit()


@contextmanager
def get_conn():
    """Context manager para conexiones con row_factory tipo dict."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------
# Medicos
# ---------------------------------------------------------------
def crear_medico(payload: dict) -> int:
    """Inserta un nuevo medico y devuelve su id."""
    with get_conn() as conn:
        cur = conn.execute("""
            INSERT INTO medicos (
                nombre_completo, edad, sexo, institucion,
                especialidad, sub_especialidad,
                anos_experiencia, anos_experiencia_ecg,
                auto_eval_habilidad, email
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            payload.get("nombre_completo"),
            payload.get("edad"),
            payload.get("sexo"),
            payload.get("institucion"),
            payload.get("especialidad"),
            payload.get("sub_especialidad"),
            payload.get("anos_experiencia"),
            payload.get("anos_experiencia_ecg"),
            payload.get("auto_eval_habilidad"),
            payload.get("email"),
        ))
        conn.commit()
        return cur.lastrowid


def get_medico(medico_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM medicos WHERE medico_id = ?", (medico_id,)
        ).fetchone()
        return dict(row) if row else None


def listar_medicos() -> list[dict]:
    with get_conn() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM medicos ORDER BY fecha_registro DESC"
        ).fetchall()]


# ---------------------------------------------------------------
# Evaluaciones
# ---------------------------------------------------------------
def crear_evaluacion(payload: dict, respuestas_ventana: list[dict]) -> int:
    """
    Inserta una evaluacion completa: la cabecera + todas las respuestas
    por ventana en una unica transaccion.

    payload["marcaciones_nota"] es un dict con:
        {marcaciones: [{start, end, color}, ...],
         pct_verde, pct_amarillo, pct_rojo, pct_sin_marcar,
         total_caracteres}
    Se serializa a JSON al guardar; opcional (puede omitirse).
    """
    marc = payload.get("marcaciones_nota")
    marc_json = json.dumps(marc, ensure_ascii=False) if marc else None

    with get_conn() as conn:
        cur = conn.execute("""
            INSERT INTO evaluaciones (
                medico_id, signal_id, llm_backend, llm_model_id,
                nota_clinica_texto, nota_semaforo_global,
                likert_exactitud, likert_coherencia, likert_utilidad,
                marcaciones_nota,
                comentarios_libres, duracion_segundos
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            payload["medico_id"],
            payload["signal_id"],
            payload["llm_backend"],
            payload.get("llm_model_id"),
            payload["nota_clinica_texto"],
            payload["nota_semaforo_global"],
            payload["likert_exactitud"],
            payload["likert_coherencia"],
            payload["likert_utilidad"],
            marc_json,
            payload.get("comentarios_libres", ""),
            payload.get("duracion_segundos"),
        ))
        eval_id = cur.lastrowid

        for resp in respuestas_ventana:
            conn.execute("""
                INSERT INTO respuestas_ventana (
                    evaluacion_id, escala, window_index,
                    prediccion_modelo, semaforo, confidence_gap
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                eval_id, resp["escala"], resp["window_index"],
                resp["prediccion_modelo"], resp["semaforo"],
                resp.get("confidence_gap"),
            ))
        conn.commit()
        return eval_id


def listar_evaluaciones(medico_id: int | None = None,
                        signal_id: str | None = None) -> list[dict]:
    where, params = [], []
    if medico_id is not None:
        where.append("medico_id = ?")
        params.append(medico_id)
    if signal_id is not None:
        where.append("signal_id = ?")
        params.append(signal_id)
    where_sql = (" WHERE " + " AND ".join(where)) if where else ""
    with get_conn() as conn:
        rows = conn.execute(f"""
            SELECT e.*, m.nombre_completo, m.especialidad, m.anos_experiencia_ecg
            FROM evaluaciones e
            JOIN medicos m ON m.medico_id = e.medico_id
            {where_sql}
            ORDER BY e.fecha_evaluacion DESC
        """, params).fetchall()
        return [dict(r) for r in rows]


def get_respuestas(eval_id: int) -> list[dict]:
    with get_conn() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM respuestas_ventana WHERE evaluacion_id = ?", (eval_id,)
        ).fetchall()]


# ---------------------------------------------------------------
# Aggregates for analytics
# ---------------------------------------------------------------
def stats_globales() -> dict:
    """Estadisticas agregadas para el dashboard."""
    with get_conn() as conn:
        n_medicos = conn.execute("SELECT COUNT(*) FROM medicos").fetchone()[0]
        n_evaluaciones = conn.execute("SELECT COUNT(*) FROM evaluaciones").fetchone()[0]
        n_respuestas = conn.execute("SELECT COUNT(*) FROM respuestas_ventana").fetchone()[0]
        return {
            "n_medicos": n_medicos,
            "n_evaluaciones": n_evaluaciones,
            "n_respuestas_ventana": n_respuestas,
        }


def stats_por_escala() -> dict:
    """% verde/amarillo/rojo por escala (beat vs rhythm)."""
    out = {}
    with get_conn() as conn:
        for escala in ["beat", "rhythm"]:
            rows = conn.execute("""
                SELECT semaforo, COUNT(*) c
                FROM respuestas_ventana
                WHERE escala = ?
                GROUP BY semaforo
            """, (escala,)).fetchall()
            total = sum(r["c"] for r in rows) or 1
            out[escala] = {r["semaforo"]: round(r["c"] / total * 100, 2)
                           for r in rows}
    return out


def stats_por_clase(escala: str) -> dict:
    """% verde/amarillo/rojo desglosado por clase predicha (NORMAL, PAC / NSR, AFIB)."""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT prediccion_modelo, semaforo, COUNT(*) c
            FROM respuestas_ventana
            WHERE escala = ?
            GROUP BY prediccion_modelo, semaforo
        """, (escala,)).fetchall()
    out = {}
    for r in rows:
        cls = r["prediccion_modelo"]
        out.setdefault(cls, {"verde": 0, "amarillo": 0, "rojo": 0, "_total": 0})
        out[cls][r["semaforo"]] = r["c"]
        out[cls]["_total"] += r["c"]
    for cls in out:
        total = out[cls].pop("_total")
        for k in ["verde", "amarillo", "rojo"]:
            out[cls][k] = round(out[cls][k] / total * 100, 2) if total else 0.0
    return out


def stats_por_backend() -> list[dict]:
    """Notas LLM: distribuciones Likert + semaforo global por backend."""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT llm_backend,
                   COUNT(*) n,
                   AVG(likert_exactitud)  prom_exactitud,
                   AVG(likert_coherencia) prom_coherencia,
                   AVG(likert_utilidad)   prom_utilidad
            FROM evaluaciones
            GROUP BY llm_backend
        """).fetchall()
    out = []
    for r in rows:
        with get_conn() as conn:
            sem_rows = conn.execute("""
                SELECT nota_semaforo_global, COUNT(*) c
                FROM evaluaciones
                WHERE llm_backend = ?
                GROUP BY nota_semaforo_global
            """, (r["llm_backend"],)).fetchall()
        total = sum(s["c"] for s in sem_rows) or 1
        sem_dist = {s["nota_semaforo_global"]: round(s["c"] / total * 100, 2)
                    for s in sem_rows}
        out.append({
            "backend": r["llm_backend"],
            "n_evaluaciones": r["n"],
            "prom_exactitud": round(r["prom_exactitud"], 2) if r["prom_exactitud"] else None,
            "prom_coherencia": round(r["prom_coherencia"], 2) if r["prom_coherencia"] else None,
            "prom_utilidad": round(r["prom_utilidad"], 2) if r["prom_utilidad"] else None,
            "semaforo_distribucion": sem_dist,
        })
    return out
