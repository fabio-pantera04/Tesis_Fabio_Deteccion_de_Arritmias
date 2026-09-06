"""
Capa fina de acceso a la base SQLite. Toda la logica de queries vive aqui.
Incluye funciones especiales para el panel de administrador:
  - Agregados para dashboard
  - Kappa inter-evaluador
  - Borrado individual y reset completo
  - Exportacion a CSV
"""
import csv
import io
import json
import sqlite3
from collections import defaultdict
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
        cols = [r[1] for r in conn.execute("PRAGMA table_info(evaluaciones)")]
        if "marcaciones_nota" not in cols:
            conn.execute(
                "ALTER TABLE evaluaciones ADD COLUMN marcaciones_nota TEXT"
            )
        conn.commit()


@contextmanager
def get_conn():
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
    """Lista todos los medicos con conteo de sus evaluaciones."""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT m.*,
                   COUNT(e.evaluacion_id) as n_evaluaciones,
                   ROUND(AVG(e.likert_exactitud), 2)  as prom_exactitud,
                   ROUND(AVG(e.likert_coherencia), 2) as prom_coherencia,
                   ROUND(AVG(e.likert_utilidad), 2)   as prom_utilidad,
                   ROUND(AVG(e.duracion_segundos), 1) as prom_duracion
            FROM medicos m
            LEFT JOIN evaluaciones e ON e.medico_id = m.medico_id
            GROUP BY m.medico_id
            ORDER BY m.fecha_registro DESC
        """).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------
# Evaluaciones
# ---------------------------------------------------------------
def crear_evaluacion(payload: dict, respuestas_ventana: list[dict]) -> int:
    marc = payload.get("marcaciones_nota")
    marc_json = json.dumps(marc, ensure_ascii=False) if marc else None
    with get_conn() as conn:
        cur = conn.execute("""
            INSERT INTO evaluaciones (
                medico_id, signal_id, llm_backend, llm_model_id,
                nota_clinica_texto, nota_semaforo_global,
                likert_exactitud, likert_coherencia, likert_utilidad,
                marcaciones_nota, comentarios_libres, duracion_segundos
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            payload["medico_id"], payload["signal_id"],
            payload["llm_backend"], payload.get("llm_model_id"),
            payload["nota_clinica_texto"], payload["nota_semaforo_global"],
            payload["likert_exactitud"], payload["likert_coherencia"],
            payload["likert_utilidad"], marc_json,
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
        where.append("e.medico_id = ?")
        params.append(medico_id)
    if signal_id is not None:
        where.append("e.signal_id = ?")
        params.append(signal_id)
    where_sql = (" WHERE " + " AND ".join(where)) if where else ""
    with get_conn() as conn:
        rows = conn.execute(f"""
            SELECT e.*, m.nombre_completo, m.especialidad,
                   m.anos_experiencia_ecg
            FROM evaluaciones e
            JOIN medicos m ON m.medico_id = e.medico_id
            {where_sql}
            ORDER BY e.fecha_evaluacion DESC
        """, params).fetchall()
        return [dict(r) for r in rows]


def get_evaluacion_completa(eval_id: int) -> dict | None:
    """Devuelve la evaluacion + medico + respuestas + marcaciones parseadas."""
    with get_conn() as conn:
        row = conn.execute("""
            SELECT e.*, m.nombre_completo, m.especialidad,
                   m.anos_experiencia_ecg, m.institucion
            FROM evaluaciones e
            JOIN medicos m ON m.medico_id = e.medico_id
            WHERE e.evaluacion_id = ?
        """, (eval_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["respuestas"] = [dict(r) for r in conn.execute(
            """SELECT * FROM respuestas_ventana
               WHERE evaluacion_id = ?
               ORDER BY escala, window_index""",
            (eval_id,)
        ).fetchall()]
        if d.get("marcaciones_nota"):
            try:
                d["marcaciones_nota_parsed"] = json.loads(d["marcaciones_nota"])
            except Exception:
                d["marcaciones_nota_parsed"] = None
        else:
            d["marcaciones_nota_parsed"] = None
        return d


def get_respuestas(eval_id: int) -> list[dict]:
    with get_conn() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM respuestas_ventana WHERE evaluacion_id = ?",
            (eval_id,)
        ).fetchall()]


# ---------------------------------------------------------------
# BORRADO (uso exclusivo del admin, para pruebas antes de
# validacion con cardiologos reales)
# ---------------------------------------------------------------
def eliminar_evaluacion(eval_id: int) -> bool:
    """Elimina una evaluacion individual y sus respuestas asociadas."""
    with get_conn() as conn:
        conn.execute("DELETE FROM respuestas_ventana WHERE evaluacion_id = ?",
                     (eval_id,))
        cur = conn.execute("DELETE FROM evaluaciones WHERE evaluacion_id = ?",
                            (eval_id,))
        conn.commit()
        return cur.rowcount > 0


def eliminar_medico(medico_id: int) -> bool:
    """Elimina un medico y en cascada todas sus evaluaciones + respuestas."""
    with get_conn() as conn:
        eval_ids = [r[0] for r in conn.execute(
            "SELECT evaluacion_id FROM evaluaciones WHERE medico_id = ?",
            (medico_id,)
        ).fetchall()]
        for eid in eval_ids:
            conn.execute("DELETE FROM respuestas_ventana WHERE evaluacion_id = ?",
                         (eid,))
        conn.execute("DELETE FROM evaluaciones WHERE medico_id = ?",
                     (medico_id,))
        cur = conn.execute("DELETE FROM medicos WHERE medico_id = ?",
                            (medico_id,))
        conn.commit()
        return cur.rowcount > 0


def reset_completo_bd() -> dict:
    """Vacia las 3 tablas por completo (medicos, evaluaciones, respuestas).
       Devuelve el conteo eliminado."""
    with get_conn() as conn:
        n_resp = conn.execute("SELECT COUNT(*) FROM respuestas_ventana").fetchone()[0]
        n_eval = conn.execute("SELECT COUNT(*) FROM evaluaciones").fetchone()[0]
        n_med = conn.execute("SELECT COUNT(*) FROM medicos").fetchone()[0]
        conn.execute("DELETE FROM respuestas_ventana")
        conn.execute("DELETE FROM evaluaciones")
        conn.execute("DELETE FROM medicos")
        # Reset autoincrement
        conn.execute("DELETE FROM sqlite_sequence WHERE name IN "
                     "('medicos', 'evaluaciones', 'respuestas_ventana')")
        conn.commit()
        return {"respuestas_ventana": n_resp,
                "evaluaciones": n_eval,
                "medicos": n_med}


# ---------------------------------------------------------------
# AGREGADOS PARA DASHBOARD ADMIN
# ---------------------------------------------------------------
def stats_globales() -> dict:
    with get_conn() as conn:
        n_medicos = conn.execute("SELECT COUNT(*) FROM medicos").fetchone()[0]
        n_evaluaciones = conn.execute("SELECT COUNT(*) FROM evaluaciones").fetchone()[0]
        n_respuestas = conn.execute("SELECT COUNT(*) FROM respuestas_ventana").fetchone()[0]
        n_signals = conn.execute(
            "SELECT COUNT(DISTINCT signal_id) FROM evaluaciones"
        ).fetchone()[0]
        avg_time = conn.execute(
            "SELECT ROUND(AVG(duracion_segundos), 1) FROM evaluaciones"
        ).fetchone()[0]
        avg_lik = conn.execute("""
            SELECT ROUND(AVG(likert_exactitud), 2)  as exactitud,
                   ROUND(AVG(likert_coherencia), 2) as coherencia,
                   ROUND(AVG(likert_utilidad), 2)   as utilidad
            FROM evaluaciones
        """).fetchone()
        return {
            "n_medicos": n_medicos,
            "n_evaluaciones": n_evaluaciones,
            "n_respuestas_ventana": n_respuestas,
            "n_signals_evaluadas": n_signals,
            "tiempo_promedio_seg": avg_time,
            "likert_promedio_exactitud":  avg_lik["exactitud"],
            "likert_promedio_coherencia": avg_lik["coherencia"],
            "likert_promedio_utilidad":   avg_lik["utilidad"],
        }


def stats_por_escala() -> dict:
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
            # Asegurar que las 3 llaves existan
            for k in ["verde", "amarillo", "rojo"]:
                out[escala].setdefault(k, 0.0)
    return out


def stats_por_clase(escala: str) -> dict:
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


def stats_semaforo_nota_global() -> dict:
    """Distribucion de veredictos globales sobre la nota (correcta/parcial/erronea)."""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT nota_semaforo_global, COUNT(*) c
            FROM evaluaciones
            WHERE nota_semaforo_global IS NOT NULL
            GROUP BY nota_semaforo_global
        """).fetchall()
        total = sum(r["c"] for r in rows) or 1
        out = {r["nota_semaforo_global"]: round(r["c"] / total * 100, 2)
               for r in rows}
        for k in ["verde", "amarillo", "rojo"]:
            out.setdefault(k, 0.0)
        return out


def stats_evaluaciones_por_senal() -> list[dict]:
    """Cuantas veces se ha evaluado cada senal."""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT signal_id, COUNT(*) n,
                   ROUND(AVG(likert_exactitud), 2)  as prom_exactitud,
                   ROUND(AVG(likert_coherencia), 2) as prom_coherencia,
                   ROUND(AVG(likert_utilidad), 2)   as prom_utilidad
            FROM evaluaciones
            GROUP BY signal_id
            ORDER BY signal_id
        """).fetchall()
        return [dict(r) for r in rows]


def stats_marcaciones_nota() -> dict:
    """Promedios de porcentajes de subrayado en las notas."""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT marcaciones_nota
            FROM evaluaciones
            WHERE marcaciones_nota IS NOT NULL
        """).fetchall()
    if not rows:
        return {"pct_verde_prom": 0, "pct_amarillo_prom": 0,
                "pct_rojo_prom": 0, "pct_sin_marcar_prom": 0, "n": 0}
    sums = {"pct_verde": 0, "pct_amarillo": 0, "pct_rojo": 0, "pct_sin_marcar": 0}
    n = 0
    for r in rows:
        try:
            m = json.loads(r["marcaciones_nota"])
            for k in sums:
                sums[k] += m.get(k, 0)
            n += 1
        except Exception:
            continue
    if n == 0:
        return {"pct_verde_prom": 0, "pct_amarillo_prom": 0,
                "pct_rojo_prom": 0, "pct_sin_marcar_prom": 0, "n": 0}
    return {
        "pct_verde_prom":       round(sums["pct_verde"] / n, 2),
        "pct_amarillo_prom":    round(sums["pct_amarillo"] / n, 2),
        "pct_rojo_prom":        round(sums["pct_rojo"] / n, 2),
        "pct_sin_marcar_prom":  round(sums["pct_sin_marcar"] / n, 2),
        "n": n,
    }


# ---------------------------------------------------------------
# KAPPA INTER-EVALUADOR
# ---------------------------------------------------------------
def _cohens_kappa(rater_a: list, rater_b: list) -> float | None:
    """
    Kappa de Cohen para dos listas de categorias del mismo largo.
    Devuelve None si no hay suficientes datos.
    """
    if len(rater_a) != len(rater_b) or len(rater_a) < 2:
        return None
    # Categorias unicas
    cats = sorted(set(rater_a) | set(rater_b))
    if len(cats) < 2:
        # Solo una categoria => acuerdo perfecto por definicion
        return 1.0
    n = len(rater_a)
    # Acuerdo observado
    p_o = sum(1 for a, b in zip(rater_a, rater_b) if a == b) / n
    # Acuerdo esperado
    p_e = 0.0
    for c in cats:
        pa = rater_a.count(c) / n
        pb = rater_b.count(c) / n
        p_e += pa * pb
    if p_e >= 1.0:
        return 1.0 if p_o == 1.0 else 0.0
    return round((p_o - p_e) / (1 - p_e), 4)


def kappa_inter_evaluador() -> dict:
    """
    Calcula kappa de Cohen entre todos los pares de cardiologos
    que hayan evaluado señales en comun, por escala (beat/rhythm)
    y sobre el veredicto global de la nota.

    Devuelve:
      {
        "pares": [
          {"med_a_id": 1, "med_a_nombre": "...",
           "med_b_id": 2, "med_b_nombre": "...",
           "senales_comunes": ["sig_001", "sig_003"],
           "kappa_beat": 0.65, "n_beat": 120,
           "kappa_rhythm": 0.72, "n_rhythm": 30,
           "kappa_nota_global": 0.83, "n_nota": 2}
        ],
        "resumen": {
          "prom_kappa_beat": 0.68,
          "prom_kappa_rhythm": 0.75,
          "prom_kappa_nota": 0.80,
          "n_pares": 10
        }
      }
    """
    with get_conn() as conn:
        # Todos los medicos con al menos 1 evaluacion
        medicos = conn.execute("""
            SELECT DISTINCT m.medico_id, m.nombre_completo
            FROM medicos m
            JOIN evaluaciones e ON e.medico_id = m.medico_id
            ORDER BY m.medico_id
        """).fetchall()
        medicos = [dict(m) for m in medicos]

        pares = []
        for i in range(len(medicos)):
            for j in range(i + 1, len(medicos)):
                med_a, med_b = medicos[i], medicos[j]
                # Senales comunes evaluadas por ambos
                signals_a = set(r[0] for r in conn.execute(
                    "SELECT signal_id FROM evaluaciones WHERE medico_id = ?",
                    (med_a["medico_id"],)
                ).fetchall())
                signals_b = set(r[0] for r in conn.execute(
                    "SELECT signal_id FROM evaluaciones WHERE medico_id = ?",
                    (med_b["medico_id"],)
                ).fetchall())
                comunes = sorted(signals_a & signals_b)
                if not comunes:
                    continue

                # Para cada señal comun: obtener las respuestas de ambos
                beat_a, beat_b = [], []
                rhythm_a, rhythm_b = [], []
                nota_a, nota_b = [], []
                for sig_id in comunes:
                    # Evaluaciones mas recientes de cada uno para esta señal
                    eval_a = conn.execute("""
                        SELECT evaluacion_id, nota_semaforo_global
                        FROM evaluaciones
                        WHERE medico_id = ? AND signal_id = ?
                        ORDER BY fecha_evaluacion DESC LIMIT 1
                    """, (med_a["medico_id"], sig_id)).fetchone()
                    eval_b = conn.execute("""
                        SELECT evaluacion_id, nota_semaforo_global
                        FROM evaluaciones
                        WHERE medico_id = ? AND signal_id = ?
                        ORDER BY fecha_evaluacion DESC LIMIT 1
                    """, (med_b["medico_id"], sig_id)).fetchone()
                    if not eval_a or not eval_b:
                        continue
                    # Nota global
                    if eval_a["nota_semaforo_global"] and eval_b["nota_semaforo_global"]:
                        nota_a.append(eval_a["nota_semaforo_global"])
                        nota_b.append(eval_b["nota_semaforo_global"])
                    # Respuestas ventana emparejadas por (escala, window_index)
                    resp_a = {(r["escala"], r["window_index"]): r["semaforo"]
                              for r in conn.execute(
                                  "SELECT * FROM respuestas_ventana WHERE evaluacion_id = ?",
                                  (eval_a["evaluacion_id"],))}
                    resp_b = {(r["escala"], r["window_index"]): r["semaforo"]
                              for r in conn.execute(
                                  "SELECT * FROM respuestas_ventana WHERE evaluacion_id = ?",
                                  (eval_b["evaluacion_id"],))}
                    keys_comunes = set(resp_a) & set(resp_b)
                    for k in keys_comunes:
                        escala = k[0]
                        if escala == "beat":
                            beat_a.append(resp_a[k])
                            beat_b.append(resp_b[k])
                        elif escala == "rhythm":
                            rhythm_a.append(resp_a[k])
                            rhythm_b.append(resp_b[k])

                pares.append({
                    "med_a_id":       med_a["medico_id"],
                    "med_a_nombre":   med_a["nombre_completo"],
                    "med_b_id":       med_b["medico_id"],
                    "med_b_nombre":   med_b["nombre_completo"],
                    "senales_comunes": comunes,
                    "kappa_beat":     _cohens_kappa(beat_a, beat_b),
                    "n_beat":         len(beat_a),
                    "kappa_rhythm":   _cohens_kappa(rhythm_a, rhythm_b),
                    "n_rhythm":       len(rhythm_a),
                    "kappa_nota_global": _cohens_kappa(nota_a, nota_b),
                    "n_nota":         len(nota_a),
                })

        # Resumen
        def avg(vals):
            vals = [v for v in vals if v is not None]
            return round(sum(vals) / len(vals), 3) if vals else None

        resumen = {
            "prom_kappa_beat":   avg([p["kappa_beat"] for p in pares]),
            "prom_kappa_rhythm": avg([p["kappa_rhythm"] for p in pares]),
            "prom_kappa_nota":   avg([p["kappa_nota_global"] for p in pares]),
            "n_pares": len(pares),
        }
        return {"pares": pares, "resumen": resumen}


# ---------------------------------------------------------------
# EXPORTACION CSV
# ---------------------------------------------------------------
def _rows_to_csv(rows: list[dict], columns: list[str]) -> str:
    """Convierte lista de dicts a CSV string."""
    buf = io.StringIO()
    writer = csv.writer(buf, quoting=csv.QUOTE_MINIMAL)
    writer.writerow(columns)
    for row in rows:
        writer.writerow([row.get(c, "") for c in columns])
    return buf.getvalue()


def exportar_medicos_csv() -> str:
    columns = ["medico_id", "nombre_completo", "edad", "sexo", "institucion",
               "especialidad", "sub_especialidad", "anos_experiencia",
               "anos_experiencia_ecg", "auto_eval_habilidad", "email",
               "fecha_registro", "n_evaluaciones",
               "prom_exactitud", "prom_coherencia", "prom_utilidad",
               "prom_duracion"]
    return _rows_to_csv(listar_medicos(), columns)


def exportar_evaluaciones_csv() -> str:
    rows = listar_evaluaciones()
    # Enriquecer con porcentajes de marcaciones
    for r in rows:
        if r.get("marcaciones_nota"):
            try:
                m = json.loads(r["marcaciones_nota"])
                r["pct_verde"] = m.get("pct_verde")
                r["pct_amarillo"] = m.get("pct_amarillo")
                r["pct_rojo"] = m.get("pct_rojo")
                r["pct_sin_marcar"] = m.get("pct_sin_marcar")
            except Exception:
                pass
    columns = ["evaluacion_id", "medico_id", "nombre_completo", "especialidad",
               "signal_id", "llm_backend", "llm_model_id",
               "nota_semaforo_global",
               "likert_exactitud", "likert_coherencia", "likert_utilidad",
               "pct_verde", "pct_amarillo", "pct_rojo", "pct_sin_marcar",
               "comentarios_libres", "duracion_segundos", "fecha_evaluacion"]
    return _rows_to_csv(rows, columns)


def exportar_respuestas_ventana_csv() -> str:
    with get_conn() as conn:
        rows = [dict(r) for r in conn.execute("""
            SELECT rv.*, e.medico_id, e.signal_id,
                   m.nombre_completo
            FROM respuestas_ventana rv
            JOIN evaluaciones e ON e.evaluacion_id = rv.evaluacion_id
            JOIN medicos m ON m.medico_id = e.medico_id
            ORDER BY rv.evaluacion_id, rv.escala, rv.window_index
        """).fetchall()]
    columns = ["respuesta_id", "evaluacion_id", "medico_id", "nombre_completo",
               "signal_id", "escala", "window_index",
               "prediccion_modelo", "semaforo", "confidence_gap"]
    return _rows_to_csv(rows, columns)
