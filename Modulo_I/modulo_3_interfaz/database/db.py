"""
Capa fina de acceso a la base SQLite.
Incluye:
  - Migracion idempotente que agrega las columnas nuevas si faltan
  - Funciones para la comparativa admin con ground truth
  - Todas las funciones anteriores del panel administrativo
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

# Ruta al archivo de etiquetas ground truth (config editable)
LABELS_PATH = ROOT.parent / "data" / "signal_labels.json"


# ---------------------------------------------------------------
# Bootstrap con migracion idempotente
# ---------------------------------------------------------------
def _add_column_if_missing(conn, table: str, column: str, definition: str):
    """Agrega columna si no existe. Idempotente."""
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def init_db():
    """Crea las tablas si no existen y aplica migraciones idempotentes."""
    with open(SCHEMA_PATH) as f:
        schema = f.read()
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(schema)

        # -- Migraciones idempotentes en tabla medicos --
        _add_column_if_missing(conn, "medicos", "acepto_terminos", "INTEGER DEFAULT 0")
        _add_column_if_missing(conn, "medicos", "fecha_aceptacion", "TEXT")

        # -- Migraciones idempotentes en tabla evaluaciones --
        _add_column_if_missing(conn, "evaluaciones", "marcaciones_nota", "TEXT")
        _add_column_if_missing(conn, "evaluaciones", "nota_clasificador_texto", "TEXT")
        _add_column_if_missing(conn, "evaluaciones", "nota_clasificador_semaforo", "TEXT")
        _add_column_if_missing(conn, "evaluaciones", "nota_clasificador_marcaciones", "TEXT")
        _add_column_if_missing(conn, "evaluaciones", "comentarios_clasificador", "TEXT")
        _add_column_if_missing(conn, "evaluaciones", "comentarios_nota", "TEXT")

        # Renombrar comentarios_libres a comentarios_nota si existe la vieja
        cols_eval = [r[1] for r in conn.execute("PRAGMA table_info(evaluaciones)")]
        if "comentarios_libres" in cols_eval and "comentarios_nota" in cols_eval:
            # Migra datos si ambos existen: pasa contenido a comentarios_nota
            conn.execute("""
                UPDATE evaluaciones
                SET comentarios_nota = COALESCE(comentarios_nota, comentarios_libres)
                WHERE comentarios_nota IS NULL AND comentarios_libres IS NOT NULL
            """)
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
# Ground truth loader
# ---------------------------------------------------------------
def load_signal_labels() -> dict:
    """Devuelve el dict con las etiquetas reales de las senales."""
    if not LABELS_PATH.exists():
        return {}
    try:
        with open(LABELS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def get_signal_label(signal_id: str) -> dict:
    """Devuelve el ground truth de una senal, o dict vacio si no existe."""
    labels = load_signal_labels()
    entry = labels.get(signal_id, {})
    if isinstance(entry, dict):
        return entry
    return {}


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
                auto_eval_habilidad, email,
                acepto_terminos, fecha_aceptacion
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            payload.get("nombre_completo"),
            payload.get("edad"),                  # ahora es rango (string)
            payload.get("sexo"),
            payload.get("institucion"),
            payload.get("especialidad"),
            payload.get("sub_especialidad"),
            payload.get("anos_experiencia"),
            payload.get("anos_experiencia_ecg"),
            payload.get("auto_eval_habilidad"),
            payload.get("email"),
            1 if payload.get("acepto_terminos") else 0,
            payload.get("fecha_aceptacion"),
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
    marc_clf = payload.get("nota_clasificador_marcaciones")
    marc_clf_json = json.dumps(marc_clf, ensure_ascii=False) if marc_clf else None

    with get_conn() as conn:
        cur = conn.execute("""
            INSERT INTO evaluaciones (
                medico_id, signal_id, llm_backend, llm_model_id,
                nota_clinica_texto, nota_semaforo_global,
                likert_exactitud, likert_coherencia, likert_utilidad,
                marcaciones_nota, comentarios_nota,
                nota_clasificador_texto, nota_clasificador_semaforo,
                nota_clasificador_marcaciones, comentarios_clasificador,
                duracion_segundos
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            payload["medico_id"], payload["signal_id"],
            payload["llm_backend"], payload.get("llm_model_id"),
            payload["nota_clinica_texto"], payload["nota_semaforo_global"],
            payload["likert_exactitud"], payload["likert_coherencia"],
            payload["likert_utilidad"], marc_json,
            payload.get("comentarios_nota", ""),
            payload.get("nota_clasificador_texto"),
            payload.get("nota_clasificador_semaforo"),
            marc_clf_json,
            payload.get("comentarios_clasificador", ""),
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
        # Parsear marcaciones de nota clinica
        if d.get("marcaciones_nota"):
            try:
                d["marcaciones_nota_parsed"] = json.loads(d["marcaciones_nota"])
            except Exception:
                d["marcaciones_nota_parsed"] = None
        else:
            d["marcaciones_nota_parsed"] = None
        # Parsear marcaciones de nota del clasificador
        if d.get("nota_clasificador_marcaciones"):
            try:
                d["marcaciones_clasificador_parsed"] = json.loads(d["nota_clasificador_marcaciones"])
            except Exception:
                d["marcaciones_clasificador_parsed"] = None
        else:
            d["marcaciones_clasificador_parsed"] = None
        return d


# ---------------------------------------------------------------
# COMPARATIVA POR SEÑAL (nueva vista admin)
# ---------------------------------------------------------------
def comparativa_signal(signal_id: str) -> dict:
    """
    Devuelve para una senal:
      - Ground truth de las bases de datos
      - Distribucion de predicciones del clasificador por ventana
      - Evaluaciones agregadas de todos los medicos por ventana
      - Metricas de concordancia derivadas
    """
    ground_truth = get_signal_label(signal_id)

    with get_conn() as conn:
        # Todas las evaluaciones de esta senal, con sus datos de medico
        evaluaciones = [dict(r) for r in conn.execute("""
            SELECT e.evaluacion_id, e.medico_id, m.nombre_completo,
                   e.nota_semaforo_global, e.nota_clasificador_semaforo,
                   e.likert_exactitud, e.likert_coherencia, e.likert_utilidad,
                   e.marcaciones_nota, e.nota_clasificador_marcaciones,
                   e.comentarios_nota, e.comentarios_clasificador,
                   e.nota_clinica_texto, e.nota_clasificador_texto,
                   e.fecha_evaluacion
            FROM evaluaciones e
            JOIN medicos m ON m.medico_id = e.medico_id
            WHERE e.signal_id = ?
            ORDER BY e.fecha_evaluacion ASC
        """, (signal_id,)).fetchall()]

        # Parsear marcaciones de cada evaluacion
        for ev in evaluaciones:
            for campo in ["marcaciones_nota", "nota_clasificador_marcaciones"]:
                if ev.get(campo):
                    try:
                        ev[campo + "_parsed"] = json.loads(ev[campo])
                    except Exception:
                        ev[campo + "_parsed"] = None
                else:
                    ev[campo + "_parsed"] = None

        # Todas las respuestas por ventana, agrupadas por (escala, window_index)
        rows = [dict(r) for r in conn.execute("""
            SELECT rv.*, e.medico_id, m.nombre_completo
            FROM respuestas_ventana rv
            JOIN evaluaciones e ON e.evaluacion_id = rv.evaluacion_id
            JOIN medicos m ON m.medico_id = e.medico_id
            WHERE e.signal_id = ?
            ORDER BY rv.escala, rv.window_index, e.medico_id
        """, (signal_id,)).fetchall()]

    # Agregar por ventana
    ventanas = defaultdict(lambda: {
        "prediccion_modelo": None,
        "confidence_gap": None,
        "veredictos_medicos": [],
        "n_verde": 0, "n_amarillo": 0, "n_rojo": 0,
    })
    for r in rows:
        key = (r["escala"], r["window_index"])
        ventanas[key]["prediccion_modelo"] = r["prediccion_modelo"]
        ventanas[key]["confidence_gap"] = r.get("confidence_gap")
        ventanas[key]["veredictos_medicos"].append({
            "medico_id": r["medico_id"],
            "nombre": r["nombre_completo"],
            "semaforo": r["semaforo"],
        })
        ventanas[key][f"n_{r['semaforo']}"] += 1

    beat_windows = []
    rhythm_windows = []
    for (escala, idx), data in sorted(ventanas.items()):
        entry = {
            "window_index": idx,
            "prediccion_modelo": data["prediccion_modelo"],
            "confidence_gap": data["confidence_gap"],
            "n_verde": data["n_verde"],
            "n_amarillo": data["n_amarillo"],
            "n_rojo": data["n_rojo"],
            "n_total": data["n_verde"] + data["n_amarillo"] + data["n_rojo"],
            "veredictos_medicos": data["veredictos_medicos"],
        }
        if entry["n_total"] > 0:
            entry["pct_verde"] = round(data["n_verde"] / entry["n_total"] * 100, 2)
            entry["pct_amarillo"] = round(data["n_amarillo"] / entry["n_total"] * 100, 2)
            entry["pct_rojo"] = round(data["n_rojo"] / entry["n_total"] * 100, 2)
        else:
            entry["pct_verde"] = entry["pct_amarillo"] = entry["pct_rojo"] = 0
        if escala == "beat":
            beat_windows.append(entry)
        else:
            rhythm_windows.append(entry)

    # Metricas agregadas: concordancia global medicos <-> clasificador
    def agg_metrics(windows):
        if not windows:
            return {}
        total_v = sum(w["n_verde"] for w in windows)
        total_a = sum(w["n_amarillo"] for w in windows)
        total_r = sum(w["n_rojo"] for w in windows)
        total_n = total_v + total_a + total_r
        if total_n == 0:
            return {}
        return {
            "n_veredictos": total_n,
            "accuracy_estricta":  round(total_v / total_n * 100, 2),
            "accuracy_permisiva": round((total_v + total_a) / total_n * 100, 2),
            "error_rate":         round(total_r / total_n * 100, 2),
            "pct_verde":          round(total_v / total_n * 100, 2),
            "pct_amarillo":       round(total_a / total_n * 100, 2),
            "pct_rojo":           round(total_r / total_n * 100, 2),
        }

    # Distribucion de clases predichas
    pred_beat_counts = defaultdict(int)
    for w in beat_windows:
        pred_beat_counts[w["prediccion_modelo"]] += 1
    pred_rhythm_counts = defaultdict(int)
    for w in rhythm_windows:
        pred_rhythm_counts[w["prediccion_modelo"]] += 1

    return {
        "signal_id": signal_id,
        "ground_truth": ground_truth,
        "n_evaluaciones": len(evaluaciones),
        "evaluaciones": evaluaciones,
        "beat_windows": beat_windows,
        "rhythm_windows": rhythm_windows,
        "metricas_beat": agg_metrics(beat_windows),
        "metricas_rhythm": agg_metrics(rhythm_windows),
        "distribucion_predicciones_beat": dict(pred_beat_counts),
        "distribucion_predicciones_rhythm": dict(pred_rhythm_counts),
    }


def listar_signals_disponibles() -> list[str]:
    """Devuelve las signal_id que tienen al menos una evaluacion."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT DISTINCT signal_id FROM evaluaciones ORDER BY signal_id"
        ).fetchall()
    return [r[0] for r in rows]


# ---------------------------------------------------------------
# BORRADO
# ---------------------------------------------------------------
def eliminar_evaluacion(eval_id: int) -> bool:
    with get_conn() as conn:
        conn.execute("DELETE FROM respuestas_ventana WHERE evaluacion_id = ?", (eval_id,))
        cur = conn.execute("DELETE FROM evaluaciones WHERE evaluacion_id = ?", (eval_id,))
        conn.commit()
        return cur.rowcount > 0


def eliminar_medico(medico_id: int) -> bool:
    with get_conn() as conn:
        eval_ids = [r[0] for r in conn.execute(
            "SELECT evaluacion_id FROM evaluaciones WHERE medico_id = ?", (medico_id,)
        ).fetchall()]
        for eid in eval_ids:
            conn.execute("DELETE FROM respuestas_ventana WHERE evaluacion_id = ?", (eid,))
        conn.execute("DELETE FROM evaluaciones WHERE medico_id = ?", (medico_id,))
        cur = conn.execute("DELETE FROM medicos WHERE medico_id = ?", (medico_id,))
        conn.commit()
        return cur.rowcount > 0


def reset_completo_bd() -> dict:
    with get_conn() as conn:
        n_resp = conn.execute("SELECT COUNT(*) FROM respuestas_ventana").fetchone()[0]
        n_eval = conn.execute("SELECT COUNT(*) FROM evaluaciones").fetchone()[0]
        n_med = conn.execute("SELECT COUNT(*) FROM medicos").fetchone()[0]
        conn.execute("DELETE FROM respuestas_ventana")
        conn.execute("DELETE FROM evaluaciones")
        conn.execute("DELETE FROM medicos")
        conn.execute("DELETE FROM sqlite_sequence WHERE name IN "
                     "('medicos', 'evaluaciones', 'respuestas_ventana')")
        conn.commit()
        return {"respuestas_ventana": n_resp,
                "evaluaciones": n_eval, "medicos": n_med}


# ---------------------------------------------------------------
# AGREGADOS DASHBOARD (sin cambios significativos)
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
            out[escala] = {r["semaforo"]: round(r["c"] / total * 100, 2) for r in rows}
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
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT nota_semaforo_global, COUNT(*) c
            FROM evaluaciones
            WHERE nota_semaforo_global IS NOT NULL
            GROUP BY nota_semaforo_global
        """).fetchall()
        total = sum(r["c"] for r in rows) or 1
        out = {r["nota_semaforo_global"]: round(r["c"] / total * 100, 2) for r in rows}
        for k in ["verde", "amarillo", "rojo"]:
            out.setdefault(k, 0.0)
        return out


def stats_evaluaciones_por_senal() -> list[dict]:
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
# PRECISION MEDICO
# ---------------------------------------------------------------
def precision_medico_vs_clasificador(medico_id: int) -> dict:
    with get_conn() as conn:
        rows = [dict(r) for r in conn.execute("""
            SELECT rv.escala, rv.prediccion_modelo, rv.semaforo, COUNT(*) as n
            FROM respuestas_ventana rv
            JOIN evaluaciones e ON e.evaluacion_id = rv.evaluacion_id
            WHERE e.medico_id = ?
            GROUP BY rv.escala, rv.prediccion_modelo, rv.semaforo
        """, (medico_id,)).fetchall()]

    def build_metrics(subset):
        by_class = defaultdict(lambda: {"verde": 0, "amarillo": 0, "rojo": 0})
        for r in subset:
            by_class[r["prediccion_modelo"]][r["semaforo"]] += r["n"]
        by_class_metrics = {}
        for cls, counts in by_class.items():
            total = counts["verde"] + counts["amarillo"] + counts["rojo"]
            if total == 0:
                continue
            by_class_metrics[cls] = {
                "n": total, "verde": counts["verde"],
                "amarillo": counts["amarillo"], "rojo": counts["rojo"],
                "accuracy_estricta":   round(counts["verde"] / total * 100, 2),
                "accuracy_permisiva":  round((counts["verde"] + counts["amarillo"]) / total * 100, 2),
                "error_rate":          round(counts["rojo"] / total * 100, 2),
            }
        total_v = sum(counts["verde"] for counts in by_class.values())
        total_a = sum(counts["amarillo"] for counts in by_class.values())
        total_r = sum(counts["rojo"] for counts in by_class.values())
        total_n = total_v + total_a + total_r
        global_metrics = {
            "n": total_n, "verde": total_v, "amarillo": total_a, "rojo": total_r,
            "accuracy_estricta":  round(total_v / total_n * 100, 2) if total_n else 0,
            "accuracy_permisiva": round((total_v + total_a) / total_n * 100, 2) if total_n else 0,
            "error_rate":         round(total_r / total_n * 100, 2) if total_n else 0,
        }
        return {"global": global_metrics, "por_clase": by_class_metrics}

    return {
        "beat":   build_metrics([r for r in rows if r["escala"] == "beat"]),
        "rhythm": build_metrics([r for r in rows if r["escala"] == "rhythm"]),
    }


def metricas_notas_medico(medico_id: int) -> dict:
    with get_conn() as conn:
        rows = [dict(r) for r in conn.execute("""
            SELECT nota_semaforo_global, likert_exactitud,
                   likert_coherencia, likert_utilidad, marcaciones_nota
            FROM evaluaciones
            WHERE medico_id = ?
        """, (medico_id,)).fetchall()]
    n = len(rows)
    if n == 0:
        return {"n": 0, "veredicto": {}, "likert": {},
                "subrayado": {}, "interpretacion": "Sin evaluaciones."}
    veredicto_counts = {"verde": 0, "amarillo": 0, "rojo": 0}
    for r in rows:
        if r["nota_semaforo_global"]:
            veredicto_counts[r["nota_semaforo_global"]] += 1
    veredicto_pct = {k: round(v / n * 100, 2) for k, v in veredicto_counts.items()}

    def avg(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return round(sum(vals) / len(vals), 2) if vals else None

    likert = {
        "exactitud":  avg("likert_exactitud"),
        "coherencia": avg("likert_coherencia"),
        "utilidad":   avg("likert_utilidad"),
    }
    sums = {"pct_verde": 0.0, "pct_amarillo": 0.0,
            "pct_rojo": 0.0, "pct_sin_marcar": 0.0}
    n_marc = 0
    for r in rows:
        if not r["marcaciones_nota"]:
            continue
        try:
            m = json.loads(r["marcaciones_nota"])
            for k in sums:
                sums[k] += m.get(k, 0)
            n_marc += 1
        except Exception:
            continue
    subrayado = {}
    if n_marc > 0:
        subrayado = {k + "_prom": round(v / n_marc, 2) for k, v in sums.items()}
        subrayado["n_evaluaciones_con_marcas"] = n_marc
    else:
        subrayado = {"n_evaluaciones_con_marcas": 0}
    if veredicto_pct["verde"] >= 60:
        interp = "El medico considera que las notas del LLM son en general de buena calidad."
    elif veredicto_pct["rojo"] >= 40:
        interp = "El medico ha identificado problemas serios en varias notas."
    else:
        interp = "Percepcion mixta de las notas del LLM (verde/amarillo/rojo relativamente equilibrados)."
    return {
        "n": n,
        "veredicto": {"conteos": veredicto_counts, "porcentajes": veredicto_pct},
        "likert": likert, "subrayado": subrayado, "interpretacion": interp,
    }


# ---------------------------------------------------------------
# KAPPA INTER-EVALUADOR (sin cambios)
# ---------------------------------------------------------------
def _cohens_kappa(rater_a: list, rater_b: list) -> float | None:
    if len(rater_a) != len(rater_b) or len(rater_a) < 2:
        return None
    cats = sorted(set(rater_a) | set(rater_b))
    if len(cats) < 2:
        return 1.0
    n = len(rater_a)
    p_o = sum(1 for a, b in zip(rater_a, rater_b) if a == b) / n
    p_e = 0.0
    for c in cats:
        pa = rater_a.count(c) / n
        pb = rater_b.count(c) / n
        p_e += pa * pb
    if p_e >= 1.0:
        return 1.0 if p_o == 1.0 else 0.0
    return round((p_o - p_e) / (1 - p_e), 4)


def kappa_inter_evaluador() -> dict:
    with get_conn() as conn:
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
                signals_a = set(r[0] for r in conn.execute(
                    "SELECT signal_id FROM evaluaciones WHERE medico_id = ?",
                    (med_a["medico_id"],)).fetchall())
                signals_b = set(r[0] for r in conn.execute(
                    "SELECT signal_id FROM evaluaciones WHERE medico_id = ?",
                    (med_b["medico_id"],)).fetchall())
                comunes = sorted(signals_a & signals_b)
                if not comunes:
                    continue
                beat_a, beat_b = [], []
                rhythm_a, rhythm_b = [], []
                nota_a, nota_b = [], []
                for sig_id in comunes:
                    eval_a = conn.execute("""
                        SELECT evaluacion_id, nota_semaforo_global
                        FROM evaluaciones WHERE medico_id = ? AND signal_id = ?
                        ORDER BY fecha_evaluacion DESC LIMIT 1
                    """, (med_a["medico_id"], sig_id)).fetchone()
                    eval_b = conn.execute("""
                        SELECT evaluacion_id, nota_semaforo_global
                        FROM evaluaciones WHERE medico_id = ? AND signal_id = ?
                        ORDER BY fecha_evaluacion DESC LIMIT 1
                    """, (med_b["medico_id"], sig_id)).fetchone()
                    if not eval_a or not eval_b:
                        continue
                    if eval_a["nota_semaforo_global"] and eval_b["nota_semaforo_global"]:
                        nota_a.append(eval_a["nota_semaforo_global"])
                        nota_b.append(eval_b["nota_semaforo_global"])
                    resp_a = {(r["escala"], r["window_index"]): r["semaforo"]
                              for r in conn.execute(
                                  "SELECT * FROM respuestas_ventana WHERE evaluacion_id = ?",
                                  (eval_a["evaluacion_id"],))}
                    resp_b = {(r["escala"], r["window_index"]): r["semaforo"]
                              for r in conn.execute(
                                  "SELECT * FROM respuestas_ventana WHERE evaluacion_id = ?",
                                  (eval_b["evaluacion_id"],))}
                    for k in set(resp_a) & set(resp_b):
                        escala = k[0]
                        if escala == "beat":
                            beat_a.append(resp_a[k]); beat_b.append(resp_b[k])
                        elif escala == "rhythm":
                            rhythm_a.append(resp_a[k]); rhythm_b.append(resp_b[k])
                pares.append({
                    "med_a_id": med_a["medico_id"],
                    "med_a_nombre": med_a["nombre_completo"],
                    "med_b_id": med_b["medico_id"],
                    "med_b_nombre": med_b["nombre_completo"],
                    "senales_comunes": comunes,
                    "kappa_beat": _cohens_kappa(beat_a, beat_b),
                    "n_beat": len(beat_a),
                    "kappa_rhythm": _cohens_kappa(rhythm_a, rhythm_b),
                    "n_rhythm": len(rhythm_a),
                    "kappa_nota_global": _cohens_kappa(nota_a, nota_b),
                    "n_nota": len(nota_a),
                })
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
               "acepto_terminos", "fecha_aceptacion",
               "fecha_registro", "n_evaluaciones",
               "prom_exactitud", "prom_coherencia", "prom_utilidad",
               "prom_duracion"]
    return _rows_to_csv(listar_medicos(), columns)


def exportar_evaluaciones_csv() -> str:
    rows = listar_evaluaciones()
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
               "nota_clasificador_semaforo",
               "likert_exactitud", "likert_coherencia", "likert_utilidad",
               "pct_verde", "pct_amarillo", "pct_rojo", "pct_sin_marcar",
               "comentarios_nota", "comentarios_clasificador",
               "duracion_segundos", "fecha_evaluacion"]
    return _rows_to_csv(rows, columns)


def exportar_respuestas_ventana_csv() -> str:
    with get_conn() as conn:
        rows = [dict(r) for r in conn.execute("""
            SELECT rv.*, e.medico_id, e.signal_id, m.nombre_completo
            FROM respuestas_ventana rv
            JOIN evaluaciones e ON e.evaluacion_id = rv.evaluacion_id
            JOIN medicos m ON m.medico_id = e.medico_id
            ORDER BY rv.evaluacion_id, rv.escala, rv.window_index
        """).fetchall()]
    columns = ["respuesta_id", "evaluacion_id", "medico_id", "nombre_completo",
               "signal_id", "escala", "window_index",
               "prediccion_modelo", "semaforo", "confidence_gap"]
    return _rows_to_csv(rows, columns)
