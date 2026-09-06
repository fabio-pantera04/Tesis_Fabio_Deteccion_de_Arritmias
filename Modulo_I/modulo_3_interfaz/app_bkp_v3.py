"""
Modulo III - Plataforma de validacion clinica human-in-the-loop.

Rutas MEDICO:
  /                       -> redirige a /registro (si no logueado) o /seleccion
  /registro               -> formulario cardiologo
  /seleccion              -> lista de 5 senales
  /evaluar/<signal_id>    -> evaluacion completa
  /mis_evaluaciones       -> historial PROPIO del cardiologo (no ve otros)
  /api/llm_note/<sid>     -> nota LLM (persistida > cache > backend)
  /api/guardar            -> POST: persiste evaluacion
  /logout                 -> cierra sesion cardiologo

Rutas ADMIN (protegidas por contrasena):
  /admin/login, /admin/logout, /admin/dashboard, /admin/medicos,
  /admin/medico/<mid>, /admin/evaluaciones, /admin/evaluacion/<eid>,
  /admin/inter-evaluador, /admin/exportar, /admin/exportar/<tipo>,
  /admin/eliminar_evaluacion/<eid>, /admin/eliminar_medico/<mid>,
  /admin/reset_bd, /admin/api/dashboard_data, /admin/api/kappa_data,
  /admin/api/medico_precision/<mid>  <-- nuevo endpoint

Rutas legacy protegidas por admin:
  /analytics -> ahora requiere @require_admin
"""
import io
import json
import os
import time
from datetime import datetime
from functools import wraps
from pathlib import Path

from flask import (
    Flask, render_template, request, redirect, url_for,
    session, jsonify, abort, send_file, Response,
)

import config
from database import db
from llm_backends import get_backend, build_prompt

app = Flask(__name__)
app.config["SECRET_KEY"] = config.SECRET_KEY

_NOTE_CACHE: dict[tuple, dict] = {}


@app.before_request
def ensure_db():
    if not hasattr(ensure_db, "_done"):
        db.init_db()
        ensure_db._done = True


# ---------------------------------------------------------------
# Utilidades: senales y notas LLM
# ---------------------------------------------------------------
def load_signal(signal_id: str) -> dict:
    path = config.SIGNALS_DIR / f"{signal_id}.json"
    if not path.exists():
        abort(404, description=f"Senal {signal_id} no encontrada.")
    with open(path) as f:
        return json.load(f)


def load_persisted_note(signal_id: str) -> dict | None:
    path = config.NOTES_DIR / f"nota_{signal_id}.json"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        app.logger.warning(f"Error leyendo nota {signal_id}: {e}")
        return None


def save_generated_note(signal_id: str, resp, signal_json: dict) -> None:
    registro = {
        "signal_id":         signal_id,
        "generated_at":      datetime.utcnow().isoformat() + "Z",
        "model":             resp.model_id,
        "prompt_version":    "v3",
        "latency_seconds":   round(resp.latency_seconds, 2),
        "signal_metadata":   signal_json["signal_metadata"],
        "aggregate_summary": signal_json["aggregate_summary"],
        "clinical_note":     resp.text,
    }
    path = config.NOTES_DIR / f"nota_{signal_id}.json"
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(registro, f, ensure_ascii=False, indent=2)
    except OSError as e:
        app.logger.warning(f"No se pudo persistir nota {signal_id}: {e}")


def list_signals() -> list[dict]:
    out = []
    for path in sorted(config.SIGNALS_DIR.glob("sig_*.json")):
        with open(path) as f:
            data = json.load(f)
        meta = data["signal_metadata"]
        sid = meta["signal_id"]
        note_ready = (config.NOTES_DIR / f"nota_{sid}.json").exists()
        out.append({
            "signal_id":            sid,
            "label":                meta["label"],
            "scenario_note":        meta["scenario_note"],
            "patient_age_estimate": meta["patient_age_estimate"],
            "patient_sex":          meta["patient_sex"],
            "duration_seconds":     meta["duration_seconds"],
            "note_ready":           note_ready,
        })
    return out


# ---------------------------------------------------------------
# Middlewares
# ---------------------------------------------------------------
def require_medico():
    return session.get("medico_id")


def require_admin(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not session.get("is_admin"):
            return redirect(url_for("admin_login"))
        return f(*args, **kwargs)
    return wrapper


def _safe_int(s, default=None):
    try:
        if s is None or str(s).strip() == "":
            return default
        return int(s)
    except (ValueError, TypeError):
        return default


# ===============================================================
# RUTAS MEDICO
# ===============================================================
@app.route("/")
def index():
    if session.get("medico_id"):
        return redirect(url_for("seleccion"))
    return redirect(url_for("registro"))


@app.route("/registro", methods=["GET", "POST"])
def registro():
    if request.method == "POST":
        especialidad_raw = (request.form.get("especialidad") or "").strip()
        especialidad_otra = (request.form.get("especialidad_otra") or "").strip()
        especialidad_final = especialidad_otra if especialidad_raw == "Otra" else especialidad_raw

        payload = {
            "nombre_completo": (request.form.get("nombre_completo") or "").strip(),
            "edad": _safe_int(request.form.get("edad")),
            "sexo": (request.form.get("sexo") or "").strip(),
            "institucion": (request.form.get("institucion") or "").strip(),
            "especialidad": especialidad_final,
            "sub_especialidad": (request.form.get("sub_especialidad") or "").strip() or None,
            "anos_experiencia": _safe_int(request.form.get("anos_experiencia")),
            "anos_experiencia_ecg": _safe_int(request.form.get("anos_experiencia_ecg")),
            "auto_eval_habilidad": _safe_int(request.form.get("auto_eval_habilidad")),
            "email": (request.form.get("email") or "").strip(),
        }
        required = {
            "nombre_completo": "Nombre completo",
            "edad": "Edad", "sexo": "Sexo", "institucion": "Institución",
            "especialidad": "Especialidad" if especialidad_raw != "Otra"
                            else "Especificación de 'Otra' especialidad",
            "anos_experiencia": "Años de experiencia clínica",
            "anos_experiencia_ecg": "Años de experiencia con ECG",
            "auto_eval_habilidad": "Auto-evaluación de habilidad",
            "email": "Correo de contacto",
        }
        missing = [label for key, label in required.items()
                   if payload[key] in (None, "", 0)]
        if payload["anos_experiencia"] == 0:
            missing = [m for m in missing if m != required["anos_experiencia"]]
        if payload["anos_experiencia_ecg"] == 0:
            missing = [m for m in missing if m != required["anos_experiencia_ecg"]]
        if missing:
            preserved = dict(request.form)
            return render_template("registro.html",
                error="Faltan campos obligatorios: " + ", ".join(missing) + ".",
                form=preserved), 400

        mid = db.crear_medico(payload)
        session["medico_id"] = mid
        session["medico_nombre"] = payload["nombre_completo"]
        return redirect(url_for("seleccion"))
    return render_template("registro.html", form={}, error=None)


@app.route("/seleccion")
def seleccion():
    if not require_medico():
        return redirect(url_for("registro"))
    return render_template("seleccion.html",
        senales=list_signals(),
        medico_nombre=session.get("medico_nombre", ""),
        backend_activo=config.LLM_BACKEND)


@app.route("/evaluar/<signal_id>")
def evaluar(signal_id):
    if not require_medico():
        return redirect(url_for("registro"))
    signal = load_signal(signal_id)
    all_ids = sorted([p.stem for p in config.SIGNALS_DIR.glob("sig_*.json")])
    try:
        signal_number = all_ids.index(signal_id) + 1
    except ValueError:
        signal_number = 1
    return render_template("evaluar.html",
        signal=signal,
        signal_number=signal_number,
        signal_json_str=json.dumps(signal),
        backend_activo=config.LLM_BACKEND,
        medico_nombre=session.get("medico_nombre", ""))


@app.route("/mis_evaluaciones")
def mis_evaluaciones():
    """Vista privada del cardiologo: solo SUS evaluaciones, no ve las de otros."""
    mid = require_medico()
    if not mid:
        return redirect(url_for("registro"))
    mis_evals = db.listar_evaluaciones(medico_id=mid)
    signals_evaluadas = set(e["signal_id"] for e in mis_evals)
    signals_disponibles = sorted([p.stem for p in config.SIGNALS_DIR.glob("sig_*.json")])
    pendientes = [s for s in signals_disponibles if s not in signals_evaluadas]
    return render_template("mis_evaluaciones.html",
        evaluaciones=mis_evals,
        signals_pendientes=pendientes,
        medico_nombre=session.get("medico_nombre", ""))


@app.route("/api/llm_note/<signal_id>")
def api_llm_note(signal_id):
    backend_name = request.args.get("backend", config.LLM_BACKEND)
    force_regen = request.args.get("force") == "1"
    cache_key = (signal_id, backend_name)

    if not force_regen and cache_key in _NOTE_CACHE:
        return jsonify(_NOTE_CACHE[cache_key])

    if not force_regen:
        persisted = load_persisted_note(signal_id)
        if persisted:
            payload = {
                "text":            persisted["clinical_note"],
                "backend_name":    persisted.get("model", backend_name),
                "model_id":        persisted.get("model", backend_name),
                "latency_seconds": persisted.get("latency_seconds", 0),
                "error":           None,
                "source":          "persisted",
                "generated_at":    persisted.get("generated_at"),
                "prompt_version":  persisted.get("prompt_version"),
            }
            _NOTE_CACHE[cache_key] = payload
            return jsonify(payload)

    signal = load_signal(signal_id)
    try:
        backend = get_backend(backend_name)
    except Exception as e:
        return jsonify({"error": str(e), "text": "",
                        "backend_name": backend_name}), 500

    resp = backend.generate(signal)
    payload = {
        "text": resp.text, "backend_name": resp.backend_name,
        "model_id": resp.model_id,
        "latency_seconds": round(resp.latency_seconds, 3),
        "error": resp.error, "source": "generated",
    }
    if not resp.error:
        _NOTE_CACHE[cache_key] = payload
        save_generated_note(signal_id, resp, signal)
    return jsonify(payload)


@app.route("/api/guardar", methods=["POST"])
def api_guardar():
    mid = require_medico()
    if not mid:
        return jsonify({"error": "Sesion no encontrada."}), 401
    payload = request.get_json(force=True)
    eval_payload = {
        "medico_id":            mid,
        "signal_id":            payload["signal_id"],
        "llm_backend":          payload["llm_backend"],
        "llm_model_id":         payload.get("llm_model_id"),
        "nota_clinica_texto":   payload["nota_clinica_texto"],
        "nota_semaforo_global": payload["nota_semaforo_global"],
        "likert_exactitud":     payload["likert_exactitud"],
        "likert_coherencia":    payload["likert_coherencia"],
        "likert_utilidad":      payload["likert_utilidad"],
        "marcaciones_nota":     payload.get("marcaciones_nota"),
        "comentarios_libres":   payload.get("comentarios_libres", ""),
        "duracion_segundos":    payload.get("duracion_segundos"),
    }
    respuestas = payload["respuestas_ventana"]
    eval_id = db.crear_evaluacion(eval_payload, respuestas)
    return jsonify({"ok": True, "evaluacion_id": eval_id})


@app.route("/logout")
def logout():
    session.pop("medico_id", None)
    session.pop("medico_nombre", None)
    return redirect(url_for("registro"))


# ===============================================================
# RUTAS ADMIN
# ===============================================================
@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        password = (request.form.get("password") or "").strip()
        if password == config.ADMIN_PASSWORD:
            session["is_admin"] = True
            return redirect(url_for("admin_dashboard"))
        return render_template("admin_login.html",
            error="Contraseña incorrecta."), 401
    return render_template("admin_login.html", error=None)


@app.route("/admin/logout")
def admin_logout():
    session.pop("is_admin", None)
    return redirect(url_for("admin_login"))


@app.route("/admin/dashboard")
@require_admin
def admin_dashboard():
    return render_template("admin_dashboard.html",
        stats=db.stats_globales())


@app.route("/admin/api/dashboard_data")
@require_admin
def admin_api_dashboard():
    return jsonify({
        "globales":          db.stats_globales(),
        "por_escala":        db.stats_por_escala(),
        "por_clase_beat":    db.stats_por_clase("beat"),
        "por_clase_rhythm":  db.stats_por_clase("rhythm"),
        "semaforo_nota":     db.stats_semaforo_nota_global(),
        "por_senal":         db.stats_evaluaciones_por_senal(),
        "marcaciones":       db.stats_marcaciones_nota(),
    })


@app.route("/admin/medicos")
@require_admin
def admin_medicos():
    return render_template("admin_medicos.html",
        medicos=db.listar_medicos())


@app.route("/admin/medico/<int:medico_id>")
@require_admin
def admin_medico_detalle(medico_id):
    medico = db.get_medico(medico_id)
    if not medico:
        abort(404)
    evaluaciones = db.listar_evaluaciones(medico_id=medico_id)
    precision = db.precision_medico_vs_clasificador(medico_id)
    metricas_notas = db.metricas_notas_medico(medico_id)
    return render_template("admin_medico_detalle.html",
        medico=medico, evaluaciones=evaluaciones,
        precision=precision, metricas_notas=metricas_notas)


@app.route("/admin/api/medico_precision/<int:medico_id>")
@require_admin
def admin_api_medico_precision(medico_id):
    return jsonify({
        "precision": db.precision_medico_vs_clasificador(medico_id),
        "metricas_notas": db.metricas_notas_medico(medico_id),
    })


@app.route("/admin/eliminar_medico/<int:medico_id>", methods=["POST"])
@require_admin
def admin_eliminar_medico(medico_id):
    db.eliminar_medico(medico_id)
    return redirect(url_for("admin_medicos"))


@app.route("/admin/evaluaciones")
@require_admin
def admin_evaluaciones():
    return render_template("admin_evaluaciones.html",
        evaluaciones=db.listar_evaluaciones())


@app.route("/admin/evaluacion/<int:eval_id>")
@require_admin
def admin_evaluacion_detalle(eval_id):
    ev = db.get_evaluacion_completa(eval_id)
    if not ev:
        abort(404)
    return render_template("admin_evaluacion_detalle.html", ev=ev)


@app.route("/admin/eliminar_evaluacion/<int:eval_id>", methods=["POST"])
@require_admin
def admin_eliminar_evaluacion(eval_id):
    db.eliminar_evaluacion(eval_id)
    return redirect(url_for("admin_evaluaciones"))


@app.route("/admin/inter-evaluador")
@require_admin
def admin_inter_evaluador():
    return render_template("admin_inter_evaluador.html")


@app.route("/admin/api/kappa_data")
@require_admin
def admin_api_kappa():
    return jsonify(db.kappa_inter_evaluador())


@app.route("/admin/exportar")
@require_admin
def admin_exportar():
    return render_template("admin_exportar.html")


@app.route("/admin/exportar/<tipo>")
@require_admin
def admin_exportar_csv(tipo):
    if tipo == "medicos":
        csv_data = db.exportar_medicos_csv()
        fname = f"medicos_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    elif tipo == "evaluaciones":
        csv_data = db.exportar_evaluaciones_csv()
        fname = f"evaluaciones_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    elif tipo == "respuestas_ventana":
        csv_data = db.exportar_respuestas_ventana_csv()
        fname = f"respuestas_ventana_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    elif tipo == "backup_bd":
        return send_file(str(db.DB_PATH), as_attachment=True,
                         download_name=f"evaluaciones_backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.db")
    else:
        abort(404)
    return Response(
        csv_data,
        mimetype="text/csv; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename={fname}"},
    )


@app.route("/admin/reset_bd", methods=["POST"])
@require_admin
def admin_reset_bd():
    conf1 = request.form.get("confirmacion_1") == "SI"
    conf2 = request.form.get("confirmacion_2") == "BORRAR TODO"
    if not (conf1 and conf2):
        return "Confirmaciones no completadas correctamente.", 400
    result = db.reset_completo_bd()
    return render_template("admin_reset_result.html", result=result)


# ===============================================================
# LEGACY - ahora PROTEGIDO por admin
# ===============================================================
@app.route("/analytics")
@require_admin
def analytics():
    """Dashboard basico legacy. Ahora requiere admin."""
    return render_template("analytics.html")


@app.route("/api/analytics_data")
@require_admin
def api_analytics_data():
    """API legacy. Ahora requiere admin."""
    return jsonify({
        "globales": db.stats_globales(),
        "por_escala": db.stats_por_escala(),
        "por_clase_beat": db.stats_por_clase("beat"),
        "por_clase_rhythm": db.stats_por_clase("rhythm"),
        "medicos": db.listar_medicos(),
        "evaluaciones": db.listar_evaluaciones(),
    })


# ---------------------------------------------------------------
if __name__ == "__main__":
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
