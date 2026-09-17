"""
Flask app: Sistema de Validacion Clinica ECG.
Version con:
  - Rangos de edad en registro
  - T&C con checkbox obligatorio
  - Guardado automatico (localStorage) - solo frontend
  - Nota clinica + nota del clasificador separadas
  - Comentarios duales (nota + clasificador)
  - Vista admin comparativa por senal con ground truth
"""
import json
import os
from datetime import datetime
from functools import wraps
from pathlib import Path

from flask import (Flask, render_template, request, redirect, url_for,
                   session, jsonify, make_response, abort)

# Imports locales
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "database"))
import db

# Backend LLM (opcional en modo web)
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "modulo_2_agente"))
    from agente_notas import generar_nota_para_signal, BACKENDS_DISPONIBLES
    LLM_DISPONIBLE = True
except Exception:
    LLM_DISPONIBLE = False
    BACKENDS_DISPONIBLES = {}

# --------------------------------------------------------------
# Config
# --------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
SIGNALS_DIR = ROOT / "data" / "signals_processed"
NOTES_DIR = ROOT / "data" / "notas_llm_a"

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret-change-in-prod")
app.config["JSON_AS_ASCII"] = False

# Backend LLM por defecto (cargado por el usuario en el admin)
DEFAULT_BACKEND = "llm_a"
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "Admin_Fabio_Cimat")

# Bootstrap de la BD
db.init_db()


# ==============================================================
# HELPERS
# ==============================================================
def _load_signal(signal_id: str) -> dict:
    """Carga un JSON de senal procesada."""
    path = SIGNALS_DIR / f"{signal_id}.json"
    if not path.exists():
        abort(404, f"Senal {signal_id} no encontrada")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_llm_note(signal_id: str) -> dict:
    """Carga la nota del LLM persistida."""
    path = NOTES_DIR / f"nota_{signal_id}.json"
    if not path.exists():
        return {"clinical_note": "[No hay nota persistida para esta senal]",
                "classifier_note": "",
                "model": "N/A", "latency_seconds": 0}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _list_all_signals() -> list[dict]:
    """Lista todas las senales disponibles con metadatos basicos."""
    signals = []
    for path in sorted(SIGNALS_DIR.glob("sig_*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        signals.append({
            "signal_id": data["signal_metadata"]["signal_id"],
            "duration_seconds": data["signal_metadata"]["duration_seconds"],
        })
    return signals


def require_medico(f):
    """Decorator: verifica que hay medico en sesion."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "medico_id" not in session:
            return redirect(url_for("registro"))
        return f(*args, **kwargs)
    return wrapper


def require_admin(f):
    """Decorator: verifica login admin."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not session.get("is_admin"):
            return redirect(url_for("admin_login"))
        return f(*args, **kwargs)
    return wrapper


# ==============================================================
# RUTAS PUBLICAS (VISTA MEDICO)
# ==============================================================
@app.route("/")
def index():
    return redirect(url_for("registro"))


@app.route("/registro", methods=["GET"])
def registro():
    return render_template("registro.html")


@app.route("/api/registro", methods=["POST"])
def api_registro():
    try:
        payload = request.get_json()
        if not payload.get("acepto_terminos"):
            return jsonify({"ok": False,
                            "error": "Debe aceptar los terminos y condiciones."}), 400

        medico_id = db.crear_medico(payload)
        session["medico_id"] = medico_id
        return jsonify({"ok": True, "medico_id": medico_id})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/seleccion")
@require_medico
def seleccion():
    senales = _list_all_signals()
    return render_template("seleccion.html", senales=senales)


@app.route("/evaluar/<signal_id>")
@require_medico
def evaluar(signal_id):
    signal_data = _load_signal(signal_id)
    all_signals = _list_all_signals()
    signal_ids = [s["signal_id"] for s in all_signals]
    signal_number = signal_ids.index(signal_id) + 1 if signal_id in signal_ids else 1
    signal_json_str = json.dumps(signal_data, ensure_ascii=False)
    return render_template(
        "evaluar.html",
        signal=signal_data,
        signal_number=signal_number,
        signal_json_str=signal_json_str,
        backend_activo=DEFAULT_BACKEND,
    )


@app.route("/api/llm_note/<signal_id>")
@require_medico
def api_llm_note(signal_id):
    """Devuelve la nota clinica + nota del clasificador (ya separadas)."""
    backend = request.args.get("backend", DEFAULT_BACKEND)
    try:
        note_data = _load_llm_note(signal_id)
        return jsonify({
            "clinical_note": note_data.get("clinical_note", ""),
            "classifier_note": note_data.get("classifier_note", ""),
            "text": note_data.get("clinical_note", ""),  # backward compat
            "backend_name": backend,
            "model_id": note_data.get("model", "N/A"),
            "latency_seconds": note_data.get("latency_seconds", 0),
        })
    except Exception as e:
        return jsonify({"error": str(e), "backend_name": backend}), 500


@app.route("/api/guardar", methods=["POST"])
@require_medico
def api_guardar():
    try:
        payload = request.get_json()
        respuestas = payload.pop("respuestas_ventana", [])
        payload["medico_id"] = session["medico_id"]
        eval_id = db.crear_evaluacion(payload, respuestas)
        return jsonify({"ok": True, "evaluacion_id": eval_id})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/mis_evaluaciones")
@require_medico
def mis_evaluaciones():
    medico_id = session["medico_id"]
    evaluaciones = db.listar_evaluaciones(medico_id=medico_id)
    medico = db.get_medico(medico_id)
    return render_template("mis_evaluaciones.html",
                           evaluaciones=evaluaciones, medico=medico)


@app.route("/logout")
def logout():
    session.pop("medico_id", None)
    return redirect(url_for("registro"))


# ==============================================================
# ADMIN
# ==============================================================
@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        if request.form.get("password") == ADMIN_PASSWORD:
            session["is_admin"] = True
            return redirect(url_for("admin_dashboard"))
        return render_template("admin_login.html", error="Contrasena incorrecta")
    return render_template("admin_login.html")


@app.route("/admin/logout")
def admin_logout():
    session.pop("is_admin", None)
    return redirect(url_for("admin_login"))


@app.route("/admin")
@app.route("/admin/dashboard")
@require_admin
def admin_dashboard():
    stats = db.stats_globales()
    return render_template("admin_dashboard.html", stats=stats)


@app.route("/admin/api/dashboard_data")
@require_admin
def admin_api_dashboard_data():
    return jsonify({
        "por_escala": db.stats_por_escala(),
        "semaforo_nota": db.stats_semaforo_nota_global(),
        "por_clase_beat": db.stats_por_clase("beat"),
        "por_clase_rhythm": db.stats_por_clase("rhythm"),
        "por_senal": db.stats_evaluaciones_por_senal(),
        "marcaciones": db.stats_marcaciones_nota(),
    })


@app.route("/admin/medicos")
@require_admin
def admin_medicos():
    return render_template("admin_medicos.html", medicos=db.listar_medicos())


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


@app.route("/admin/medico/<int:medico_id>/eliminar", methods=["POST"])
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


@app.route("/admin/evaluacion/<int:eval_id>/eliminar", methods=["POST"])
@require_admin
def admin_eliminar_evaluacion(eval_id):
    db.eliminar_evaluacion(eval_id)
    return redirect(url_for("admin_evaluaciones"))


@app.route("/admin/inter_evaluador")
@require_admin
def admin_inter_evaluador():
    return render_template("admin_inter_evaluador.html")


@app.route("/admin/api/kappa_data")
@require_admin
def admin_api_kappa_data():
    return jsonify(db.kappa_inter_evaluador())


# --------------------------------------------------------------
# NUEVA VISTA COMPARATIVA POR SENAL
# --------------------------------------------------------------
@app.route("/admin/comparativa")
@require_admin
def admin_comparativa():
    """Lista las senales disponibles para vista comparativa."""
    signals_all = _list_all_signals()
    labels = db.load_signal_labels()
    signals_con_evals = set(db.listar_signals_disponibles())
    lista = []
    for s in signals_all:
        sid = s["signal_id"]
        entry = labels.get(sid, {})
        lista.append({
            "signal_id": sid,
            "dataset": entry.get("dataset", "N/A"),
            "rhythm_label": entry.get("rhythm_label", "N/A"),
            "predominant_beat_label": entry.get("predominant_beat_label", "N/A"),
            "duration_seconds": s.get("duration_seconds"),
            "tiene_evaluaciones": sid in signals_con_evals,
        })
    return render_template("admin_comparativa.html", signals=lista)


@app.route("/admin/comparativa/<signal_id>")
@require_admin
def admin_comparativa_signal(signal_id):
    """Vista detallada comparativa de una senal."""
    data = db.comparativa_signal(signal_id)
    signal_data = _load_signal(signal_id)
    note_data = _load_llm_note(signal_id)
    data["clinical_note"] = note_data.get("clinical_note", "")
    data["classifier_note"] = note_data.get("classifier_note", "")
    data["signal_meta"] = signal_data.get("signal_metadata", {})
    return render_template("admin_comparativa_signal.html", data=data,
                           signal_id=signal_id)


@app.route("/admin/api/comparativa/<signal_id>")
@require_admin
def admin_api_comparativa(signal_id):
    return jsonify(db.comparativa_signal(signal_id))


# --------------------------------------------------------------
# EXPORTACION
# --------------------------------------------------------------
@app.route("/admin/exportar")
@require_admin
def admin_exportar():
    stats = db.stats_globales()
    return render_template("admin_exportar.html", stats=stats)


@app.route("/admin/exportar/<tipo>")
@require_admin
def admin_exportar_csv(tipo):
    if tipo == "medicos":
        csv_data = db.exportar_medicos_csv()
        nombre = "medicos.csv"
    elif tipo == "evaluaciones":
        csv_data = db.exportar_evaluaciones_csv()
        nombre = "evaluaciones.csv"
    elif tipo == "respuestas":
        csv_data = db.exportar_respuestas_ventana_csv()
        nombre = "respuestas_ventana.csv"
    else:
        abort(404)
    response = make_response(csv_data)
    response.headers["Content-Type"] = "text/csv; charset=utf-8"
    response.headers["Content-Disposition"] = f"attachment; filename={nombre}"
    return response


@app.route("/admin/reset_bd", methods=["POST"])
@require_admin
def admin_reset_bd():
    """Doble confirmacion via form."""
    conf1 = request.form.get("confirmar")
    conf2 = request.form.get("confirmar2")
    if conf1 != "SI" or conf2 != "BORRAR TODO":
        return redirect(url_for("admin_exportar"))
    resumen = db.reset_completo_bd()
    return render_template("admin_exportar.html",
                           stats=db.stats_globales(), reset_ok=True,
                           reset_resumen=resumen)


# ==============================================================
# ARRANQUE
# ==============================================================
if __name__ == "__main__":
    app.run(debug=True, port=5000)
