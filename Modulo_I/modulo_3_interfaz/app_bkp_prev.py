"""
Modulo III - Plataforma de validacion clinica human-in-the-loop.

Rutas:
  /                       -> redirige a /registro o a /seleccion segun sesion
  /registro               -> formulario de registro de cardiologo
  /seleccion              -> elige una de las 5 senales para evaluar
  /evaluar/<signal_id>    -> pagina principal de evaluacion
  /api/llm_note/<sid>     -> nota LLM (persistida en disco > cache > backend)
  /api/guardar            -> POST: persiste evaluacion completa
  /analytics              -> dashboard de metricas agregadas
  /api/analytics_data     -> JSON con stats para charts
  /logout                 -> limpia sesion

Persistencia de notas LLM:
  Prioridad de lectura del endpoint /api/llm_note/<sid>:
    1. Cache en memoria (_NOTE_CACHE)
    2. Archivo persistido en config.NOTES_DIR/nota_<sid>.json
    3. Fallback: generar con el backend LLM y guardar en disco

  Parametros query:
    - backend=<nombre> : forzar backend LLM especifico
    - force=1          : ignorar cache y disco, regenerar con LLM
"""
import json
import os
import time
from datetime import datetime
from pathlib import Path

from flask import (
    Flask, render_template, request, redirect, url_for,
    session, jsonify, abort,
)

import config
from database import db
from llm_backends import get_backend, build_prompt

app = Flask(__name__)
app.config["SECRET_KEY"] = config.SECRET_KEY

# Cache en memoria de notas LLM: {(signal_id, backend): payload_dict}
_NOTE_CACHE: dict[tuple, dict] = {}


# ---------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------
@app.before_request
def ensure_db():
    """Inicializar BD en la primera peticion."""
    if not hasattr(ensure_db, "_done"):
        db.init_db()
        ensure_db._done = True


# ---------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------
def load_signal(signal_id: str) -> dict:
    """Carga un JSON pre-procesado desde data/signals_processed/."""
    path = config.SIGNALS_DIR / f"{signal_id}.json"
    if not path.exists():
        abort(404, description=f"Senal {signal_id} no encontrada.")
    with open(path) as f:
        return json.load(f)


def load_persisted_note(signal_id: str) -> dict | None:
    """
    Lee una nota pre-generada desde config.NOTES_DIR/nota_<signal_id>.json.
    Devuelve None si no existe (el endpoint hara fallback al backend LLM).

    Formato del JSON persistido (compatible con generar_y_guardar_notas.py):
      {
        "signal_id": "sig_001",
        "generated_at": ISO datetime,
        "model": "gemini-2.5-flash",
        "prompt_version": "v3",
        "latency_seconds": float,
        "signal_metadata": {...},
        "aggregate_summary": {...},
        "clinical_note": "texto de la nota"
      }
    """
    path = config.NOTES_DIR / f"nota_{signal_id}.json"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        app.logger.warning(f"Error leyendo nota persistida {signal_id}: {e}")
        return None


def save_generated_note(signal_id: str, resp, signal_json: dict) -> None:
    """
    Guarda una nota recien generada por el backend LLM para consultas
    futuras. Estructura compatible con generar_y_guardar_notas.py.
    """
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
    """Lista resumida de las senales disponibles con estado de nota."""
    out = []
    for path in sorted(config.SIGNALS_DIR.glob("sig_*.json")):
        with open(path) as f:
            data = json.load(f)
        meta = data["signal_metadata"]
        signal_id = meta["signal_id"]
        # Indica si la nota LLM ya esta persistida (para badge en UI)
        note_ready = (config.NOTES_DIR / f"nota_{signal_id}.json").exists()
        out.append({
            "signal_id":            signal_id,
            "label":                meta["label"],
            "scenario_note":        meta["scenario_note"],
            "patient_age_estimate": meta["patient_age_estimate"],
            "patient_sex":          meta["patient_sex"],
            "duration_seconds":     meta["duration_seconds"],
            "note_ready":           note_ready,
        })
    return out


def require_medico():
    """Devuelve el medico_id de sesion o None si no hay sesion."""
    mid = session.get("medico_id")
    if not mid:
        return None
    return mid


def _safe_int(s, default=None):
    """Convierte string a int de forma defensiva."""
    try:
        if s is None or str(s).strip() == "":
            return default
        return int(s)
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------
@app.route("/")
def index():
    if session.get("medico_id"):
        return redirect(url_for("seleccion"))
    return redirect(url_for("registro"))


@app.route("/registro", methods=["GET", "POST"])
def registro():
    if request.method == "POST":
        # Si la especialidad es "Otra", usar el valor del input de texto
        especialidad_raw = (request.form.get("especialidad") or "").strip()
        especialidad_otra = (request.form.get("especialidad_otra") or "").strip()
        if especialidad_raw == "Otra":
            especialidad_final = especialidad_otra
        else:
            especialidad_final = especialidad_raw

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

        # Todos obligatorios EXCEPTO sub_especialidad
        required = {
            "nombre_completo": "Nombre completo",
            "edad": "Edad",
            "sexo": "Sexo",
            "institucion": "Institución",
            "especialidad": "Especialidad" if especialidad_raw != "Otra"
                            else "Especificación de 'Otra' especialidad",
            "anos_experiencia": "Años de experiencia clínica",
            "anos_experiencia_ecg": "Años de experiencia con ECG",
            "auto_eval_habilidad": "Auto-evaluación de habilidad",
            "email": "Correo de contacto",
        }
        missing = [label for key, label in required.items()
                   if payload[key] in (None, "", 0)]
        # Special case: 0 es valido para anos_experiencia (residente nuevo)
        if payload["anos_experiencia"] == 0:
            missing = [m for m in missing if m != required["anos_experiencia"]]
        if payload["anos_experiencia_ecg"] == 0:
            missing = [m for m in missing if m != required["anos_experiencia_ecg"]]

        if missing:
            # Re-render preservando lo capturado
            preserved = dict(request.form)
            return render_template(
                "registro.html",
                error="Faltan campos obligatorios: " + ", ".join(missing) + ".",
                form=preserved,
            ), 400

        mid = db.crear_medico(payload)
        session["medico_id"] = mid
        session["medico_nombre"] = payload["nombre_completo"]
        return redirect(url_for("seleccion"))
    return render_template("registro.html", form={}, error=None)


@app.route("/seleccion")
def seleccion():
    mid = require_medico()
    if not mid:
        return redirect(url_for("registro"))
    return render_template(
        "seleccion.html",
        senales=list_signals(),
        medico_nombre=session.get("medico_nombre", ""),
        backend_activo=config.LLM_BACKEND,
    )


@app.route("/evaluar/<signal_id>")
def evaluar(signal_id):
    mid = require_medico()
    if not mid:
        return redirect(url_for("registro"))
    signal = load_signal(signal_id)
    # Numero secuencial (Señal 1, Señal 2, ...) basado en orden alfabetico
    all_ids = sorted([p.stem for p in config.SIGNALS_DIR.glob("sig_*.json")])
    try:
        signal_number = all_ids.index(signal_id) + 1
    except ValueError:
        signal_number = 1
    return render_template(
        "evaluar.html",
        signal=signal,
        signal_number=signal_number,
        signal_json_str=json.dumps(signal),
        backend_activo=config.LLM_BACKEND,
        medico_nombre=session.get("medico_nombre", ""),
    )


@app.route("/api/llm_note/<signal_id>")
def api_llm_note(signal_id):
    """
    Devuelve la nota clinica del backend activo.
    Prioridad de lectura:
      1. Cache en memoria (~1 ms)
      2. Nota persistida en disco (~10 ms)
      3. Backend LLM en vivo (~15 s), luego guardar en disco

    Query params:
      backend=<nombre> - backend LLM (default: config.LLM_BACKEND)
      force=1          - ignorar cache y disco, regenerar con LLM
    """
    backend_name = request.args.get("backend", config.LLM_BACKEND)
    force_regen = request.args.get("force") == "1"
    cache_key = (signal_id, backend_name)

    # 1. Cache en memoria (fast path)
    if not force_regen and cache_key in _NOTE_CACHE:
        return jsonify(_NOTE_CACHE[cache_key])

    # 2. Nota persistida en disco
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

    # 3. Fallback: generar con el backend LLM y persistir
    signal = load_signal(signal_id)
    try:
        backend = get_backend(backend_name)
    except Exception as e:
        return jsonify({
            "error": str(e),
            "text": "",
            "backend_name": backend_name,
        }), 500

    resp = backend.generate(signal)
    payload = {
        "text":            resp.text,
        "backend_name":    resp.backend_name,
        "model_id":        resp.model_id,
        "latency_seconds": round(resp.latency_seconds, 3),
        "error":           resp.error,
        "source":          "generated",
    }

    if not resp.error:
        _NOTE_CACHE[cache_key] = payload
        # Persistir para futuras consultas
        save_generated_note(signal_id, resp, signal)

    return jsonify(payload)


@app.route("/api/guardar", methods=["POST"])
def api_guardar():
    """Persiste la evaluacion completa."""
    mid = require_medico()
    if not mid:
        return jsonify({"error": "Sesion no encontrada."}), 401

    payload = request.get_json(force=True)
    eval_payload = {
        "medico_id": mid,
        "signal_id": payload["signal_id"],
        "llm_backend": payload["llm_backend"],
        "llm_model_id": payload.get("llm_model_id"),
        "nota_clinica_texto": payload["nota_clinica_texto"],
        "nota_semaforo_global": payload["nota_semaforo_global"],
        "likert_exactitud": payload["likert_exactitud"],
        "likert_coherencia": payload["likert_coherencia"],
        "likert_utilidad": payload["likert_utilidad"],
        "comentarios_libres": payload.get("comentarios_libres", ""),
        "duracion_segundos": payload.get("duracion_segundos"),
    }
    respuestas = payload["respuestas_ventana"]
    eval_id = db.crear_evaluacion(eval_payload, respuestas)
    return jsonify({"ok": True, "evaluacion_id": eval_id})


@app.route("/analytics")
def analytics():
    return render_template("analytics.html")


@app.route("/api/analytics_data")
def api_analytics_data():
    return jsonify({
        "globales": db.stats_globales(),
        "por_escala": db.stats_por_escala(),
        "por_clase_beat": db.stats_por_clase("beat"),
        "por_clase_rhythm": db.stats_por_clase("rhythm"),
        "por_backend": db.stats_por_backend(),
        "medicos": db.listar_medicos(),
        "evaluaciones": db.listar_evaluaciones(),
    })


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("registro"))


# ---------------------------------------------------------------
if __name__ == "__main__":
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)