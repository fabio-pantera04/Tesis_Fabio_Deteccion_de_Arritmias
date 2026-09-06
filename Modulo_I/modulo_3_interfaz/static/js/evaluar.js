/* ============================================================
   evaluar.js - Lógica de la página de evaluación clínica
   ------------------------------------------------------------
   Módulos:
   1. Renderizado de 6 segmentos ECG (10s c/u) con hover
      clínico (tiempo + voltaje) y tracks BEAT/RHYTHM alineados.
   2. Sistema de subrayado sobre la nota LLM (4 colores).
   3. Contador de progreso en tiempo real y botón Guardar
      deshabilitado hasta que toda la información esté completa.
   ============================================================ */

const SIGNAL = JSON.parse(document.getElementById("signal-data").textContent);
const N_SEGMENTS = 6;
const SEG_DURATION = 10;
const RHYTHM_WINDOW_S = 4;

const state = {
    beat_semaforos:   new Array(SIGNAL.beat_windows.length).fill(null),
    rhythm_semaforos: new Array(SIGNAL.rhythm_windows.length).fill(null),
    llm_note:         null,
    nota_semaforo:    null,
    likert_exactitud:  null,
    likert_coherencia: null,
    likert_utilidad:   null,
    nota_texto_original: "",
    marcaciones: [],
    modo_marcador: null,
    t_start: Date.now(),
};

// ============================================================
// 1. RENDERIZADO ECG POR SEGMENTOS CON HOVER DE VOLTAJE
// ============================================================

function drawECGSegments() {
    const container = document.getElementById("segments-container");
    container.innerHTML = "";

    const fs = SIGNAL.signal_metadata.sampling_rate_hz;
    const samplesPerSegment = Math.round(SEG_DURATION * fs);
    const totalSamples = SIGNAL.raw_signal.length;

    for (let segIdx = 0; segIdx < N_SEGMENTS; segIdx++) {
        const segStart = segIdx * SEG_DURATION;
        const segEnd = segStart + SEG_DURATION;
        const sampleStart = segIdx * samplesPerSegment;
        const sampleEnd = Math.min(sampleStart + samplesPerSegment, totalSamples);

        const block = document.createElement("div");
        block.className = "segment-block";
        block.dataset.segment = segIdx;

        const plotDiv = document.createElement("div");
        plotDiv.className = "ecg-strip";
        plotDiv.id = `ecg-strip-${segIdx}`;
        block.appendChild(plotDiv);

        const tracksDiv = document.createElement("div");
        tracksDiv.className = "segment-tracks";
        tracksDiv.appendChild(buildSegmentTrackRow("beat", segIdx, segStart, segEnd));
        tracksDiv.appendChild(buildSegmentTrackRow("rhythm", segIdx, segStart, segEnd));
        block.appendChild(tracksDiv);

        container.appendChild(block);

        const yVals = SIGNAL.raw_signal.slice(sampleStart, sampleEnd);
        const xVals = yVals.map((_, i) => segStart + i / fs);

        let sigMin = Infinity, sigMax = -Infinity;
        for (const v of yVals) {
            if (v < sigMin) sigMin = v;
            if (v > sigMax) sigMax = v;
        }
        const pad = (sigMax - sigMin) * 0.1 || 1;
        const yRange = [sigMin - pad, sigMax + pad];

        // NUEVO: hover clínico que muestra tiempo (s) y voltaje (mV)
        const trace = {
            x: xVals, y: yVals, mode: "lines", type: "scatter",
            line: { color: "#B22222", width: 1.2 },
            hovertemplate: "t = %{x:.3f} s<br>V = %{y:.3f} mV<extra></extra>",
            name: "",
        };
        const layout = {
            margin: { l: 70, r: 18, t: 8, b: 30 },
            xaxis: {
                range: [segStart, segEnd],
                showgrid: true, gridcolor: "rgba(178, 34, 34, 0.10)",
                dtick: 1, tickfont: { family: "JetBrains Mono", size: 10, color: "#8A7560" },
                showspikes: true,          // línea vertical al pasar mouse
                spikemode: "across",
                spikecolor: "#B22222",
                spikethickness: 1,
                spikedash: "dot",
            },
            yaxis: {
                range: yRange, showgrid: true, gridcolor: "rgba(178, 34, 34, 0.10)",
                tickfont: { family: "JetBrains Mono", size: 10, color: "#8A7560" },
                title: {
                    text: `${segStart}-${segEnd}s`,
                    font: { family: "JetBrains Mono", size: 11, color: "#5C453A" },
                    standoff: 8,
                },
            },
            paper_bgcolor: "rgba(0,0,0,0)",
            plot_bgcolor: "rgba(0,0,0,0)",
            showlegend: false,
            height: 130,
            hovermode: "x unified",        // tooltip cerca del cursor
            hoverlabel: {
                bgcolor: "#FBF5E5",
                bordercolor: "#B22222",
                font: { family: "JetBrains Mono", size: 11, color: "#2A1810" },
            },
        };
        Plotly.newPlot(plotDiv.id, [trace], layout, {
            displayModeBar: false, responsive: true,
        });
    }
}

function buildSegmentTrackRow(escala, segIdx, segStart, segEnd) {
    const row = document.createElement("div");
    row.className = "segment-track-row";

    const labelSpan = document.createElement("div");
    labelSpan.className = "segment-track-label";
    labelSpan.textContent = escala === "beat" ? "LATIDO" : "RITMO";
    row.appendChild(labelSpan);

    const cellsContainer = document.createElement("div");
    cellsContainer.className = "segment-track-cells";

    if (escala === "beat") {
        for (let s = 0; s < SEG_DURATION; s++) {
            const globalIdx = segStart + s;
            if (globalIdx >= SIGNAL.beat_windows.length) break;
            const win = SIGNAL.beat_windows[globalIdx];
            const cell = buildLabelCell("beat", globalIdx, win);
            cell.style.gridColumn = `${s + 1} / span 1`;
            cellsContainer.appendChild(cell);
        }
    } else {
        const firstIdx = Math.floor(segStart / RHYTHM_WINDOW_S);
        const lastIdx = Math.ceil(segEnd / RHYTHM_WINDOW_S) - 1;
        for (let i = firstIdx; i <= lastIdx; i++) {
            if (i < 0 || i >= SIGNAL.rhythm_windows.length) continue;
            const win = SIGNAL.rhythm_windows[i];
            const winStart = i * RHYTHM_WINDOW_S;
            const winEnd = winStart + RHYTHM_WINDOW_S;
            const localStart = Math.max(0, winStart - segStart);
            const localEnd = Math.min(SEG_DURATION, winEnd - segStart);
            if (localEnd <= localStart) continue;
            const cell = buildLabelCell("rhythm", i, win);
            cell.style.gridColumn = `${Math.round(localStart) + 1} / ${Math.round(localEnd) + 1}`;
            cellsContainer.appendChild(cell);
        }
    }
    row.appendChild(cellsContainer);
    return row;
}

function buildLabelCell(escala, globalIdx, win) {
    const cell = document.createElement("div");
    cell.className = `label-cell ${win.prediction}` +
        (win.escalation_flag ? " escalation" : "");
    cell.dataset.escala = escala;
    cell.dataset.idx = globalIdx;

    const lab = document.createElement("div");
    lab.className = "label-cell-label";
    lab.textContent = win.prediction;
    lab.title = `Confidence gap: ${win.confidence_gap?.toFixed(2) ?? "N/A"}`;
    cell.appendChild(lab);

    const sem = document.createElement("div");
    sem.className = "semaforo";
    for (const color of ["verde", "amarillo", "rojo"]) {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = color;
        btn.dataset.color = color;
        btn.title = color;
        const stateArr = escala === "beat" ? state.beat_semaforos : state.rhythm_semaforos;
        if (stateArr[globalIdx] === color) btn.classList.add("active");
        btn.addEventListener("click", () => {
            stateArr[globalIdx] = color;
            document.querySelectorAll(
                `.label-cell[data-escala="${escala}"][data-idx="${globalIdx}"]`
            ).forEach(otherCell => {
                otherCell.querySelectorAll(".semaforo button").forEach(b => {
                    b.classList.toggle("active", b.dataset.color === color);
                });
            });
            actualizarProgreso();
        });
        sem.appendChild(btn);
    }
    cell.appendChild(sem);
    return cell;
}

// ============================================================
// 2. NOTA LLM + SUBRAYADO
// ============================================================

async function fetchLLMNote() {
    const backendStamp = document.getElementById("backend-stamp");
    const noteText = document.getElementById("note-text");
    backendStamp.textContent = "consultando modelo…";
    noteText.textContent = "Generando la nota clínica…";
    try {
        const resp = await fetch(`/api/llm_note/${SIGNAL.signal_metadata.signal_id}?backend=${window.BACKEND_ACTIVO}`);
        const data = await resp.json();
        if (data.error) {
            noteText.textContent = "Error al generar nota: " + data.error;
            backendStamp.textContent = window.BACKEND_ACTIVO + " · error";
            return;
        }
        state.llm_note = data;
        state.nota_texto_original = data.text;
        state.marcaciones = [];
        renderNotaConMarcas();
        actualizarDistribucion();
        backendStamp.textContent =
            `${data.backend_name} · ${data.model_id} · ${data.latency_seconds}s`;
    } catch (e) {
        noteText.textContent = "Error de red al obtener la nota.";
        backendStamp.textContent = window.BACKEND_ACTIVO + " · error de red";
    }
}

function renderNotaConMarcas() {
    const container = document.getElementById("note-text");
    const texto = state.nota_texto_original;
    const marks = [...state.marcaciones].sort((a, b) => a.start - b.start);
    if (marks.length === 0) {
        container.textContent = texto;
        return;
    }
    let html = "";
    let cursor = 0;
    for (const m of marks) {
        if (cursor < m.start) html += escapeHTML(texto.slice(cursor, m.start));
        html += `<span class="mark-${m.color}">${escapeHTML(texto.slice(m.start, m.end))}</span>`;
        cursor = m.end;
    }
    if (cursor < texto.length) html += escapeHTML(texto.slice(cursor));
    container.innerHTML = html;
}

function escapeHTML(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function aplicarMarca(newStart, newEnd, newColor) {
    if (newStart >= newEnd) return;
    const result = [];
    for (const m of state.marcaciones) {
        if (m.end <= newStart || m.start >= newEnd) { result.push(m); continue; }
        if (m.start >= newStart && m.end <= newEnd) continue;
        if (m.start < newStart) result.push({ start: m.start, end: newStart, color: m.color });
        if (m.end > newEnd) result.push({ start: newEnd, end: m.end, color: m.color });
    }
    if (newColor !== "borrar") {
        result.push({ start: newStart, end: newEnd, color: newColor });
    }
    state.marcaciones = result.sort((a, b) => a.start - b.start);
    renderNotaConMarcas();
    actualizarDistribucion();
}

function getSelectionOffsets(container) {
    const sel = window.getSelection();
    if (!sel || sel.rangeCount === 0) return null;
    const range = sel.getRangeAt(0);
    if (!container.contains(range.commonAncestorContainer)) return null;
    if (range.collapsed) return null;
    const startRange = document.createRange();
    startRange.selectNodeContents(container);
    startRange.setEnd(range.startContainer, range.startOffset);
    const start = startRange.toString().length;
    const endRange = document.createRange();
    endRange.selectNodeContents(container);
    endRange.setEnd(range.endContainer, range.endOffset);
    const end = endRange.toString().length;
    return { start, end };
}

function actualizarDistribucion() {
    const total = state.nota_texto_original.length || 1;
    const counts = { verde: 0, amarillo: 0, rojo: 0 };
    for (const m of state.marcaciones) counts[m.color] += (m.end - m.start);
    const marked = counts.verde + counts.amarillo + counts.rojo;
    document.getElementById("pct-verde").textContent = (counts.verde / total * 100).toFixed(1) + "%";
    document.getElementById("pct-amarillo").textContent = (counts.amarillo / total * 100).toFixed(1) + "%";
    document.getElementById("pct-rojo").textContent = (counts.rojo / total * 100).toFixed(1) + "%";
    document.getElementById("pct-neutral").textContent = ((total - marked) / total * 100).toFixed(1) + "%";
}

function setupSubrayado() {
    const noteText = document.getElementById("note-text");
    const toolbar = document.getElementById("mark-toolbar");
    toolbar.querySelectorAll(".mark-btn[data-mark]").forEach(btn => {
        btn.addEventListener("click", () => {
            const color = btn.dataset.mark;
            if (state.modo_marcador === color) {
                state.modo_marcador = null;
                btn.classList.remove("active");
                document.body.classList.remove("marking-mode");
            } else {
                state.modo_marcador = color;
                toolbar.querySelectorAll(".mark-btn").forEach(b => b.classList.remove("active"));
                btn.classList.add("active");
                document.body.classList.add("marking-mode");
            }
        });
    });
    document.getElementById("btn-limpiar-marcas").addEventListener("click", () => {
        if (state.marcaciones.length === 0) return;
        if (confirm("¿Borrar todas las marcaciones de la nota?")) {
            state.marcaciones = [];
            renderNotaConMarcas();
            actualizarDistribucion();
        }
    });
    noteText.addEventListener("mouseup", () => {
        if (!state.modo_marcador) return;
        const offsets = getSelectionOffsets(noteText);
        if (!offsets) return;
        aplicarMarca(offsets.start, offsets.end, state.modo_marcador);
        window.getSelection().removeAllRanges();
    });
}

// ============================================================
// 3. VALORACIONES + CONTADOR DE PROGRESO EN TIEMPO REAL
// ============================================================

function setupSemaforoNota() {
    const container = document.getElementById("semaforo-nota");
    container.querySelectorAll("button[data-color]").forEach(btn => {
        btn.addEventListener("click", () => {
            container.querySelectorAll("button").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            state.nota_semaforo = btn.dataset.color;
            actualizarProgreso();
        });
    });
}

function setupLikert(containerId, stateKey) {
    const container = document.getElementById(containerId);
    container.querySelectorAll("button[data-val]").forEach(btn => {
        btn.addEventListener("click", () => {
            container.querySelectorAll("button").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            state[stateKey] = parseInt(btn.dataset.val);
            actualizarProgreso();
        });
    });
}

/**
 * Cuenta cuántos elementos ya están completados de un total esperado,
 * actualiza el indicador visual y habilita/deshabilita el botón Guardar.
 */
function actualizarProgreso() {
    const totalBeat = state.beat_semaforos.length;
    const totalRhythm = state.rhythm_semaforos.length;
    const doneBeat = state.beat_semaforos.filter(x => x !== null).length;
    const doneRhythm = state.rhythm_semaforos.filter(x => x !== null).length;
    const doneNota = state.nota_semaforo ? 1 : 0;
    const doneLikert = [state.likert_exactitud, state.likert_coherencia, state.likert_utilidad]
                      .filter(x => x != null).length;

    const totalGeneral = totalBeat + totalRhythm + 1 /* nota */ + 3 /* likert */;
    const doneGeneral = doneBeat + doneRhythm + doneNota + doneLikert;
    const completo = doneGeneral === totalGeneral;

    // Elementos DOM del contador
    const barra = document.getElementById("progress-bar-fill");
    const texto = document.getElementById("progress-text");
    const detalle = document.getElementById("progress-detail");
    const btnGuardar = document.getElementById("btn-guardar");

    if (barra) barra.style.width = `${(doneGeneral / totalGeneral * 100).toFixed(1)}%`;
    if (texto) texto.textContent = `${doneGeneral} / ${totalGeneral} elementos completados`;
    if (detalle) {
        detalle.textContent =
            `LATIDO: ${doneBeat}/${totalBeat} · ` +
            `RITMO: ${doneRhythm}/${totalRhythm} · ` +
            `Nota: ${doneNota}/1 · ` +
            `Likert: ${doneLikert}/3`;
    }
    if (btnGuardar) {
        btnGuardar.disabled = !completo;
        btnGuardar.classList.toggle("btn-ready", completo);
    }
}

// ============================================================
// 4. TIMER
// ============================================================

function startElapsedTimer() {
    setInterval(() => {
        const elapsed = Math.floor((Date.now() - state.t_start) / 1000);
        document.getElementById("elapsed").textContent = elapsed;
    }, 1000);
}

// ============================================================
// 5. GUARDAR
// ============================================================

function setupSubmit() {
    document.getElementById("btn-guardar").addEventListener("click", async () => {
        const msg = document.getElementById("submit-msg");
        // Validación defensiva (por si el disabled del botón falla)
        const missing = [];
        if (!state.nota_semaforo) missing.push("semáforo global de la nota");
        if (state.likert_exactitud == null) missing.push("Likert exactitud");
        if (state.likert_coherencia == null) missing.push("Likert coherencia");
        if (state.likert_utilidad == null) missing.push("Likert utilidad");
        const bm = state.beat_semaforos.filter(x => x === null).length;
        const rm = state.rhythm_semaforos.filter(x => x === null).length;
        if (bm > 0) missing.push(`${bm} ventanas LATIDO`);
        if (rm > 0) missing.push(`${rm} ventanas RITMO`);
        if (missing.length > 0) {
            msg.innerHTML = `<div class="alert alert-error">Faltan: ${missing.join(", ")}.</div>`;
            return;
        }

        const respuestas = [];
        SIGNAL.beat_windows.forEach((w, i) => {
            respuestas.push({
                escala: "beat", window_index: i,
                prediccion_modelo: w.prediction,
                semaforo: state.beat_semaforos[i],
                confidence_gap: w.confidence_gap,
            });
        });
        SIGNAL.rhythm_windows.forEach((w, i) => {
            respuestas.push({
                escala: "rhythm", window_index: i,
                prediccion_modelo: w.prediction,
                semaforo: state.rhythm_semaforos[i],
                confidence_gap: w.confidence_gap,
            });
        });

        const total = state.nota_texto_original.length || 1;
        const counts = { verde: 0, amarillo: 0, rojo: 0 };
        for (const m of state.marcaciones) counts[m.color] += (m.end - m.start);
        const marcaciones_nota = {
            marcaciones: state.marcaciones,
            pct_verde:    +(counts.verde / total * 100).toFixed(2),
            pct_amarillo: +(counts.amarillo / total * 100).toFixed(2),
            pct_rojo:     +(counts.rojo / total * 100).toFixed(2),
            pct_sin_marcar: +((total - counts.verde - counts.amarillo - counts.rojo) / total * 100).toFixed(2),
            total_caracteres: total,
        };

        const payload = {
            signal_id: SIGNAL.signal_metadata.signal_id,
            llm_backend: state.llm_note?.backend_name || window.BACKEND_ACTIVO,
            llm_model_id: state.llm_note?.model_id || null,
            nota_clinica_texto: state.nota_texto_original,
            nota_semaforo_global: state.nota_semaforo,
            likert_exactitud: state.likert_exactitud,
            likert_coherencia: state.likert_coherencia,
            likert_utilidad: state.likert_utilidad,
            marcaciones_nota: marcaciones_nota,
            comentarios_libres: document.getElementById("comentarios_libres").value || "",
            duracion_segundos: (Date.now() - state.t_start) / 1000,
            respuestas_ventana: respuestas,
        };

        try {
            const resp = await fetch("/api/guardar", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            const data = await resp.json();
            if (data.ok) {
                msg.innerHTML = `<div class="alert alert-success">Evaluación guardada correctamente (id ${data.evaluacion_id}). Regresando al listado…</div>`;
                setTimeout(() => window.location.href = "/seleccion", 1600);
            } else {
                msg.innerHTML = `<div class="alert alert-error">Error al guardar: ${data.error || "desconocido"}</div>`;
            }
        } catch (e) {
            msg.innerHTML = `<div class="alert alert-error">Error de red al guardar: ${e.message}</div>`;
        }
    });
}

// ============================================================
// INIT
// ============================================================

document.addEventListener("DOMContentLoaded", () => {
    drawECGSegments();
    setupSubrayado();
    setupSemaforoNota();
    setupLikert("likert-exactitud", "likert_exactitud");
    setupLikert("likert-coherencia", "likert_coherencia");
    setupLikert("likert-utilidad", "likert_utilidad");
    startElapsedTimer();
    setupSubmit();
    fetchLLMNote();
    actualizarProgreso();   // primera actualización con estado vacío
});