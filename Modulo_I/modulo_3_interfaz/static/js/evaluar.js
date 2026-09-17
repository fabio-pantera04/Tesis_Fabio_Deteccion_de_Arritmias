// =================================================================
// Pagina de evaluacion — version con tracks intercalados por tira,
// sincronizacion de rhythms compartidos entre tiras, y orden nuevo:
//   1) ECG + tracks integrados por tira
//   2) Nota del clasificador
//   3) Nota clinica
//   4) Comentarios opcionales (uno para cada nota)
// =================================================================
const SIGNAL = JSON.parse(document.getElementById("signal-data").textContent);
const FS = SIGNAL.signal_metadata.sampling_rate_hz;
const DURATION = SIGNAL.signal_metadata.duration_seconds;
const N_BEAT_WIN = SIGNAL.beat_windows.length;
const N_RHYTHM_WIN = SIGNAL.rhythm_windows.length;

// Cada tira son 10 segundos
const STRIP_DURATION = 10;
const N_STRIPS = Math.ceil(DURATION / STRIP_DURATION);

// LocalStorage key (borrador por medico + senal)
const MEDICO_ID = window.MEDICO_ID || "anon";
const STORAGE_KEY = `eval_${MEDICO_ID}_${window.SIGNAL_ID}`;
const AUTOSAVE_DEBOUNCE_MS = 400;

const T_START = Date.now();
setInterval(() => {
    const el = document.getElementById("elapsed");
    if (el) el.textContent = Math.round((Date.now() - T_START) / 1000);
}, 1000);

// ------------------------------------------------------------
// Estado de la evaluacion
// ------------------------------------------------------------
const state = {
    beat_semaforos:   new Array(N_BEAT_WIN).fill(null),
    rhythm_semaforos: new Array(N_RHYTHM_WIN).fill(null),

    // Nota clinica
    nota_semaforo: null,
    likert_exactitud: null,
    likert_coherencia: null,
    likert_utilidad: null,
    marcaciones_nota: null,
    comentarios_nota: "",

    // Nota del clasificador
    nota_clasificador_semaforo: null,
    marcaciones_clasificador: null,
    comentarios_clasificador: "",

    llm_payload: null,
};

const LABEL_ABBR = { NORMAL: "N", PAC: "P", NSR: "NSR", AFIB: "AF" };

// ------------------------------------------------------------
// Utilidades de subrayado
// ------------------------------------------------------------
class HighlightManager {
    constructor(containerId, storageField, onChange) {
        this.container = document.getElementById(containerId);
        this.storageField = storageField;
        this.onChange = onChange;
        this.marks = [];
        this.originalText = "";
    }

    setText(text) {
        this.originalText = text;
        this.render();
    }

    render() {
        const text = this.originalText;
        if (!text) { this.container.innerHTML = ""; return; }
        const sorted = [...this.marks].sort((a, b) => a.start - b.start);
        const esc = s => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
        let html = "", cursor = 0;
        for (const m of sorted) {
            if (cursor < m.start) html += esc(text.slice(cursor, m.start));
            html += `<span class="mark-${m.color}">${esc(text.slice(m.start, m.end))}</span>`;
            cursor = m.end;
        }
        if (cursor < text.length) html += esc(text.slice(cursor));
        this.container.innerHTML = html;
    }

    getSelectionRange() {
        const sel = window.getSelection();
        if (!sel || sel.rangeCount === 0) return null;
        const range = sel.getRangeAt(0);
        if (!this.container.contains(range.commonAncestorContainer)) return null;
        const pre = range.cloneRange();
        pre.selectNodeContents(this.container);
        pre.setEnd(range.startContainer, range.startOffset);
        const start = pre.toString().length;
        const end = start + range.toString().length;
        if (start === end) return null;
        return { start, end };
    }

    applyMark(color) {
        const sel = this.getSelectionRange();
        if (!sel) return false;
        this.marks = this.marks.filter(m => m.end <= sel.start || m.start >= sel.end);
        if (color !== "borrar") {
            this.marks.push({ start: sel.start, end: sel.end, color });
        }
        this.render();
        this.updateDistribution();
        window.getSelection().removeAllRanges();
        if (this.onChange) this.onChange();
    }

    clearAll() {
        this.marks = [];
        this.render();
        this.updateDistribution();
        if (this.onChange) this.onChange();
    }

    computeDistribution() {
        const total = this.originalText.length;
        if (total === 0) return { pct_verde: 0, pct_amarillo: 0, pct_rojo: 0, pct_sin_marcar: 100 };
        const counts = { verde: 0, amarillo: 0, rojo: 0 };
        for (const m of this.marks) counts[m.color] += (m.end - m.start);
        const pct_verde = (counts.verde / total) * 100;
        const pct_amarillo = (counts.amarillo / total) * 100;
        const pct_rojo = (counts.rojo / total) * 100;
        const pct_sin_marcar = 100 - pct_verde - pct_amarillo - pct_rojo;
        return {
            pct_verde: Math.round(pct_verde * 10) / 10,
            pct_amarillo: Math.round(pct_amarillo * 10) / 10,
            pct_rojo: Math.round(pct_rojo * 10) / 10,
            pct_sin_marcar: Math.round(pct_sin_marcar * 10) / 10,
        };
    }

    updateDistribution() {
        const dist = this.computeDistribution();
        const scope = this.container.closest("section") || document;
        const setPct = (id, val) => {
            const el = scope.querySelector(`#${id}`);
            if (el) el.textContent = val + "%";
        };
        if (this.storageField === "marcaciones_nota") {
            setPct("pct-verde", dist.pct_verde);
            setPct("pct-amarillo", dist.pct_amarillo);
            setPct("pct-rojo", dist.pct_rojo);
            setPct("pct-neutral", dist.pct_sin_marcar);
        } else {
            setPct("pct-clf-verde", dist.pct_verde);
            setPct("pct-clf-amarillo", dist.pct_amarillo);
            setPct("pct-clf-rojo", dist.pct_rojo);
            setPct("pct-clf-neutral", dist.pct_sin_marcar);
        }
        state[this.storageField] = {
            marcaciones: this.marks,
            ...dist,
        };
    }
}

let hlNotaClinica = null;
let hlNotaClasificador = null;

// ------------------------------------------------------------
// 1. ECG + tracks intercalados por tira
// ------------------------------------------------------------
function drawECGWithTracks() {
    const container = document.getElementById("ecg-container");
    container.innerHTML = "";

    const signal = SIGNAL.raw_signal;
    let sigMin = Infinity, sigMax = -Infinity;
    for (let i = 0; i < signal.length; i++) {
        if (signal[i] < sigMin) sigMin = signal[i];
        if (signal[i] > sigMax) sigMax = signal[i];
    }
    const pad = (sigMax - sigMin) * 0.08;
    const yRange = [sigMin - pad, sigMax + pad];
    const cream = "#FBF5E5";
    const ecgRed = "#B22222";
    const gridRed = "rgba(178, 34, 34, 0.18)";
    const inkSoft = "#5C453A";

    for (let s = 0; s < N_STRIPS; s++) {
        const stripStart = s * STRIP_DURATION;
        const stripEnd = Math.min((s + 1) * STRIP_DURATION, DURATION);

        // Bloque contenedor de la tira + tracks
        const block = document.createElement("div");
        block.className = "strip-block";
        container.appendChild(block);

        // 1a. ECG plot (Plotly)
        const ecgDiv = document.createElement("div");
        ecgDiv.id = `ecg-strip-${s}`;
        ecgDiv.className = "ecg-strip";
        block.appendChild(ecgDiv);

        const startSample = s * STRIP_DURATION * FS;
        const endSample = Math.min(startSample + STRIP_DURATION * FS, signal.length);
        const stripSig = signal.slice(startSample, endSample);
        const t = stripSig.map((_, k) => stripStart + k / FS);

        const trace = {
            x: t, y: stripSig,
            type: "scattergl", mode: "lines",
            line: { color: ecgRed, width: 1.4 },
            hovertemplate: "t=%{x:.2f}s<br>amp=%{y:.2f}<extra></extra>",
        };
        const layout = {
            margin: { l: 70, r: 18, t: 2, b: 26 },
            paper_bgcolor: cream, plot_bgcolor: cream,
            xaxis: {
                range: [stripStart, stripStart + STRIP_DURATION],
                gridcolor: gridRed,
                tickfont: { color: inkSoft, size: 10 },
                dtick: 1, showgrid: true, zeroline: false, showspikes: false,
            },
            yaxis: {
                range: yRange, gridcolor: gridRed,
                tickfont: { color: inkSoft, size: 9 },
                title: {
                    text: `<b>${Math.floor(stripStart)}-${Math.floor(stripEnd)}s</b>`,
                    font: { size: 11, color: inkSoft, family: "JetBrains Mono, monospace" },
                    standoff: 8,
                },
                showgrid: true, zeroline: false,
            },
            showlegend: false,
        };
        Plotly.newPlot(ecgDiv.id, [trace], layout, {
            displayModeBar: false, responsive: true, staticPlot: false,
        });

        // 1b. Track de LATIDO para esta tira
        const beatTrack = document.createElement("div");
        beatTrack.className = "beat-track-strip";
        beatTrack.dataset.strip = s;
        block.appendChild(beatTrack);
        renderBeatTrackForStrip(beatTrack, s, stripStart, stripEnd);

        // 1c. Track de RITMO para esta tira
        const rhythmTrack = document.createElement("div");
        rhythmTrack.className = "rhythm-track-strip";
        rhythmTrack.dataset.strip = s;
        block.appendChild(rhythmTrack);
        renderRhythmTrackForStrip(rhythmTrack, s, stripStart, stripEnd);
    }
}

// ------------------------------------------------------------
// 2. Beat track para una tira (10 celdas, 1 por segundo)
// ------------------------------------------------------------
function renderBeatTrackForStrip(container, stripIdx, stripStart, stripEnd) {
    container.innerHTML = "";
    // Header pequeño con el nombre del track
    const header = document.createElement("div");
    header.className = "track-header";
    header.innerHTML = `<span class="track-name">LATIDO</span>
        <span class="track-hint">${Math.floor(stripStart)}-${Math.floor(stripEnd)}s</span>`;
    container.appendChild(header);

    // Fila 1: labels
    const row1 = document.createElement("div");
    row1.className = "label-row";
    row1.style.gridTemplateColumns = `repeat(${STRIP_DURATION}, minmax(0, 1fr))`;
    // Fila 2: semaforos
    const row2 = document.createElement("div");
    row2.className = "label-row";
    row2.style.gridTemplateColumns = `repeat(${STRIP_DURATION}, minmax(0, 1fr))`;

    for (let i = 0; i < STRIP_DURATION; i++) {
        const beatIdx = stripIdx * STRIP_DURATION + i;
        if (beatIdx >= N_BEAT_WIN) break;
        const w = SIGNAL.beat_windows[beatIdx];

        // Label cell
        const cell = document.createElement("div");
        cell.className = `label-cell ${w.prediction}${w.escalation_flag ? " escalation" : ""}`;
        cell.title = `Ventana ${beatIdx}: ${w.prediction}\nNivel de certeza: ${w.confidence_gap.toFixed(2)}`;
        cell.textContent = LABEL_ABBR[w.prediction] || w.prediction;
        row1.appendChild(cell);

        // Semaforo cell
        const sem = document.createElement("div");
        sem.className = "semaforo beat-semaforo";
        ["verde", "amarillo", "rojo"].forEach((color) => {
            const btn = document.createElement("button");
            btn.className = color;
            btn.title = color;
            btn.dataset.beatIndex = beatIdx;
            btn.dataset.color = color;
            btn.addEventListener("click", () => {
                state.beat_semaforos[beatIdx] = color;
                sem.querySelectorAll("button").forEach((b) => b.classList.remove("active"));
                btn.classList.add("active");
                autosave();
            });
            sem.appendChild(btn);
        });
        row2.appendChild(sem);
    }
    container.appendChild(row1);
    container.appendChild(row2);
}

// ------------------------------------------------------------
// 3. Rhythm track para una tira
// Rhythms de 4s. Si un rhythm cruza dos tiras, aparece en ambas
// con celdas mas cortas, pero el semaforo se sincroniza.
// ------------------------------------------------------------
function renderRhythmTrackForStrip(container, stripIdx, stripStart, stripEnd) {
    container.innerHTML = "";
    // Header
    const header = document.createElement("div");
    header.className = "track-header";
    header.innerHTML = `<span class="track-name">RITMO</span>
        <span class="track-hint">${Math.floor(stripStart)}-${Math.floor(stripEnd)}s</span>`;
    container.appendChild(header);

    // Grid con STRIP_DURATION*2 columnas (0.5s cada una) para posicionamiento fino
    const N_COLS = STRIP_DURATION * 2;  // 20 columnas
    const row1 = document.createElement("div");
    row1.className = "label-row";
    row1.style.gridTemplateColumns = `repeat(${N_COLS}, minmax(0, 1fr))`;
    const row2 = document.createElement("div");
    row2.className = "label-row";
    row2.style.gridTemplateColumns = `repeat(${N_COLS}, minmax(0, 1fr))`;

    // Recorrer todos los rhythms y encontrar los que se solapen con esta tira
    SIGNAL.rhythm_windows.forEach((w, i) => {
        const winStart = i * 4;
        const winEnd = winStart + 4;
        const overlapStart = Math.max(winStart, stripStart);
        const overlapEnd = Math.min(winEnd, stripEnd);
        if (overlapEnd <= overlapStart) return;

        // Convertir a columnas (0.5s por columna)
        const colStart = Math.round((overlapStart - stripStart) * 2) + 1;
        const colEnd = Math.round((overlapEnd - stripStart) * 2) + 1;
        if (colEnd <= colStart) return;

        // Label cell
        const cell = document.createElement("div");
        cell.className = `label-cell ${w.prediction}${w.escalation_flag ? " escalation" : ""}`;
        cell.style.gridColumn = `${colStart} / ${colEnd}`;
        cell.title = `Ventana ${i}: ${w.prediction} (${winStart}-${winEnd}s)\n`
                   + `Nivel de certeza: ${w.confidence_gap.toFixed(2)}`;
        cell.textContent = LABEL_ABBR[w.prediction] || w.prediction;
        row1.appendChild(cell);

        // Semaforo cell (sincronizado entre tiras con mismo rhythm-index)
        const sem = document.createElement("div");
        sem.className = "semaforo rhythm-semaforo";
        sem.style.gridColumn = `${colStart} / ${colEnd}`;
        ["verde", "amarillo", "rojo"].forEach((color) => {
            const btn = document.createElement("button");
            btn.className = color;
            btn.title = color;
            btn.dataset.rhythmIndex = i;
            btn.dataset.color = color;
            btn.addEventListener("click", () => {
                state.rhythm_semaforos[i] = color;
                // Sincronizar TODOS los botones de este rhythm en cualquier tira
                document.querySelectorAll(
                    `.rhythm-semaforo button[data-rhythm-index="${i}"]`
                ).forEach(b => b.classList.remove("active"));
                document.querySelectorAll(
                    `.rhythm-semaforo button[data-rhythm-index="${i}"][data-color="${color}"]`
                ).forEach(b => b.classList.add("active"));
                autosave();
            });
            sem.appendChild(btn);
        });
        row2.appendChild(sem);
    });

    container.appendChild(row1);
    container.appendChild(row2);
}

// ------------------------------------------------------------
// 4. LLM notes (clinica + clasificador)
// ------------------------------------------------------------
async function loadLLMNotes() {
    const noteBox = document.getElementById("note-text");
    const clfBox  = document.getElementById("classifier-note-text");
    const stamp   = document.getElementById("backend-stamp");
    try {
        const resp = await fetch(`/api/llm_note/${window.SIGNAL_ID}?backend=${window.BACKEND_ACTIVO}`);
        const data = await resp.json();
        if (data.error) {
            noteBox.textContent = `[Error del backend: ${data.error}]`;
            stamp.textContent = `${data.backend_name} (error)`;
            return;
        }
        hlNotaClinica.setText(data.clinical_note || data.text || "");
        hlNotaClinica.updateDistribution();

        const classifierText = data.classifier_note || "";
        if (classifierText) {
            hlNotaClasificador.setText(classifierText);
            hlNotaClasificador.updateDistribution();
            document.getElementById("classifier-note-section").style.display = "";
            document.getElementById("classifier-note-instructions").style.display = "";
            document.getElementById("comentarios-clasificador-wrap").style.display = "";
        } else {
            document.getElementById("classifier-note-section").style.display = "none";
            document.getElementById("classifier-note-instructions").style.display = "none";
            document.getElementById("comentarios-clasificador-wrap").style.display = "none";
        }
        stamp.textContent = `${data.backend_name} - ${data.model_id} - ${data.latency_seconds}s`;
        state.llm_payload = data;
    } catch (e) {
        noteBox.textContent = `[Error de red: ${e.message}]`;
        stamp.textContent = "-";
    }
}

// ------------------------------------------------------------
// 5. Semaforos globales (notas)
// ------------------------------------------------------------
function setupNoteSemaforos() {
    const contNota = document.getElementById("semaforo-nota");
    if (contNota) {
        contNota.querySelectorAll("button").forEach((btn) => {
            btn.addEventListener("click", () => {
                state.nota_semaforo = btn.dataset.color;
                contNota.querySelectorAll("button").forEach((b) => b.classList.remove("active"));
                btn.classList.add("active");
                autosave();
            });
        });
    }
    const contClf = document.getElementById("semaforo-clasificador");
    if (contClf) {
        contClf.querySelectorAll("button").forEach((btn) => {
            btn.addEventListener("click", () => {
                state.nota_clasificador_semaforo = btn.dataset.color;
                contClf.querySelectorAll("button").forEach((b) => b.classList.remove("active"));
                btn.classList.add("active");
                autosave();
            });
        });
    }
}

// ------------------------------------------------------------
// 6. Toolbars de marcado
// ------------------------------------------------------------
function setupMarkToolbars() {
    document.querySelectorAll("#mark-toolbar [data-mark]").forEach((btn) => {
        btn.addEventListener("click", () => {
            hlNotaClinica.applyMark(btn.dataset.mark);
            autosave();
        });
    });
    const btnLimpiarNota = document.getElementById("btn-limpiar-marcas");
    if (btnLimpiarNota) {
        btnLimpiarNota.addEventListener("click", () => {
            hlNotaClinica.clearAll();
            autosave();
        });
    }
    document.querySelectorAll("#mark-toolbar-clf [data-mark]").forEach((btn) => {
        btn.addEventListener("click", () => {
            hlNotaClasificador.applyMark(btn.dataset.mark);
            autosave();
        });
    });
    const btnLimpiarClf = document.getElementById("btn-limpiar-marcas-clf");
    if (btnLimpiarClf) {
        btnLimpiarClf.addEventListener("click", () => {
            hlNotaClasificador.clearAll();
            autosave();
        });
    }
}

// ------------------------------------------------------------
// 7. Likert
// ------------------------------------------------------------
function setupLikert(elementId, stateKey) {
    const container = document.getElementById(elementId);
    if (!container) return;
    container.querySelectorAll(".scale button").forEach((btn) => {
        btn.addEventListener("click", () => {
            state[stateKey] = parseInt(btn.dataset.val, 10);
            container.querySelectorAll(".scale button").forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");
            autosave();
        });
    });
}

// ------------------------------------------------------------
// 8. Comentarios
// ------------------------------------------------------------
function setupComentarios() {
    const inputNota = document.getElementById("comentarios_nota");
    if (inputNota) {
        inputNota.addEventListener("input", () => {
            state.comentarios_nota = inputNota.value;
            autosave();
        });
    }
    const inputClf = document.getElementById("comentarios_clasificador");
    if (inputClf) {
        inputClf.addEventListener("input", () => {
            state.comentarios_clasificador = inputClf.value;
            autosave();
        });
    }
}

// ------------------------------------------------------------
// 9. AUTOSAVE en localStorage
// ------------------------------------------------------------
let autosaveTimer = null;
function autosave() {
    if (autosaveTimer) clearTimeout(autosaveTimer);
    autosaveTimer = setTimeout(() => {
        try {
            const snapshot = {
                ...state,
                _timestamp: Date.now(),
                _hl_clinica_marks: hlNotaClinica ? hlNotaClinica.marks : [],
                _hl_clf_marks: hlNotaClasificador ? hlNotaClasificador.marks : [],
            };
            localStorage.setItem(STORAGE_KEY, JSON.stringify(snapshot));
            const indicator = document.getElementById("autosave-indicator");
            if (indicator) {
                indicator.textContent = "Guardado local: " + new Date().toLocaleTimeString();
                indicator.classList.add("visible");
                clearTimeout(indicator._t);
                indicator._t = setTimeout(() => indicator.classList.remove("visible"), 1500);
            }
            updateProgress();
        } catch (e) {
            console.warn("No se pudo guardar en localStorage:", e);
        }
    }, AUTOSAVE_DEBOUNCE_MS);
}

function restoreFromLocalStorage() {
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) return;
        const snap = JSON.parse(raw);

        if (Array.isArray(snap.beat_semaforos))   state.beat_semaforos   = snap.beat_semaforos;
        if (Array.isArray(snap.rhythm_semaforos)) state.rhythm_semaforos = snap.rhythm_semaforos;
        state.nota_semaforo               = snap.nota_semaforo || null;
        state.nota_clasificador_semaforo  = snap.nota_clasificador_semaforo || null;
        state.likert_exactitud            = snap.likert_exactitud || null;
        state.likert_coherencia           = snap.likert_coherencia || null;
        state.likert_utilidad             = snap.likert_utilidad || null;
        state.comentarios_nota            = snap.comentarios_nota || "";
        state.comentarios_clasificador    = snap.comentarios_clasificador || "";

        // Restaurar UI de beats
        for (let i = 0; i < N_BEAT_WIN; i++) {
            if (state.beat_semaforos[i]) {
                const btn = document.querySelector(
                    `.beat-semaforo button[data-beat-index="${i}"][data-color="${state.beat_semaforos[i]}"]`);
                if (btn) btn.classList.add("active");
            }
        }
        // Restaurar UI de rhythms (marcar TODOS los botones con el mismo rhythm-index)
        for (let i = 0; i < N_RHYTHM_WIN; i++) {
            if (state.rhythm_semaforos[i]) {
                document.querySelectorAll(
                    `.rhythm-semaforo button[data-rhythm-index="${i}"][data-color="${state.rhythm_semaforos[i]}"]`
                ).forEach(b => b.classList.add("active"));
            }
        }
        // Semaforos globales
        if (state.nota_semaforo) {
            const b = document.querySelector(`#semaforo-nota button[data-color="${state.nota_semaforo}"]`);
            if (b) b.classList.add("active");
        }
        if (state.nota_clasificador_semaforo) {
            const b = document.querySelector(`#semaforo-clasificador button[data-color="${state.nota_clasificador_semaforo}"]`);
            if (b) b.classList.add("active");
        }
        // Likert
        [["exactitud", state.likert_exactitud],
         ["coherencia", state.likert_coherencia],
         ["utilidad", state.likert_utilidad]].forEach(([k, v]) => {
            if (v) {
                const b = document.querySelector(`#likert-${k} .scale button[data-val="${v}"]`);
                if (b) b.classList.add("active");
            }
        });
        // Comentarios
        const cn = document.getElementById("comentarios_nota");
        if (cn && state.comentarios_nota) cn.value = state.comentarios_nota;
        const cc = document.getElementById("comentarios_clasificador");
        if (cc && state.comentarios_clasificador) cc.value = state.comentarios_clasificador;

        // Marcas de subrayado (aplicar despues de cargar las notas del LLM)
        window._pendingHlMarks = {
            clinica: snap._hl_clinica_marks || [],
            clasificador: snap._hl_clf_marks || [],
        };

        console.log("Restaurado desde localStorage (borrador guardado)");
    } catch (e) {
        console.warn("No se pudo restaurar de localStorage:", e);
    }
}

function applyPendingMarks() {
    if (!window._pendingHlMarks) return;
    if (hlNotaClinica && window._pendingHlMarks.clinica.length) {
        hlNotaClinica.marks = window._pendingHlMarks.clinica;
        hlNotaClinica.render();
        hlNotaClinica.updateDistribution();
    }
    if (hlNotaClasificador && window._pendingHlMarks.clasificador.length) {
        hlNotaClasificador.marks = window._pendingHlMarks.clasificador;
        hlNotaClasificador.render();
        hlNotaClasificador.updateDistribution();
    }
    window._pendingHlMarks = null;
}

// ------------------------------------------------------------
// 10. Progreso
// ------------------------------------------------------------
function updateProgress() {
    const beatOK   = state.beat_semaforos.filter(v => v !== null).length;
    const rhythmOK = state.rhythm_semaforos.filter(v => v !== null).length;
    const notaOK   = state.nota_semaforo ? 1 : 0;
    const clfOK    = state.nota_clasificador_semaforo ? 1 : 0;
    const likertOK = (state.likert_exactitud ? 1 : 0) +
                     (state.likert_coherencia ? 1 : 0) +
                     (state.likert_utilidad ? 1 : 0);
    const clfSection = document.getElementById("classifier-note-section");
    const requiereClf = clfSection && clfSection.style.display !== "none";
    const totalRequired = N_BEAT_WIN + N_RHYTHM_WIN + 1 + 3 + (requiereClf ? 1 : 0);
    const totalDone = beatOK + rhythmOK + notaOK + likertOK + (requiereClf ? clfOK : 0);
    const pct = Math.round((totalDone / totalRequired) * 100);
    const fill = document.getElementById("progress-bar-fill");
    if (fill) fill.style.width = pct + "%";
    const text = document.getElementById("progress-text");
    if (text) text.textContent = `${totalDone} / ${totalRequired} elementos completados (${pct}%)`;
    const detail = document.getElementById("progress-detail");
    if (detail) {
        detail.textContent = `LATIDO: ${beatOK}/${N_BEAT_WIN} - RITMO: ${rhythmOK}/${N_RHYTHM_WIN} - `
            + `Nota: ${notaOK}/1 - Likert: ${likertOK}/3`
            + (requiereClf ? ` - Clasificador: ${clfOK}/1` : "");
    }
    const btn = document.getElementById("btn-guardar");
    if (btn) btn.disabled = (totalDone < totalRequired);
}

// ------------------------------------------------------------
// 11. Guardar (submission definitivo)
// ------------------------------------------------------------
function setupSubmit() {
    document.getElementById("btn-guardar").addEventListener("click", async () => {
        const msg = document.getElementById("submit-msg");
        msg.innerHTML = "";
        const errors = [];
        if (state.beat_semaforos.some((v) => v === null))
            errors.push("Falta calificar todas las ventanas de LATIDO.");
        if (state.rhythm_semaforos.some((v) => v === null))
            errors.push("Falta calificar todas las ventanas de RITMO.");
        if (!state.nota_semaforo)
            errors.push("Falta calificar la nota clinica con el semaforo global.");
        if (!state.likert_exactitud || !state.likert_coherencia || !state.likert_utilidad)
            errors.push("Falta calificar los tres ejes Likert.");
        const clfSection = document.getElementById("classifier-note-section");
        const requiereClf = clfSection && clfSection.style.display !== "none";
        if (requiereClf && !state.nota_clasificador_semaforo)
            errors.push("Falta calificar la nota del clasificador con el semaforo.");
        if (errors.length) {
            msg.innerHTML = `<div class="error">${errors.join("<br>")}</div>`;
            window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
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
        const payload = {
            signal_id: window.SIGNAL_ID,
            llm_backend: state.llm_payload?.backend_name || window.BACKEND_ACTIVO,
            llm_model_id: state.llm_payload?.model_id || null,
            nota_clinica_texto: hlNotaClinica ? hlNotaClinica.originalText : "",
            nota_semaforo_global: state.nota_semaforo,
            likert_exactitud: state.likert_exactitud,
            likert_coherencia: state.likert_coherencia,
            likert_utilidad: state.likert_utilidad,
            marcaciones_nota: state.marcaciones_nota,
            comentarios_nota: state.comentarios_nota,
            nota_clasificador_texto: hlNotaClasificador ? hlNotaClasificador.originalText : "",
            nota_clasificador_semaforo: state.nota_clasificador_semaforo,
            nota_clasificador_marcaciones: state.marcaciones_clasificador,
            comentarios_clasificador: state.comentarios_clasificador,
            duracion_segundos: (Date.now() - T_START) / 1000,
            respuestas_ventana: respuestas,
        };
        const resp = await fetch("/api/guardar", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        const data = await resp.json();
        if (data.ok) {
            try { localStorage.removeItem(STORAGE_KEY); } catch (e) {}
            msg.innerHTML = `<div class="success">Evaluacion guardada (ID: ${data.evaluacion_id}). Gracias.</div>`;
            setTimeout(() => { window.location.href = "/seleccion"; }, 1800);
        } else {
            msg.innerHTML = `<div class="error">Error al guardar: ${data.error || "desconocido"}</div>`;
        }
    });
}

// ------------------------------------------------------------
// Init
// ------------------------------------------------------------
hlNotaClinica = new HighlightManager("note-text", "marcaciones_nota", updateProgress);
hlNotaClasificador = new HighlightManager("classifier-note-text", "marcaciones_clasificador", updateProgress);

drawECGWithTracks();

restoreFromLocalStorage();

loadLLMNotes().then(() => {
    applyPendingMarks();
    updateProgress();
});

setupNoteSemaforos();
setupMarkToolbars();
setupLikert("likert-exactitud",  "likert_exactitud");
setupLikert("likert-coherencia", "likert_coherencia");
setupLikert("likert-utilidad",   "likert_utilidad");
setupComentarios();
setupSubmit();
