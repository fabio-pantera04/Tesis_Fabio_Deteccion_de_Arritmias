// =================================================================
// Página de evaluación
// =================================================================
const SIGNAL = JSON.parse(document.getElementById("signal-data").textContent);
const FS = SIGNAL.signal_metadata.sampling_rate_hz;       // 250 Hz
const DURATION = SIGNAL.signal_metadata.duration_seconds; // 60 s
const N_BEAT_WIN = SIGNAL.beat_windows.length;            // 60
const N_RHYTHM_WIN = SIGNAL.rhythm_windows.length;        // 15

const T_START = Date.now();
setInterval(() => {
    document.getElementById("elapsed").textContent = Math.round((Date.now() - T_START) / 1000);
}, 1000);

// State of the evaluation form
const state = {
    beat_semaforos:   new Array(N_BEAT_WIN).fill(null),
    rhythm_semaforos: new Array(N_RHYTHM_WIN).fill(null),
    nota_semaforo: null,
    likert_exactitud: null,
    likert_coherencia: null,
    likert_utilidad: null,
    llm_payload: null,
};

// ---------------------------------------------------------
// 1. ECG en 6 tiras de 10 segundos (formato clínico estándar)
// ---------------------------------------------------------
function drawECG() {
    const signal = SIGNAL.raw_signal;
    const N_STRIPS = 6;
    const STRIP_DURATION = 10;                // segundos por tira
    const STRIP_SAMPLES = STRIP_DURATION * FS; // 2500 muestras por tira

    const container = document.getElementById("ecg-strips");
    container.innerHTML = "";

    // Rango y compartido entre todas las tiras (para que escalas sean
    // comparables visualmente — un latido alto en la tira 1 se ve igual
    // de alto en la tira 6).
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

    for (let i = 0; i < N_STRIPS; i++) {
        const stripDiv = document.createElement("div");
        stripDiv.id = `ecg-strip-${i}`;
        stripDiv.className = "ecg-strip";
        container.appendChild(stripDiv);

        const startSample = i * STRIP_SAMPLES;
        const endSample = Math.min(startSample + STRIP_SAMPLES, signal.length);
        const stripSig = signal.slice(startSample, endSample);
        const t = stripSig.map((_, k) => i * STRIP_DURATION + k / FS);

        const trace = {
            x: t, y: stripSig,
            type: "scattergl", mode: "lines",
            line: { color: ecgRed, width: 1.4 },
            hovertemplate: "t=%{x:.2f}s<br>amp=%{y:.2f}<extra></extra>",
        };

        const layout = {
            margin: { l: 70, r: 18, t: 2, b: 26 },
            paper_bgcolor: cream,
            plot_bgcolor: cream,
            xaxis: {
                range: [i * STRIP_DURATION, (i + 1) * STRIP_DURATION],
                gridcolor: gridRed,
                tickfont: { color: inkSoft, size: 10 },
                dtick: 1,             // marca cada 1 segundo (cuadro grande de 0.2s × 5)
                showgrid: true,
                zeroline: false,
                showspikes: false,
            },
            yaxis: {
                range: yRange,
                gridcolor: gridRed,
                tickfont: { color: inkSoft, size: 9 },
                title: {
                    text: `<b>${i * 10}–${(i + 1) * 10}s</b>`,
                    font: { size: 11, color: inkSoft, family: "JetBrains Mono, monospace" },
                    standoff: 8,
                },
                showgrid: true,
                zeroline: false,
            },
            showlegend: false,
        };

        Plotly.newPlot(stripDiv.id, [trace], layout, {
            displayModeBar: false, responsive: true, staticPlot: false,
        });
    }
}

// ---------------------------------------------------------
// 2. Label tracks (BEAT y RHYTHM)
// ---------------------------------------------------------
function buildLabelTrack(containerId, windows, totalWindows, scale) {
    const container = document.getElementById(containerId);
    container.innerHTML = "";

    // Two rows: top labels, bottom semaforos
    // minmax(0,1fr) forces equal partition; without minmax(0,...) the cells
    // expand to fit their text and overflow horizontally.
    const row1 = document.createElement("div");
    row1.className = "label-row";
    row1.style.gridTemplateColumns = `repeat(${totalWindows}, minmax(0, 1fr))`;

    const row2 = document.createElement("div");
    row2.className = "label-row";
    row2.style.gridTemplateColumns = `repeat(${totalWindows}, minmax(0, 1fr))`;

    const LABEL_ABBR = { NORMAL: "N", PAC: "PAC", NSR: "NSR", AFIB: "AF" };
    windows.forEach((w, i) => {
        // Label cell
        const cell = document.createElement("div");
        cell.className = `label-cell ${w.prediction}${w.escalation_flag ? " escalation" : ""}`;
        cell.title = `Ventana ${i}: ${w.prediction}\nConfidence gap: ${w.confidence_gap.toFixed(2)}`;
        cell.textContent = LABEL_ABBR[w.prediction] || w.prediction;
        row1.appendChild(cell);

        // Semaforo cell
        const sem = document.createElement("div");
        sem.className = "semaforo";
        ["verde", "amarillo", "rojo"].forEach((color) => {
            const btn = document.createElement("button");
            btn.className = color;
            btn.title = color;
            btn.dataset.windowIndex = i;
            btn.dataset.color = color;
            btn.addEventListener("click", () => {
                state[`${scale}_semaforos`][i] = color;
                Array.from(sem.children).forEach((b) => b.classList.remove("active"));
                btn.classList.add("active");
            });
            sem.appendChild(btn);
        });
        row2.appendChild(sem);
    });

    container.appendChild(row1);
    container.appendChild(row2);
}

// ---------------------------------------------------------
// 3. LLM note loader
// ---------------------------------------------------------
async function loadLLMNote() {
    const noteBox = document.getElementById("note-text");
    const stamp = document.getElementById("backend-stamp");
    try {
        const resp = await fetch(`/api/llm_note/${window.SIGNAL_ID}?backend=${window.BACKEND_ACTIVO}`);
        const data = await resp.json();
        if (data.error) {
            noteBox.textContent = `[Error del backend: ${data.error}]`;
            stamp.textContent = `${data.backend_name} (error)`;
            return;
        }
        noteBox.textContent = data.text;
        stamp.textContent = `${data.backend_name} · ${data.model_id} · ${data.latency_seconds}s`;
        state.llm_payload = data;
    } catch (e) {
        noteBox.textContent = `[Error de red: ${e.message}]`;
        stamp.textContent = "—";
    }
}

// ---------------------------------------------------------
// 4. Semaforo grande de la nota
// ---------------------------------------------------------
function setupNoteSemaforo() {
    const container = document.getElementById("semaforo-nota");
    const noteText = document.getElementById("note-text");
    container.querySelectorAll("button").forEach((btn) => {
        btn.addEventListener("click", () => {
            const color = btn.dataset.color;
            state.nota_semaforo = color;
            container.querySelectorAll("button").forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");
            // Highlight note background
            noteText.classList.remove("verde-bg", "amarillo-bg", "rojo-bg");
            noteText.classList.add(`${color}-bg`);
        });
    });
}

// ---------------------------------------------------------
// 5. Likert
// ---------------------------------------------------------
function setupLikert(elementId, stateKey) {
    const container = document.getElementById(elementId);
    container.querySelectorAll(".scale button").forEach((btn) => {
        btn.addEventListener("click", () => {
            state[stateKey] = parseInt(btn.dataset.val, 10);
            container.querySelectorAll(".scale button").forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");
        });
    });
}

// ---------------------------------------------------------
// 6. Save submission
// ---------------------------------------------------------
function setupSubmit() {
    document.getElementById("btn-guardar").addEventListener("click", async () => {
        const msg = document.getElementById("submit-msg");
        msg.innerHTML = "";

        // Validate completeness
        const errors = [];
        if (state.beat_semaforos.some((v) => v === null))
            errors.push("Falta calificar todas las ventanas de LATIDO.");
        if (state.rhythm_semaforos.some((v) => v === null))
            errors.push("Falta calificar todas las ventanas de RITMO.");
        if (!state.nota_semaforo)
            errors.push("Falta calificar la nota clínica con el semáforo global.");
        if (!state.likert_exactitud || !state.likert_coherencia || !state.likert_utilidad)
            errors.push("Falta calificar los tres ejes Likert.");

        if (errors.length) {
            msg.innerHTML = `<div class="error">${errors.join("<br>")}</div>`;
            window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
            return;
        }

        // Build respuestas_ventana payload
        const respuestas = [];
        SIGNAL.beat_windows.forEach((w, i) => {
            respuestas.push({
                escala: "beat",
                window_index: i,
                prediccion_modelo: w.prediction,
                semaforo: state.beat_semaforos[i],
                confidence_gap: w.confidence_gap,
            });
        });
        SIGNAL.rhythm_windows.forEach((w, i) => {
            respuestas.push({
                escala: "rhythm",
                window_index: i,
                prediccion_modelo: w.prediction,
                semaforo: state.rhythm_semaforos[i],
                confidence_gap: w.confidence_gap,
            });
        });

        const payload = {
            signal_id: window.SIGNAL_ID,
            llm_backend: state.llm_payload?.backend_name || window.BACKEND_ACTIVO,
            llm_model_id: state.llm_payload?.model_id || null,
            nota_clinica_texto: document.getElementById("note-text").textContent,
            nota_semaforo_global: state.nota_semaforo,
            likert_exactitud: state.likert_exactitud,
            likert_coherencia: state.likert_coherencia,
            likert_utilidad: state.likert_utilidad,
            comentarios_libres: document.getElementById("comentarios_libres").value,
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
            msg.innerHTML = `<div class="success">Evaluación guardada (ID: ${data.evaluacion_id}). Gracias.</div>`;
            setTimeout(() => { window.location.href = "/seleccion"; }, 1800);
        } else {
            msg.innerHTML = `<div class="error">Error al guardar: ${data.error || "desconocido"}</div>`;
        }
    });
}

// ---------------------------------------------------------
// Init
// ---------------------------------------------------------
drawECG();
buildLabelTrack("track-beat",   SIGNAL.beat_windows,   N_BEAT_WIN,   "beat");
buildLabelTrack("track-rhythm", SIGNAL.rhythm_windows, N_RHYTHM_WIN, "rhythm");
loadLLMNote();
setupNoteSemaforo();
setupLikert("likert-exactitud",  "likert_exactitud");
setupLikert("likert-coherencia", "likert_coherencia");
setupLikert("likert-utilidad",   "likert_utilidad");
setupSubmit();
