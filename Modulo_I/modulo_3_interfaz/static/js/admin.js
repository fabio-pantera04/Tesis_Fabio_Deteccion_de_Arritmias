/* ============================================================
   admin.js - Gráficos Plotly del Panel de Administración
   ============================================================ */

const COLORS = {
    verde:    "#3A8F3A",
    amarillo: "#D6A317",
    rojo:     "#B22222",
    ink:      "#2A1810",
    inkSoft:  "#5C453A",
    cream:    "#FBF5E5",
};

const COMMON_LAYOUT = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor:  "rgba(0,0,0,0)",
    font: { family: "Inter, sans-serif", color: COLORS.ink, size: 12 },
    margin: { l: 50, r: 20, t: 10, b: 40 },
    xaxis: { gridcolor: "rgba(178,34,34,0.10)" },
    yaxis: { gridcolor: "rgba(178,34,34,0.10)" },
    legend: {
        orientation: "h",
        y: -0.15,
        x: 0.5,
        xanchor: "center",
        font: { size: 11 },
    },
};

async function loadDashboardCharts() {
    try {
        const resp = await fetch("/admin/api/dashboard_data");
        const data = await resp.json();

        drawSemaforoPorEscala(data.por_escala);
        drawSemaforoNota(data.semaforo_nota);
        drawSemaforoPorClase("chart-clase-beat", data.por_clase_beat);
        drawSemaforoPorClase("chart-clase-rhythm", data.por_clase_rhythm);
        drawEvaluacionesPorSenal(data.por_senal);
        drawMarcaciones(data.marcaciones);
    } catch (e) {
        console.error("Error cargando dashboard:", e);
    }
}

function drawSemaforoPorEscala(data) {
    const escalas = ["beat", "rhythm"];
    const traces = [
        { name: "Correcto (verde)",   type: "bar", x: escalas.map(e => e.toUpperCase()),
          y: escalas.map(e => data[e]?.verde || 0), marker: { color: COLORS.verde },
          text: escalas.map(e => `${(data[e]?.verde || 0).toFixed(1)}%`), textposition: "auto" },
        { name: "Parcial (amarillo)", type: "bar", x: escalas.map(e => e.toUpperCase()),
          y: escalas.map(e => data[e]?.amarillo || 0), marker: { color: COLORS.amarillo },
          text: escalas.map(e => `${(data[e]?.amarillo || 0).toFixed(1)}%`), textposition: "auto" },
        { name: "Incorrecto (rojo)",  type: "bar", x: escalas.map(e => e.toUpperCase()),
          y: escalas.map(e => data[e]?.rojo || 0), marker: { color: COLORS.rojo },
          text: escalas.map(e => `${(data[e]?.rojo || 0).toFixed(1)}%`), textposition: "auto" },
    ];
    Plotly.newPlot("chart-semaforo-escala", traces, {
        ...COMMON_LAYOUT, barmode: "stack",
        yaxis: { ...COMMON_LAYOUT.yaxis, range: [0, 100], title: "%" },
    }, { displayModeBar: false, responsive: true });
}

function drawSemaforoNota(data) {
    const traces = [{
        type: "pie",
        labels: ["Correcta", "Parcial", "Errónea"],
        values: [data.verde || 0, data.amarillo || 0, data.rojo || 0],
        marker: { colors: [COLORS.verde, COLORS.amarillo, COLORS.rojo] },
        textinfo: "label+percent",
        hovertemplate: "%{label}: %{value:.1f}%<extra></extra>",
    }];
    Plotly.newPlot("chart-semaforo-nota", traces, {
        ...COMMON_LAYOUT,
        margin: { l: 20, r: 20, t: 20, b: 20 },
        showlegend: false,
    }, { displayModeBar: false, responsive: true });
}

function drawSemaforoPorClase(divId, data) {
    const clases = Object.keys(data).sort();
    if (clases.length === 0) {
        document.getElementById(divId).innerHTML =
            '<p style="text-align:center; color:#8A7560; padding:40px;">Sin datos aún.</p>';
        return;
    }
    const traces = [
        { name: "Correcto",   type: "bar", x: clases,
          y: clases.map(c => data[c]?.verde || 0), marker: { color: COLORS.verde },
          text: clases.map(c => `${(data[c]?.verde || 0).toFixed(0)}%`), textposition: "inside" },
        { name: "Parcial",    type: "bar", x: clases,
          y: clases.map(c => data[c]?.amarillo || 0), marker: { color: COLORS.amarillo },
          text: clases.map(c => `${(data[c]?.amarillo || 0).toFixed(0)}%`), textposition: "inside" },
        { name: "Incorrecto", type: "bar", x: clases,
          y: clases.map(c => data[c]?.rojo || 0), marker: { color: COLORS.rojo },
          text: clases.map(c => `${(data[c]?.rojo || 0).toFixed(0)}%`), textposition: "inside" },
    ];
    Plotly.newPlot(divId, traces, {
        ...COMMON_LAYOUT, barmode: "stack",
        yaxis: { ...COMMON_LAYOUT.yaxis, range: [0, 100], title: "%" },
    }, { displayModeBar: false, responsive: true });
}

function drawEvaluacionesPorSenal(data) {
    if (!data || data.length === 0) {
        document.getElementById("chart-por-senal").innerHTML =
            '<p style="text-align:center; color:#8A7560; padding:40px;">Sin datos aún.</p>';
        return;
    }
    const signals = data.map(d => d.signal_id);
    const traces = [
        { name: "N evaluaciones", type: "bar", x: signals,
          y: data.map(d => d.n), marker: { color: COLORS.ink }, yaxis: "y",
          text: data.map(d => d.n), textposition: "auto" },
        { name: "Likert Exactitud",  type: "scatter", mode: "lines+markers",
          x: signals, y: data.map(d => d.prom_exactitud),
          line: { color: COLORS.rojo, width: 2 }, marker: { size: 8 }, yaxis: "y2" },
        { name: "Likert Coherencia", type: "scatter", mode: "lines+markers",
          x: signals, y: data.map(d => d.prom_coherencia),
          line: { color: COLORS.amarillo, width: 2 }, marker: { size: 8 }, yaxis: "y2" },
        { name: "Likert Utilidad",   type: "scatter", mode: "lines+markers",
          x: signals, y: data.map(d => d.prom_utilidad),
          line: { color: COLORS.verde, width: 2 }, marker: { size: 8 }, yaxis: "y2" },
    ];
    Plotly.newPlot("chart-por-senal", traces, {
        ...COMMON_LAYOUT,
        yaxis:  { ...COMMON_LAYOUT.yaxis, title: "N evaluaciones", side: "left" },
        yaxis2: { title: "Likert promedio", range: [0, 5],
                  overlaying: "y", side: "right",
                  gridcolor: "rgba(0,0,0,0)" },
    }, { displayModeBar: false, responsive: true });
}

function drawMarcaciones(data) {
    if (!data || data.n === 0) {
        document.getElementById("chart-marcaciones").innerHTML =
            '<p style="text-align:center; color:#8A7560; padding:40px;">Sin marcaciones registradas aún.</p>';
        return;
    }
    const traces = [{
        type: "bar", orientation: "h",
        y: ["Sin marcar", "Incorrecto", "Parcial", "Correcto"],
        x: [data.pct_sin_marcar_prom, data.pct_rojo_prom,
            data.pct_amarillo_prom, data.pct_verde_prom],
        marker: {
            color: ["#8A7560", COLORS.rojo, COLORS.amarillo, COLORS.verde],
        },
        text: [
            `${data.pct_sin_marcar_prom.toFixed(1)}%`,
            `${data.pct_rojo_prom.toFixed(1)}%`,
            `${data.pct_amarillo_prom.toFixed(1)}%`,
            `${data.pct_verde_prom.toFixed(1)}%`,
        ],
        textposition: "auto",
    }];
    Plotly.newPlot("chart-marcaciones", traces, {
        ...COMMON_LAYOUT,
        xaxis: { ...COMMON_LAYOUT.xaxis, range: [0, 100], title: "% del texto de la nota" },
        showlegend: false,
        margin: { l: 100, r: 20, t: 10, b: 40 },
    }, { displayModeBar: false, responsive: true });
}
