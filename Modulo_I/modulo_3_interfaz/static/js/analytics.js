// =================================================================
// Dashboard de analytics
// =================================================================
function kpiCard(label, value, unit = "") {
    return `
        <div class="kpi">
            <div class="label">${label}</div>
            <div class="value">${value}<span class="unit">${unit}</span></div>
        </div>`;
}

function bar(pct, color) {
    return `
        <span class="stat-bar">
            <span class="fill ${color}" style="width: ${pct}%;"></span>
        </span>
        <span style="font-family: var(--font-mono); font-size: 12px;">${pct.toFixed(1)}%</span>`;
}

async function load() {
    const resp = await fetch("/api/analytics_data");
    const data = await resp.json();

    // KPIs
    const g = data.globales;
    document.getElementById("kpi-row").innerHTML =
        kpiCard("Cardiólogos registrados", g.n_medicos) +
        kpiCard("Evaluaciones completas", g.n_evaluaciones) +
        kpiCard("Ventanas calificadas", g.n_respuestas_ventana);

    // Concordancia por escala
    const tbE = document.querySelector("#tabla-escala tbody");
    ["beat", "rhythm"].forEach((escala) => {
        const d = data.por_escala[escala] || {};
        const v = d.verde || 0, a = d.amarillo || 0, r = d.rojo || 0;
        const total = (v + a + r);
        tbE.innerHTML += `
            <tr>
                <td><b>${escala === "beat" ? "LATIDO (250 muestras)" : "RITMO (1000 muestras)"}</b></td>
                <td>${bar(v, "verde")}</td>
                <td>${bar(a, "amarillo")}</td>
                <td>${bar(r, "rojo")}</td>
                <td style="color: var(--ink-soft);">≈ ${Math.round(total / (v + a + r || 1) * g.n_respuestas_ventana / 2)} ventanas</td>
            </tr>`;
    });

    // Por clase BEAT
    const tbCB = document.querySelector("#tabla-clase-beat tbody");
    Object.entries(data.por_clase_beat || {}).forEach(([cls, d]) => {
        tbCB.innerHTML += `
            <tr>
                <td><b>${cls}</b></td>
                <td>${bar(d.verde, "verde")}</td>
                <td>${bar(d.amarillo, "amarillo")}</td>
                <td>${bar(d.rojo, "rojo")}</td>
            </tr>`;
    });

    // Por clase RHYTHM
    const tbCR = document.querySelector("#tabla-clase-rhythm tbody");
    Object.entries(data.por_clase_rhythm || {}).forEach(([cls, d]) => {
        tbCR.innerHTML += `
            <tr>
                <td><b>${cls}</b></td>
                <td>${bar(d.verde, "verde")}</td>
                <td>${bar(d.amarillo, "amarillo")}</td>
                <td>${bar(d.rojo, "rojo")}</td>
            </tr>`;
    });

    // Backends
    const tbB = document.querySelector("#tabla-backend tbody");
    data.por_backend.forEach((b) => {
        const sd = b.semaforo_distribucion || {};
        tbB.innerHTML += `
            <tr>
                <td><b>${b.backend}</b></td>
                <td>${b.n_evaluaciones}</td>
                <td>${b.prom_exactitud ?? "—"}</td>
                <td>${b.prom_coherencia ?? "—"}</td>
                <td>${b.prom_utilidad ?? "—"}</td>
                <td>${bar(sd.verde || 0, "verde")}</td>
                <td>${bar(sd.amarillo || 0, "amarillo")}</td>
                <td>${bar(sd.rojo || 0, "rojo")}</td>
            </tr>`;
    });

    // Médicos
    const tbM = document.querySelector("#tabla-medicos tbody");
    data.medicos.forEach((m) => {
        tbM.innerHTML += `
            <tr>
                <td>${m.nombre_completo}</td>
                <td>${m.institucion || "—"}</td>
                <td>${m.especialidad || "—"}</td>
                <td>${m.anos_experiencia_ecg ?? "—"}</td>
                <td>${m.auto_eval_habilidad ?? "—"} / 5</td>
                <td style="font-family: var(--font-mono); font-size: 11px; color: var(--ink-soft);">${m.fecha_registro?.split(".")[0] || ""}</td>
            </tr>`;
    });
}

load();
