"""
Backend MOCK que produce notas clinicas plausibles sin llamar a una API.
Util para desarrollo, demos sin red, y para los datos de la primera ronda
de validacion mientras se decide el LLM definitivo.
"""
import time

from .base import LLMBackend, LLMResponse


class MockBackend(LLMBackend):
    name = "mock"
    model_id = "mock-v1"

    def generate(self, signal_json: dict) -> LLMResponse:
        t0 = time.time()
        s = signal_json["aggregate_summary"]
        meta = signal_json["signal_metadata"]

        # Build a note deterministically from the JSON
        sentences = []

        # 1. Patient & recording
        sentences.append(
            f"Registro ECG de paciente de aproximadamente {meta['patient_age_estimate']} "
            f"anos, sexo {meta['patient_sex']}, derivacion {meta['lead']}, "
            f"duracion {meta['duration_seconds']} segundos."
        )

        # 2. Rhythm finding
        rhythm = s["predominant_rhythm_class"]
        if rhythm == "AFIB":
            sentences.append(
                f"Se observa fibrilacion auricular predominante a lo largo del registro "
                f"({s['rhythm_class_distribution_pct'].get('AFIB', 0):.0f}% de las ventanas)."
            )
        elif "AFIB" in s["rhythm_class_distribution_pct"]:
            sentences.append(
                f"Ritmo sinusal predominante con episodios paroxisticos de "
                f"fibrilacion auricular detectados en "
                f"{s['rhythm_class_distribution_pct'].get('AFIB', 0):.0f}% de las ventanas."
            )
        else:
            sentences.append("Ritmo sinusal normal a lo largo de todo el registro.")

        # 3. Beat finding
        pac_pct = s["beat_class_distribution_pct"].get("PAC", 0)
        if pac_pct > 5:
            sentences.append(
                f"Se identifican contracciones auriculares prematuras (PAC) en "
                f"{pac_pct:.1f}% de los latidos analizados."
            )
        elif pac_pct > 0:
            sentences.append(
                f"PAC aisladas identificadas ({pac_pct:.1f}% de latidos)."
            )
        else:
            sentences.append("No se identifican latidos ectopicos en el registro.")

        # 4. Anomalies summary
        if s["anomalies_detected"]:
            sentences.append(
                f"Total de {len(s['anomalies_detected'])} eventos anomalos catalogados."
            )

        # 5. Escalation
        if s["global_escalation_required"]:
            sentences.append(
                "Confianza diagnostica reducida en al menos una ventana; "
                "se recomienda revision por cardiologo especialista."
            )

        text = " ".join(sentences)

        return LLMResponse(
            text=text,
            backend_name=self.name,
            model_id=self.model_id,
            latency_seconds=time.time() - t0,
            prompt_tokens=None,
            completion_tokens=None,
        )
